"""R8 scoring: the flow-matching denoiser (Q3.3) and posterior (Q3.4).

Both are scored on the q3_fiber grid's frozen sets against its TCAN-44k
cells, through :func:`~smi.experiments.evaluate.evaluate`, so every RMSE here
is the same statistic as the grid's own records:

* Q3.3: the free-space (``nominal``) TCAN on posterior-mean denoised fiber
  traces, against the ``fiber`` TCAN on the raw fiber traces. The free-space
  TCAN on raw fiber traces (no denoiser) and on the free-space set bracket it.
* Q3.4: the posterior's mean is the point estimate; its draws give 90% band
  coverage, CRPS, the fraction of multimodal per-sample marginals and the
  fraction of shots whose draws split in global sign. A deep ensemble of every
  trained TCAN-44k seed on the same sim, when there are at least two, is the
  uncertainty baseline (Gaussian predictive from its mean and spread).

The prediction these test (data-processing argument, continuous broadening)
is in the R8 plan; it is not restated here.
"""

import dataclasses
import importlib
import pkgutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from torch import Tensor

from smi.core.metrics import (
    band_coverage,
    best_sign,
    crps_gaussian,
    crps_samples,
    per_shot_rmse,
    summarize_per_shot,
)
from smi.data.datasets import select_pd_keys
from smi.data.frozen import load_or_create
from smi.data.schema import VELOCITY_KEY
from smi.experiments import grids
from smi.experiments.evaluate import CheckpointEstimator, evaluate
from smi.experiments.grid import Grid
from smi.experiments.runner import Cell, expand_grid
from smi.learning.flow_lit_module import FlowLitModule
from smi.theory.forward import SimConfig

BAND_LEVEL = 0.9
FAMILY = 'tcan-44k'
# Every 64th sample: 256 marginals per shot is enough for a fraction, and the
# GMM fits are the cost.
MODALITY_STRIDE = 64
# A shot is sign-split when at least this share of its draws has the minority
# global sign relative to draw 0.
SIGN_SPLIT_MIN = 0.1


def multimodal_fraction(draws: np.ndarray) -> float:
    """Share of columns of ``draws`` ``[S, N]`` a 2-component GMM fits better.

    "Better" is a lower BIC than one Gaussian, which charges the mixture its
    three extra parameters, so sampling noise in S draws is not a mode.
    """
    multi = 0
    for column in draws.T:
        x = column[:, None]
        bic = [GaussianMixture(k, random_state=0).fit(x).bic(x) for k in (1, 2)]
        multi += bic[1] < bic[0]
    return multi / draws.shape[1]


def _draws(
    module: FlowLitModule,
    cond: Tensor,
    n_samples: int,
    n_steps: int,
    per_call: int,
    generator: torch.Generator,
) -> Tensor:
    """``[S, B, C, L]`` draws, ``per_call`` at a time to bound activation memory."""
    cond = cond.to(module.device)
    chunks = []
    for start in range(0, n_samples, per_call):
        n = min(per_call, n_samples - start)
        chunks.append(module.sample(cond, n, n_steps, generator).cpu())
    return torch.cat(chunks)


def _check_wavelengths(module: FlowLitModule, wavelengths_um: tuple[float, ...]):
    if tuple(module.sim_config.wavelengths_um) != tuple(wavelengths_um):
        raise ValueError(
            f'flow trained on {module.sim_config.wavelengths_um} um, asked for '
            f'{wavelengths_um} um'
        )


@dataclasses.dataclass
class Sampling:
    """How draws are taken; recorded with the results."""

    n_samples: int
    n_steps: int
    per_call: int
    seed: int


class DenoisedEstimator:
    """A regressor applied to the denoiser's posterior-mean trace."""

    causal = False
    receptive_field = None

    def __init__(
        self, flow: FlowLitModule, regressor: CheckpointEstimator, sampling: Sampling
    ):
        self.flow = flow
        self.regressor = regressor
        self.sampling = sampling
        self.generator = torch.Generator(flow.device).manual_seed(sampling.seed)
        self.name = f'denoised:{regressor.name}'
        self.model_type = f'flow+{regressor.model_type}'
        self.total_params = regressor.total_params + sum(
            p.numel() for p in flow.parameters()
        )
        self.device = regressor.device
        self.config = {'regressor': regressor.config, **dataclasses.asdict(sampling)}

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """Posterior-mean free-space trace of ``signals``, then the regressor."""
        _check_wavelengths(self.flow, wavelengths_um)
        s = self.sampling
        draws = _draws(
            self.flow, signals, s.n_samples, s.n_steps, s.per_call, self.generator
        )
        return self.regressor.predict_velocity(draws.mean(dim=0), wavelengths_um)


class PosteriorEstimator:
    """The velocity posterior's mean, keeping every draw for the band scores."""

    causal = False
    receptive_field = None

    def __init__(self, flow: FlowLitModule, sampling: Sampling, name: str):
        self.flow = flow
        self.sampling = sampling
        self.generator = torch.Generator(flow.device).manual_seed(sampling.seed)
        self.name = name
        self.model_type = 'flow-posterior'
        self.total_params = sum(p.numel() for p in flow.parameters())
        self.device = str(flow.device)
        self.config = dataclasses.asdict(sampling)
        self.draws: list[Tensor] = []

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """Mean over draws; the ``[S, b, L]`` draws are appended to ``draws``."""
        _check_wavelengths(self.flow, wavelengths_um)
        s = self.sampling
        draws = _draws(
            self.flow, signals, s.n_samples, s.n_steps, s.per_call, self.generator
        )[:, :, 0]
        self.draws.append(draws)
        return draws.mean(dim=0)


def _velocity(dataset: Path, n_shots: int) -> Tensor:
    with h5py.File(dataset, 'r') as f:
        return torch.from_numpy(np.asarray(f[VELOCITY_KEY][:n_shots], np.float32))


def _signals(dataset: Path, wavelengths_um: tuple[float, ...], n_shots: int) -> Tensor:
    with h5py.File(dataset, 'r') as f:
        keys = select_pd_keys(f, wavelengths_um)
        return torch.from_numpy(
            np.stack([np.asarray(f[k][:n_shots], np.float32) for k in keys], axis=1)
        )


def band_scores(draws: Tensor, truth: Tensor) -> dict[str, Any]:
    """Coverage, CRPS, modality and sign split of ``[S, B, L]`` draws."""
    crps = crps_samples(draws, truth).mean(dim=-1).numpy()
    # Global sign of each draw against draw 0 of its shot.
    signs = torch.stack([best_sign(d, draws[0]) for d in draws])  # [S, B]
    minority = (signs < 0).double().mean(dim=0)
    minority = torch.minimum(minority, 1 - minority)
    sub = draws[:, :, ::MODALITY_STRIDE].reshape(draws.shape[0], -1).double().numpy()
    return {
        'band_level': BAND_LEVEL,
        'coverage': band_coverage(draws, truth, BAND_LEVEL),
        'crps_um_per_ms': summarize_per_shot(crps),
        'multimodal_fraction': multimodal_fraction(sub),
        'modality_stride': MODALITY_STRIDE,
        'sign_split_fraction': float((minority >= SIGN_SPLIT_MIN).double().mean()),
        'sign_split_min': SIGN_SPLIT_MIN,
    }


def ensemble_scores(
    members: list[CheckpointEstimator],
    dataset: Path,
    wavelengths_um: tuple[float, ...],
    n_shots: int,
) -> dict[str, Any]:
    """Deep-ensemble baseline: Gaussian predictive from the members' spread."""
    signals = _signals(dataset, wavelengths_um, n_shots)
    truth = _velocity(dataset, n_shots)
    preds = torch.stack(
        [
            torch.cat(
                [m.predict_velocity(s, wavelengths_um) for s in signals.split(16)]
            )
            for m in members
        ]
    )
    mean, std = preds.mean(dim=0), preds.std(dim=0)
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.5 + BAND_LEVEL / 2))
    inside = (truth - mean).abs() <= z * std
    return {
        'members': [m.name for m in members],
        'coverage': float(inside.double().mean()),
        'crps_um_per_ms': summarize_per_shot(
            crps_gaussian(mean, std, truth).mean(dim=-1).numpy()
        ),
        'velocity_rmse': summarize_per_shot(per_shot_rmse(mean, truth).numpy()),
    }


def tcan_cells(sim: SimConfig, runs_dir: Path) -> list[tuple[Cell, Path]]:
    """Every TCAN-44k cell on ``sim`` with a checkpoint, one per seed.

    Searched across all grids, because seeds of one sim are spread over grids
    (q1_channels, q1_r1_count/seeds, q2_r4_noise, q3_fiber).
    """
    found: dict[int, tuple[Cell, Path]] = {}
    want = sim.hash()
    for info in sorted(pkgutil.iter_modules(grids.__path__), key=lambda m: m.name):
        for cell in expand_grid(_grid(info.name)):
            ckpt = runs_dir / info.name / f'{cell.key}.ckpt'
            if (
                cell.family == FAMILY
                and cell.seed is not None
                and cell.seed not in found
                and cell.sim.hash() == want
                and ckpt.exists()
            ):
                found[cell.seed] = (cell, ckpt)
    return [found[s] for s in sorted(found)]


def _cell(grid: Grid, sim_name: str) -> Cell:
    (cell,) = [
        c for c in expand_grid(grid) if c.family == FAMILY and c.sim_name == sim_name
    ]
    return cell


def _rmse(record: Any) -> dict[str, float]:
    return record.metrics['velocity_rmse']


def _grid(name: str) -> Grid:
    return importlib.import_module(f'{grids.__name__}.{name}').GRID


def score_denoise(
    flow: FlowLitModule,
    grid_name: str,
    runs_dir: Path,
    frozen_root: Path,
    sampling: Sampling,
    device: str,
) -> dict[str, Any]:
    """Q3.3 on the grid's ``fiber`` frozen set; RMSEs in um/ms."""
    grid = _grid(grid_name)
    grid_runs = runs_dir / grid_name
    test = grid.test_set
    nominal, fiber = _cell(grid, 'nominal'), _cell(grid, 'fiber')
    if flow.sim_config.hash() != fiber.sim.hash():
        raise ValueError('the denoiser was not trained on the fiber arm sim')
    fiber_set = load_or_create(fiber.sim, test.seed, test.n_shots, frozen_root)
    free_set = load_or_create(nominal.sim, test.seed, test.n_shots, frozen_root)
    free_tcan = CheckpointEstimator(grid_runs / f'{nominal.key}.ckpt', device)
    fiber_tcan = CheckpointEstimator(grid_runs / f'{fiber.key}.ckpt', device)
    wl = fiber.wavelengths_um

    def run(est: Any, dataset: Path) -> dict[str, float]:
        return _rmse(evaluate(est, dataset, wl, test.n_shots, True, 0).record)

    return {
        'test_set': {**dataclasses.asdict(test), 'dataset': fiber_set.stem},
        'sampling': dataclasses.asdict(sampling),
        'velocity_rmse': {
            'free_tcan_on_denoised_fiber': run(
                DenoisedEstimator(flow, free_tcan, sampling), fiber_set
            ),
            'fiber_tcan_on_fiber': run(fiber_tcan, fiber_set),
            'free_tcan_on_fiber': run(free_tcan, fiber_set),
            'free_tcan_on_free_space': run(free_tcan, free_set),
        },
        'regressors': {'free': nominal.key, 'fiber': fiber.key},
    }


def score_posterior(
    flow: FlowLitModule,
    grid_name: str,
    sim_name: str,
    runs_dir: Path,
    frozen_root: Path,
    sampling: Sampling,
    device: str,
) -> dict[str, Any]:
    """Q3.4 for one arm (``nominal`` or ``fiber``) of the grid."""
    grid = _grid(grid_name)
    test = grid.test_set
    cell = _cell(grid, sim_name)
    if flow.sim_config.hash() != cell.sim.hash():
        raise ValueError(f'the posterior was not trained on the {sim_name} sim')
    dataset = load_or_create(cell.sim, test.seed, test.n_shots, frozen_root)
    wl = cell.wavelengths_um
    est = PosteriorEstimator(flow, sampling, f'flow-posterior:{sim_name}')
    record = evaluate(est, dataset, wl, test.n_shots, True, 0).record
    out: dict[str, Any] = {
        'test_set': {**dataclasses.asdict(test), 'dataset': dataset.stem},
        'sampling': dataclasses.asdict(sampling),
        'posterior_mean_velocity_rmse': _rmse(record),
        **band_scores(torch.cat(est.draws, dim=1), _velocity(dataset, test.n_shots)),
    }
    members = tcan_cells(cell.sim, runs_dir)
    regressor = runs_dir / grid_name / f'{cell.key}.ckpt'
    out['best_regressor'] = {
        'cell': cell.key,
        'velocity_rmse': _rmse(
            evaluate(
                CheckpointEstimator(regressor, device),
                dataset,
                wl,
                test.n_shots,
                True,
                0,
            ).record
        ),
    }
    out['deep_ensemble'] = (
        ensemble_scores(
            [CheckpointEstimator(p, device) for _, p in members],
            dataset,
            wl,
            test.n_shots,
        )
        if len(members) >= 2
        else f'unavailable: {len(members)} trained TCAN-44k seed(s) on this sim'
    )
    return out
