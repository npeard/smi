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
  TCAN-44k seed trained exactly as the arm's grid cell, when there are at
  least three, is the uncertainty baseline (Gaussian predictive from its mean
  and spread).

The prediction these test (data-processing argument, continuous broadening)
is in the R8 plan; it is not restated here.
"""

import dataclasses
import hashlib
import importlib
import pkgutil
from collections.abc import Iterator
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
from smi.experiments.runner import Cell, cell_fingerprint, expand_grid, training_config
from smi.learning.flow_lit_module import FlowLitModule

BAND_LEVEL = 0.9
FAMILY = 'tcan-44k'
# Every 64th sample: 256 marginals per shot is enough for a fraction, and the
# GMM fits are the cost.
MODALITY_STRIDE = 64
# A shot is sign-split when at least this share of its draws has the minority
# global sign relative to draw 0.
SIGN_SPLIT_MIN = 0.1
# Two seeds give a spread with one degree of freedom; three is the least
# that makes the Gaussian predictive's std worth scoring.
ENSEMBLE_MIN = 3
# Q3.3 sampling checks: posterior-mean RMSE at nested draw counts (prefixes of
# one draw set at the base step count), and at STEP_CHECK_DRAWS draws with
# twice the steps. A setting has plateaued when the next one moves the mean
# RMSE by less than PLATEAU_RTOL.
DENOISE_DRAWS = (8, 16, 32)
STEP_CHECK_DRAWS = 16
PLATEAU_RTOL = 0.01
# Two-mode criterion of multimodal_fraction. 10 BIC units is "very strong"
# evidence on the Kass-Raftery scale. An equal two-Gaussian mixture turns
# bimodal at 2 sigma, but at 64 draws a 2-sigma cut still flags ~12% of
# exponential marginals; 3 sigma brings exponential to ~3% and lognormal to
# ~1% while keeping a 6-sigma mode split at 100% (measured, 400 columns).
MODE_BIC_MARGIN = 10.0
MODE_MIN_WEIGHT = 0.1
MODE_SEPARATION = 3.0


def multimodal_fraction(draws: np.ndarray) -> float:
    """Share of columns of ``draws`` ``[S, N]`` with two separated modes.

    A 2-component GMM must beat one Gaussian by ``MODE_BIC_MARGIN`` in BIC,
    give its lighter component at least ``MODE_MIN_WEIGHT``, and put the means
    more than ``MODE_SEPARATION`` of the wider std apart. BIC alone flags
    nearly every skewed unimodal marginal (exponential, lognormal) at 64
    draws, because two Gaussians fit a skewed shape better than one; the
    weight and separation tests ask for what a mode split looks like.
    """
    multi = 0
    for column in draws.T:
        x = column[:, None]
        one, two = (GaussianMixture(k, random_state=0).fit(x) for k in (1, 2))
        mu = two.means_[:, 0]
        sigma = np.sqrt(two.covariances_.reshape(-1))
        multi += (
            two.bic(x) < one.bic(x) - MODE_BIC_MARGIN
            and two.weights_.min() >= MODE_MIN_WEIGHT
            and abs(mu[0] - mu[1]) > MODE_SEPARATION * sigma.max()
        )
    return multi / draws.shape[1]


def _draw_chunks(
    module: FlowLitModule,
    cond: Tensor,
    n_samples: int,
    n_steps: int,
    per_call: int,
    generator: torch.Generator,
) -> Iterator[Tensor]:
    """``[n, B, C, L]`` CPU chunks of S draws, ``per_call`` at a time.

    Chunking bounds activation memory; the draws do not depend on it, since
    each chunk continues ``generator``'s stream.
    """
    cond = cond.to(module.device)
    for start in range(0, n_samples, per_call):
        n = min(per_call, n_samples - start)
        yield module.sample(cond, n, n_steps, generator).cpu()


def _draws(
    module: FlowLitModule,
    cond: Tensor,
    n_samples: int,
    n_steps: int,
    per_call: int,
    generator: torch.Generator,
) -> Tensor:
    """``[S, B, C, L]`` draws."""
    return torch.cat(
        list(_draw_chunks(module, cond, n_samples, n_steps, per_call, generator))
    )


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
    """A regressor applied to the mean of the denoiser's first ``n_mean`` draws.

    The means of the first ``n`` draws for every ``n`` in ``prefixes`` are
    memoized per input batch in ``cache``, so estimators that share a cache
    (and so must share ``prefixes``) score nested subsets (8, 16, 32 draws)
    of one set of draws instead of paying for each separately. Only running
    sums are kept, not the draws: 32 draws of the 64-shot fiber set are
    ~400 MB.
    """

    causal = False
    receptive_field = None

    def __init__(
        self,
        flow: FlowLitModule,
        regressor: CheckpointEstimator,
        sampling: Sampling,
        n_mean: int,
        prefixes: tuple[int, ...],
        cache: dict[str, dict[int, Tensor]],
    ):
        if n_mean not in prefixes:
            raise ValueError(f'n_mean {n_mean} is not one of prefixes {prefixes}')
        if max(prefixes) > sampling.n_samples:
            raise ValueError(f'prefixes {prefixes} > n_samples {sampling.n_samples}')
        self.flow = flow
        self.regressor = regressor
        self.sampling = sampling
        self.n_mean = n_mean
        self.prefixes = prefixes
        self.cache = cache
        self.generator = torch.Generator(flow.device).manual_seed(sampling.seed)
        self.name = f'denoised:{regressor.name}'
        self.model_type = f'flow+{regressor.model_type}'
        self.total_params = regressor.total_params + sum(
            p.numel() for p in flow.parameters()
        )
        self.device = regressor.device
        self.config = {
            'regressor': regressor.config,
            **dataclasses.asdict(sampling),
            'n_mean': n_mean,
        }

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """Posterior-mean free-space trace of ``signals``, then the regressor."""
        _check_wavelengths(self.flow, wavelengths_um)
        key = hashlib.blake2b(signals.numpy().tobytes()).hexdigest()
        if key not in self.cache:
            s = self.sampling
            means, total, count = {}, None, 0
            for chunk in _draw_chunks(
                self.flow, signals, s.n_samples, s.n_steps, s.per_call, self.generator
            ):
                for draw in chunk:
                    total = draw.clone() if total is None else total + draw
                    count += 1
                    if count in self.prefixes:
                        means[count] = total / count
            self.cache[key] = means
        return self.regressor.predict_velocity(
            self.cache[key][self.n_mean], wavelengths_um
        )


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


@dataclasses.dataclass
class Member:
    """One deep-ensemble member: a grid cell's trained checkpoint."""

    grid: str
    cell: str
    seed: int
    checkpoint: Path


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
        'coverage': float(inside.double().mean()),
        'crps_um_per_ms': summarize_per_shot(
            crps_gaussian(mean, std, truth).mean(dim=-1).numpy()
        ),
        'velocity_rmse': summarize_per_shot(per_shot_rmse(mean, truth).numpy()),
    }


def _training(cell: Cell, grid: Grid) -> dict[str, Any]:
    """What ``cell`` trained, test set excluded: seed-independent by design."""
    blocks = cell_fingerprint(cell, grid, training_config(cell, grid))
    return {k: v for k, v in blocks['fingerprint_blocks'].items() if k != 'test_set'}


def tcan_cells(reference: Cell, reference_grid: Grid, runs_dir: Path) -> list[Member]:
    """Every TCAN-44k seed trained exactly as ``reference``, one per seed.

    Searched across all grids, because seeds of one sim are spread over grids
    (q1_channels, q1_r1_count/seeds, q2_r4_noise, q3_fiber). Only members
    whose resolved training config (budget, schedule, architecture, loss,
    sim) matches the reference's count, so the ensemble's spread is seed
    variation and not a mix of training budgets.
    """
    want = _training(reference, reference_grid)
    found: dict[int, Member] = {}
    for info in sorted(pkgutil.iter_modules(grids.__path__), key=lambda m: m.name):
        grid = _grid(info.name)
        for cell in expand_grid(grid):
            ckpt = runs_dir / info.name / f'{cell.key}.ckpt'
            if (
                cell.family == FAMILY
                and cell.seed is not None
                and cell.seed not in found
                and cell.sim.hash() == reference.sim.hash()
                and ckpt.exists()
                and _training(cell, grid) == want
            ):
                found[cell.seed] = Member(info.name, cell.key, cell.seed, ckpt)
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

    def denoised(
        s: Sampling,
        n_mean: int,
        prefixes: tuple[int, ...],
        cache: dict[str, dict[int, Tensor]],
    ) -> Any:
        est = DenoisedEstimator(flow, free_tcan, s, n_mean, prefixes, cache)
        return run(est, fiber_set)

    if sampling.n_samples != DENOISE_DRAWS[-1]:
        raise ValueError(f'denoise sampling must draw {DENOISE_DRAWS[-1]}')
    shared: dict[str, dict[int, Tensor]] = {}
    by_draws = {n: denoised(sampling, n, DENOISE_DRAWS, shared) for n in DENOISE_DRAWS}
    long = dataclasses.replace(
        sampling, n_samples=STEP_CHECK_DRAWS, n_steps=2 * sampling.n_steps
    )
    long_rmse = denoised(long, STEP_CHECK_DRAWS, (STEP_CHECK_DRAWS,), {})

    def moved(a: dict[str, float], b: dict[str, float]) -> float:
        return abs(a['mean'] - b['mean']) / b['mean']

    draws_moved = moved(by_draws[DENOISE_DRAWS[-2]], by_draws[DENOISE_DRAWS[-1]])
    steps_moved = moved(by_draws[STEP_CHECK_DRAWS], long_rmse)
    return {
        'test_set': {**dataclasses.asdict(test), 'dataset': fiber_set.stem},
        'sampling': dataclasses.asdict(sampling),
        'sampling_check': {
            'velocity_rmse_by_draws': {str(n): r for n, r in by_draws.items()},
            'step_check': {
                'n_samples': STEP_CHECK_DRAWS,
                'n_steps': long.n_steps,
                'velocity_rmse': long_rmse,
            },
            'draws_rel_change': draws_moved,
            'steps_rel_change': steps_moved,
            'plateau_rtol': PLATEAU_RTOL,
            'draws_plateau': draws_moved < PLATEAU_RTOL,
            'steps_plateau': steps_moved < PLATEAU_RTOL,
        },
        'velocity_rmse': {
            # The largest draw count: the plateau value when draws_plateau.
            'free_tcan_on_denoised_fiber': by_draws[DENOISE_DRAWS[-1]],
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
    members = tcan_cells(cell, grid, runs_dir)
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
        {
            **ensemble_scores(
                [CheckpointEstimator(m.checkpoint, device) for m in members],
                dataset,
                wl,
                test.n_shots,
            ),
            'members': [
                {'grid': m.grid, 'cell': m.cell, 'seed': m.seed} for m in members
            ],
        }
        if len(members) >= ENSEMBLE_MIN
        else f'unavailable: {len(members)} TCAN-44k seed(s) trained as {cell.key}'
    )
    return out
