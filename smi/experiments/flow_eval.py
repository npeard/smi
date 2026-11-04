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
    sign_invariant_rmse,
    summarize_per_shot,
)
from smi.data.datasets import select_pd_keys
from smi.data.frozen import load_or_create
from smi.data.schema import VELOCITY_KEY, read_shots
from smi.experiments import grids
from smi.experiments.evaluate import CheckpointEstimator, evaluate
from smi.experiments.grid import EvalSet, Grid
from smi.experiments.runner import (
    Cell,
    cell_fingerprint,
    expand_grid,
    trained_cells,
    trained_checkpoint,
    training_config,
)
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


def _read(
    dataset: Path, wavelengths_um: tuple[float, ...], n_shots: int
) -> tuple[Tensor, Tensor]:
    """``[B, C, L]`` signals and ``[B, L]`` stored velocity of a frozen set."""
    with h5py.File(dataset, 'r') as f:
        keys = select_pd_keys(f, wavelengths_um)
    arrays, _ = read_shots(dataset, [*keys, VELOCITY_KEY], stop=n_shots)
    signals = np.stack([np.asarray(arrays[k], np.float32) for k in keys], axis=1)
    velocity = np.asarray(arrays[VELOCITY_KEY], np.float32)
    return torch.from_numpy(signals), torch.from_numpy(velocity)


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
    signals, truth = _read(dataset, wavelengths_um, n_shots)
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


def training_identity(cell: Cell, grid: Grid) -> dict[str, Any]:
    """What ``cell`` trained, test set excluded: seed-independent by design."""
    return _training_identity(cell_fingerprint(cell, grid, training_config(cell, grid)))


def _training_identity(fingerprint: dict[str, Any]) -> dict[str, Any]:
    blocks = fingerprint['fingerprint_blocks']
    return {k: v for k, v in blocks.items() if k != 'test_set'}


def tcan_cells(reference: Cell, reference_grid: Grid, runs_dir: Path) -> list[Member]:
    """Every TCAN-44k seed trained exactly as ``reference``, one per seed.

    Searched across all grids, because seeds of one sim are spread over grids
    (q1_channels, q1_r1_count/seeds, q2_r4_noise, q3_fiber). Only members
    whose resolved training config (budget, schedule, architecture, loss,
    sim) matches the reference's count, so the ensemble's spread is seed
    variation and not a mix of training budgets.
    """
    want = training_identity(reference, reference_grid)
    found: dict[int, Member] = {}
    for name, cell, fingerprint in trained_cells():
        ckpt = runs_dir / name / f'{cell.key}.ckpt'
        if (
            cell.family == FAMILY
            and cell.seed is not None
            and cell.seed not in found
            and cell.sim.hash() == reference.sim.hash()
            and ckpt.exists()
            and _training_identity(fingerprint) == want
        ):
            found[cell.seed] = Member(name, cell.key, cell.seed, ckpt)
    return [found[s] for s in sorted(found)]


def arm_cell(grid: Grid, sim_name: str) -> Cell:
    """The grid's one ``FAMILY`` cell on sim ``sim_name``."""
    (cell,) = [
        c for c in expand_grid(grid) if c.family == FAMILY and c.sim_name == sim_name
    ]
    return cell


def _velocity_rmse(
    est: Any, dataset: Path, wavelengths_um: tuple[float, ...], n_shots: int
) -> dict[str, float]:
    """``est``'s velocity RMSE summary, scored as the grid's records are."""
    result = evaluate(
        est, dataset, wavelengths_um, n_shots=n_shots, converged=True, train_seed=0
    )
    return result.record.metrics['velocity_rmse']


def _arm_setup(
    flow: FlowLitModule, grid_name: str, sim_name: str, frozen_root: Path, role: str
) -> tuple[EvalSet, Cell, Path]:
    """The grid's test set, its ``sim_name`` arm cell and that arm's frozen set.

    Raises:
        ValueError: If ``flow`` was not trained on the arm's sim.
    """
    grid = grids.grid(grid_name)
    test = grid.test_set
    cell = arm_cell(grid, sim_name)
    if flow.sim_config.hash() != cell.sim.hash():
        raise ValueError(f'the {role} was not trained on the {sim_name} sim')
    return test, cell, load_or_create(cell.sim, test.seed, test.n_shots, frozen_root)


def _moved(a: dict[str, float], b: dict[str, float]) -> float:
    """Relative change of the mean RMSE from ``b`` to ``a``."""
    return abs(a['mean'] - b['mean']) / b['mean']


def _sampling_check(
    by_draws: dict[int, dict[str, float]], long: Sampling, long_rmse: dict[str, float]
) -> dict[str, Any]:
    """Whether the posterior-mean RMSE has plateaued in draws and in steps."""
    draws_moved = _moved(by_draws[DENOISE_DRAWS[-2]], by_draws[DENOISE_DRAWS[-1]])
    steps_moved = _moved(by_draws[STEP_CHECK_DRAWS], long_rmse)
    return {
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
    }


def score_denoise(
    flow: FlowLitModule,
    grid_name: str,
    runs_dir: Path,
    frozen_root: Path,
    sampling: Sampling,
    device: str,
) -> dict[str, Any]:
    """Q3.3 on the grid's ``fiber`` frozen set; RMSEs in um/ms."""
    test, fiber, fiber_set = _arm_setup(
        flow, grid_name, 'fiber', frozen_root, 'denoiser'
    )
    nominal = arm_cell(grids.grid(grid_name), 'nominal')
    free_set = load_or_create(nominal.sim, test.seed, test.n_shots, frozen_root)
    free_tcan = CheckpointEstimator(
        trained_checkpoint(nominal, grid_name, runs_dir), device
    )
    fiber_tcan = CheckpointEstimator(
        trained_checkpoint(fiber, grid_name, runs_dir), device
    )
    wl = fiber.wavelengths_um

    def run(est: Any, dataset: Path) -> dict[str, float]:
        return _velocity_rmse(est, dataset, wl, test.n_shots)

    def denoised(
        s: Sampling,
        n_mean: int,
        prefixes: tuple[int, ...],
        cache: dict[str, dict[int, Tensor]],
    ) -> dict[str, float]:
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
    return {
        'test_set': {**dataclasses.asdict(test), 'dataset': fiber_set.stem},
        'sampling': dataclasses.asdict(sampling),
        'sampling_check': _sampling_check(by_draws, long, long_rmse),
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
    test, cell, dataset = _arm_setup(
        flow, grid_name, sim_name, frozen_root, 'posterior'
    )
    wl = cell.wavelengths_um
    est = PosteriorEstimator(flow, sampling, f'flow-posterior:{sim_name}')
    mean_rmse = _velocity_rmse(est, dataset, wl, test.n_shots)
    _, truth = _read(dataset, wl, test.n_shots)
    regressor = trained_checkpoint(cell, grid_name, runs_dir)
    out: dict[str, Any] = {
        'test_set': {**dataclasses.asdict(test), 'dataset': dataset.stem},
        'sampling': dataclasses.asdict(sampling),
        'posterior_mean_velocity_rmse': mean_rmse,
        **band_scores(torch.cat(est.draws, dim=1), truth),
        'best_regressor': {
            'cell': cell.key,
            'velocity_rmse': _velocity_rmse(
                CheckpointEstimator(regressor, device), dataset, wl, test.n_shots
            ),
        },
    }
    members = tcan_cells(cell, grids.grid(grid_name), runs_dir)
    if len(members) < ENSEMBLE_MIN:
        out['deep_ensemble'] = (
            f'unavailable: {len(members)} TCAN-44k seed(s) trained as {cell.key}'
        )
        return out
    out['deep_ensemble'] = {
        **ensemble_scores(
            [CheckpointEstimator(m.checkpoint, device) for m in members],
            dataset,
            wl,
            test.n_shots,
        ),
        'members': [{'grid': m.grid, 'cell': m.cell, 'seed': m.seed} for m in members],
    }
    return out


def magnitude_scores(draws: Tensor, truth: Tensor) -> dict[str, Any]:
    """Sign-blind scores of ``[S, B, L]`` velocity draws against ``[B, L]`` truth.

    Every draw is folded to ``|v|`` and scored against ``|truth|``: the point
    estimate is the posterior mean of ``|v|``, the band and CRPS are those of
    the folded draws. This asks whether the flow knows the speed even where it
    does not know the direction.
    """
    folded, target = draws.abs(), truth.abs()
    return {
        'speed_rmse': summarize_per_shot(
            per_shot_rmse(folded.mean(dim=0), target).numpy()
        ),
        'coverage': band_coverage(folded, target, BAND_LEVEL),
        'crps_um_per_ms': summarize_per_shot(
            crps_samples(folded, target).mean(dim=-1).numpy()
        ),
    }


def _segment_signs(v: Tensor, reference: Tensor, segment: int) -> Tensor:
    """``sign(v . reference)`` per ``segment``-sample block: ``[..., L // segment]``.

    A trailing partial block is dropped. Ties go to +1, as in ``best_sign``.
    """
    n = v.shape[-1] // segment
    dot = (v[..., : n * segment] * reference[..., : n * segment]).unflatten(
        -1, (n, segment)
    )
    return torch.where(dot.sum(dim=-1) >= 0, 1.0, -1.0)


def sign_structure(draws: Tensor, truth: Tensor, segment: int) -> dict[str, Any]:
    """Where the flow's sign ambiguity lives, for ``[S, B, L]`` draws.

    * ``draw_sign_agreement``: share of draws whose global sign (one per draw,
      by ``best_sign`` against truth) is right; 0.5 is a coin flip.
    * ``shot_majority_agreement``: share of shots whose majority draw sign is
      right, the per-shot sign a vote over draws would report.
    * ``segment_flip_fraction``: share of ``segment``-sample blocks whose sign
      against truth differs from their own draw's global sign. Near zero means
      each draw commits to one sign for the whole shot (a global ambiguity,
      fixable by one sign per draw); large means the sign wanders within a
      draw and needs a per-segment fix.
    * ``sample_mass_right_sign``: per sample with ``|truth|`` above its shot's
      median ``|truth|`` (where the sign is not lost in a zero crossing), the
      share of draws with the true sign; with ``sample_mass_bimodal`` the
      share of those samples whose draws put at least ``SIGN_SPLIT_MIN`` of
      their mass on each sign, i.e. a +|v| / -|v| split explained by sign.
    """
    glob = torch.stack([best_sign(d, truth)[:, 0] for d in draws])  # [S, B]
    seg = _segment_signs(draws, truth, segment)  # [S, B, n]
    flips = (seg != glob[..., None]).double().mean()
    moving = truth.abs() > truth.abs().median(dim=-1, keepdim=True).values
    right = (torch.sign(draws) == torch.sign(truth)).double().mean(dim=0)  # [B, L]
    right = right[moving]
    return {
        'draw_sign_agreement': float((glob > 0).double().mean()),
        'shot_majority_agreement': float((glob.mean(dim=0) > 0).double().mean()),
        'segment': segment,
        'segment_flip_fraction': float(flips),
        'sample_mass_right_sign': float(right.mean()),
        'sample_mass_bimodal': float(
            ((right >= SIGN_SPLIT_MIN) & (right <= 1 - SIGN_SPLIT_MIN)).double().mean()
        ),
    }


def align_signs(draws: Tensor, reference: Tensor, segment: int | None) -> Tensor:
    """Flip each draw to agree in sign with ``reference`` ``[B, L]``.

    ``segment`` None flips each draw's whole shot by one sign; an integer
    flips each ``segment``-sample block on its own (a trailing partial block
    takes its preceding block's sign). The result is a hybrid estimator:
    the flow's draws with the direction taken from ``reference``, typically a
    regressor's point estimate, not a property of the flow.
    """
    if segment is None:
        return torch.stack([best_sign(d, reference) * d for d in draws])
    signs = _segment_signs(draws, reference, segment).repeat_interleave(segment, -1)
    pad = draws.shape[-1] - signs.shape[-1]
    signs = torch.cat([signs, signs[..., -1:].expand(*signs.shape[:-1], pad)], -1)
    return signs * draws


def signed_scores(draws: Tensor, truth: Tensor) -> dict[str, Any]:
    """Posterior-mean velocity RMSE, coverage and CRPS of signed draws."""
    return {
        'posterior_mean_velocity_rmse': summarize_per_shot(
            per_shot_rmse(draws.mean(dim=0), truth).numpy()
        ),
        'coverage': band_coverage(draws, truth, BAND_LEVEL),
        'crps_um_per_ms': summarize_per_shot(
            crps_samples(draws, truth).mean(dim=-1).numpy()
        ),
    }


# Sign-structure blocks: 1024 samples is the coarsest that still resolves a
# flip within a shot (16 per 16384-sample shot); 256 and 4096 bracket it.
SIGN_SEGMENTS = (256, 1024, 4096)
# The hybrid's per-segment sign block, scored beside the per-shot sign: the
# middle of SIGN_SEGMENTS, at which the oracle (true-sign) hybrid is lowest.
ALIGN_SEGMENT = 1024


# Shots used by conditioning_sensitivity: the ratio is a mean over many
# samples per shot, so a few shots settle it.
SENSITIVITY_SHOTS = 4


def conditioning_sensitivity(
    flow: FlowLitModule, signals: Tensor, n_steps: int, seed: int
) -> float:
    """RMS change of draws when ``signals`` is zeroed, over the draws' RMS.

    The same initial noise is integrated under the real and the zeroed
    conditioning, so the ratio is what the field takes from the signals: near
    zero means the flow samples its prior and ignores the measurement.
    """
    out = []
    for cond in (signals, torch.zeros_like(signals)):
        generator = torch.Generator(flow.device).manual_seed(seed)
        with torch.no_grad():
            out.append(flow.sample(cond.to(flow.device), 2, n_steps, generator).cpu())
    real, zeroed = out
    return float((real - zeroed).pow(2).mean().sqrt() / real.pow(2).mean().sqrt())


def _predict(est: Any, signals: Tensor, wavelengths_um: tuple[float, ...]) -> Tensor:
    return torch.cat(
        [est.predict_velocity(s, wavelengths_um) for s in signals.split(16)]
    )


def score_posterior_sign(
    flow: FlowLitModule,
    grid_name: str,
    sim_name: str,
    runs_dir: Path,
    frozen_root: Path,
    sampling: Sampling,
    device: str,
) -> dict[str, Any]:
    """Sign-blind scores, sign structure and the regressor-signed hybrid.

    The posterior flows put half their mass on each sign (``score_posterior``'s
    ``sign_split_fraction``), so their mean is near zero. This separates how
    well they know the speed from how well they know the direction, and
    scores the simplest fix: each draw's sign taken from the arm's TCAN
    regressor, per shot and per ``ALIGN_SEGMENT`` block. The regressors'
    own speed RMSE is the yardstick for the first.
    """
    test, cell, dataset = _arm_setup(
        flow, grid_name, sim_name, frozen_root, 'posterior'
    )
    wl = cell.wavelengths_um
    signals, truth = _read(dataset, wl, test.n_shots)
    s = sampling
    generator = torch.Generator(flow.device).manual_seed(s.seed)
    with torch.no_grad():
        draws = torch.cat(
            [
                _draws(flow, b, s.n_samples, s.n_steps, s.per_call, generator)[:, :, 0]
                for b in signals.split(16)
            ],
            dim=1,
        )
    tcan = _predict(
        CheckpointEstimator(trained_checkpoint(cell, grid_name, runs_dir), device),
        signals,
        wl,
    )

    def speed(pred: Tensor) -> dict[str, float]:
        return summarize_per_shot(per_shot_rmse(pred.abs(), truth.abs()).numpy())

    regressors: dict[str, Any] = {
        'tcan': {
            'cell': cell.key,
            'speed_rmse': speed(tcan),
            'velocity_rmse_sign_invariant': summarize_per_shot(
                sign_invariant_rmse(tcan, truth).numpy()
            ),
            'shot_sign_agreement': float((best_sign(tcan, truth) > 0).double().mean()),
        }
    }
    members = tcan_cells(cell, grids.grid(grid_name), runs_dir)
    if len(members) >= ENSEMBLE_MIN:
        mean = torch.stack(
            [
                _predict(CheckpointEstimator(m.checkpoint, device), signals, wl)
                for m in members
            ]
        ).mean(dim=0)
        regressors['deep_ensemble_mean'] = {
            'n_members': len(members),
            'speed_rmse': speed(mean),
            'velocity_rmse_sign_invariant': summarize_per_shot(
                sign_invariant_rmse(mean, truth).numpy()
            ),
        }
    speed_mean = draws.abs().mean(dim=0).double().numpy()
    return {
        'test_set': {**dataclasses.asdict(test), 'dataset': dataset.stem},
        'sampling': dataclasses.asdict(sampling),
        'conditioning_sensitivity': conditioning_sensitivity(
            flow, signals[:SENSITIVITY_SHOTS], s.n_steps, s.seed
        ),
        # Mean over shots of corr(posterior mean |v|, |truth|) along time.
        'speed_correlation': float(
            np.mean(
                [
                    np.corrcoef(m, a)[0, 1]
                    for m, a in zip(
                        speed_mean, truth.abs().double().numpy(), strict=True
                    )
                ]
            )
        ),
        'magnitude': magnitude_scores(draws, truth),
        'sign_structure': [sign_structure(draws, truth, n) for n in SIGN_SEGMENTS],
        'regressors': regressors,
        # Oracle: signs from the truth, the ceiling of any sign fix.
        'oracle_signed': {
            'per_shot': signed_scores(align_signs(draws, truth, None), truth),
            f'per_{ALIGN_SEGMENT}': signed_scores(
                align_signs(draws, truth, ALIGN_SEGMENT), truth
            ),
        },
        'tcan_signed_hybrid': {
            'per_shot': signed_scores(align_signs(draws, tcan, None), truth),
            f'per_{ALIGN_SEGMENT}': signed_scores(
                align_signs(draws, tcan, ALIGN_SEGMENT), truth
            ),
        },
    }


class EncoderEstimator:
    """The posterior flow's encoder regression head alone, as a regressor.

    Same backbone, inputs and full-shot training as the flow, but one
    deterministic pass and no sampling: scoring it separates what the
    whole-shot backbone gives from what the flow's draws add.
    """

    causal = False

    def __init__(self, flow: FlowLitModule, name: str):
        if flow.flow.encoder is None:
            raise ValueError('this flow has no conditioning encoder')
        self.flow = flow
        self.name = name
        self.model_type = 'flow-encoder-head'
        self.receptive_field = getattr(flow.flow.encoder, 'receptive_field', None)
        self.total_params = sum(p.numel() for p in flow.flow.encoder.parameters())
        self.device = str(flow.device)
        self.config: dict[str, Any] = {}

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """``[B, L]`` velocity from the encoder's first channel."""
        _check_wavelengths(self.flow, wavelengths_um)
        with torch.no_grad():
            guess = self.flow.encoder_guess(signals.to(self.flow.device))
        return guess[:, 0].cpu()


# Spectral share used by high_frequency_fraction: the band holding 99% of the
# truth's power is where it has structure at all.
TRUTH_POWER_QUANTILE = 0.99


def high_frequency_fraction(x: Tensor, truth: Tensor) -> float:
    """Power share of ``x`` ``[L]`` above the bin holding 99% of ``truth``'s.

    A posterior mean that tracked detail the signals cannot carry would put
    more power above that cutoff than the truth does, not less.
    """
    power = torch.fft.rfft(truth.double()).abs().pow(2)
    cut = int(torch.searchsorted(power.cumsum(0) / power.sum(), TRUTH_POWER_QUANTILE))
    px = torch.fft.rfft(x.double()).abs().pow(2)
    return float(px[cut + 1 :].sum() / px.sum())


# Fresh frozen set for the memorization check: any seed the training stream
# (2^62 per-step seeds from the run seed) and the norm stats (seed 0) do not
# single out; the grid's own test set is seed 1000.
FRESH_TEST_SEED = 2000
AUDIT_SHOTS = 3


def _module_facts(est: Any, crop: int | None) -> dict[str, Any]:
    return {
        'total_params': est.total_params,
        'receptive_field': est.receptive_field,
        'causal': est.causal,
        'train_crop_length': crop,
    }


def score_audit(
    flow: FlowLitModule,
    grid_name: str,
    sim_name: str,
    runs_dir: Path,
    frozen_root: Path,
    sampling: Sampling,
    device: str,
) -> dict[str, Any]:
    """Leakage, comparator and sanity checks of one encoder posterior flow.

    * The encoder head alone (:class:`EncoderEstimator`) through ``evaluate``
      on the arm's test set and on a fresh seed, and the flow's posterior
      mean on both at ``sampling`` (a memorized test set would score worse
      on the fresh one).
    * Size, receptive field and training crop of the flow and the arm's
      regressors (TCAN and UNetTransformer-offline cells of the grid).
    * The first ``AUDIT_SHOTS`` test shots: posterior mean, encoder head and
      TCAN RMSE, posterior std, and the high-frequency power share of the
      truth, the mean and draw 0.
    """
    test, cell, dataset = _arm_setup(flow, grid_name, sim_name, frozen_root, 'audit')
    wl = cell.wavelengths_um
    fresh = load_or_create(cell.sim, FRESH_TEST_SEED, test.n_shots, frozen_root)
    head = EncoderEstimator(flow, f'flow-encoder:{sim_name}')
    flow_facts = {
        'total_params': sum(p.numel() for p in flow.flow.parameters()),
        'encoder_params': head.total_params,
        'field_params': sum(p.numel() for p in flow.flow.velocity.parameters()),
        'receptive_field': head.receptive_field,
        'causal': False,
        'train_crop_length': flow.train_crop_length,
    }
    grid = grids.grid(grid_name)
    regressors = {}
    for c in expand_grid(grid):
        if c.sim_name == sim_name:
            est = CheckpointEstimator(
                trained_checkpoint(c, grid_name, runs_dir), device
            )
            crop = est.module.hparams.get('train_crop_length')
            regressors[c.key] = _module_facts(est, crop)
    tcan = CheckpointEstimator(trained_checkpoint(cell, grid_name, runs_dir), device)

    signals, truth = _read(dataset, wl, AUDIT_SHOTS)
    generator = torch.Generator(flow.device).manual_seed(sampling.seed)
    s = sampling
    draws = _draws(flow, signals, s.n_samples, s.n_steps, s.per_call, generator)
    draws = draws[:, :, 0]
    mean, std = draws.mean(dim=0), draws.std(dim=0)
    guess = head.predict_velocity(signals, wl)
    tcan_pred = tcan.predict_velocity(signals, wl)
    shots = [
        {
            'truth_std': float(truth[i].std()),
            'posterior_mean_rmse': float(
                per_shot_rmse(mean[i : i + 1], truth[i : i + 1])
            ),
            'encoder_head_rmse': float(
                per_shot_rmse(guess[i : i + 1], truth[i : i + 1])
            ),
            'tcan_rmse': float(per_shot_rmse(tcan_pred[i : i + 1], truth[i : i + 1])),
            'posterior_std_mean': float(std[i].mean()),
            'hf_fraction_truth': high_frequency_fraction(truth[i], truth[i]),
            'hf_fraction_posterior_mean': high_frequency_fraction(mean[i], truth[i]),
            'hf_fraction_draw0': high_frequency_fraction(draws[0, i], truth[i]),
        }
        for i in range(AUDIT_SHOTS)
    ]
    return {
        'test_set': {**dataclasses.asdict(test), 'dataset': dataset.stem},
        'fresh_set': {'seed': FRESH_TEST_SEED, 'dataset': fresh.stem},
        'sampling': dataclasses.asdict(sampling),
        'encoder_head_velocity_rmse': {
            'test': _velocity_rmse(head, dataset, wl, test.n_shots),
            'fresh': _velocity_rmse(head, fresh, wl, test.n_shots),
        },
        # Both sets at this sampling, so their gap is the set and not the
        # draw count.
        'posterior_mean_velocity_rmse': {
            name: _velocity_rmse(
                PosteriorEstimator(flow, sampling, f'flow-posterior:{sim_name}'),
                path,
                wl,
                test.n_shots,
            )
            for name, path in (('test', dataset), ('fresh', fresh))
        },
        'flow': flow_facts,
        'regressors': regressors,
        'example_shots': shots,
    }
