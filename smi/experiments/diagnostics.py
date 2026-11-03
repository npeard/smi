"""Why the q3_fiber networks score at the zero predictor: a linear-Gaussian bound.

``fiber_results.json``'s Fisher matrix models the fiber path as a constant
drift *rate* u over the shot (one nuisance) and the coupling as |kappa| = 1,
and its velocity is one constant offset v per shot. The simulator instead
random-walks p(t) per sample, fades |kappa(t)|, and drifts a per-color speckle
phase arg(kappa_c(t)), while the network must return v(t) per sample. This
module bounds that per-sample task: each channel is read as a
displacement-equivalent phase track (genie demodulation: fringe order known,
detector noise mapped to phase through the Fisher information), the tracks are
differenced to velocity units, and the per-frequency multichannel Wiener
(LMMSE) posterior of v gives the RMSE. Every component is stationary Gaussian
with its PSD estimated from the simulator, so it is the best *linear*
estimator; nonlinear unwrapping errors only add to it, and the nominal arm's
network-vs-bound gap calibrates how much.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from smi.data.frozen import load_or_create
from smi.experiments.evaluate import CheckpointEstimator, evaluate
from smi.experiments.grids import GRIDS
from smi.experiments.runner import expand_grid, trained_checkpoint
from smi.theory.bounds import phase_factor
from smi.theory.fiber import simulate_fiber
from smi.theory.forward import SimConfig, noise_std, simulate_batch

_CPU = torch.device('cpu')


@dataclass(frozen=True)
class Knockouts:
    """Which fiber impairments the bound keeps (each True keeps the effect)."""

    fading: bool
    speckle: bool
    path: bool


def _psd(x: np.ndarray) -> np.ndarray:
    """Periodogram |FFT|^2 / L averaged over every leading axis but the last."""
    spec = np.abs(np.fft.fft(x, axis=-1)) ** 2 / x.shape[-1]
    return spec.reshape(-1, x.shape[-1]).mean(0)


def wiener_rmse(
    s_v: np.ndarray,
    s_common: np.ndarray,
    s_channel: np.ndarray,
    sees_target: np.ndarray,
) -> np.ndarray:
    """Per-shot RMSE of the LMMSE velocity from K channel tracks.

    Channel k reads ``a_k v + w + e_k`` per frequency, with a_k = 1 if it
    sees the target (0 for a dichroic monitor), w common to every channel and
    e_k independent. Posterior variance S_v - S_v^2 a^T Sigma^-1 a with
    Sigma = S_v a a^T + S_w 1 1^T + diag(S_e).

    Args:
        s_v: [F] velocity PSD (um/ms)^2.
        s_common: [F] PSD of the common term w.
        s_channel: [B, K, F] per-shot, per-channel independent PSDs.
        sees_target: [K] 0/1 mask a.

    Returns:
        [B] RMSE (um/ms); Parseval: mean square = mean over F of the PSD.
    """
    a = sees_target.astype(float)
    ones = np.ones_like(a)
    sigma = (
        s_v[:, None, None] * np.outer(a, a)
        + s_common[:, None, None] * np.outer(ones, ones)
    )[None] + np.einsum('bkf,kj->bfkj', s_channel, np.eye(len(a)))
    gain = np.linalg.solve(sigma, np.broadcast_to(a, sigma.shape[:-1])[..., None])
    post = s_v[None] - s_v[None] ** 2 * (gain[..., 0] @ a)
    return np.sqrt(post.clip(min=0).mean(-1))


def fiber_bound(
    cfg: SimConfig, knock: Knockouts, n_shots: int, seed: int
) -> dict[str, float]:
    """Linear-Gaussian velocity RMSE bound of ``cfg`` with ``knock`` applied.

    Detector noise enters each channel as white displacement noise of variance
    (lambda / 4 pi)^2 sigma^2 / <V^2 |kappa|^2> / phase_factor(C, alpha) per
    sample (the per-shot Fisher-average, so optimistic in deep fades); the
    per-channel phase drift and the common path walk are white in velocity;
    the speckle phase and the velocity prior are estimated from simulated shots.

    Returns:
        ``rmse`` (mean over shots of the per-shot RMSE, as the records score)
        and ``zero_predictor`` on the same shots.
    """
    gen = torch.Generator().manual_seed(seed)
    batch = simulate_batch(dataclasses.replace(cfg, fiber=None), n_shots, _CPU, gen)
    velocity = batch.velocity.double().numpy()
    dt_ms = 1e3 / cfg.sample_rate
    length = velocity.shape[-1]
    lam = np.asarray(cfg.wavelengths_um)
    to_disp = lam / (4 * math.pi)  # rad -> um of target displacement
    omega = 2 * np.pi * np.fft.fftfreq(length)
    diff_gain = 2 * (1 - np.cos(omega)) / dt_ms**2  # |1 - e^{-i w}|^2 / dt^2
    vis_lo, vis_hi = cfg.visibility_range
    mean_v2 = (vis_lo**2 + vis_lo * vis_hi + vis_hi**2) / 3

    fib = cfg.fiber
    coupling2 = np.ones((n_shots, len(lam)))
    s_speckle = np.zeros((len(lam), length))
    s_common = np.zeros(length)
    monitors: tuple[float, ...] = ()
    if fib is not None:
        monitors = fib.monitor_wavelengths_um
        draw = simulate_fiber(
            dataclasses.replace(fib, path_drift_um_per_sqrt_s=0.0),
            cfg.wavelengths_um,
            n_shots,
            length,
            cfg.sample_rate,
            _CPU,
            gen,
        )
        kappa = draw.kappa.numpy().astype(np.complex128)
        if knock.fading:
            coupling2 = (np.abs(kappa) ** 2).mean(-1)
        if knock.speckle:
            psi = np.unwrap(np.angle(kappa), axis=-1) * to_disp[None, :, None]
            s_speckle = np.stack(
                [
                    _psd(np.diff(psi[:, k], axis=-1, prepend=psi[:, k, :1]) / dt_ms)
                    for k in range(len(lam))
                ]
            )
        if knock.path:
            step = fib.path_drift_um_per_sqrt_s / math.sqrt(cfg.sample_rate)
            s_common = np.full(length, (step / dt_ms) ** 2)
    drift_step = cfg.phase_drift_rad_per_sqrt_s / math.sqrt(cfg.sample_rate)
    s_drift = (to_disp * drift_step / dt_ms) ** 2
    sigma = noise_std(cfg.snr_db)
    # C scales with |kappa|; phase_factor at the shot-mean coupling.
    factor = np.vectorize(
        lambda k2: phase_factor(cfg.feedback_c * math.sqrt(k2), cfg.alpha)
    )(coupling2)
    noise_var = to_disp[None] ** 2 * sigma**2 / (mean_v2 * coupling2 * factor)
    s_channel = (
        noise_var[..., None] * diff_gain[None, None]
        + s_drift[None, :, None]
        + s_speckle[None]
    )
    sees = np.array([lam_k not in monitors for lam_k in cfg.wavelengths_um])
    s_v = _psd(velocity)
    rmse = wiener_rmse(s_v, s_common, s_channel, sees)
    zero = np.sqrt((velocity**2).mean(-1))
    return {'rmse': float(rmse.mean()), 'zero_predictor': float(zero.mean())}


def drift_free_probe(
    checkpoint: Path, cfg: SimConfig, n_shots: int, seed: int, device: str
) -> dict[str, float]:
    """Score ``checkpoint`` on ``cfg`` as trained, without path drift, and static.

    A network that learned fringes under fading improves when the drifts it
    cannot track are removed; one still at the zero predictor on a static
    fiber never learned to read the fringes at all.
    """
    est = CheckpointEstimator(checkpoint, device)
    fib = cfg.fiber
    assert fib is not None
    variants = {
        'as_trained': fib,
        'no_path_drift': dataclasses.replace(fib, path_drift_um_per_sqrt_s=0.0),
        'static_fiber': dataclasses.replace(
            fib, path_drift_um_per_sqrt_s=0.0, drift_rad_per_sqrt_s=0.0
        ),
    }
    out = {}
    for name, variant in variants.items():
        sim = dataclasses.replace(cfg, fiber=variant)
        gen = torch.Generator().manual_seed(seed)
        batch = simulate_batch(sim, n_shots, _CPU, gen)
        pred = est.predict_velocity(batch.signals, sim.wavelengths_um)
        out[name] = float(((pred - batch.velocity) ** 2).mean(-1).sqrt().mean())
    out['zero_predictor'] = float(batch.velocity.pow(2).mean(-1).sqrt().mean())
    return out


def per_shot_table(
    cells: dict[str, list[str]], runs_dir: Path, frozen_root: Path, device: str
) -> dict[str, dict[str, object]]:
    """Per-shot velocity RMSE of each listed cell on its grid's frozen set.

    ``cells`` maps a grid name to cell keys (empty list: every cell). Keyed
    by cell key; cells of different grids with one key share a fingerprint
    (``trained_checkpoint``), so the first grid listing a key wins.
    """
    table: dict[str, dict[str, object]] = {}
    for grid_name, keys in cells.items():
        grid = GRIDS[grid_name]
        for cell in expand_grid(grid):
            if (keys and cell.key not in keys) or cell.key in table:
                continue
            test = grid.test_set
            dataset = load_or_create(cell.sim, test.seed, test.n_shots, frozen_root)
            ckpt = trained_checkpoint(cell, grid_name, runs_dir)
            result = evaluate(
                CheckpointEstimator(ckpt, device),
                dataset,
                cell.wavelengths_um,
                test.n_shots,
                converged=True,
                train_seed=cell.seed,
            )
            rmse = result.per_shot['velocity_rmse']
            table[cell.key] = {
                'grid': grid_name,
                'dataset': dataset.stem,
                'velocity_rmse': [float(x) for x in rmse],
            }
    return table
