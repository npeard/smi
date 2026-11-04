"""Why the q3_fiber networks score at the zero predictor.

Two kinds of evidence. A *linear-Gaussian perfectly-demodulated reference
RMSE* (:func:`fiber_reference`) and direct checks on simulated shots
(:func:`classical_static_check`, :func:`rescale_probe`).

``fiber_results.json``'s Fisher matrix models the fiber path as a constant
drift *rate* u over the shot (one nuisance) and the coupling as |kappa| = 1,
and its velocity is one constant offset v per shot. The simulator instead
random-walks p(t) per sample, fades |kappa(t)|, and drifts a per-color speckle
phase arg(kappa_c(t)), while the network must return v(t) per sample. This
module scores that per-sample task: each channel is read as a
displacement-equivalent phase track (genie demodulation: fringe order known,
detector noise mapped to phase through the Fisher information), the tracks are
differenced to velocity units, and the per-frequency multichannel Wiener
(LMMSE) posterior of v gives the RMSE.

It is a reference, not a bound in either direction:

- perfect demodulation (fringe order known) is optimistic;
- linear, stationary, Gaussian estimation is pessimistic for non-Gaussian
  speckle and the non-stationary chirp drive, and the velocity prior is the
  simulator's own (white dp/dt against the 1 Hz-1 kHz drive band), so the
  arm-to-arm separations rest on that prior;
- fading enters only through the shot-averaged |kappa|^2, so it is blind to
  in-shot fades and cycle slips by construction (and to P_perp, hence
  fiber-pbs equals fiber).

Arm comparisons are posterior-MSE ratios to the free-space reference, not
Fisher-information ratios.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from smi.core.metrics import per_shot_rmse
from smi.data.frozen import load_or_create
from smi.data.schema import VELOCITY_KEY
from smi.experiments.evaluate import (
    DISPLACEMENT_LIMIT_MARGIN,
    CheckpointEstimator,
    InverseFitEstimator,
    evaluate,
)
from smi.experiments.grids import GRIDS
from smi.experiments.runner import expand_grid, trained_checkpoint
from smi.theory.bounds import phase_factor
from smi.theory.fiber import simulate_fiber
from smi.theory.forward import SimConfig, noise_std, simulate_batch

_CPU = torch.device('cpu')


@dataclass(frozen=True)
class Knockouts:
    """Which fiber impairments the reference keeps (each True keeps the effect)."""

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


def fiber_reference(
    cfg: SimConfig, knock: Knockouts, n_shots: int, seed: int
) -> dict[str, float]:
    """Linear-Gaussian perfectly-demodulated reference RMSE of ``cfg``.

    Caveats in the module docstring: neither an upper nor a lower bound.

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
        out[name] = _rmse(pred, batch.velocity)
    out['zero_predictor'] = _rmse(torch.zeros_like(batch.velocity), batch.velocity)
    return out


def static_fiber(cfg: SimConfig) -> SimConfig:
    """``cfg`` with its fiber frozen: no path drift and no speckle drift."""
    assert cfg.fiber is not None
    fib = dataclasses.replace(
        cfg.fiber, path_drift_um_per_sqrt_s=0.0, drift_rad_per_sqrt_s=0.0
    )
    return dataclasses.replace(cfg, fiber=fib)


def channel_stats(
    cfg: SimConfig, n_shots: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pooled per-channel mean and std ``[C]`` of signals drawn from ``cfg``."""
    gen = torch.Generator().manual_seed(seed)
    sig = simulate_batch(cfg, n_shots, _CPU, gen).signals
    return sig.mean((0, 2)), sig.std((0, 2))


def rescale_to(
    signals: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Per shot and channel, standardize then map onto ``mean`` / ``std`` ``[C]``.

    Undoes a shot's overall coupling |kappa| (which scales its fringe std) so
    every shot presents the training distribution's per-channel level.
    """
    mu = signals.mean(-1, keepdim=True)
    sd = signals.std(-1, keepdim=True).clamp_min(1e-12)
    return (signals - mu) / sd * std[:, None] + mean[:, None]


def _rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean over shots of the per-shot RMSE, as the records score."""
    return float(per_shot_rmse(pred, true).mean())


def classical_static_check(cfg: SimConfig, n_shots: int, seed: int) -> dict[str, float]:
    """Inverse-fit (Viterbi) velocity RMSE on ``cfg``'s shots, vs zero predictor.

    ``inverse_fit`` normalizes each channel's fringe amplitude itself, so a
    static fiber's fixed |kappa| does not hurt it; a score well under the zero
    predictor means the shots carry the velocity.
    """
    gen = torch.Generator().manual_seed(seed)
    batch = simulate_batch(cfg, n_shots, _CPU, gen)
    disp = batch.displacement.double()
    excursion = (disp - disp.mean(-1, keepdim=True)).abs().max()
    est = InverseFitEstimator(
        cfg.sample_rate, float(excursion) * DISPLACEMENT_LIMIT_MARGIN
    )
    pred = est.predict_velocity(batch.signals, cfg.wavelengths_um)
    return {
        'inverse_fit': _rmse(pred, batch.velocity),
        'zero_predictor': _rmse(torch.zeros_like(batch.velocity), batch.velocity),
    }


def rescale_probe(
    checkpoint: Path,
    cfg: SimConfig,
    stats: tuple[torch.Tensor, torch.Tensor],
    n_shots: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    """Score ``checkpoint`` on ``cfg``'s shots raw and rescaled onto ``stats``."""
    est = CheckpointEstimator(checkpoint, device)
    gen = torch.Generator().manual_seed(seed)
    batch = simulate_batch(cfg, n_shots, _CPU, gen)
    scaled = rescale_to(batch.signals, *stats)
    return {
        'raw': _rmse(
            est.predict_velocity(batch.signals, cfg.wavelengths_um), batch.velocity
        ),
        'rescaled': _rmse(
            est.predict_velocity(scaled, cfg.wavelengths_um), batch.velocity
        ),
    }


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


def coupling_stats(cfg: SimConfig, n_shots: int, seed: int) -> dict[str, float]:
    """How ``cfg``'s fiber coupling kappa scales the fringes, from simulated draws.

    Fringe modulation and C both scale with |kappa|, so the fringe SNR drops by
    E|kappa|^2 (in power) and C_eff = C |kappa|. ``in_shot_fade_rel`` is the
    mean within-shot std of |kappa| over its mean (0 for a static fiber: a
    fixed scale a network or a per-shot normalization absorbs);
    ``speckle_phase_rms_um`` the within-shot std of the speckle phase
    arg(kappa) (path drift excluded) as target-displacement equivalent at the
    first color, the nuisance the velocity read must reject.
    ``path_rms_um`` is on the same footing: the expected within-shot std of
    the Brownian path walk about its shot mean, drift * sqrt(T / 6) (its RMS
    from the shot start would be sqrt(T / 2)), since a constant path offset is
    a phase a per-shot read absorbs.
    """
    fib = cfg.fiber
    assert fib is not None
    gen = torch.Generator().manual_seed(seed)
    length = cfg.acquisition.buffer_size
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
    mag = np.abs(kappa)
    psi = np.unwrap(np.angle(kappa[:, 0]), axis=-1)
    mean_k2 = float((mag**2).mean())
    return {
        'mean_kappa2': mean_k2,
        'mean_kappa': float(mag.mean()),
        'feedback_c_eff': float(cfg.feedback_c * mag.mean()),
        'snr_loss_db': float(-10 * np.log10(mean_k2)),
        'in_shot_fade_rel': float((mag.std(-1) / mag.mean(-1)).mean()),
        'speckle_phase_rms_um': float(
            psi.std(-1).mean() * cfg.wavelengths_um[0] / (4 * math.pi)
        ),
        'path_rms_um': fib.path_drift_um_per_sqrt_s
        * math.sqrt(length / cfg.sample_rate / 6),
    }


def shot_speed(path: Path) -> np.ndarray:
    """Per-shot velocity RMS of a frozen set; its mean is the zero predictor's RMSE."""
    with h5py.File(path, 'r') as f:
        return np.sqrt((f[VELOCITY_KEY][:] ** 2).mean(-1))


def explained_fraction(rmse: float, zero_rmse: float) -> float:
    """Share of the zero predictor's MSE an estimator removes: 1 - (rmse / zero)^2.

    Puts runs with different error floors (noise-free C sweep, nominal SNR
    ladder, fiber sweep) on one scale, so a knob's cost is compared as the
    share of learned signal it destroys rather than as a raw RMSE step.
    """
    return 1.0 - (rmse / zero_rmse) ** 2


def matched_retention(curve: dict[float, float], ref: float, at: float) -> float:
    """``curve`` (knob -> explained fraction) at ``at`` over its value at ``ref``.

    Linear in the knob between measured rungs; the prediction a hypothesis makes
    for a fiber point whose only change is that knob moving from ``ref`` to
    ``at``.
    """
    xs = sorted(curve)
    ys = [curve[x] for x in xs]
    return float(np.interp(at, xs, ys) / np.interp(ref, xs, ys))


def per_shot_excess(
    base: np.ndarray, faded: np.ndarray, speed: np.ndarray, amplitude: np.ndarray
) -> dict[str, float]:
    """Where a knob's extra error lands across paired shots.

    ``base``/``faded`` are per-shot RMSEs on shots sharing their motion,
    ``speed`` each shot's velocity RMS, ``amplitude`` each shot's fringe
    amplitude relative to free space. A shot-SNR (or shot-C_eff) mechanism puts
    the excess MSE on the faded shots (correlation with amplitude < 0); a
    collapse of fast-motion estimation puts it on the fast shots.
    """
    excess = faded**2 - base**2
    return {
        'corr_excess_mse_vs_speed': float(np.corrcoef(excess, speed)[0, 1]),
        'corr_excess_mse_vs_amplitude': float(np.corrcoef(excess, amplitude)[0, 1]),
        'median_rmse_base': float(np.median(base)),
        'median_rmse_faded': float(np.median(faded)),
        'mean_rmse_base': float(base.mean()),
        'mean_rmse_faded': float(faded.mean()),
    }


def colour_fading_verdict(
    three_m1: float, three_m8: float, three_m8_corr: float, one_m1: float, one_m8: float
) -> dict[str, float | str]:
    """H4 of the fiber sweep: whether its fading cost is colours fading apart.

    Arguments are explained fractions (``explained_fraction``): three colours at
    M = 1, at M = 8 fading independently and at M = 8 sharing one speckle
    draw, and 0.635 um alone at M = 1 and 8. H4 predicts the correlated arm
    recovers most of the M = 1 -> 8 loss and a lone colour keeps far more of
    its own signal than three independent ones. The thresholds (half the loss
    recovered; twice the three-colour retention) were fixed before the run.
    """
    recovery = (three_m8_corr - three_m8) / (three_m1 - three_m8)
    three, one = three_m8 / three_m1, one_m8 / one_m1
    votes = (recovery > 0.5) + (one > 2 * three)
    return {
        'corr_recovery': recovery,
        'three_colour_retention': three,
        'three_colour_corr_retention': three_m8_corr / three_m1,
        'one_colour_retention': one,
        'verdict': ('rejected', 'mixed', 'supported')[votes],
    }
