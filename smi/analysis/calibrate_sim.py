#!/usr/bin/env python3
"""Fit the simulator's detector knobs to a file in the real-data HDF5 schema.

Written in P1 and validated ONLY by a synthetic round trip (a frozen set made
with known knobs, ``tests/test_calibrate_sim.py``): the program is blinded to
the real acquisitions until P6, so this must not be run on them before then.

Every estimator reads only what an acquisition holds -- the drive voltage
``RP1_CH1``, the photodiode channels and the ``sample_rate`` attribute -- and
derives displacement from the voltage through ``CoilDriver`` exactly as the
baselines do. The frozen sets' ground-truth datasets are never read.

Units. The simulator works in units of the nominal DC light level with that
level subtracted; frozen sets store exactly that (``units='signal'``). An
acquisition stores volts from a DC-coupled photodiode (``units='volts'``), so
each channel is divided by its mean level and 1 subtracted. That treats the
whole mean as nominal light: a DC offset (ambient light, electronic offset) is
absorbed into the scale and is not separately identifiable from volts.

Knob by knob (None in the output means "no estimate", never zero):

* ``drive_gain``: ratio of the file's mean per-shot drive RMS to the
  simulator's at unit gain -- the drive voltage is recorded directly.
* ``snr_db``: white noise floor of the PD spectrum between
  ``NOISE_BAND_LOW`` x fs (and ``NOISE_BAND_FRINGE_MULTIPLE`` x each used
  shot's peak fringe rate) and ``NOISE_BAND_TOP`` x fs, after subtracting the
  fringe's random-walk phase skirt predicted from the fitted phase drift and
  fringe power (it is comparable to the noise there at the nominal drift).
* ``phase_drift_rad_per_sqrt_s``: increments of the windowed fringe phase
  (``baseline_fit.window_fringe_coefficients``, the fit inside
  ``measure_phase_drift``) between adjacent windows: 2/3 D Delta for a random
  walk averaged over windows of length Delta, read through
  E[cos dphi] = exp(-var / 2) so wrapping does not bias it. Increments rather
  than ``measure_phase_drift``'s peak-to-peak, one extreme-value statistic per
  shot that under-reads at large drift. A known PD lag is removed first.
* ``feedback_c`` and ``alpha``: second fringe harmonic relative to the first,
  in the phase the drive and the windowed fit give. To first order
  ``H2 / H1 = -(C / 2) exp(i atan(alpha))``, inverted by a table built from
  ``forward.smi_excess_phase``. The phase wandering inside a window blurs H2
  faster than H1, so the magnitude is extrapolated to zero window length from
  windows W and 2W. Because the drive fixes the phase, C is identifiable even
  at alpha = 0 (the d -> -d symmetry there concerns decoding without a drive);
  alpha comes from the harmonic's phase and needs C > 0.
* ``pd_bandwidth_hz``: the roll-off of windowed fringe amplitude with fringe
  rate when fringe rates reach the bandwidth; otherwise the knee the PD puts
  in the phase skirt of the noise band. None when neither shows.
* ``visibility_range``: per-shot windowed fringe amplitude, corrected for the
  SMI fundamental, the PD and the within-window coherence loss (same W / 2W
  extrapolation), moment-matched to U(lo, hi) after removing estimation
  scatter.
* ``amplitude_drift_rel``, ``contrast_drift_rel``: None. No estimator meets
  the bar at the published drift: the windowed-DC increment statistic reads
  many times the truth at the nominal phase drift and the windowed
  log-amplitude increment statistic reads far above it too (neither is a
  committed measurement), because the within-window phase drift
  (D, rad/sqrt(s)) leaks into both. Both are near-unbiased only
  as D -> 0, which the published 1-2.7 fringes per shot rules out.
* ``adc_bits``: the board constant, not estimated. ``adc_full_scale``: the
  board's input range over the channel's DC level in volts; None for
  ``units='signal'`` files, which carry no volts.
* ``dc_offset_range``: None (absorbed into the scale, see Units).
  ``shot_noise``: None, not estimated.

Known limits of the round trip (256 shots, 100 kHz PD, other knobs nominal;
each is an expected-failure test in ``tests/test_calibrate_sim.py`` with the
measured error): at D = 58 rad/sqrt(s) the within-window drift blurs the
harmonics, so ``feedback_c`` reads up to -32%, ``alpha`` up to +23% and the PD
knee up to +59%; at C = 0.4 the PD knee reads +19% to +59%; at snr_db = 20
the noise-band knee is buried and ``pd_bandwidth_hz`` is None. Every other
knob holds the bar at those points, and all knobs hold it at drive_gain x2,
snr_db 40, D = 21.5 and alpha = 5.
"""

import argparse
import functools
import json
import logging
import math
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from scipy.signal import periodogram

from smi.analysis.baseline_fit import (
    DRIVE_KEY,
    fit_window_basis,
    window_fringe_coefficients,
)
from smi.analysis.datasets import file_pd_channels
from smi.redpitaya.redpitaya_config import RedPitayaConfig
from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.forward import (
    NOMINAL_DRIVE_BAND_HZ,
    fringe,
    one_pole_coefficient,
    sample_trajectory,
    smi_excess_phase,
)
from smi.synthetic.waveform import Waveform

logger = logging.getLogger(__name__)

Units = Literal['signal', 'volts']

# measure_phase_drift's window: short enough that the phase is near-constant
# inside it at the published drift, long enough to sweep about a fringe.
WINDOW_SAMPLES = 512
# A window whose model phase sweeps less than this many fringes leaves cos and
# sin nearly collinear, so its amplitude and phase are noise; it is skipped.
MIN_WINDOW_FRINGES = 1.0
# The noise band starts this many times above the shot's peak fringe rate, so
# SMI harmonics (relative amplitude ~C^(n-1)) are negligible against it, and
# no lower than NOISE_BAND_LOW x fs, where the fringe's offset from the skirt
# formula's zero carrier is a small fraction of f.
NOISE_BAND_FRINGE_MULTIPLE = 4.0
NOISE_BAND_LOW = 0.1
# Upper edge of the noise band as a fraction of the sample rate (Nyquist 0.5),
# clear of the Hann window's edge effects.
NOISE_BAND_TOP = 0.45
# Shots to estimate the unit-gain drive RMS from: the per-shot U(0, 1) power
# draw makes the mean RMS a ~1/sqrt(n) estimate (2048 -> ~1.3%).
DRIVE_REFERENCE_SHOTS = 2048
# The phase-increment variance is read from the wrapped-Gaussian identity
# E[cos dphi] = exp(-var / 2), which wrapping does not bias; below this mean
# cosine (var > 3.2 rad^2) the estimate is too noisy and the drift is None.
MIN_MEAN_COS_INCREMENT = 0.2
# The amplitude roll-off is accepted only when the best bandwidth explains at
# least this fraction of the log-amplitude variance more than no roll-off
# does and lies inside the searched grid.
MIN_ROLLOFF_GAIN = 0.05
# The noise-band knee likewise must cut the band's squared residual by this
# fraction against an unfiltered skirt.
MIN_KNEE_GAIN = 0.1
PD_BANDWIDTH_GRID_HZ = np.geomspace(1e3, 2e5, 121)
# The knee is searched up to just below Nyquist (pd_lowpass's own limit).
KNEE_GRID_HZ = np.geomspace(1e4, 2.4e5, 101)


def _shot_arrays(
    path: Path, n_shots: int
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray, float]:
    """PD signals and wavelengths by key, drive voltage and sample rate."""
    with h5py.File(path, 'r') as f:
        sample_rate = float(f.attrs['sample_rate'])
        wavelengths = file_pd_channels(f)
        available = int(f[DRIVE_KEY].shape[0])
        if n_shots > available:
            raise ValueError(f'{path} holds {available} shots, asked for {n_shots}')
        voltage = f[DRIVE_KEY][:n_shots].astype(np.float64)
        signals = {k: f[k][:n_shots].astype(np.float64) for k in wavelengths}
    return signals, wavelengths, voltage, sample_rate


def _drive_gain(voltage: np.ndarray, sample_rate: float) -> float:
    """File drive RMS over the simulator's at unit gain.

    Assumes the rig's 1-1000 Hz drive band (``NOMINAL_DRIVE_BAND_HZ``): the
    acquisition schema records no band, so a file driven over another band
    would read a wrong gain.
    """
    waveform = Waveform(*NOMINAL_DRIVE_BAND_HZ)
    if waveform.acq_sample_rate != sample_rate:
        raise ValueError(
            f'sample_rate {sample_rate} does not match the rig drive grid '
            f'({waveform.acq_sample_rate} Hz)'
        )
    reference, _, _ = sample_trajectory(
        DRIVE_REFERENCE_SHOTS,
        waveform,
        CoilDriver(),
        1.0,
        sample_rate,
        torch.device('cpu'),
        torch.Generator().manual_seed(0),
    )
    ref_rms = reference.double().pow(2).mean(-1).sqrt().mean().item()
    return float(np.sqrt((voltage**2).mean(-1)).mean() / ref_rms)


def _displacement(voltage: np.ndarray, sample_rate: float) -> np.ndarray:
    """Drive-implied displacement (um), as the baselines derive it."""
    _, disp = CoilDriver().get_velocity_and_displacement_torch(
        torch.from_numpy(voltage).float(), sample_rate
    )
    return disp.double().numpy()


@functools.cache
def _harmonic_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``H2 / H1`` of ``forward.fringe`` over a (C, alpha) grid."""
    c_grid = np.linspace(0.0, 0.95, 191)
    alpha_grid = np.concatenate([np.linspace(0.0, 10.0, 201), [20.0, 50.0]])
    # 256 phase points: harmonics above ~100 (relative size C^n) cannot alias
    # onto the first two at any C in the table.
    phase = torch.linspace(0, 2 * math.pi, 257, dtype=torch.float64)[:-1]
    # fringe(phase, C, alpha) solves on psi = phase + atan(alpha), so one
    # alpha = 0 solve on a per-alpha shifted grid covers every alpha at once.
    beta = torch.from_numpy(np.arctan(alpha_grid))[:, None]
    ratio = np.empty((c_grid.size, alpha_grid.size), dtype=np.complex128)
    for i, c in enumerate(c_grid):
        delta, _ = smi_excess_phase(phase + beta, float(c), 0.0)
        spectrum = np.fft.fft(torch.cos(phase + delta).numpy(), axis=-1)
        ratio[i] = spectrum[:, 2] / spectrum[:, 1]
    return c_grid, alpha_grid, ratio


def _invert_harmonic(ratio: complex) -> tuple[float, float]:
    """Nearest (C, alpha) in the table to a measured ``H2 / H1``."""
    c_grid, alpha_grid, table = _harmonic_table()
    i, j = np.unravel_index(np.argmin(np.abs(table - ratio)), table.shape)
    return float(c_grid[i]), float(alpha_grid[j])


def _harmonic_basis(psi: np.ndarray) -> np.ndarray:
    """``[cos, sin, cos 2, sin 2, 1]`` columns for per-window phases."""
    return np.stack(
        [np.cos(psi), np.sin(psi), np.cos(2 * psi), np.sin(2 * psi), np.ones_like(psi)],
        axis=2,
    )


def _windows(x: np.ndarray, theta: np.ndarray, window: int) -> tuple[np.ndarray, ...]:
    """Per-window first and second fringe harmonics of one shot.

    The first harmonic's phase comes from ``window_fringe_coefficients`` (as
    in ``measure_phase_drift``); the refit in that phase adds the second
    harmonic, keeping the intercept.

    Returns:
        ``(ok, phase, h1, h2)`` per window; only windows whose model phase
        sweeps ``MIN_WINDOW_FRINGES`` are trusted (``ok``).
    """
    n_win = x.size // window
    coeffs = window_fringe_coefficients(x, theta, window)
    th = theta[: n_win * window].reshape(n_win, -1)
    ok = np.ptp(th, axis=1) / (2 * np.pi) >= MIN_WINDOW_FRINGES
    phase = np.arctan2(-coeffs[:, 1], coeffs[:, 0])
    psi = (th + phase[:, None]).ravel()
    c1, s1, c2, s2, _ = fit_window_basis(x, psi, window, _harmonic_basis).T
    return ok, phase, c1 - 1j * s1, c2 - 1j * s2


def _pooled_ratio(
    harmonics: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    pd_bandwidth_hz: float | None,
) -> complex:
    """``H2 / H1`` pooled over windows, weighted by ``|H1|^2``.

    A PD of known bandwidth passes the second harmonic at 2f as H(2f) while
    the fitted phase has absorbed H(f)'s lag, so each window's ratio is
    multiplied back by ``|H(f)| exp(2i arg H(f)) / H(2f)`` (signed fringe
    rate f). H is the analog one-pole here because the correction is only
    needed at fringe rates far below the sample rate, where it matches the
    simulator's discrete pole to second order in f / fs;
    ``_one_pole_power`` integrates to Nyquist, where they differ.
    """
    h1 = np.concatenate([h[0] for h in harmonics])
    h2 = np.concatenate([h[1] for h in harmonics])
    if pd_bandwidth_hz is not None:
        f = np.concatenate([h[2] for h in harmonics])
        resp1 = 1 / (1 + 1j * f / pd_bandwidth_hz)
        resp2 = 1 / (1 + 2j * f / pd_bandwidth_hz)
        h2 = h2 * np.abs(resp1) * np.exp(2j * np.angle(resp1)) / resp2
    return complex(np.sum(h2 * np.conj(h1)) / np.sum(np.abs(h1) ** 2))


def _one_pole_power(
    freqs: np.ndarray, cutoff_hz: float | None, fs: float
) -> np.ndarray:
    """|H(f)|^2 of ``forward.pd_lowpass``; 1 for an unlimited bandwidth.

    The discrete pole, not the analog one used in ``_pooled_ratio``: the noise
    floor model spans the whole band up to Nyquist, where the two differ.
    """
    if cutoff_hz is None:
        return np.ones_like(freqs)
    p = one_pole_coefficient(cutoff_hz, fs)
    return (1 - p) ** 2 / (1 - 2 * p * np.cos(2 * np.pi * freqs / fs) + p * p)


def _pd_rolloff(amps: list[np.ndarray], rates: list[np.ndarray]) -> float | None:
    """One-pole bandwidth from the roll-off of amplitude with fringe rate.

    Per shot the log amplitude is compared up to a constant, so visibility
    drops out; identifiable only when fringe rates reach a sizeable fraction
    of the bandwidth (the best fit must beat no roll-off by
    ``MIN_ROLLOFF_GAIN`` of the variance and lie inside the grid).
    """
    if not amps:
        return None

    def sse(fc: float) -> float:
        err = 0.0
        for a, f in zip(amps, rates, strict=True):
            model = np.log(a) + 0.5 * np.log1p((f / fc) ** 2)
            err += float(np.sum((model - model.mean()) ** 2))
        return err

    total = sse(math.inf)
    curve = np.array([sse(fc) for fc in PD_BANDWIDTH_GRID_HZ])
    best = int(np.argmin(curve))
    gain = (total - curve[best]) / total if total > 0 else 0.0
    if gain < MIN_ROLLOFF_GAIN or best in (0, curve.size - 1):
        return None
    return float(PD_BANDWIDTH_GRID_HZ[best])


def _noise_floor(
    x: np.ndarray,
    max_fringe_hz: np.ndarray,
    fringe_power: float,
    drift_var: float,
    rolloff_hz: float | None,
    fs: float,
) -> tuple[float | None, float | None]:
    """White-noise variance and PD bandwidth from the PSD above the fringes.

    The random-walk phase spreads the fringe into a skirt that still reaches
    the band: one-sided ``2 P D / (2 fs sin(pi f / fs))^2`` for fringe power P
    and phase diffusion D (rad^2/s), then low-passed by the PD while the noise
    is not (the simulator adds it after the PD). The band mean is fit as
    ``n0 + skirt |H(f; fc)|^2``: ``n0`` is the noise and, when the skirt is
    large enough to show its knee, ``fc`` the PD bandwidth. A bandwidth
    already found from the amplitude roll-off is held fixed.

    Returns:
        ``(noise variance or None, knee bandwidth or None)``.
    """
    use = NOISE_BAND_FRINGE_MULTIPLE * max_fringe_hz <= NOISE_BAND_LOW * fs
    if not use.any():
        return None, None
    f_lo = NOISE_BAND_LOW * fs
    xs = x[use] - x[use].mean(axis=-1, keepdims=True)
    freqs, psd = periodogram(xs, fs, window='hann', axis=-1)
    band = (freqs >= f_lo) & (freqs <= NOISE_BAND_TOP * fs)
    f, level = freqs[band], psd[:, band].mean(axis=0)
    skirt = 2 * fringe_power * drift_var / (2 * fs * np.sin(np.pi * f / fs)) ** 2

    def fit(fc: float | None) -> tuple[float, float]:
        rest = level - skirt * _one_pole_power(f, fc, fs)
        n0 = float(rest.mean())
        return n0, float(np.sum((rest - n0) ** 2))

    if rolloff_hz is not None:
        n0, _ = fit(rolloff_hz)
        knee = None
    else:
        n0_flat, sse_flat = fit(None)
        fits = [fit(float(fc)) for fc in KNEE_GRID_HZ]
        best = int(np.argmin([e for _, e in fits]))
        n0, knee = n0_flat, None
        if fits[best][1] < (1 - MIN_KNEE_GAIN) * sse_flat and best < len(fits) - 1:
            n0, knee = fits[best][0], float(KNEE_GRID_HZ[best])
    # One-sided density of white noise of variance s2 is 2 s2 / fs.
    return (n0 * fs / 2 if n0 > 0 else None), knee


def _channel(
    x: np.ndarray, theta: np.ndarray, sample_rate: float
) -> dict[str, float | tuple[float, float] | None]:
    """Every per-channel estimate for signals ``x`` [n, L] in signal units."""
    n_shots, length = x.shape
    n_win = length // WINDOW_SAMPLES
    delta_s = WINDOW_SAMPLES / sample_rate
    signed_rate = np.gradient(theta, axis=-1) * sample_rate / (2 * np.pi)
    fringe_rate = np.abs(signed_rate)
    window_shape = (n_shots, n_win, WINDOW_SAMPLES)
    usable = n_win * WINDOW_SAMPLES
    window_rate = fringe_rate[:, :usable].reshape(window_shape).mean(-1)
    window_signed = signed_rate[:, :usable].reshape(window_shape).mean(-1)

    phases, pairs, amps, rate_w, h_short, h_long = [], [], [], [], [], []
    for s in range(n_shots):
        ok, phase, h1, h2 = _windows(x[s], theta[s], WINDOW_SAMPLES)
        phases.append(phase)
        pairs.append(ok[1:] & ok[:-1])
        if ok.sum() < 2:
            continue
        amps.append(np.abs(h1[ok]))
        rate_w.append(window_rate[s, ok])
        h_short.append((h1[ok], h2[ok], window_signed[s, ok]))
        ok2, _, g1, g2 = _windows(x[s], theta[s], 2 * WINDOW_SAMPLES)
        long_rate = signed_rate[s, : (n_win // 2) * 2 * WINDOW_SAMPLES]
        long_rate = long_rate.reshape(n_win // 2, -1).mean(-1)
        h_long.append((g1[ok2], g2[ok2], long_rate[ok2]))
    out: dict[str, float | tuple[float, float] | None] = dict.fromkeys(
        (
            'snr_db',
            'phase_drift_rad_per_sqrt_s',
            'feedback_c',
            'alpha',
            'pd_bandwidth_hz',
            'visibility_range',
        )
    )
    # No estimator meets the bar at the published phase drift (module doc).
    out['amplitude_drift_rel'] = out['contrast_drift_rel'] = None
    if not h_short:
        return out

    rolloff = _pd_rolloff(amps, rate_w)
    # A PD lags the fringe by atan(f / fc), which moves with the fringe rate
    # and would read as drift; remove it where the bandwidth is known.
    lag = np.zeros_like(window_signed)
    if rolloff is not None:
        lag = np.arctan(window_signed / rolloff)
    inc = np.concatenate(
        [np.diff(ph + lg)[pr] for ph, lg, pr in zip(phases, lag, pairs, strict=True)]
    )
    drift_var = 0.0
    mean_cos = float(np.mean(np.cos(inc)))
    if mean_cos > MIN_MEAN_COS_INCREMENT:
        # Window averages of a random walk: increments of variance 2/3 D Delta.
        drift_var = -2 * math.log(mean_cos) / (2 / 3 * delta_s)
        out['phase_drift_rad_per_sqrt_s'] = math.sqrt(drift_var)

    # The phase wandering inside a window costs both harmonics coherence in
    # proportion to the window length (a random walk's within-window variance
    # grows with it), so log-magnitudes at W and 2W extrapolate to W = 0.
    r_short = _pooled_ratio(h_short, rolloff)
    r_long = _pooled_ratio(h_long, rolloff)
    out['feedback_c'], out['alpha'] = _invert_harmonic(
        r_short * abs(r_short) / abs(r_long)
    )
    a_short = np.median(np.concatenate([np.abs(h[0]) for h in h_short]))
    a_long = np.median(np.concatenate([np.abs(h[0]) for h in h_long]))
    coherence = float(a_long / a_short)

    power = 0.5 * float(np.mean(np.concatenate(amps) ** 2)) / coherence**2
    sigma2, knee = _noise_floor(
        x, fringe_rate.max(-1), power, drift_var, rolloff, sample_rate
    )
    out['snr_db'] = None if sigma2 is None else 10 * math.log10(0.5 / sigma2)
    out['pd_bandwidth_hz'] = rolloff if rolloff is not None else knee
    out['visibility_range'] = _visibility_range(amps, rate_w, out, coherence)
    return out


def _visibility_range(
    amps: list[np.ndarray],
    rates: list[np.ndarray],
    est: dict[str, float | tuple[float, float] | None],
    coherence: float,
) -> tuple[float, float] | None:
    """U(lo, hi) whose mean and variance match the per-shot amplitudes."""
    if len(amps) < 2:
        return None
    fc = est['pd_bandwidth_hz']
    c, alpha = est['feedback_c'], est['alpha']
    # The SMI fundamental is not exactly the visibility; nor is an amplitude
    # attenuated by the PD or by the phase wandering inside the window.
    h1 = 1.0
    if isinstance(c, float) and isinstance(alpha, float) and c > 0:
        phase = torch.linspace(0, 2 * math.pi, 4097)[:-1]
        h1 = 2 * abs(np.fft.fft(fringe(phase, c, alpha).double().numpy())[1]) / 4096
    per_shot, scatter = [], []
    for a, f in zip(amps, rates, strict=True):
        corrected = a / (h1 * coherence)
        if isinstance(fc, float):
            corrected = corrected * np.sqrt(1 + (f / fc) ** 2)
        per_shot.append(np.median(corrected))
        # Median of n draws: variance ~ (pi / 2) var / n.
        scatter.append(np.pi / 2 * corrected.var() / corrected.size)
    m = float(np.mean(per_shot))
    var = max(float(np.var(per_shot)) - float(np.mean(scatter)), 0.0)
    half = math.sqrt(3 * var)
    return (max(m - half, 0.0), min(m + half, 1.0))


def calibrate(path: Path, n_shots: int, units: Units) -> dict:
    """Estimate every identifiable ``SimConfig`` knob from the first shots.

    Args:
        path: HDF5 file in the real-data schema.
        n_shots: Number of leading shots to use.
        units: ``'signal'`` for frozen synthetic sets (nominal-DC units, DC
            subtracted), ``'volts'`` for acquisitions. See the module doc.

    Returns:
        JSON-ready dict: file-level ``sample_rate``, ``drive_gain``,
        ``adc_bits``, and ``channels`` mapping each PD key to its wavelength
        and per-channel estimates (None where not identifiable).
    """
    signals, wavelengths, voltage, sample_rate = _shot_arrays(Path(path), n_shots)
    displacement = _displacement(voltage, sample_rate)
    channels = {}
    for key, lam in wavelengths.items():
        x = signals[key]
        full_scale = None
        if units == 'volts':
            dc_level = float(x.mean())
            x = x / dc_level - 1
            full_scale = RedPitayaConfig.INPUT_FULL_SCALE_LV_V / dc_level
        theta = 4 * np.pi / lam * displacement
        est = _channel(x, theta, sample_rate)
        channels[key] = {
            'wavelength_um': lam,
            **est,
            'adc_full_scale': full_scale,
            'dc_offset_range': None,
            'shot_noise': None,
        }
    return {
        'units': units,
        'n_shots': n_shots,
        'sample_rate': sample_rate,
        'drive_gain': _drive_gain(voltage, sample_rate),
        'adc_bits': RedPitayaConfig.ADC_BITS,
        'channels': channels,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI: write the calibration JSON. Do not run on real data before P6."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-shots', type=int, required=True)
    parser.add_argument('--units', choices=('signal', 'volts'), required=True)
    args = parser.parse_args(argv)
    result = calibrate(args.dataset, args.n_shots, args.units)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    logger.info('wrote %s', args.out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
