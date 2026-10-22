#!/usr/bin/env python3
"""Non-deep-learning baselines for displacement/velocity estimation.

This module answers the "establishing baselines" question: before claiming a
neural network is doing something useful, how well does classical, parametric
interferometry do on the same data?

Two distinct measurements live here, and it matters which is which.

**Method 1 -- forward-model residual** (:func:`forward_model_residual`).
Question answered: *does the Michelson forward model explain this data at
all?* We take the displacement implied by the known speaker drive voltage
(via :class:`~smi.synthetic.coil_driver.CoilDriver`), push it through
``cos(4*pi/lambda * d(t) + phi)``, and fit only the per-shot phase offset and
amplitude. Because the model is linear in its ``cos``/``sin`` components once
the displacement is fixed, this is a closed-form least-squares -- there is no
iterative optimizer and no local minimum to worry about. The reported number
is a residual RMSE on the photodiode signal, in volts. This is the honest
floor: no inverse method can beat the forward model's own ability to describe
the measurement.

One global ``(cos, sin, 1)`` fit over the whole 33.6 ms record is however a
*rigid* estimator: real shots drift by 1-2.7 fringes within a record from
speckle and path drift, and a single phase cannot absorb that, so the global
R^2 understates the model's agreement with the data.
:func:`windowed_forward_model_residual` refits per window of W samples and
pools the variances into one comparable R^2 per shot. Both are reported. A
windowed R^2 must always be quoted beside :func:`scrambled_theta_control`,
which does the identical fit against a *different* shot's displacement at the
*same* W, and so measures the windowing's fitting capacity on its own.

W itself is fringe-matched (:func:`fringe_matched_window`): it is resolved per
shot per channel so every window spans about two fringes, because a fixed W
holds a third of a fringe on the quietest shots and a dozen on the loudest and
so does not mean the same thing across excursion bins. The artifact the
baseline report reads is generated this way, so one estimator backs every
table and figure in it.

**Method 2 -- inverse fit** (:func:`inverse_fit`).
Question answered: *how well can a classical estimator recover displacement
and velocity from the photodiode signals alone?* Only the three PD channels
are used; the drive voltage is held out and used purely as ground truth for
scoring. Displacement is recovered by multi-wavelength decoding across the
635 / 675 / 515 nm channels, then differentiated for velocity. The reported
numbers are displacement RMSE in microns and velocity RMSE in microns per
millisecond -- the same units the network is scored in. **This is the number
that belongs beside the network's RMSE.**

The pairing is the point. A bad method-2 number on its own is ambiguous
between "the inverse algorithm is bad" and "the data does not obey the
assumed physics". Method 1 disambiguates: if method 1 also fits badly, the
model does not describe the data and no inverse -- classical or learned --
can do better than that ceiling.

Under Gaussian residuals the per-shot MSE *is* the negative log-likelihood up
to an additive constant, so "compute the likelihood of the data under the
Michelson model" and "fit each shot and report MSE" are the same measurement.
That is why the simple thing here is also the rigorous thing.

Unit conventions follow the rest of the package: wavelength and displacement
in microns, velocity in microns per millisecond (um/ms), time in seconds,
photodiode signals in volts. Velocity is um/ms because that is what
:class:`~smi.synthetic.coil_driver.CoilDriver` emits and what the network is
trained and scored on; sample rates remain in Hz.

The two methods deliberately run at different shot counts: method 1 is
milliseconds per shot and runs at N = 1000, while method 2's GPU Viterbi
decode is ~5.85 s per shot and stays at N = 200. Both are recorded in the
artifact's ``config`` block; never quote one N over the other's table.

Run as a script to regenerate the results artifact -- the defaults are the
settings it was produced with::

    pixi run -e dev python -m smi.analysis.baseline_fit
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.signal import savgol_filter

from smi.synthetic.coil_driver import MS_PER_S, CoilDriver

logger = logging.getLogger(__name__)

# Photodiode channel -> laser wavelength in microns. RP1_CH1 is the speaker
# drive voltage and is not a photodiode.
PD_WAVELENGTHS_UM: dict[str, float] = {
    'RP1_CH2': 0.635,
    'RP2_CH1': 0.675,
    'RP2_CH2': 0.515,
}
DRIVE_KEY = 'RP1_CH1'

# Default location of the acquired datasets, relative to this file.
DATA_DIR = Path(__file__).resolve().parent / 'data'
DEFAULT_DATASETS: dict[str, str] = {
    'free-space': 'free-space-synchro_10k.h5',
    'mmfiber': 'mmfiber-synchro_10k.h5',
}
DEFAULT_RESULTS_PATH = DATA_DIR / 'baseline_results.json'


def _as_float64_zero_mean(signal: np.ndarray) -> np.ndarray:
    """Cast to float64 and remove the DC offset.

    The forward model produces a DC-free interference term (see
    ``MichelsonInterferometer.get_interferometer_output``), so the measured
    signal must be DC-removed before it can be compared to one.

    Args:
        signal: Signal samples of any float dtype.

    Returns:
        Zero-mean float64 copy of the input.
    """
    out = np.asarray(signal, dtype=np.float64)
    return out - out.mean()


# ---------------------------------------------------------------------------
# Method 1: forward-model residual
# ---------------------------------------------------------------------------


@dataclass
class PhaseAmplitudeFit:
    """Result of the closed-form phase/amplitude fit for one channel.

    Attributes:
        amplitude: Fitted fringe amplitude (V). Non-negative by construction.
        phase: Fitted phase offset (radians), wrapped to (-pi, pi].
        offset: Fitted constant offset (V); nominally zero for DC-removed data.
        rmse: Root-mean-square residual between fit and data (V).
        r_squared: Fraction of the signal variance explained by the fit.
    """

    amplitude: float
    phase: float
    offset: float
    rmse: float
    r_squared: float


def fit_phase_amplitude(signal: np.ndarray, theta: np.ndarray) -> PhaseAmplitudeFit:
    """Fit ``A * cos(theta + phi) + c`` to a signal in closed form.

    The model is nonlinear in ``phi`` but linear in the pair
    ``(A*cos(phi), -A*sin(phi))``, because::

        A * cos(theta + phi) = [A cos(phi)] * cos(theta) - [A sin(phi)] * sin(theta)

    So fitting reduces to an ordinary least-squares solve on the design matrix
    ``[cos(theta), sin(theta), 1]``, and the amplitude and phase are read back
    off the two coefficients. There is no iterative optimizer here and no
    local minimum: the least-squares solution is the global one.

    Args:
        signal: Measured samples (V), shape ``(n,)``.
        theta: Known model phase argument (radians), shape ``(n,)``. For the
            Michelson model this is ``4*pi/wavelength * displacement``.

    Returns:
        The fitted amplitude, phase, offset and residual statistics.

    Raises:
        ValueError: If the inputs do not have matching one-dimensional shapes.
    """
    signal = np.asarray(signal, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if signal.ndim != 1 or theta.ndim != 1 or signal.shape != theta.shape:
        raise ValueError(
            f'signal and theta must be 1-D with matching shape, '
            f'got {signal.shape} and {theta.shape}'
        )

    design = np.stack([np.cos(theta), np.sin(theta), np.ones_like(theta)], axis=1)
    coeffs, *_ = np.linalg.lstsq(design, signal, rcond=None)
    a_cos, b_sin, offset = (float(c) for c in coeffs)

    # signal ~ a*cos(theta) + b*sin(theta) = A*cos(theta + phi)
    # with A = hypot(a, b) and phi = atan2(-b, a).
    amplitude = float(np.hypot(a_cos, b_sin))
    phase = float(np.arctan2(-b_sin, a_cos))

    residual = signal - design @ coeffs
    rmse = float(np.sqrt(np.mean(residual**2)))
    variance = float(np.var(signal))
    r_squared = float(1.0 - np.var(residual) / variance) if variance > 0 else 0.0

    return PhaseAmplitudeFit(
        amplitude=amplitude, phase=phase, offset=offset, rmse=rmse, r_squared=r_squared
    )


def forward_model_residual(
    pd_signal: np.ndarray, displacement: np.ndarray, wavelength_um: float
) -> PhaseAmplitudeFit:
    """Method 1: fit the Michelson forward model to one channel of one shot.

    The displacement is *given* (derived from the speaker drive voltage), so
    the only free parameters are the per-shot fringe amplitude and phase
    offset. Whatever residual is left over is the part of the photodiode
    signal that the Michelson model plus the coil-driver calibration cannot
    explain -- detector noise, calibration error, fiber-induced distortion,
    speckle, or physics the model omits.

    Of those, the calibration term has been measured:
    :mod:`smi.analysis.gain_sweep` repeats this fit over a grid of
    coil-driver gains and finds the live ``k`` is 5-13% low across the six
    acquisition/channel pairs (N = 1000). It also shows that correcting the
    gain does not close the gap above 3 um of excursion, so calibration is a
    real but secondary term here.

    Args:
        pd_signal: Photodiode samples for one shot (V), shape ``(n,)``.
        displacement: Mirror displacement for the same shot (microns), shape
            ``(n,)``. Note this is the mirror displacement, not the optical
            path difference, which is twice as large.
        wavelength_um: Laser wavelength for this channel (microns).

    Returns:
        The closed-form fit, whose ``rmse`` is the method-1 residual in volts
        and whose ``r_squared`` is the explained variance fraction.
    """
    signal = _as_float64_zero_mean(pd_signal)
    theta = 4.0 * np.pi / wavelength_um * np.asarray(displacement, dtype=np.float64)
    return fit_phase_amplitude(signal, theta)


DEFAULT_WINDOW_SAMPLES = 1024

# cos, sin and a constant: the parameter count per window, which the
# scrambled-theta control has to match for the comparison to be fair.
_N_FIT_PARAMS = 3


@dataclass
class WindowedFit:
    """Pooled result of refitting ``(cos, sin, 1)`` per window of one shot.

    Attributes:
        r_squared: One R^2 for the whole shot, formed by pooling the residual
            and total sums of squares over all windows. Pooled rather than
            averaged over per-window R^2 values so the number stays directly
            comparable to the global single-window fit.
        rmse: Root-mean-square residual over the whole shot (V).
        n_windows: Number of windows the record was split into.
        window_samples: Window length actually used (samples).
    """

    r_squared: float
    rmse: float
    n_windows: int
    window_samples: int


def _window_bounds(n: int, window_samples: int | None) -> list[tuple[int, int]]:
    """Split ``n`` samples into consecutive windows of ``window_samples``.

    A record length that is not a multiple of the window leaves a short tail.
    The tail is kept as its own window rather than dropped: dropping it would
    silently score a different span than the global fit does, and the two
    numbers are meant to be compared. A tail shorter than the three fitted
    parameters is merged into the previous window instead, because a window
    with fewer samples than parameters fits exactly and would inflate R^2.

    Args:
        n: Record length in samples.
        window_samples: Target window length, or ``None`` for a single window
            spanning the whole record.

    Returns:
        ``(begin, end)`` index pairs covering ``[0, n)`` exactly once.

    Raises:
        ValueError: If ``window_samples`` is not positive.
    """
    if window_samples is None:
        return [(0, n)]
    if window_samples <= 0:
        raise ValueError(f'window_samples must be positive, got {window_samples}')
    bounds = [(b, min(b + window_samples, n)) for b in range(0, n, window_samples)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < _N_FIT_PARAMS + 1:
        begin, _ = bounds.pop(-2)
        bounds[-1] = (begin, bounds[-1][1])
    return bounds


def windowed_forward_model_residual(
    pd_signal: np.ndarray,
    displacement: np.ndarray,
    wavelength_um: float,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
) -> WindowedFit:
    """Method 1 with the fringe amplitude and phase refitted per window.

    The global fit in :func:`forward_model_residual` forces one amplitude,
    phase and offset across the whole 33.6 ms record. Real shots drift by
    1-2.7 fringes within a record (speckle and path drift), which a single
    phase cannot absorb, so the global R^2 measures the estimator's rigidity
    as much as the model's agreement with the data. Refitting in windows
    removes that limitation while keeping the displacement fixed and given.

    The per-window residual and total sums of squares are **pooled** into one
    R^2 for the shot. Averaging per-window R^2 values would not be comparable
    to the global fit, because a window whose signal barely varies would
    contribute a near-meaningless ratio with the same weight as a window that
    swings over several fringes.

    Interpreting the result requires the scrambled-theta control: short
    windows can fit almost anything, so run this again with ``displacement``
    taken from a *different shot* and report the two side by side. See
    :func:`scrambled_theta_control`.

    Args:
        pd_signal: Photodiode samples for one shot (V), shape ``(n,)``.
        displacement: Mirror displacement for the same shot (microns), shape
            ``(n,)``. For the control, pass another shot's displacement.
        wavelength_um: Laser wavelength for this channel (microns).
        window_samples: Window length in samples. ``None`` reproduces the
            global fit exactly, so the pre-windowing number stays available.

    Returns:
        The pooled fit statistics for this shot.

    Raises:
        ValueError: If the inputs do not have matching one-dimensional shapes.
    """
    signal = _as_float64_zero_mean(pd_signal)
    theta = 4.0 * np.pi / wavelength_um * np.asarray(displacement, dtype=np.float64)
    if signal.ndim != 1 or theta.ndim != 1 or signal.shape != theta.shape:
        raise ValueError(
            f'signal and displacement must be 1-D with matching shape, '
            f'got {signal.shape} and {theta.shape}'
        )

    bounds = _window_bounds(signal.size, window_samples)
    design = np.stack([np.cos(theta), np.sin(theta), np.ones_like(theta)], axis=1)

    # The global fit measures variance about the whole record's mean, so the
    # pooled total sum of squares must use that same mean. Summing each
    # window's variance about its own mean instead would drop the
    # between-window offset spread out of the denominator, inflating R^2 for
    # variance the per-window offset column merely re-baselined rather than
    # explained -- and the two columns would no longer be comparable, which is
    # the only reason to report them together.
    residual_ss = 0.0
    grand_mean = float(signal.mean())
    total_ss = float(np.sum((signal - grand_mean) ** 2))
    for begin, end in bounds:
        block = signal[begin:end]
        coeffs, *_ = np.linalg.lstsq(design[begin:end], block, rcond=None)
        residual_ss += float(np.sum((block - design[begin:end] @ coeffs) ** 2))

    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
    return WindowedFit(
        r_squared=float(r_squared),
        rmse=float(np.sqrt(residual_ss / signal.size)),
        n_windows=len(bounds),
        window_samples=int(window_samples) if window_samples else int(signal.size),
    )


def scrambled_theta_control(
    pd_signal: np.ndarray,
    other_displacement: np.ndarray,
    wavelength_um: float,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
) -> WindowedFit:
    """Windowed fit against a *different* shot's displacement.

    This is the mandatory control for every windowed R^2. It has the identical
    window count and per-window parameter count, so whatever variance it
    explains is the fitting capacity of the windowing itself rather than
    agreement between the model and the data. A windowed R^2 reported without
    this number beside it is not evidence.

    Args:
        pd_signal: Photodiode samples for one shot (V), shape ``(n,)``.
        other_displacement: Displacement from a different shot (microns),
            shape ``(n,)``.
        wavelength_um: Laser wavelength for this channel (microns).
        window_samples: Window length in samples, matching the real fit.

    Returns:
        The pooled control statistics.
    """
    return windowed_forward_model_residual(
        pd_signal, other_displacement, wavelength_um, window_samples=window_samples
    )


# ---------------------------------------------------------------------------
# Method 1 variants: the three levers on the amplitude bias
# ---------------------------------------------------------------------------
#
# A least-squares amplitude attenuates toward zero when the regressor is
# wrong. Projecting the data onto a cosine whose phase is off by ``eps``
# recovers ``A*cos(eps)``, not ``A``; average that over a window in which the
# phase error wanders and the fitted amplitude comes out low. So the fitted
# amplitude running systematically under the data's local amplitude is a
# *symptom of phase error*, not an independent gain defect. Measured on
# free-space 635 nm, the ratio of fitted amplitude to ``sqrt(2)*std`` of the
# window is below 1 in 94% of high-excursion windows (median 0.586) while the
# same test on exact-model synthetic fringes sits at 1.058 -- the estimator is
# unbiased, so the bias is in how well the phase regressor matches the data.
#
# Three levers follow, in increasing order of how directly they attack the
# phase error rather than its symptom:
#
# 1. ``fringe_matched_window`` -- shorten the window where the fringes are
#    fast, so less phase error accumulates inside one fit.
# 2. ``envelope_forward_model_residual`` -- let the amplitude and phase drift
#    smoothly across the record instead of stepping per window.
# 3. ``joint_gain_fit`` -- fix the displacement scale itself using all three
#    wavelengths at once, which is the only one of the three that attacks the
#    cause.
#
# Each adds fitting freedom, so each must be scored against a scrambled-theta
# control run under that *same* freedom. Comparing a freer fit to a stiffer
# control measures the freedom, not the agreement.


@dataclass(frozen=True)
class JointGainFit:
    """Result of the shared-gain fit across several wavelengths.

    Attributes:
        gain: The one displacement scale factor shared by all channels
            (dimensionless, multiplying the drive-implied displacement).
        amplitudes: Per-channel fringe amplitude (V), in channel order.
        phases: Per-channel phase offset (radians), in channel order.
        r_squared: Explained variance fraction pooled over all channels.
        per_channel_r_squared: Each channel's own R^2 at the shared gain.
    """

    gain: float
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]
    r_squared: float
    per_channel_r_squared: tuple[float, ...]


# Search bounds for the shared gain, as a multiplier on the drive-implied
# displacement. The gain sweep found the live coil-driver k is 5-13% low, so
# the interesting range sits just above 1; the bounds are wide enough to show
# an optimum inside them rather than pinned at an edge.
DEFAULT_GAIN_BOUNDS = (0.8, 1.5)

# Coarse-then-fine grid over the gain. The objective is multi-modal in gain --
# a gain wrong by one fringe order over the record can still line up locally --
# so a gradient method would find whichever basin it started in. A grid finds
# the global best on the bracket, and the refinement costs one more pass.
_GAIN_COARSE_POINTS = 281
_GAIN_REFINE_POINTS = 81
_GAIN_REFINE_PASSES = 3


def _joint_residual_at_gain(
    signals: np.ndarray,
    displacement: np.ndarray,
    wavelengths_um: Sequence[float],
    gain: float,
    window_samples: int | None,
) -> tuple[float, list[PhaseAmplitudeFit]]:
    """Total residual sum of squares at one fixed gain.

    With the gain fixed, each channel's ``(A, phi)`` is the existing
    closed-form solve, so the joint objective reduces to a sum of independent
    linear solves. That is the whole reason the search is 1-D.

    Args:
        signals: Zero-mean channel signals, shape ``(n_channels, n)``.
        displacement: Drive-implied displacement (um), shape ``(n,)``.
        wavelengths_um: One wavelength per channel (microns).
        gain: Displacement scale factor to evaluate.
        window_samples: Refit per window of this length, or ``None`` for one
            fit spanning the record.

    Returns:
        Tuple of (summed residual sum of squares, per-channel whole-record
        fits at this gain).
    """
    scaled = gain * displacement
    residual_ss = 0.0
    fits: list[PhaseAmplitudeFit] = []
    for channel, wavelength_um in enumerate(wavelengths_um):
        theta = 4.0 * np.pi / wavelength_um * scaled
        signal = signals[channel]
        fits.append(fit_phase_amplitude(signal, theta))
        if window_samples is None:
            design = np.stack(
                [np.cos(theta), np.sin(theta), np.ones_like(theta)], axis=1
            )
            coeffs, *_ = np.linalg.lstsq(design, signal, rcond=None)
            residual_ss += float(np.sum((signal - design @ coeffs) ** 2))
        else:
            for begin, end in _window_bounds(signal.size, window_samples):
                block, block_theta = signal[begin:end], theta[begin:end]
                design = np.stack(
                    [
                        np.cos(block_theta),
                        np.sin(block_theta),
                        np.ones_like(block_theta),
                    ],
                    axis=1,
                )
                coeffs, *_ = np.linalg.lstsq(design, block, rcond=None)
                residual_ss += float(np.sum((block - design @ coeffs) ** 2))
    return residual_ss, fits


def joint_gain_fit(
    signals: np.ndarray,
    displacement: np.ndarray,
    wavelengths_um: Sequence[float],
    gain_bounds: tuple[float, float] = DEFAULT_GAIN_BOUNDS,
    window_samples: int | None = None,
) -> JointGainFit:
    """Fit one shared displacement gain across several wavelengths.

    The physical constraint, which is what makes this worth doing: the mirror
    moves once, and the three lasers observe that single motion at different
    wavelengths through different optics and detectors. So the displacement
    scale is *shared* and the fringe amplitude and phase offset are *not*::

        minimize over (g, A_c, phi_c):
          sum_c || y_c - A_c * cos(4*pi*g*d/lambda_c + phi_c) ||^2

    Because ``(A_c, phi_c)`` is closed-form once ``g`` is fixed
    (:func:`fit_phase_amplitude`), this is a 1-D search over ``g`` with a
    linear solve inside -- not a general nonlinear optimization. The search is
    a grid refined three times rather than a descent, because the objective is
    multi-modal in ``g``: a gain wrong by a whole fringe order across the
    record still lines the fringes up locally, so a descent finds whichever
    basin it starts in.

    Three wavelengths constrain ``g`` far better than one does. A single
    channel trades a gain error against a phase offset almost freely; the
    three channels' fringe rates differ by 31% (0.515 against 0.675 um), so a
    gain that satisfies all three at once is pinned.

    Args:
        signals: Photodiode samples, shape ``(n_channels, n)`` (V). Each row
            is DC-removed internally.
        displacement: Drive-implied mirror displacement (microns), shape
            ``(n,)``. Mirror displacement, not optical path difference.
        wavelengths_um: One laser wavelength per row of ``signals``.
        gain_bounds: Inclusive ``(low, high)`` bracket for the search. Passing
            ``(x, x)`` evaluates exactly that gain, which is how the unit-gain
            reference is obtained.
        window_samples: Refit amplitude and phase per window of this length,
            or ``None`` (default) for one fit per channel over the record.
            The shared gain is global either way.

    Returns:
        The shared gain with each channel's amplitude, phase and R^2.

    Raises:
        ValueError: If the shapes do not line up, if there is not one
            wavelength per channel, or if ``gain_bounds`` is inverted.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    displacement = np.asarray(displacement, dtype=np.float64)
    if displacement.ndim != 1 or signals.ndim != 2:
        raise ValueError(
            f'signals must be 2-D and displacement 1-D, '
            f'got {signals.shape} and {displacement.shape}'
        )
    if signals.shape[1] != displacement.size:
        raise ValueError(
            f'signals and displacement must agree in length, '
            f'got {signals.shape[1]} and {displacement.size}'
        )
    if len(wavelengths_um) != signals.shape[0]:
        raise ValueError(
            f'need one wavelength per channel, got {len(wavelengths_um)} '
            f'for {signals.shape[0]} channels'
        )
    low, high = float(gain_bounds[0]), float(gain_bounds[1])
    if high < low:
        raise ValueError(f'gain_bounds must be (low, high), got {gain_bounds}')

    signals = signals - signals.mean(axis=1, keepdims=True)

    n_points = _GAIN_COARSE_POINTS
    best_gain = low
    for _ in range(_GAIN_REFINE_PASSES + 1):
        grid = np.linspace(low, high, n_points) if high > low else np.array([low])
        scores = [
            _joint_residual_at_gain(
                signals, displacement, wavelengths_um, float(g), window_samples
            )[0]
            for g in grid
        ]
        index = int(np.argmin(scores))
        best_gain = float(grid[index])
        if high <= low:
            break
        # Re-bracket to the winning cell and its neighbors, so the next pass
        # cannot step outside a region the coarse pass already rejected.
        step = float(grid[1] - grid[0])
        low, high = best_gain - step, best_gain + step
        n_points = _GAIN_REFINE_POINTS

    residual_ss, fits = _joint_residual_at_gain(
        signals, displacement, wavelengths_um, best_gain, window_samples
    )
    total_ss = float(np.sum(signals**2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0

    return JointGainFit(
        gain=best_gain,
        amplitudes=tuple(f.amplitude for f in fits),
        phases=tuple(f.phase for f in fits),
        r_squared=float(r_squared),
        per_channel_r_squared=tuple(f.r_squared for f in fits),
    )


# Fringe-matched window bounds. Below 256 the three fitted parameters start to
# explain a meaningful share of the window by themselves, so the number stops
# measuring agreement. Measured on the free-space above-3 um cohort (54 shots
# of the leading 400), median control against median real fit:
#
#     W       635 nm         675 nm         515 nm
#     256     0.665 / 0.880  0.708 / 0.871  0.562 / 0.690
#     1024    0.239 / 0.536  0.251 / 0.423  0.193 / 0.280
#     4096    0.067 / 0.155  0.065 / 0.129  0.050 / 0.082
#     16384   0.010 / 0.030  0.009 / 0.025  0.006 / 0.014
#
# At 256 the control is two thirds of the real fit: most of that R^2 is
# fitting capacity and it must never be quoted as agreement. Above 4096 there
# are only four windows in a record and the phase drift the windowing exists
# to absorb comes back, taking both curves toward zero. The clip therefore
# brackets the range over which the gap is worth reading at all.
MIN_FRINGE_WINDOW = 256
MAX_FRINGE_WINDOW = 4096
DEFAULT_FRINGES_PER_WINDOW = 2.0


def fringe_matched_window(
    displacement: np.ndarray,
    wavelength_um: float,
    fringes_per_window: float = DEFAULT_FRINGES_PER_WINDOW,
) -> int:
    """Window length that spans a fixed fringe count rather than a fixed time.

    A fixed ``W = 1024`` spans a third of a fringe on the quietest shots and a
    dozen on the loudest, so one window's worth of phase error is not
    comparable between them. Scaling ``W`` by the shot's own fringe rate makes
    it comparable: every window holds about ``fringes_per_window`` fringes, so
    the phase has about the same room to wander inside each one regardless of
    excursion.

    The fringe count over the record is the total phase travel divided by
    ``2*pi``, using the summed absolute phase increment rather than the
    peak-to-peak, because the mirror reverses many times and a peak-to-peak
    would undercount every reversal.

    Args:
        displacement: Drive-implied displacement for one shot (um), shape
            ``(n,)``.
        wavelength_um: Laser wavelength (microns).
        fringes_per_window: Target fringes spanned by one window.

    Returns:
        The window length in samples, clipped to
        ``[MIN_FRINGE_WINDOW, MAX_FRINGE_WINDOW]``.

    Raises:
        ValueError: If ``fringes_per_window`` is not positive.
    """
    if fringes_per_window <= 0:
        raise ValueError(
            f'fringes_per_window must be positive, got {fringes_per_window}'
        )
    displacement = np.asarray(displacement, dtype=np.float64)
    theta = 4.0 * np.pi / wavelength_um * displacement
    total_fringes = float(np.sum(np.abs(np.diff(theta)))) / (2.0 * np.pi)
    if total_fringes <= 0:
        return MAX_FRINGE_WINDOW
    samples_per_fringe = displacement.size / total_fringes
    window = round(samples_per_fringe * fringes_per_window)
    return int(np.clip(window, MIN_FRINGE_WINDOW, MAX_FRINGE_WINDOW))


# Polynomial order of the amplitude and phase envelope across the record.
# Degree 2 was chosen as the smallest order that can follow both a taper and a
# turnaround; degree 0 reduces exactly to the fixed-amplitude windowed fit,
# which is what makes the comparison between them a nested one.
DEFAULT_ENVELOPE_DEGREE = 2


def envelope_forward_model_residual(
    pd_signal: np.ndarray,
    displacement: np.ndarray,
    wavelength_um: float,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    degree: int = DEFAULT_ENVELOPE_DEGREE,
) -> WindowedFit:
    """Windowed fit whose amplitude and phase vary polynomially in time.

    The fixed-``W`` fit steps its amplitude and phase discontinuously at every
    window edge, which is not how either actually behaves: speckle and path
    drift move them smoothly. Here each window still gets its own solve, but
    the ``cos`` and ``sin`` columns are each multiplied by ``1, t, ..., t^d``
    in the window's own normalized time, so the recovered amplitude and phase
    are smooth polynomials within the window rather than constants.

    Both components share the same expansion, so amplitude *and* phase pick up
    the freedom together -- that is what makes degree 0 reduce exactly to
    :func:`windowed_forward_model_residual` and the comparison nested.

    This is the second of the three levers on the amplitude bias, and the
    weaker one in principle: it lets the estimator *describe* a varying
    amplitude but does nothing about the phase error that causes the
    attenuation. Its scrambled control must use the same ``degree``.

    Args:
        pd_signal: Photodiode samples for one shot (V), shape ``(n,)``.
        displacement: Mirror displacement for the same shot (um), shape
            ``(n,)``. Pass another shot's for the control.
        wavelength_um: Laser wavelength (microns).
        window_samples: Window length, or ``None`` for one window.
        degree: Polynomial order of the envelope. 0 is the fixed fit.

    Returns:
        Pooled fit statistics, directly comparable to the fixed-amplitude fit
        because the total sum of squares is taken about the same record mean.

    Raises:
        ValueError: If the shapes mismatch or ``degree`` is negative.
    """
    if degree < 0:
        raise ValueError(f'degree must be >= 0, got {degree}')
    signal = _as_float64_zero_mean(pd_signal)
    displacement = np.asarray(displacement, dtype=np.float64)
    if signal.ndim != 1 or displacement.ndim != 1 or signal.shape != displacement.shape:
        raise ValueError(
            f'signal and displacement must be 1-D with matching shape, '
            f'got {signal.shape} and {displacement.shape}'
        )
    theta = 4.0 * np.pi / wavelength_um * displacement

    bounds = _window_bounds(signal.size, window_samples)
    residual_ss = 0.0
    grand_mean = float(signal.mean())
    total_ss = float(np.sum((signal - grand_mean) ** 2))
    for begin, end in bounds:
        block, block_theta = signal[begin:end], theta[begin:end]
        # Normalized to [-1, 1] within the window so the powers stay
        # well-conditioned regardless of window length.
        local_t = np.linspace(-1.0, 1.0, block.size)
        powers = np.stack([local_t**k for k in range(degree + 1)], axis=1)
        columns = [
            powers * np.cos(block_theta)[:, None],
            powers * np.sin(block_theta)[:, None],
            np.ones((block.size, 1)),
        ]
        design = np.concatenate(columns, axis=1)
        if design.shape[0] <= design.shape[1]:
            # More parameters than samples fits exactly and would inflate R^2
            # for free. Fall back to the constant envelope on such a tail.
            design = np.stack(
                [np.cos(block_theta), np.sin(block_theta), np.ones_like(block)], axis=1
            )
        coeffs, *_ = np.linalg.lstsq(design, block, rcond=None)
        residual_ss += float(np.sum((block - design @ coeffs) ** 2))

    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 0.0
    return WindowedFit(
        r_squared=float(r_squared),
        rmse=float(np.sqrt(residual_ss / signal.size)),
        n_windows=len(bounds),
        window_samples=int(window_samples) if window_samples else int(signal.size),
    )


@dataclass(frozen=True)
class PhaseDrift:
    """How far a shot's fringe phase and amplitude wander within one record.

    Attributes:
        drift_fringes: Peak-to-peak excursion of the unwrapped per-window
            fitted phase, in fringes (1 fringe = 2*pi radians).
        amplitude_cv: Standard deviation of the per-window fitted amplitude
            divided by its mean. Zero means a constant fringe visibility.
        n_windows: Number of windows the record was split into.
    """

    drift_fringes: float
    amplitude_cv: float
    n_windows: int


def measure_phase_drift(
    pd_signal: np.ndarray,
    displacement: np.ndarray,
    wavelength_um: float,
    window_samples: int = 512,
) -> PhaseDrift:
    """Measure within-shot phase and amplitude wander directly.

    This is the *mechanism* behind the windowed refit, measured on its own
    rather than inferred from the fact that windowing raises R^2. The global
    fit assumes one amplitude and one phase describe the whole 33.6 ms record;
    this quantifies how badly that assumption fails by fitting each window
    independently and tracking how the fitted values move.

    The phase is unwrapped across windows before its excursion is taken, so a
    drift that crosses the atan2 branch cut is not reported as a jump. A window
    of 512 samples (1.05 ms) is short enough that the phase is near-constant
    within it and long enough to sweep roughly a fringe, which is what makes
    the per-window fit well conditioned.

    Args:
        pd_signal: Photodiode samples for one shot (V), shape ``(n,)``.
        displacement: Mirror displacement for the same shot (microns), shape
            ``(n,)``.
        wavelength_um: Laser wavelength for this channel (microns).
        window_samples: Window length in samples.

    Returns:
        The drift in fringes and the amplitude coefficient of variation.

    Raises:
        ValueError: If fewer than two windows fit in the record, since a single
            window cannot exhibit drift and reporting 0.0 would be misleading.
    """
    signal = _as_float64_zero_mean(pd_signal)
    theta = 4.0 * np.pi / wavelength_um * np.asarray(displacement, dtype=np.float64)
    n_windows = signal.size // int(window_samples)
    if n_windows < 2:
        raise ValueError(
            f'need at least 2 windows to measure drift, got {n_windows} for '
            f'{signal.size} samples at window_samples={window_samples}'
        )

    usable = n_windows * int(window_samples)
    blocks = signal[:usable].reshape(n_windows, -1)
    thetas = theta[:usable].reshape(n_windows, -1)
    blocks = blocks - blocks.mean(axis=1, keepdims=True)

    design = np.stack([np.cos(thetas), np.sin(thetas), np.ones_like(thetas)], axis=2)
    gram = design.transpose(0, 2, 1) @ design
    rhs = design.transpose(0, 2, 1) @ blocks[:, :, None]
    coeffs = np.linalg.solve(gram, rhs)[:, :, 0]

    # Same (a, b) -> phase convention as fit_phase_amplitude.
    phase = np.unwrap(np.arctan2(-coeffs[:, 1], coeffs[:, 0]))
    amplitude = np.hypot(coeffs[:, 0], coeffs[:, 1])
    mean_amplitude = float(amplitude.mean())

    return PhaseDrift(
        drift_fringes=float(np.ptp(phase) / (2.0 * np.pi)),
        amplitude_cv=(
            float(amplitude.std() / mean_amplitude) if mean_amplitude > 0 else 0.0
        ),
        n_windows=int(n_windows),
    )


# ---------------------------------------------------------------------------
# Method 2: inverse fit by multi-wavelength decoding
# ---------------------------------------------------------------------------


def _viterbi(
    cost: torch.Tensor, grid_step: float, max_step: int, smooth_weight: float
) -> torch.Tensor:
    """Minimum-cost monotone-in-time path through a displacement grid.

    Solves, over a batch of shots simultaneously::

        argmin_{j_0..j_{n-1}}  sum_i cost[b, j_i, i]
                             + smooth_weight * sum_i ((j_i - j_{i-1}) * dg)^2

    subject to ``|j_i - j_{i-1}| <= max_step``. The band constraint encodes a
    maximum plausible mirror speed, and the quadratic penalty makes the
    trajectory prefer smooth motion -- which is what breaks the
    ``cos`` ambiguity that defeats single-channel phase unwrapping when the
    velocity changes sign.

    Args:
        cost: Per-sample data cost, shape ``(batch, n_grid, n_time)``.
        grid_step: Spacing between adjacent displacement grid points (microns).
        max_step: Maximum grid points the path may move per time sample.
        smooth_weight: Weight on the squared-displacement-change penalty.

    Returns:
        Grid indices of the optimal path, shape ``(batch, n_time)``, dtype long.
    """
    batch, _n_grid, n_time = cost.shape
    device = cost.device
    offsets = torch.arange(-max_step, max_step + 1, device=device, dtype=cost.dtype)
    penalty = smooth_weight * (offsets * grid_step) ** 2

    running = cost[:, :, 0].clone()
    backpointer = torch.empty(
        (batch, cost.shape[1], n_time), dtype=torch.int16, device=device
    )
    for i in range(1, n_time):
        padded = torch.nn.functional.pad(
            running, (max_step, max_step), value=float('inf')
        )
        # window[b, j, k] == running[b, j + k - max_step]
        window = padded.unfold(1, 2 * max_step + 1, 1)
        best, arg = (window + penalty).min(dim=2)
        running = best + cost[:, :, i]
        backpointer[:, :, i] = (arg - max_step).to(torch.int16)

    node = running.argmin(dim=1)
    path = torch.empty((batch, n_time), dtype=torch.long, device=device)
    path[:, -1] = node
    rows = torch.arange(batch, device=device)
    for i in range(n_time - 1, 0, -1):
        node = node + backpointer[rows, node, i].long()
        path[:, i - 1] = node
    return path


def _channel_costs(
    signals: torch.Tensor,
    grid: torch.Tensor,
    wavelengths: torch.Tensor,
    phases: torch.Tensor,
) -> torch.Tensor:
    """Squared-error data cost of every grid displacement at every time sample.

    Args:
        signals: Amplitude-normalized PD signals, shape ``(batch, n_ch, n_time)``.
        grid: Candidate displacements (microns), shape ``(n_grid,)``.
        wavelengths: Channel wavelengths (microns), shape ``(n_ch,)``.
        phases: Per-shot per-channel phase offsets, shape ``(batch, n_ch)``.

    Returns:
        Cost tensor of shape ``(batch, n_grid, n_time)``.
    """
    batch, n_channels, n_time = signals.shape
    cost = torch.zeros(
        (batch, grid.numel(), n_time), dtype=signals.dtype, device=signals.device
    )
    # Accumulate channel by channel rather than materializing the full
    # (batch, n_ch, n_grid, n_time) difference, which is the peak-memory term.
    for k in range(n_channels):
        # model[b, g] = cos(4*pi/lambda_k * grid_g + phi_bk)
        model = torch.cos(
            4.0 * torch.pi / wavelengths[k] * grid[None, :] + phases[:, k, None]
        )
        cost += (model[:, :, None] - signals[:, k, None, :]) ** 2
    return cost


def _normalize_fringe_amplitude(signals: np.ndarray) -> np.ndarray:
    """Scale each channel so its fringes span roughly [-1, 1].

    A pure sinusoid has standard deviation ``A / sqrt(2)``, so dividing by
    ``sqrt(2) * std`` puts a clean fringe train on unit amplitude, which is
    what the ``cos`` template in the decoder assumes.

    Args:
        signals: PD signals, shape ``(..., n_ch, n_time)``.

    Returns:
        Zero-mean, unit-fringe-amplitude signals of the same shape.
    """
    out = np.asarray(signals, dtype=np.float64)
    out = out - out.mean(axis=-1, keepdims=True)
    scale = np.sqrt(2.0) * out.std(axis=-1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return out / scale


_Decoded = tuple[torch.Tensor, torch.Tensor]


def _phase_candidates(
    resolution: int, n_channels: int, device: torch.device
) -> torch.Tensor:
    """Enumerate the relative-phase grid searched by :func:`inverse_fit`.

    Only two relative phases are enumerated. An overall phase shift common to
    all channels is degenerate with a shift of the displacement origin, and
    that origin is itself unobservable from DC-removed fringes, so searching
    a third phase would only re-explore the same solutions.

    Args:
        resolution: Grid points per relative-phase dimension.
        n_channels: Number of photodiode channels.
        device: Device to build the tensor on.

    Returns:
        Candidate phase offsets, shape ``(resolution ** 2, n_channels)``, with
        channel 0 pinned to zero.
    """
    axis = torch.linspace(
        0.0, 2.0 * torch.pi, resolution + 1, dtype=torch.float32, device=device
    )[:-1]
    first, second = torch.meshgrid(axis, axis, indexing='ij')
    candidates = torch.zeros(
        (axis.numel() ** 2, n_channels), dtype=torch.float32, device=device
    )
    if n_channels > 1:
        candidates[:, 1] = first.reshape(-1)
    if n_channels > 2:
        candidates[:, 2] = second.reshape(-1)
    return candidates


def _decode_batched(
    signals: torch.Tensor,
    phases: torch.Tensor,
    *,
    grid: torch.Tensor,
    wavelengths: torch.Tensor,
    grid_step: float,
    max_step: int,
    smooth_weight: float,
    chunk_elements: int,
) -> _Decoded:
    """Viterbi-decode a batch and return its total path cost and path.

    The cost tensor is ``(rows, n_grid, n_time)`` float32 and the decoder
    transiently needs several multiples of it, so rows are processed in
    memory-bounded chunks. Rows never interact, so chunking cannot change the
    result -- it only bounds peak memory.

    Args:
        signals: Amplitude-normalized signals, ``(rows, n_ch, n_time)``.
        phases: Per-row phase offsets, ``(rows, n_ch)``.
        grid: Candidate displacements (microns).
        wavelengths: Channel wavelengths (microns).
        grid_step: Spacing between grid points (microns).
        max_step: Maximum grid points traversed per time sample.
        smooth_weight: Weight on the squared-step penalty.
        chunk_elements: Soft cap on ``rows * n_grid * n_time`` per call.

    Returns:
        Tuple of total path cost per row and grid-index path per row.
    """
    per_row = max(1, grid.numel() * signals.shape[2])
    chunk = max(1, int(chunk_elements // per_row))
    costs: list[torch.Tensor] = []
    paths: list[torch.Tensor] = []
    for begin in range(0, signals.shape[0], chunk):
        stop = begin + chunk
        cost = _channel_costs(
            signals[begin:stop], grid, wavelengths, phases[begin:stop]
        )
        path = _viterbi(cost, grid_step, max_step, smooth_weight)
        data = cost.gather(1, path[:, None, :]).squeeze(1).sum(dim=1)
        step = (path[:, 1:] - path[:, :-1]).to(cost.dtype) * grid_step
        costs.append(data + smooth_weight * (step**2).sum(dim=1))
        paths.append(path)
        del cost
    return torch.cat(costs), torch.cat(paths)


@dataclass
class InverseFitResult:
    """Displacement and velocity recovered from photodiode signals alone.

    Attributes:
        displacement: Estimated displacement (microns), shape ``(batch, n_time)``,
            zero-mean (the absolute offset is unobservable from fringes).
        velocity: Estimated velocity (microns/ms), same shape.
        phases: Per-shot per-channel phase offsets chosen by the search
            (radians), shape ``(batch, n_channels)``.
    """

    displacement: np.ndarray
    velocity: np.ndarray
    phases: np.ndarray


def inverse_fit(
    pd_signals: np.ndarray,
    sample_rate: float,
    wavelengths_um: np.ndarray | list[float],
    displacement_limit_um: float,
    grid_points: int = 1201,
    max_speed_um_per_ms: float = 4.0,
    smooth_scale: float = 0.5,
    phase_grid_points: int = 16,
    phase_search_samples: int = 1024,
    phase_refine_candidates: int = 8,
    velocity_smooth_window: int = 251,
    decode_chunk_elements: int = 200_000_000,
    device: str | torch.device = 'cpu',
) -> InverseFitResult:
    """Method 2: recover displacement from the photodiode signals alone.

    The drive voltage is deliberately *not* used. Each wavelength wraps its
    fringe phase at a different rate, so the three channels taken together
    pin down the displacement modulo a much longer synthetic beat period than
    any single channel could. Concretely, for each candidate displacement on a
    grid we score how well all three ``cos`` templates match the measured
    samples, then pick the time trajectory minimizing total data cost plus a
    smoothness penalty (Viterbi). The smoothness term is essential: it is what
    lets the decoder walk through velocity sign reversals, which is exactly
    where naive per-channel Hilbert phase unwrapping fails.

    The per-channel phase offsets are unknown per shot. They are found by a
    coarse grid search on the two *relative* phases (the third is degenerate
    with the unobservable displacement offset), scored by the Viterbi path
    cost. Using the smoothness-constrained cost rather than the per-sample
    minimum is what makes this search work; the per-sample minimum is happy to
    let the trajectory jump around and so cannot rank phases. The search runs
    on a short prefix to shortlist candidates, then rescores the shortlist on
    the full shot, because a short prefix ranks phases too noisily to trust
    outright.

    Velocity is obtained by central-difference differentiation of the decoded
    displacement.

    Args:
        pd_signals: Photodiode signals, shape ``(batch, n_channels, n_time)``
            or ``(n_channels, n_time)`` for a single shot.
        sample_rate: Acquisition sample rate (Hz).
        wavelengths_um: Channel wavelengths (microns), length ``n_channels``.
        displacement_limit_um: Half-width of the displacement search grid
            (microns); the grid spans ``[-limit, +limit]``.
        grid_points: Number of displacement grid points.
        max_speed_um_per_ms: Largest mirror speed the decoder will allow
            (microns per millisecond). Sets the Viterbi band width.
        smooth_scale: Dimensionless weight on the smoothness penalty relative
            to the maximum allowed per-sample step.
        phase_grid_points: Grid resolution per relative-phase dimension. The
            coarse search evaluates ``phase_grid_points ** 2`` candidates.
        phase_search_samples: Number of leading time samples used to shortlist
            phase candidates. Keeps the coarse search cheap.
        phase_refine_candidates: How many shortlisted candidates are rescored
            on the full shot before picking a winner.
        velocity_smooth_window: Savitzky-Golay window (samples) used for the
            derivative. At 488 kHz, 251 samples is 0.5 ms -- far shorter than
            a period of the 1 kHz upper drive frequency, so it suppresses
            grid-quantization noise without attenuating the signal.
        decode_chunk_elements: Soft cap on ``rows * grid_points * n_time``
            per Viterbi call, to bound peak memory. Does not change results.
        device: Torch device for the decode. ``'cuda:0'`` is much faster for
            batches.

    Returns:
        The decoded displacement, velocity and per-shot phases.

    Raises:
        ValueError: If the signal array is not 2-D or 3-D, or the number of
            channels does not match the wavelength list.
    """
    signals = np.asarray(pd_signals, dtype=np.float64)
    if signals.ndim == 2:
        signals = signals[None, ...]
    if signals.ndim != 3:
        raise ValueError(f'pd_signals must be 2-D or 3-D, got shape {signals.shape}')
    wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
    if signals.shape[1] != wavelengths.size:
        raise ValueError(
            f'got {signals.shape[1]} channels but {wavelengths.size} wavelengths'
        )

    batch, n_channels, n_time = signals.shape
    normalized = _normalize_fringe_amplitude(signals)

    torch_device = torch.device(device)
    sig_t = torch.tensor(normalized, dtype=torch.float32, device=torch_device)
    wl_t = torch.tensor(wavelengths, dtype=torch.float32, device=torch_device)
    grid = torch.linspace(
        -displacement_limit_um,
        displacement_limit_um,
        grid_points,
        dtype=torch.float32,
        device=torch_device,
    )
    grid_step = float(grid[1] - grid[0])
    # sample_rate is in Hz, so the speed must be per-second to divide into it
    # and yield microns per sample; max_speed is per-millisecond.
    max_speed_um_per_s = max_speed_um_per_ms * MS_PER_S
    max_step = max(1, int(np.ceil(max_speed_um_per_s / sample_rate / grid_step)))
    smooth_weight = smooth_scale / (grid_step * max_step) ** 2

    def decode(signals_t: torch.Tensor, phases: torch.Tensor) -> _Decoded:
        return _decode_batched(
            signals_t,
            phases,
            grid=grid,
            wavelengths=wl_t,
            grid_step=grid_step,
            max_step=max_step,
            smooth_weight=smooth_weight,
            chunk_elements=decode_chunk_elements,
        )

    # --- coarse phase search on a short prefix ----------------------------
    prefix = min(phase_search_samples, n_time)
    candidates = _phase_candidates(phase_grid_points, n_channels, torch_device)
    n_cand = candidates.shape[0]

    # (batch * n_cand, ...): every shot paired with every phase candidate, so
    # the whole search is a handful of kernel launches, not one per candidate.
    prefix_signals = sig_t[:, :, :prefix].repeat_interleave(n_cand, dim=0)
    coarse_cost, _ = decode(prefix_signals, candidates.repeat(batch, 1))
    coarse_cost = coarse_cost.view(batch, n_cand)

    # --- rescore the best candidates on the full shot ---------------------
    # The prefix is short enough that its ranking is noisy, so the prefix is
    # used only to shortlist. The winner is decided by full-shot cost, which
    # is the quantity we actually care about minimizing.
    keep = min(phase_refine_candidates, n_cand)
    shortlist = coarse_cost.topk(keep, dim=1, largest=False).indices  # (batch, keep)
    full_signals = sig_t.repeat_interleave(keep, dim=0)
    full_phases = candidates[shortlist.reshape(-1)]
    full_cost, full_path = decode(full_signals, full_phases)
    full_cost = full_cost.view(batch, keep)
    winner = full_cost.argmin(dim=1)

    rows = torch.arange(batch, device=torch_device)
    flat = rows * keep + winner
    path = full_path[flat]
    best_phases = full_phases[flat]

    displacement = grid[path]
    displacement = displacement - displacement.mean(dim=1, keepdim=True)

    disp_np = displacement.detach().cpu().numpy().astype(np.float64)

    # The decoded displacement is quantized to the grid, so a bare finite
    # difference at 488 kHz differentiates a staircase and buries the answer
    # in quantization noise. A Savitzky-Golay derivative fits a local cubic
    # and differentiates that instead. The window is long in samples but
    # short in time (see velocity_smooth_window), and the speaker is driven
    # only to 1 kHz, so this removes quantization noise rather than signal.
    # savgol_filter requires an odd window. The min() bounds the length
    # against short shots, but an even velocity_smooth_window would survive it
    # and raise inside scipy, so force parity on the result rather than only
    # on the bound.
    window = min(velocity_smooth_window, n_time - (1 - n_time % 2))
    window -= 1 - window % 2
    # The time step is in milliseconds, so differentiating displacement in um
    # yields um/ms -- the unit CoilDriver emits and the network is scored in.
    # A per-second step here would make the returned velocity, and hence the
    # RMSE against the ground truth, 1000x too large.
    dt_ms = MS_PER_S / sample_rate
    if window >= 5:
        velocity_np = savgol_filter(
            disp_np, window, polyorder=3, deriv=1, delta=dt_ms, axis=1
        )
    else:
        velocity_np = np.gradient(disp_np, dt_ms, axis=1)

    return InverseFitResult(
        displacement=disp_np,
        velocity=velocity_np,
        phases=best_phases.detach().cpu().numpy().astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Evaluation over real acquired data
# ---------------------------------------------------------------------------


@dataclass
class DatasetBaseline:
    """Aggregated baseline numbers for one acquired dataset.

    Attributes:
        dataset: Short dataset label, e.g. ``'free-space'``.
        file: File name the numbers came from.
        n_shots: Number of shots method 1 evaluated.
        n_shots_method2: Number of shots method 2 evaluated. Lower than
            ``n_shots`` on purpose -- see :func:`evaluate_dataset`.
        n_samples_per_shot: Time samples per shot.
        sample_rate_hz: Acquisition sample rate (Hz).
        method1: Per-channel forward-model residual statistics (volts).
        method2: Inverse-fit displacement/velocity error statistics.
        signal_scale: Reference scales used to make the RMSE numbers readable.
        decode_grid: Decode grid extent and speed band, derived per dataset at
            run time and therefore not recoverable from the static config.
        runtime_s: Wall-clock seconds spent evaluating this dataset.
    """

    dataset: str
    file: str
    n_shots: int
    n_shots_method2: int
    n_samples_per_shot: int
    sample_rate_hz: float
    method1: dict[str, Any] = field(default_factory=dict)
    method2: dict[str, Any] = field(default_factory=dict)
    signal_scale: dict[str, Any] = field(default_factory=dict)
    decode_grid: dict[str, float] = field(default_factory=dict)
    runtime_s: float = 0.0


def _summary(values: np.ndarray) -> dict[str, float]:
    """Mean/median/quartile summary of a 1-D array of per-shot scores."""
    values = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'min': float(values.min()),
        'max': float(values.max()),
    }


# Bin edges in microns peak-to-peak. Fixed rather than quantile-derived so
# the bins mean the same thing across datasets and across reruns with a
# different --n-shots, which is the whole point of persisting them.
# The 2-3 um and >3 um ends behave differently -- a gain correction
# recovers much of the former and little of the latter -- so a single
# '2+' top bin hid the distinction that matters.
PTP_BIN_EDGES_UM = (0.0, 0.3, 1.0, 2.0, 3.0, np.inf)

# How the windowed measure picks its window length. 'fringe-matched' resolves
# W per shot per channel from that shot's own fringe rate, which is what the
# report's artifact uses so every table in it means one estimator. 'fixed'
# holds W at ``window_samples``, which is what the known-answer tests want:
# their fringe rate is synthetic, so matching a window to it measures nothing.
_WINDOWING_MODES = frozenset({'fringe-matched', 'fixed'})


def binned_by_amplitude(
    scores: np.ndarray, ptp_um: np.ndarray
) -> list[dict[str, float | int | str]]:
    """Per-shot ``scores`` grouped by drive peak-to-peak excursion.

    Each bin reports its own ``n``: a bin holding two shots is not evidence
    of a trend, and the reader needs to see that without recomputing it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    ptp_um = np.asarray(ptp_um, dtype=np.float64)
    bins: list[dict[str, float | int | str]] = []
    for low, high in itertools.pairwise(PTP_BIN_EDGES_UM):
        mask = (ptp_um >= low) & (ptp_um < high)
        n = int(mask.sum())
        label = f'{low:g}-{high:g}' if np.isfinite(high) else f'{low:g}+'
        entry: dict[str, float | int | str] = {'ptp_um': label, 'n': n}
        if n:
            entry['median'] = float(np.median(scores[mask]))
            entry['mean'] = float(scores[mask].mean())
        bins.append(entry)
    return bins


def _method1_channel(
    signals: np.ndarray,
    true_disp: np.ndarray,
    scramble: np.ndarray,
    wavelength_um: float,
    ptp_um: np.ndarray,
    windowing: str,
    window_samples: int | None,
) -> dict[str, Any]:
    """Method-1 statistics for one channel over every shot.

    Args:
        signals: Photodiode samples for this channel, shape ``(n_shots, n)``.
        true_disp: Drive-implied displacement per shot (um), same shape.
        scramble: Shot index each shot borrows a control displacement from.
        wavelength_um: This channel's laser wavelength (microns).
        ptp_um: Per-shot displacement peak-to-peak, for the excursion bins.
        windowing: ``'fringe-matched'`` or ``'fixed'``.
        window_samples: Window length used when ``windowing='fixed'``.

    Returns:
        The per-channel block written into the artifact.
    """
    n_shots, n_time = signals.shape
    rmses = np.empty(n_shots)
    r2s = np.empty(n_shots)
    rel = np.empty(n_shots)
    windowed_r2s = np.empty(n_shots)
    control_r2s = np.empty(n_shots)
    resolved_w = np.empty(n_shots, dtype=np.int64)
    for i in range(n_shots):
        fit = forward_model_residual(signals[i], true_disp[i], wavelength_um)
        rmses[i] = fit.rmse
        r2s[i] = fit.r_squared
        signal_rms = float(np.std(signals[i]))
        rel[i] = fit.rmse / signal_rms if signal_rms > 0 else np.nan
        # One W per shot per channel, resolved from *this* shot's own
        # displacement and this channel's wavelength: a shot's fringe rate
        # differs by 31% between 515 and 675 nm, so a per-shot W shared
        # across channels would be fringe-matched for none of them.
        shot_w = (
            fringe_matched_window(true_disp[i], wavelength_um)
            if windowing == 'fringe-matched'
            else window_samples
        )
        resolved_w[i] = shot_w if shot_w is not None else n_time
        windowed_r2s[i] = windowed_forward_model_residual(
            signals[i], true_disp[i], wavelength_um, window_samples=shot_w
        ).r_squared
        # The control gets the *same* W as the fit it controls, resolved from
        # this shot rather than from the displacement it borrows. Scoring a
        # fit against a control that had a different amount of fitting
        # freedom measures the freedom, not the agreement.
        control_r2s[i] = scrambled_theta_control(
            signals[i], true_disp[scramble[i]], wavelength_um, window_samples=shot_w
        ).r_squared

    return {
        'wavelength_um': wavelength_um,
        # What window the numbers below came from. A fringe-matched W is per
        # shot, so no single number states it: the median and the range are
        # what let a reader see the window without holding the per-shot array.
        'window_samples_resolved': {
            'median': float(np.median(resolved_w)),
            'min': int(resolved_w.min()),
            'max': int(resolved_w.max()),
            'p25': float(np.percentile(resolved_w, 25)),
            'p75': float(np.percentile(resolved_w, 75)),
        },
        'residual_rmse_volts': _summary(rmses),
        'residual_rmse_fraction_of_signal_rms': _summary(rel),
        'r_squared': _summary(r2s),
        # The windowed refit measures model/data agreement and the global
        # single-window fit is the contrast that motivates windowing, so both
        # are reported side by side. The control is not optional -- short
        # windows can fit almost anything, so a windowed R^2 reported without
        # it is not evidence.
        'r_squared_windowed': _summary(windowed_r2s),
        'r_squared_windowed_scrambled': _summary(control_r2s),
        # Binned by drive excursion because the aggregate R^2 hides the trend
        # that matters: the forward model tracks small excursions better than
        # large ones. Persisted rather than left to an ad-hoc session, so the
        # claim can be checked against the artifact instead of taken on trust.
        'r_squared_by_drive_ptp': binned_by_amplitude(r2s, ptp_um),
        'r_squared_windowed_by_drive_ptp': binned_by_amplitude(windowed_r2s, ptp_um),
        'r_squared_windowed_scrambled_by_drive_ptp': binned_by_amplitude(
            control_r2s, ptp_um
        ),
    }


def evaluate_dataset(
    file_path: Path,
    label: str,
    n_shots: int = 1000,
    n_shots_method2: int = 200,
    batch_size: int = 16,
    device: str = 'cpu',
    grid_points: int = 1201,
    phase_grid_points: int = 16,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    windowing: str = 'fringe-matched',
) -> DatasetBaseline:
    """Run both baseline methods over a subset of one acquired HDF5 file.

    Shots are read as a contiguous leading slice; the files are 0.66 GB per
    channel, so the whole array is never materialized.

    **The two methods run at different shot counts on purpose.** Method 1 is a
    handful of least-squares solves per shot, about a millisecond each, so it
    can afford N = 1000 -- and it needs it, because that is where the
    large-excursion claims live and the extreme bins are otherwise too thin to
    carry a trend. Method 2 is a sequential GPU Viterbi decode at ~5.85 s per
    shot; N = 1000 would cost ~3.3 hours across both acquisitions to put a
    tighter error bar on displacement and velocity RMSE numbers that nobody
    disputes. Both counts are recorded so a reader can tell which N produced
    which table.

    **The windowed measure is fringe-matched by default.** A fixed ``W``
    spans a third of a fringe on the quietest shots and a dozen on the
    loudest, so one window's worth of phase error is not comparable between
    them and the measure means something different in each excursion bin.
    :func:`fringe_matched_window` scales ``W`` by each shot's own fringe rate
    instead, so every window holds about the same number of fringes. Pass
    ``windowing='fixed'`` for the fixed-``W`` behaviour, which is what the
    known-answer tests want: their fringe rate is synthetic, so matching a
    window to it measures nothing.

    Args:
        file_path: Path to the acquisition HDF5 file.
        label: Short label recorded in the result.
        n_shots: Number of shots for method 1.
        n_shots_method2: Number of shots for method 2, capped at ``n_shots``.
        batch_size: Shots decoded per Viterbi batch. Larger amortizes GPU
            kernel launch overhead; memory scales with
            ``batch_size * grid_points * n_time``.
        device: Torch device for the method-2 decode.
        grid_points: Displacement grid resolution for method 2.
        phase_grid_points: Phase search resolution for method 2.
        window_samples: Window length for method 1's windowed refit, used
            only when ``windowing='fixed'``.
        windowing: ``'fringe-matched'`` to resolve ``W`` per shot per channel
            from that shot's own fringe rate, or ``'fixed'`` to use
            ``window_samples`` for every shot.

    Returns:
        Aggregated statistics for this dataset.

    Raises:
        ValueError: If fewer than two shots are available, which leaves the
            scrambled-theta control no other shot to borrow a displacement
            from, or if ``windowing`` is not one of the two accepted values.
    """
    if windowing not in _WINDOWING_MODES:
        raise ValueError(
            f'windowing must be one of {sorted(_WINDOWING_MODES)}, got {windowing!r}'
        )
    start = time.perf_counter()
    coil = CoilDriver()
    channels = list(PD_WAVELENGTHS_UM)
    wavelengths = np.array([PD_WAVELENGTHS_UM[c] for c in channels])

    with h5py.File(file_path, 'r') as handle:
        sample_rate = float(handle.attrs['sample_rate'])
        n_shots = min(n_shots, handle[DRIVE_KEY].shape[0])
        n_shots_method2 = min(n_shots_method2, n_shots)
        n_time = int(handle[DRIVE_KEY].shape[1])
        drive = np.asarray(handle[DRIVE_KEY][:n_shots], dtype=np.float64)
        pd_raw = np.stack(
            [np.asarray(handle[c][:n_shots], dtype=np.float64) for c in channels],
            axis=1,
        )

    # Ground truth from the drive voltage.
    true_disp = np.empty((n_shots, n_time))
    true_vel = np.empty((n_shots, n_time))
    for i in range(n_shots):
        # One shared spectrum: get_displacement and get_velocity each call
        # get_displacement_spectrum internally, so calling both computes the
        # same FFT twice.
        true_vel[i], true_disp[i] = coil.get_velocity_and_displacement(
            drive[i], sample_rate
        )
    true_disp_centered = true_disp - true_disp.mean(axis=1, keepdims=True)

    # --- Method 1 ---------------------------------------------------------
    # The scrambled-theta control pairs each shot's signal with the next
    # shot's displacement (the last wraps to the first). A fixed rotation
    # rather than a random pairing keeps the artifact reproducible, and a
    # neighbouring shot comes from the same acquisition conditions, which is
    # what makes it the right control: it differs from the real pairing only
    # in that the displacement no longer belongs to this signal.
    #
    # With a single shot the rotation maps it onto itself, so the "control"
    # would be the real fit under a different name -- a low number that looks
    # like evidence of no overfitting and is nothing of the kind. Refuse
    # rather than emit it.
    if n_shots < 2:
        raise ValueError(
            f'the scrambled-theta control needs at least 2 shots to borrow a '
            f'displacement from, got {n_shots}'
        )
    scramble = np.roll(np.arange(n_shots), -1)

    ptp_um = np.ptp(true_disp_centered, axis=1)
    m1: dict[str, Any] = {
        channel: _method1_channel(
            pd_raw[:, k],
            true_disp,
            scramble,
            float(wavelengths[k]),
            ptp_um,
            windowing,
            window_samples,
        )
        for k, channel in enumerate(channels)
    }

    # --- Method 2 ---------------------------------------------------------
    # The search grid must cover the true excursion with margin; use the
    # observed ground-truth range for this subset.
    # Derived from the shots method 2 actually scores, not from method 1's
    # larger subset: a grid wider than this subset needs would waste decode
    # time, and a speed band set by shots outside the subset could not be
    # reproduced from the numbers the artifact records.
    m2_disp = true_disp_centered[:n_shots_method2]
    m2_vel = true_vel[:n_shots_method2]
    limit = float(np.abs(m2_disp).max()) * 1.15
    max_speed = float(np.abs(m2_vel).max()) * 1.25

    disp_rmse = np.empty(n_shots_method2)
    vel_rmse = np.empty(n_shots_method2)
    disp_corr = np.empty(n_shots_method2)
    for begin in range(0, n_shots_method2, batch_size):
        end = min(begin + batch_size, n_shots_method2)
        result = inverse_fit(
            pd_raw[begin:end],
            sample_rate=sample_rate,
            wavelengths_um=wavelengths,
            displacement_limit_um=limit,
            grid_points=grid_points,
            max_speed_um_per_ms=max_speed,
            phase_grid_points=phase_grid_points,
            device=device,
        )
        for j, i in enumerate(range(begin, end)):
            # Fringe sign is not observable: cos is even, so d(t) and -d(t)
            # produce identical signals. Score against whichever sign fits
            # better rather than penalizing an unobservable degree of freedom.
            candidates = [
                (result.displacement[j], result.velocity[j]),
                (-result.displacement[j], -result.velocity[j]),
            ]
            errors = [
                float(np.sqrt(np.mean((d - true_disp_centered[i]) ** 2)))
                for d, _ in candidates
            ]
            pick = int(np.argmin(errors))
            est_d, est_v = candidates[pick]
            disp_rmse[i] = errors[pick]
            vel_rmse[i] = float(np.sqrt(np.mean((est_v - true_vel[i]) ** 2)))
            denom = float(np.std(est_d) * np.std(true_disp_centered[i]))
            disp_corr[i] = (
                float(np.mean(est_d * true_disp_centered[i]) / denom)
                if denom > 0
                else 0.0
            )

    return DatasetBaseline(
        dataset=label,
        file=file_path.name,
        n_shots=int(n_shots),
        n_shots_method2=int(n_shots_method2),
        n_samples_per_shot=n_time,
        sample_rate_hz=sample_rate,
        method1=m1,
        method2={
            'displacement_rmse_um': _summary(disp_rmse),
            'velocity_rmse_um_per_ms': _summary(vel_rmse),
            'displacement_correlation': _summary(disp_corr),
            'displacement_nrmse': _summary(disp_rmse / np.std(m2_disp, axis=1)),
            'velocity_nrmse': _summary(vel_rmse / np.std(m2_vel, axis=1)),
        },
        signal_scale={
            # Over method 1's N, which is the larger subset, so these scales
            # describe the cohort the R^2 tables are computed on.
            'true_displacement_rms_um': _summary(np.std(true_disp_centered, axis=1)),
            'true_velocity_rms_um_per_ms': _summary(np.std(true_vel, axis=1)),
            'true_displacement_ptp_um': _summary(np.ptp(true_disp_centered, axis=1)),
        },
        # Derived from this subset's ground truth at run time, so they vary
        # with --n-shots and with the file. They set the decode grid's extent
        # and the Viterbi step band, which means a run cannot be reproduced
        # from the static config block alone -- record them per dataset.
        decode_grid={'displacement_limit_um': limit, 'max_speed_um_per_ms': max_speed},
        runtime_s=time.perf_counter() - start,
    )


def run_baselines(
    n_shots: int = 1000,
    n_shots_method2: int = 200,
    device: str = 'cpu',
    batch_size: int = 16,
    output: Path | None = None,
    datasets: dict[str, str] | None = None,
    grid_points: int = 1201,
    phase_grid_points: int = 16,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    windowing: str = 'fringe-matched',
) -> dict[str, Any]:
    """Evaluate both baselines on every configured dataset and write JSON.

    Args:
        n_shots: Shots per dataset for method 1.
        n_shots_method2: Shots per dataset for method 2, which is ~5.85 s per
            shot and so runs on a smaller subset (see
            :func:`evaluate_dataset`).
        device: Torch device for the method-2 decode.
        batch_size: Shots per decode batch.
        output: Destination JSON path. Defaults to
            ``smi/analysis/data/baseline_results.json``.
        datasets: Mapping of label to file name under ``smi/analysis/data``.
        grid_points: Displacement grid resolution.
        phase_grid_points: Phase search resolution.
        window_samples: Window length for method 1's windowed refit, used
            only when ``windowing='fixed'``.
        windowing: ``'fringe-matched'`` (default) or ``'fixed'``; see
            :func:`evaluate_dataset`.

    Returns:
        The results dictionary that was written to disk.
    """
    datasets = datasets or DEFAULT_DATASETS
    output = output or DEFAULT_RESULTS_PATH

    results: dict[str, Any] = {
        # 3: method 1 gained the windowed refit and its scrambled-theta
        # control, the excursion bins split 2+ into 2-3 and 3+, and the two
        # methods no longer share one shot count.
        # 4: the windowed measure and its control are fringe-matched per shot
        # per channel, so `window_samples` no longer describes them and each
        # channel carries a `window_samples_resolved` distribution instead.
        'schema_version': 4,
        'config': {
            'n_shots': n_shots,
            'n_shots_method2': n_shots_method2,
            # What the windowed measure is. A reader who sees only
            # `window_samples` would take the windowed columns for a
            # fixed-window measure; under 'fringe-matched' that field is
            # inactive and the per-channel `window_samples_resolved` block is
            # what states the window the numbers came from.
            'windowing': windowing,
            'window_samples': window_samples if windowing == 'fixed' else None,
            'fringes_per_window': (
                DEFAULT_FRINGES_PER_WINDOW if windowing == 'fringe-matched' else None
            ),
            'window_clip_samples': (
                [MIN_FRINGE_WINDOW, MAX_FRINGE_WINDOW]
                if windowing == 'fringe-matched'
                else None
            ),
            'device': device,
            'batch_size': batch_size,
            'grid_points': grid_points,
            'phase_grid_points': phase_grid_points,
            'pd_wavelengths_um': dict(PD_WAVELENGTHS_UM),
        },
        'datasets': {},
    }
    for label, name in datasets.items():
        path = DATA_DIR / name
        if not path.exists():
            logger.warning('skipping %s: %s not found', label, path)
            continue
        logger.info('evaluating %s (%s)...', label, name)
        baseline = evaluate_dataset(
            path,
            label,
            n_shots=n_shots,
            n_shots_method2=n_shots_method2,
            batch_size=batch_size,
            device=device,
            grid_points=grid_points,
            phase_grid_points=phase_grid_points,
            window_samples=window_samples,
            windowing=windowing,
        )
        results['datasets'][label] = asdict(baseline)
        m2 = baseline.method2
        for channel, stats in baseline.method1.items():
            # Log the windowed number with its control attached, so even the
            # console output cannot quote one without the other.
            logger.info(
                '  %s %s: method-1 median R^2 global %.3f, windowed %.3f '
                '(scrambled control %.3f) at median W = %.0f',
                label,
                channel,
                stats['r_squared']['median'],
                stats['r_squared_windowed']['median'],
                stats['r_squared_windowed_scrambled']['median'],
                stats['window_samples_resolved']['median'],
            )
        logger.info(
            '  %s: method-2 median displacement RMSE %.4f um, '
            'median velocity RMSE %.4f um/ms (%.1f s)',
            label,
            m2['displacement_rmse_um']['median'],
            m2['velocity_rmse_um_per_ms']['median'],
            baseline.runtime_s,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    logger.info('wrote %s', output)
    return results


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for regenerating the baseline artifact."""
    # These defaults are the settings the committed artifact was produced
    # with, so a bare `python -m smi.analysis.baseline_fit` reproduces it.
    # They previously differed (cpu, 1201 grid points), which meant the
    # documented regeneration command silently returned different numbers
    # than the ones in the file it overwrote -- the config block records
    # them, but only a reader who compared would have noticed.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--n-shots',
        type=int,
        default=1000,
        help='Shots for method 1, which costs about a millisecond per shot.',
    )
    parser.add_argument(
        '--n-shots-method2',
        type=int,
        default=200,
        help='Shots for method 2. Kept well below --n-shots because the '
        'Viterbi decode is ~5.85 s per shot on cuda:0.',
    )
    parser.add_argument(
        '--windowing',
        choices=sorted(_WINDOWING_MODES),
        default='fringe-matched',
        help="How method 1's windowed measure picks its window length. "
        "'fringe-matched' (the default, and what the committed artifact "
        "uses) scales W by each shot's own fringe rate; 'fixed' uses "
        '--window-samples for every shot.',
    )
    parser.add_argument(
        '--window-samples',
        type=int,
        default=DEFAULT_WINDOW_SAMPLES,
        help="Window length for method 1's windowed refit under "
        '--windowing=fixed. Pass 0 to report only the global fit.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Decode device. The Viterbi pass is a long sequential loop: '
        'seconds per shot on a GPU, minutes on CPU.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grid-points', type=int, default=1601)
    parser.add_argument('--phase-grid-points', type=int, default=8)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    run_baselines(
        n_shots=args.n_shots,
        n_shots_method2=args.n_shots_method2,
        device=args.device,
        batch_size=args.batch_size,
        output=args.output,
        grid_points=args.grid_points,
        phase_grid_points=args.phase_grid_points,
        window_samples=args.window_samples or None,
        windowing=args.windowing,
    )


if __name__ == '__main__':
    main()
