"""Method 1: closed-form Michelson forward-model fits of photodiode signals.

Given a displacement-implied phase ``theta = 4*pi*d/lambda``, every estimator
here fits fringe amplitude and phase by linear least squares: globally, per
window, with a shared gain, or with a polynomial envelope. The same fringe
basis also measures within-shot phase drift. Arrays in, arrays out; the
dataset-level evaluation that drives these lives in
:mod:`smi.experiments.baselines`, whose docstring explains what method 1
measures and why.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


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

    design = fringe_basis(theta)
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
    :mod:`smi.estimators.gain_sweep` repeats this fit over a grid of
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
    design = fringe_basis(theta)

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
            design = fringe_basis(theta)
            coeffs, *_ = np.linalg.lstsq(design, signal, rcond=None)
            residual_ss += float(np.sum((signal - design @ coeffs) ** 2))
        else:
            for begin, end in _window_bounds(signal.size, window_samples):
                block, block_theta = signal[begin:end], theta[begin:end]
                design = fringe_basis(block_theta)
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
            design = fringe_basis(block_theta)
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


def fringe_basis(thetas: np.ndarray) -> np.ndarray:
    """``[cos, sin, 1]`` columns on a new last axis, for phases of any shape."""
    return np.stack([np.cos(thetas), np.sin(thetas), np.ones_like(thetas)], axis=-1)


def fit_window_basis(
    signal: np.ndarray,
    theta: np.ndarray,
    window_samples: int,
    basis: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Batched least squares of ``signal`` on ``basis(theta)`` in each whole window.

    Trailing samples that do not fill a window are dropped. The normal
    equations of every window are solved in one batched call.

    Args:
        signal: Samples for one shot, shape ``(n,)``.
        theta: Phase for the same samples (radians), shape ``(n,)``.
        window_samples: Window length in samples.
        basis: Maps phases ``(n_windows, w)`` to a design ``(n_windows, w, k)``.

    Returns:
        ``(n_windows, k)`` coefficients per window.
    """
    n_windows = signal.size // int(window_samples)
    usable = n_windows * int(window_samples)
    blocks = np.asarray(signal[:usable], dtype=np.float64).reshape(n_windows, -1)
    thetas = np.asarray(theta[:usable], dtype=np.float64).reshape(n_windows, -1)
    design = basis(thetas)
    gram = design.transpose(0, 2, 1) @ design
    rhs = design.transpose(0, 2, 1) @ blocks[:, :, None]
    return np.linalg.solve(gram, rhs)[:, :, 0]


def window_fringe_coefficients(
    signal: np.ndarray, theta: np.ndarray, window_samples: int
) -> np.ndarray:
    """Least-squares ``a cos(theta) + b sin(theta) + c`` in each whole window.

    The intercept column makes ``a`` and ``b`` independent of any constant
    added to a window, so ``c`` is that window's DC level and the fringe terms
    need no mean removal. Trailing samples that do not fill a window are
    dropped.

    Args:
        signal: Samples for one shot, shape ``(n,)``.
        theta: Model phase for the same samples (radians), shape ``(n,)``.
        window_samples: Window length in samples.

    Returns:
        ``(n_windows, 3)`` array of ``[a, b, c]`` per window.
    """
    return fit_window_basis(signal, theta, window_samples, fringe_basis)


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

    coeffs = window_fringe_coefficients(signal, theta, window_samples)

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


# Bin edges in microns peak-to-peak. Fixed rather than quantile-derived so
# the bins mean the same thing across datasets and across reruns with a
# different --n-shots, which is the whole point of persisting them.
# The 2-3 um and >3 um ends behave differently -- a gain correction
# recovers much of the former and little of the latter -- so a single
# '2+' top bin hid the distinction that matters.
PTP_BIN_EDGES_UM = (0.0, 0.3, 1.0, 2.0, 3.0, np.inf)


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
