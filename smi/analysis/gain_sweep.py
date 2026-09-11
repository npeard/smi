#!/usr/bin/env python3
"""Is the coil-driver gain constant ``k`` right? A committed answer.

The displacement targets this project trains on are not measured -- they are
derived from the speaker drive voltage through
:class:`~smi.synthetic.coil_driver.CoilDriver`, whose amplitude transfer
function carries a single scalar gain ``k`` (microns per volt at DC, scaled by
a resonance). If ``k`` is wrong, every displacement and velocity target is
wrong by that factor, so the question deserves a measurement rather than an
assumption.

The measurement is method 1 from :mod:`smi.analysis.baseline_fit`, repeated
over a grid of gains. For each candidate gain, the displacement implied by the
drive voltage is pushed through ``cos(4*pi*d/lambda + phi)`` and the per-shot
fringe amplitude and phase are fitted in closed form; the reported score is
the median R^2 across shots, per photodiode channel. A wrong gain smears the
fringe phase and the R^2 collapses, so the curve peaks at the gain the data
actually implies. The peak is what this module reports.

Two structural facts make the sweep cheap, and both are load-bearing:

1. **The transfer function is linear in ``k``.** So the displacement is
   computed once with ``k = 1.0`` and simply scaled per gain. Rebuilding a
   :class:`CoilDriver` and recomputing the FFT for each of the 101 default
   gains would repeat the same 2.7 s of transfer-function work 101 times for
   an identical answer.
2. **The per-shot least-squares solve is already fast.** A batched
   normal-equations rewrite was measured at 0.77 ms per fit against 0.64 ms
   for the loop over :func:`~smi.analysis.baseline_fit.fit_phase_amplitude`,
   agreeing to 2.2e-16 in R^2: LAPACK handles the 16384x3 solve well and
   building the design tensor dominates. The loop stays.

Scores are reported in both method-1 measures. The global single-``(A, B,
offset)`` fit over the whole 33.6 ms record is what the original finding used;
the windowed refit
(:func:`~smi.analysis.baseline_fit.windowed_forward_model_residual`) is the
honest one, because one phase cannot span a record over which real shots drift
by 1-2.7 fringes. Both columns appear per gain so the gain conclusion can be
read in whichever measure the reader trusts.

Results are binned by drive peak-to-peak excursion, using the same edges as
the baseline artifact, because the aggregate median hides the trend that
matters: the gain correction recovers much of the 1-3 um cohort and little of
the >3 um cohort, which is what separates "scale error" from "unmodeled
within-shot drift".

What it found, at N = 1000 on both acquisitions (the numbers in
``gain_sweep_results.json``, global fit):

- The curve is sharply peaked. Best ``k`` is **16.00-16.50** free space and
  **15.25-15.75** mm fiber, against the live 14.553 -- so the live gain is
  **5-13% low**. The three wavelengths within one acquisition agree to ~3%,
  which is what a real mechanical displacement error looks like: the speaker
  does not know which laser is watching it.
- The 2024 calibration's ``k = 33.42``, still commented out in
  ``calib_params.py``, is **refuted**: R^2 0.012-0.029, i.e. noise. The
  file's "Incorrect fringe counts in 2024 calibration?" was right to doubt it.
- Fixing the gain does not fix the large-excursion collapse. On the 635 nm
  channel, live -> best ``k`` moves the 1-2 um bin 0.19 -> 0.56 (free space)
  and 0.54 -> 0.71 (fiber), but the >3 um bin only 0.03 -> 0.12 and
  0.05 -> 0.23.
- **A per-shot optimal gain does not rescue it either**, and this is the
  strongest single result here. Given its own gain, every free-space shot
  above 3 um still scores 0.143 / 0.094 / 0.052 (635 / 675 / 515 nm), against
  0.122 / 0.076 / 0.020 at the best global gain. Since a per-shot gain is
  strictly more freedom than any recalibrated constant could have, the
  leftover error is not a scale error at all -- which is what points at
  unmodeled within-shot phase drift.

One artifact of the per-shot fit worth knowing before reading its numbers:
the mm-fiber 675 nm channel has a per-shot gain IQR of 15.50-30.50, i.e. a
large minority of shots prefer a gain near the refuted legacy value. Those
are spurious fringe-order aliases that a per-shot fit is free to chase and a
single global gain is not, so that channel's per-shot R^2 (0.336 vs 0.230
global) is inflated rather than informative. The median gain, 15.75, is the
trustworthy number there.

Two candidates are ruled out by the same run:

- **Aliasing.** Peak fringe rate over all 1000 shots is 9.9-13.6 kHz
  (evaluated at each channel's best gain; 8.8-13.0 kHz at the live gain)
  against a 244 kHz Nyquist -- 18-25x headroom. The 515 nm channel is the
  binding case, as expected: it crosses the most fringes per micron.
- **A drive-to-photodiode time delay.** Every channel's R^2-versus-shift
  curve declines monotonically across tau in [-8, +8] samples, with no peak:
  the maximum sits at or beside the window's own boundary and stands only
  0.008-0.034 above the value at zero shift. A real delay would rise to an
  interior maximum, so there is none. See :attr:`DelayCurve.peak_is_interior`
  for why "the argmax is not an endpoint" is the wrong test.

Two caveats that must travel with those numbers:

1. **The two acquisitions disagree on the best ``k``** by ~6% (16.50 vs 15.50
   at 635 nm). Drive-to-displacement gain is a property of the speaker, not
   the optical path, so a scalar ``k`` is absorbing something else.
2. So ``k`` is deliberately **left alone** in ``calib_params.py``. Changing it
   would shift every training target, and a constant the two acquisitions
   disagree about has not earned that.

Units follow the package: displacement in microns, velocity in microns per
millisecond, wavelengths in microns, sample rates in Hz. ``k`` is in the same
units the calibration file uses.

Run as a script to regenerate the artifact::

    pixi run -e dev python -m smi.analysis.gain_sweep

This deliberately writes its own JSON file rather than extending
``baseline_results.json``: that artifact's method 2 is a ~20 min GPU Viterbi
decode per acquisition, and a gain sweep has no business triggering it.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from smi.analysis.baseline_fit import (
    DATA_DIR,
    DEFAULT_DATASETS,
    DEFAULT_WINDOW_SAMPLES,
    DRIVE_KEY,
    PD_WAVELENGTHS_UM,
    binned_by_amplitude,
    forward_model_residual,
    windowed_forward_model_residual,
)
from smi.synthetic.calib_params import CalibrationParameters
from smi.synthetic.coil_driver import MS_PER_S, CoilDriver

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = DATA_DIR / 'gain_sweep_results.json'

# The live constant, for ratio reporting. Read from the calibration file rather
# than copied, so this module cannot drift away from the value the training
# targets are actually built with.
LIVE_K = CalibrationParameters().k

# Sweep range and step. The range brackets the live 14.553 generously on both
# sides and reaches the 2024 calibration's 33.42, so that value is refuted by
# the same curve rather than by a separate run. A 0.25 step is 1.7% of the
# live gain, so it resolves a peak to well inside the ~6% the two acquisitions
# disagree by -- a coarser step could not have shown that disagreement is real.
DEFAULT_GAIN_LOW = 10.0
DEFAULT_GAIN_HIGH = 35.0
DEFAULT_GAIN_STEP = 0.25

# The 2024 calibration's gain, kept commented out in calib_params.py with the
# note "Incorrect fringe counts in 2024 calibration?". Included in the default
# grid so the sweep adjudicates it.
LEGACY_K = 33.42493417407945


def unit_gain_displacement(
    drive_voltage: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and displacement for a coil driver with ``k = 1``.

    The amplitude transfer function is ``k * f0^2 / sqrt(...)`` -- strictly
    proportional to ``k`` -- so a sweep over gains is a sweep over scalar
    multiples of this one result. Computing it once and scaling is exact, not
    an approximation; :mod:`tests.test_gain_sweep` pins that against a rebuilt
    driver.

    Args:
        drive_voltage: Speaker drive voltage for one shot (V), shape ``(n,)``.
        sample_rate: Acquisition sample rate (Hz).

    Returns:
        Tuple of (velocity per unit gain in um/ms, displacement per unit gain
        in um), each shape ``(n,)``. Multiply either by a gain to get the
        physical quantity at that gain.
    """
    driver = CoilDriver(CalibrationParameters(k=1.0))
    return driver.get_velocity_and_displacement(drive_voltage, sample_rate)


# ---------------------------------------------------------------------------
# Two candidate explanations this module rules out, committed rather than
# recalled. Neither survives, but "ruled out" is only a claim if the check
# that ruled it out can be rerun -- and each check is only evidence if it
# would have *detected* the effect it dismisses, which is what the tests pin.
# ---------------------------------------------------------------------------


def peak_fringe_rate_hz(velocity_um_per_ms: np.ndarray, wavelength_um: float) -> float:
    """Fastest instantaneous fringe frequency implied by a velocity record.

    Aliasing is a candidate explanation for a poor forward-model fit: if the
    mirror crosses fringes faster than the digitizer can sample, the recorded
    signal is not the signal the model predicts. The Michelson phase is
    ``4*pi*d/lambda``, so its time derivative is ``4*pi*v/lambda`` rad/s and
    the instantaneous fringe frequency is ``2*v/lambda`` in Hz. The peak over
    the record is what must be compared to Nyquist.

    Args:
        velocity_um_per_ms: Mirror velocity for one shot (um/ms), shape
            ``(n,)``. Any shape is accepted; the maximum is taken over all of
            it.
        wavelength_um: Laser wavelength (microns).

    Returns:
        Peak fringe rate in Hz.
    """
    # velocity is um/ms, so multiplying by MS_PER_S gives um/s and the result
    # comes out in Hz rather than kHz. Getting this wrong would understate the
    # rate by 1000x and make any headroom claim vacuous.
    speed_um_per_s = np.abs(np.asarray(velocity_um_per_ms, dtype=np.float64)) * MS_PER_S
    return float(2.0 * speed_um_per_s.max() / wavelength_um)


def fringe_rate_headroom(
    velocity_um_per_ms: np.ndarray, wavelength_um: float, sample_rate: float
) -> float:
    """Ratio of the Nyquist frequency to the peak fringe rate.

    A value above 1 means the fringes are sampled; well above 1 means aliasing
    cannot be the explanation for anything.

    Args:
        velocity_um_per_ms: Mirror velocity for one shot (um/ms).
        wavelength_um: Laser wavelength (microns).
        sample_rate: Acquisition sample rate (Hz).

    Returns:
        Nyquist frequency divided by the peak fringe rate. ``inf`` for a
        stationary mirror.
    """
    rate = peak_fringe_rate_hz(velocity_um_per_ms, wavelength_um)
    return float('inf') if rate == 0.0 else float(sample_rate / 2.0 / rate)


@dataclass
class DelayCurve:
    """R^2 of the forward-model fit versus an integer sample delay.

    Attributes:
        shifts: Sample shifts applied to the displacement, increasing, from
            ``-max_shift`` to ``+max_shift``.
        r_squared: Global-fit R^2 at each shift.
    """

    shifts: np.ndarray
    r_squared: np.ndarray

    @property
    def best_shift(self) -> int:
        """Shift maximizing R^2."""
        return int(self.shifts[int(np.argmax(self.r_squared))])

    # Smallest R^2 rise on both sides of the argmax that counts as a peak
    # rather than a ripple. The measured curves ripple by ~3e-4 at their flat
    # end, and their whole-window span is ~0.03, so 2e-3 sits an order of
    # magnitude above the noise and an order below any effect worth acting on.
    _MIN_PEAK_PROMINENCE = 2e-3

    @property
    def peak_is_interior(self) -> bool:
        """Whether the curve actually peaks, which is the delay signature.

        A real time delay makes R^2 *rise* to a maximum at some nonzero shift
        and fall away on both sides. Everything else is not a delay:

        - an argmax at zero means no misalignment;
        - an argmax pinned to the window's boundary means the curve was still
          trending when the sweep ran out of window;
        - an argmax one index in from a nearly-flat edge is that same trend
          with a ripple on it.

        The third case is why this is not simply "the argmax is not an
        endpoint". The measured curves are flat to ~3e-4 at their left edge
        and decline monotonically after; on two of six channels the ripple
        alone moved the argmax off the boundary, which an endpoint test then
        reported as a real delay on a curve that never rises. So the test is
        prominence: the maximum must exceed both of the curve's own endpoints
        by :data:`_MIN_PEAK_PROMINENCE`.

        Returns:
            True only if the maximum is at a nonzero shift and stands clear of
            both endpoints by more than the prominence floor.
        """
        best = self.best_shift
        if best == 0:
            return False
        peak = float(self.r_squared.max())
        rise_from_left = peak - float(self.r_squared[0])
        rise_from_right = peak - float(self.r_squared[-1])
        return bool(min(rise_from_left, rise_from_right) > self._MIN_PEAK_PROMINENCE)

    def r_squared_at(self, shift: int) -> float:
        """R^2 at one specific shift.

        Args:
            shift: Sample shift, which must be inside the swept window.

        Returns:
            The R^2 recorded at that shift.

        Raises:
            ValueError: If ``shift`` was not swept.
        """
        matches = np.nonzero(self.shifts == shift)[0]
        if matches.size == 0:
            raise ValueError(f'shift {shift} was not swept')
        return float(self.r_squared[matches[0]])


def delay_sweep(
    pd_signal: np.ndarray,
    displacement: np.ndarray,
    wavelength_um: float,
    max_shift: int = 8,
) -> DelayCurve:
    """Score the forward model against the displacement shifted in time.

    A trigger or acquisition delay between the drive channel and the
    photodiode channels would mean the model is being asked to explain the
    signal with a misaligned displacement. Such a delay has a signature: R^2
    peaks at a nonzero shift. Its absence -- a monotone, near-flat curve with
    its maximum at the window edge or at zero -- is what rules a delay out.

    Args:
        pd_signal: Photodiode samples for one shot (V), shape ``(n,)``.
        displacement: Mirror displacement for the same shot (microns), shape
            ``(n,)``.
        wavelength_um: Laser wavelength for this channel (microns).
        max_shift: Largest shift in samples, swept in both directions.

    Returns:
        The R^2-versus-shift curve.

    Raises:
        ValueError: If ``max_shift`` is negative.
    """
    if max_shift < 0:
        raise ValueError(f'max_shift must be non-negative, got {max_shift}')
    shifts = np.arange(-max_shift, max_shift + 1)
    scores = np.empty(shifts.size)
    for j, shift in enumerate(shifts):
        # np.roll wraps rather than truncating. At |shift| <= 8 out of 16384
        # samples the wrapped tail is 0.05% of the record, far too little to
        # move an R^2, and truncating instead would score each shift on a
        # slightly different span -- which is worse, because then the curve's
        # shape would partly reflect the changing denominator.
        scores[j] = forward_model_residual(
            pd_signal, np.roll(displacement, shift), wavelength_um
        ).r_squared
    return DelayCurve(shifts=shifts, r_squared=scores)


def gain_grid(low: float, high: float, step: float) -> np.ndarray:
    """Inclusive grid of candidate gains.

    Args:
        low: Smallest gain to score.
        high: Largest gain to score; included even when ``high - low`` is not
            an exact multiple of ``step``, because the endpoint is usually
            chosen to hit a specific value of interest.
        step: Spacing between candidate gains.

    Returns:
        Increasing 1-D array of gains from ``low`` to ``high``.

    Raises:
        ValueError: If ``step`` is not positive or ``high`` does not exceed
            ``low``.
    """
    if step <= 0:
        raise ValueError(f'step must be positive, got {step}')
    if high <= low:
        raise ValueError(f'high must exceed low, got low={low}, high={high}')
    n = int(np.floor(round((high - low) / step, 9))) + 1
    grid = low + step * np.arange(n, dtype=np.float64)
    if grid[-1] < high - 1e-9:
        grid = np.append(grid, high)
    return grid


@dataclass
class ChannelGainCurve:
    """Median R^2 versus gain for one photodiode channel.

    Attributes:
        wavelength_um: Laser wavelength for this channel (microns).
        median_r_squared: Median global-fit R^2 across shots, one entry per
            swept gain.
        median_r_squared_windowed: The same in the windowed measure, or
            ``None`` when windowing was disabled.
        binned_r_squared: Global-fit R^2 grouped by drive excursion, one list
            of bins per swept gain. The excursion is evaluated at the gain
            being scored, so bin membership shifts with the gain -- which is
            correct, since a gain change is a change in how far the mirror
            moved.
        binned_r_squared_windowed: The windowed equivalent, or ``None``.
        peak_fringe_rate_hz: Highest fringe rate any shot reaches at the best
            gain, in Hz. Compare to Nyquist to rule aliasing in or out.
        fringe_rate_headroom: Nyquist divided by ``peak_fringe_rate_hz``.
        delay_shifts: Sample shifts the delay check swept, or ``None`` if it
            was not run.
        delay_median_r_squared: Median R^2 at each swept shift, or ``None``.
    """

    wavelength_um: float
    median_r_squared: list[float] = field(default_factory=list)
    median_r_squared_windowed: list[float] | None = None
    binned_r_squared: list[list[dict[str, float | int | str]]] = field(
        default_factory=list
    )
    binned_r_squared_windowed: list[list[dict[str, float | int | str]]] | None = None
    peak_fringe_rate_hz: float | None = None
    fringe_rate_headroom: float | None = None
    delay_shifts: list[int] | None = None
    delay_median_r_squared: list[float] | None = None


@dataclass
class GainSweep:
    """Outcome of scoring the Michelson forward model over a grid of gains.

    Attributes:
        dataset: Short dataset label.
        file: File name the signals came from, or ``'<in-memory>'``.
        n_shots: Shots scored at every gain.
        n_samples_per_shot: Time samples per shot.
        window_samples: Window length used for the windowed measure, or
            ``None`` if only the global fit was computed.
        gains: The swept gains, increasing.
        curves: Per-channel median-R^2 curves, keyed by channel name.
        nyquist_hz: Half the sample rate, or ``None`` if no sample rate was
            supplied and so the bandwidth check did not run.
        per_shot: Upper bound from giving each shot its own optimal gain, or
            ``None`` if it was not computed. Carried here rather than returned
            separately so the caveat cannot be quoted apart from the claim it
            qualifies.
        runtime_s: Wall-clock seconds spent on the sweep.
    """

    dataset: str
    file: str
    n_shots: int
    n_samples_per_shot: int
    window_samples: int | None
    gains: np.ndarray
    curves: dict[str, ChannelGainCurve] = field(default_factory=dict)
    nyquist_hz: float | None = None
    per_shot: PerShotGainSweep | None = None
    runtime_s: float = 0.0

    @property
    def channels(self) -> list[str]:
        """Channel names in the order they were scored."""
        return list(self.curves)

    def _curve(self, channel: str, windowed: bool) -> list[float]:
        """The requested median-R^2 curve, or a clear error if absent."""
        curve = self.curves[channel]
        if not windowed:
            return curve.median_r_squared
        if curve.median_r_squared_windowed is None:
            raise ValueError(
                'no windowed measure in this sweep; rerun with window_samples set'
            )
        return curve.median_r_squared_windowed

    def median_r_squared(self, channel: str) -> list[float]:
        """Global-fit median R^2 per swept gain for one channel."""
        return self.curves[channel].median_r_squared

    def median_r_squared_windowed(self, channel: str) -> list[float] | None:
        """Windowed median R^2 per swept gain, or ``None`` if not computed."""
        return self.curves[channel].median_r_squared_windowed

    def binned_r_squared(
        self, channel: str, windowed: bool = False
    ) -> list[list[dict[str, float | int | str]]]:
        """Excursion-binned R^2 per swept gain for one channel.

        Args:
            channel: Channel name.
            windowed: Report the windowed measure instead of the global fit.

        Returns:
            One list of bin dictionaries per swept gain.

        Raises:
            ValueError: If ``windowed`` is requested but was not computed.
        """
        curve = self.curves[channel]
        if not windowed:
            return curve.binned_r_squared
        if curve.binned_r_squared_windowed is None:
            raise ValueError(
                'no windowed measure in this sweep; rerun with window_samples set'
            )
        return curve.binned_r_squared_windowed

    def best_gain(self, channel: str, windowed: bool = False) -> float:
        """Gain maximizing median R^2 for one channel.

        Args:
            channel: Channel name.
            windowed: Use the windowed measure instead of the global fit.

        Returns:
            The gain at the curve's peak.

        Raises:
            ValueError: If ``windowed`` is requested but was not computed.
        """
        curve = np.asarray(self._curve(channel, windowed), dtype=np.float64)
        return float(self.gains[int(np.argmax(curve))])

    def best_r_squared(self, channel: str, windowed: bool = False) -> float:
        """Peak median R^2 for one channel.

        Args:
            channel: Channel name.
            windowed: Use the windowed measure instead of the global fit.

        Returns:
            The maximum of the median-R^2 curve.

        Raises:
            ValueError: If ``windowed`` is requested but was not computed.
        """
        return float(np.max(self._curve(channel, windowed)))

    def r_squared_at(self, channel: str, gain: float, windowed: bool = False) -> float:
        """Median R^2 at the swept gain nearest ``gain``.

        Nearest rather than exact so a caller can ask about the live constant
        without having to place it on the grid by hand.

        Args:
            channel: Channel name.
            gain: Gain of interest.
            windowed: Use the windowed measure instead of the global fit.

        Returns:
            Median R^2 at the closest swept gain.

        Raises:
            ValueError: If ``windowed`` is requested but was not computed.
        """
        index = int(np.argmin(np.abs(np.asarray(self.gains) - gain)))
        return float(self._curve(channel, windowed)[index])

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary, including the derived headline numbers.

        The best gain and its ratio against the live constant are stored
        rather than left for a reader to recompute: the whole point of this
        phase is that the claim is checkable against a committed artifact, and
        a claim that requires re-deriving an argmax is one step less checkable.

        Returns:
            Nested plain-Python dictionary ready for :func:`json.dumps`.
        """
        channels: dict[str, Any] = {}
        for name, curve in self.curves.items():
            entry: dict[str, Any] = {
                'wavelength_um': curve.wavelength_um,
                'median_r_squared': list(curve.median_r_squared),
                'median_r_squared_windowed': (
                    list(curve.median_r_squared_windowed)
                    if curve.median_r_squared_windowed is not None
                    else None
                ),
                'binned_r_squared': curve.binned_r_squared,
                'binned_r_squared_windowed': curve.binned_r_squared_windowed,
                'best_gain': self.best_gain(name),
                'best_gain_ratio_vs_live': self.best_gain(name) / LIVE_K,
                'best_r_squared': self.best_r_squared(name),
                'r_squared_at_live_k': self.r_squared_at(name, LIVE_K),
                'r_squared_at_legacy_k': self.r_squared_at(name, LEGACY_K),
            }
            if curve.median_r_squared_windowed is not None:
                entry['best_gain_windowed'] = self.best_gain(name, windowed=True)
                entry['best_gain_windowed_ratio_vs_live'] = (
                    self.best_gain(name, windowed=True) / LIVE_K
                )
                entry['best_r_squared_windowed'] = self.best_r_squared(
                    name, windowed=True
                )
                entry['r_squared_windowed_at_live_k'] = self.r_squared_at(
                    name, LIVE_K, windowed=True
                )
            if curve.peak_fringe_rate_hz is not None:
                entry['peak_fringe_rate_hz'] = curve.peak_fringe_rate_hz
                entry['fringe_rate_headroom'] = curve.fringe_rate_headroom
            if curve.delay_shifts is not None:
                assert curve.delay_median_r_squared is not None
                # The whole curve, not just its argmax. "No interior peak" is
                # the evidence that rules a delay out, and a reader can only
                # see that from the shape.
                pooled = DelayCurve(
                    shifts=np.asarray(curve.delay_shifts),
                    r_squared=np.asarray(curve.delay_median_r_squared),
                )
                entry['delay_sweep'] = {
                    'shifts': list(curve.delay_shifts),
                    'median_r_squared': list(curve.delay_median_r_squared),
                    'best_shift': pooled.best_shift,
                    # The verdict, not just the argmax. An argmax pinned to
                    # the window edge is a monotone trend, and recording only
                    # "best_shift: -8" would read as a detected delay.
                    'peak_is_interior': pooled.peak_is_interior,
                    'r_squared_at_zero': pooled.r_squared_at(0),
                }
            channels[name] = entry
        return {
            'dataset': self.dataset,
            'file': self.file,
            'n_shots': self.n_shots,
            'n_samples_per_shot': self.n_samples_per_shot,
            'window_samples': self.window_samples,
            'live_k': LIVE_K,
            'legacy_k': LEGACY_K,
            'nyquist_hz': self.nyquist_hz,
            'gains': [float(g) for g in self.gains],
            'channels': channels,
            # Sits inside the same dataset block as the best-global-gain
            # numbers it qualifies: a per-shot gain is the most generous
            # scalar correction possible, so where it still fails, no
            # recalibration of k can succeed.
            'per_shot_gain': (
                self.per_shot.to_dict() if self.per_shot is not None else None
            ),
            'runtime_s': self.runtime_s,
        }


def _check_sweep_shapes(
    signals: np.ndarray, unit: np.ndarray, wavelengths: np.ndarray
) -> None:
    """Reject inputs whose shapes cannot be scored together.

    Args:
        signals: Photodiode signals, expected ``(n_shots, n_channels, n_time)``.
        unit: Displacement per unit gain, expected ``(n_shots, n_time)``.
        wavelengths: Channel wavelengths, expected length ``n_channels``.

    Raises:
        ValueError: If the signal array is not 3-D, the shot counts disagree,
            the channel count does not match the wavelength list, or the
            record lengths disagree.
    """
    if signals.ndim != 3:
        raise ValueError(f'pd_signals must be 3-D, got shape {signals.shape}')
    if unit.ndim != 2 or unit.shape[0] != signals.shape[0]:
        raise ValueError(
            f'pd_signals and unit_displacement must agree on shots, got '
            f'{signals.shape[0]} and {unit.shape[0] if unit.ndim == 2 else unit.shape}'
        )
    if signals.shape[1] != wavelengths.size:
        raise ValueError(
            f'got {signals.shape[1]} channels but {wavelengths.size} wavelengths'
        )
    if unit.shape[1] != signals.shape[2]:
        raise ValueError(
            f'unit_displacement has {unit.shape[1]} samples but pd_signals has '
            f'{signals.shape[2]}'
        )


def _attach_ruled_out_checks(
    curves: dict[str, ChannelGainCurve],
    names: Sequence[str],
    signals: np.ndarray,
    unit: np.ndarray,
    unit_velocity: np.ndarray,
    wavelengths: np.ndarray,
    gains: np.ndarray,
    sample_rate: float,
    delay_max_shift: int,
) -> None:
    """Fill in the aliasing and time-delay checks on each channel's curve.

    Both are evaluated at the gain that channel's own curve peaked at, not at
    the live constant: a gain error depresses R^2 at every shift and misstates
    the fringe rate, so checking at the live gain would confound the two
    candidates with the one that is actually real.

    Args:
        curves: Per-channel curves to mutate in place.
        names: Channel names in scoring order.
        signals: Photodiode signals, ``(n_shots, n_channels, n_time)``.
        unit: Displacement per unit gain (microns), ``(n_shots, n_time)``.
        unit_velocity: Velocity per unit gain (um/ms), same shape as ``unit``.
        wavelengths: Channel wavelengths (microns).
        gains: The swept gains.
        sample_rate: Acquisition sample rate (Hz).
        delay_max_shift: Half-width of the delay sweep; 0 skips it.
    """
    n_shots = signals.shape[0]
    for k, name in enumerate(names):
        curve = curves[name]
        wavelength = float(wavelengths[k])
        best = float(gains[int(np.argmax(curve.median_r_squared))])
        curve.peak_fringe_rate_hz = peak_fringe_rate_hz(
            best * unit_velocity, wavelength
        )
        curve.fringe_rate_headroom = fringe_rate_headroom(
            best * unit_velocity, wavelength, sample_rate
        )
        if delay_max_shift <= 0:
            continue
        # One curve per channel, median over shots. A single shot's curve is
        # noisy enough to show a spurious interior bump, and "no interior
        # peak" is the whole content of the conclusion.
        displacement = best * unit
        per_shot = np.empty((n_shots, 2 * delay_max_shift + 1))
        shifts = np.arange(-delay_max_shift, delay_max_shift + 1)
        for i in range(n_shots):
            per_shot[i] = delay_sweep(
                signals[i, k], displacement[i], wavelength, max_shift=delay_max_shift
            ).r_squared
        curve.delay_shifts = [int(s) for s in shifts]
        curve.delay_median_r_squared = [float(v) for v in np.median(per_shot, axis=0)]


def sweep_gain(
    pd_signals: np.ndarray,
    unit_displacement: np.ndarray,
    wavelengths_um: Sequence[float],
    gains: np.ndarray,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    channel_names: Sequence[str] | None = None,
    dataset: str = 'in-memory',
    file: str = '<in-memory>',
    unit_velocity: np.ndarray | None = None,
    sample_rate: float | None = None,
    delay_max_shift: int = 8,
) -> GainSweep:
    """Score the forward model at each gain and report median R^2 per channel.

    For every gain the displacement is ``gain * unit_displacement``, the
    method-1 closed-form fit runs per shot per channel, and the per-shot R^2
    values are reduced to a median (robust to the few shots whose fringe
    contrast is poor) and to excursion bins.

    When ``unit_velocity`` and ``sample_rate`` are both supplied, two
    competing explanations for a poor fit are also checked *at the winning
    gain* and recorded: aliasing (peak fringe rate against Nyquist) and a
    drive-to-photodiode time delay (R^2 versus an integer sample shift). They
    are evaluated at the best gain rather than the live one because a gain
    error would otherwise contaminate both -- the wrong displacement scale
    gives the wrong fringe rate and a depressed R^2 at every shift.

    Args:
        pd_signals: Photodiode signals, shape ``(n_shots, n_channels, n_time)``.
        unit_displacement: Displacement per unit gain (microns), shape
            ``(n_shots, n_time)``, as returned by
            :func:`unit_gain_displacement`.
        wavelengths_um: Channel wavelengths (microns), length ``n_channels``.
        gains: Candidate gains to score, shape ``(n_gains,)``.
        window_samples: Window length for the windowed measure. ``None``
            computes the global fit only, which roughly halves the runtime.
        channel_names: Names to key the result by. Defaults to the photodiode
            channel names in :data:`PD_WAVELENGTHS_UM` order.
        dataset: Label recorded in the result.
        file: File name recorded in the result.
        unit_velocity: Velocity per unit gain (um/ms), same shape as
            ``unit_displacement``. Required for the bandwidth check.
        sample_rate: Acquisition sample rate (Hz). Required for the bandwidth
            check, which needs a Nyquist frequency to compare against.
        delay_max_shift: Half-width of the delay sweep in samples. Pass 0 to
            skip it.

    Returns:
        The per-channel gain curves, plus the ruled-out checks when the inputs
        for them were supplied.

    Raises:
        ValueError: If the shot counts disagree, or the channel count does not
            match the wavelength list.
    """
    start = time.perf_counter()
    signals = np.asarray(pd_signals, dtype=np.float64)
    unit = np.asarray(unit_displacement, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)

    _check_sweep_shapes(signals, unit, wavelengths)

    names = (
        list(channel_names)
        if channel_names
        else list(PD_WAVELENGTHS_UM)[: signals.shape[1]]
    )
    n_shots, _n_channels, n_time = signals.shape

    # Excursion is a property of the displacement, so it is measured once per
    # gain and shared across channels rather than recomputed three times.
    unit_centered = unit - unit.mean(axis=1, keepdims=True)
    unit_ptp = np.ptp(unit_centered, axis=1)

    curves = {
        name: ChannelGainCurve(
            wavelength_um=float(wavelengths[k]),
            median_r_squared_windowed=None if window_samples is None else [],
            binned_r_squared_windowed=None if window_samples is None else [],
        )
        for k, name in enumerate(names)
    }

    for gain in gains:
        displacement = gain * unit
        # Bin membership follows the gain: at gain g a shot's excursion is
        # g * unit_ptp, so scoring at the unit gain and binning at the live one
        # (or vice versa) would attribute every number to the wrong cohort.
        ptp = gain * unit_ptp
        for k, name in enumerate(names):
            r2 = np.empty(n_shots)
            windowed = np.empty(n_shots) if window_samples is not None else None
            for i in range(n_shots):
                r2[i] = forward_model_residual(
                    signals[i, k], displacement[i], wavelengths[k]
                ).r_squared
                if windowed is not None:
                    windowed[i] = windowed_forward_model_residual(
                        signals[i, k],
                        displacement[i],
                        wavelengths[k],
                        window_samples=window_samples,
                    ).r_squared
            curve = curves[name]
            curve.median_r_squared.append(float(np.median(r2)))
            curve.binned_r_squared.append(binned_by_amplitude(r2, ptp))
            if windowed is not None:
                assert curve.median_r_squared_windowed is not None
                assert curve.binned_r_squared_windowed is not None
                curve.median_r_squared_windowed.append(float(np.median(windowed)))
                curve.binned_r_squared_windowed.append(
                    binned_by_amplitude(windowed, ptp)
                )

    # --- the two ruled-out candidates, at the winning gain ------------------
    nyquist_hz: float | None = None
    if unit_velocity is not None and sample_rate is not None:
        nyquist_hz = sample_rate / 2.0
        _attach_ruled_out_checks(
            curves,
            names,
            signals,
            unit,
            np.asarray(unit_velocity, dtype=np.float64),
            wavelengths,
            gains,
            sample_rate,
            delay_max_shift,
        )

    return GainSweep(
        dataset=dataset,
        file=file,
        n_shots=int(n_shots),
        n_samples_per_shot=int(n_time),
        window_samples=window_samples,
        gains=gains,
        curves=curves,
        nyquist_hz=nyquist_hz,
        runtime_s=time.perf_counter() - start,
    )


@dataclass
class PerShotChannelGains:
    """Per-shot optimal gains and the R^2 they buy, for one channel.

    Attributes:
        wavelength_um: Laser wavelength for this channel (microns).
        best_gains: The gain maximizing R^2 for each shot, in shot order.
        r_squared: The R^2 achieved at each shot's own best gain.
        binned_r_squared: Those R^2 values grouped by excursion, each shot
            binned at its own fitted gain.
    """

    wavelength_um: float
    best_gains: list[float] = field(default_factory=list)
    r_squared: list[float] = field(default_factory=list)
    binned_r_squared: list[dict[str, float | int | str]] = field(default_factory=list)


@dataclass
class PerShotGainSweep:
    """Upper bound on what any scalar gain correction can achieve.

    This exists to answer one question the global sweep cannot: is the
    residual amplitude dependence a *scale* error at all? Giving every shot
    its own optimal gain is the most generous scale correction possible -- far
    more freedom than a recalibrated constant would have. If the
    large-excursion cohort still collapses under that, the leftover error is
    not a scale error, which is what points at unmodeled within-shot drift.

    Attributes:
        dataset: Short dataset label.
        n_shots: Shots fitted.
        gains: The candidate gains each shot chose from.
        curves: Per-channel results, keyed by channel name.
        runtime_s: Wall-clock seconds spent fitting.
    """

    dataset: str
    n_shots: int
    gains: np.ndarray
    curves: dict[str, PerShotChannelGains] = field(default_factory=dict)
    runtime_s: float = 0.0

    @property
    def channels(self) -> list[str]:
        """Channel names in the order they were fitted."""
        return list(self.curves)

    def best_gains(self, channel: str) -> list[float]:
        """Each shot's own optimal gain, in shot order."""
        return self.curves[channel].best_gains

    def median_r_squared(self, channel: str) -> float:
        """Median over shots of the R^2 at each shot's own best gain."""
        return float(np.median(self.curves[channel].r_squared))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary.

        Returns:
            Nested plain-Python dictionary ready for :func:`json.dumps`.
        """
        return {
            'dataset': self.dataset,
            'n_shots': self.n_shots,
            'gain_low': float(self.gains[0]),
            'gain_high': float(self.gains[-1]),
            'live_k': LIVE_K,
            'channels': {
                name: {
                    'wavelength_um': curve.wavelength_um,
                    'best_gains': list(curve.best_gains),
                    'median_best_gain': float(np.median(curve.best_gains)),
                    'p25_best_gain': float(np.percentile(curve.best_gains, 25)),
                    'p75_best_gain': float(np.percentile(curve.best_gains, 75)),
                    'median_r_squared': self.median_r_squared(name),
                    'binned_r_squared': curve.binned_r_squared,
                }
                for name, curve in self.curves.items()
            },
            'runtime_s': self.runtime_s,
        }


def per_shot_best_gain(
    pd_signals: np.ndarray,
    unit_displacement: np.ndarray,
    wavelengths_um: Sequence[float],
    gains: np.ndarray,
    channel_names: Sequence[str] | None = None,
    dataset: str = 'in-memory',
) -> PerShotGainSweep:
    """Give every shot its own optimal gain and report what that buys.

    The result is an upper bound, not a proposal: nobody could deploy a
    per-shot gain, since it requires knowing the answer. Its purpose is
    falsification. If the >3 um cohort stays low even here, then no scalar
    recalibration of ``k`` -- however carefully fitted -- can explain the
    residual, and the remaining error must be something a scale cannot reach.

    Args:
        pd_signals: Photodiode signals, shape ``(n_shots, n_channels, n_time)``.
        unit_displacement: Displacement per unit gain (microns), shape
            ``(n_shots, n_time)``.
        wavelengths_um: Channel wavelengths (microns), length ``n_channels``.
        gains: Candidate gains each shot may choose from.
        channel_names: Names to key the result by. Defaults to the photodiode
            channel names in :data:`PD_WAVELENGTHS_UM` order.
        dataset: Label recorded in the result.

    Returns:
        Each shot's fitted gain and the R^2 it achieves, per channel.

    Raises:
        ValueError: If the shot counts disagree, or the channel count does not
            match the wavelength list.
    """
    start = time.perf_counter()
    signals = np.asarray(pd_signals, dtype=np.float64)
    unit = np.asarray(unit_displacement, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    _check_sweep_shapes(signals, unit, wavelengths)

    names = (
        list(channel_names)
        if channel_names
        else list(PD_WAVELENGTHS_UM)[: signals.shape[1]]
    )
    n_shots = signals.shape[0]
    unit_ptp = np.ptp(unit - unit.mean(axis=1, keepdims=True), axis=1)

    curves: dict[str, PerShotChannelGains] = {}
    for k, name in enumerate(names):
        scores = np.empty((n_shots, gains.size))
        for j, gain in enumerate(gains):
            for i in range(n_shots):
                scores[i, j] = forward_model_residual(
                    signals[i, k], gain * unit[i], wavelengths[k]
                ).r_squared
        pick = np.argmax(scores, axis=1)
        best = gains[pick]
        r2 = scores[np.arange(n_shots), pick]
        curves[name] = PerShotChannelGains(
            wavelength_um=float(wavelengths[k]),
            best_gains=[float(g) for g in best],
            r_squared=[float(v) for v in r2],
            # Each shot is binned at its own fitted gain, since that is the
            # excursion its reported R^2 was achieved at.
            binned_r_squared=binned_by_amplitude(r2, best * unit_ptp),
        )

    return PerShotGainSweep(
        dataset=dataset,
        n_shots=int(n_shots),
        gains=gains,
        curves=curves,
        runtime_s=time.perf_counter() - start,
    )


def sweep_dataset(
    file_path: Path,
    label: str,
    n_shots: int = 1000,
    gains: np.ndarray | None = None,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    delay_max_shift: int = 8,
    include_per_shot: bool = True,
) -> GainSweep:
    """Sweep the gain over a leading slice of one acquisition HDF5 file.

    N = 1000 by default, which is worth the cost: N = 200 left the extreme
    excursion bins at n = 14-72, too thin to carry the trend that separates a
    scale error from unmodeled drift, while N = 1000 gives n = 84-310 per bin.
    Measured cost with every option on (101 gains, windowed measure, delay
    check, per-shot fit) is ~9.3 min per acquisition on CPU; the global fit
    alone is ~3 min. Method 2's N = 200 in :mod:`smi.analysis.baseline_fit` is
    a different budget for a ~5.85 s/shot GPU decode and must not be
    conflated with this one.

    Args:
        file_path: Path to the acquisition HDF5 file.
        label: Short label recorded in the result.
        n_shots: Shots to score, capped at the file's shot count.
        gains: Candidate gains. Defaults to :func:`gain_grid` over
            ``[DEFAULT_GAIN_LOW, DEFAULT_GAIN_HIGH]``, which brackets both the
            live constant and the 2024 calibration's value.
        window_samples: Window length for the windowed measure, or ``None``
            for the global fit only.
        delay_max_shift: Half-width of the delay check in samples, swept at
            the winning gain. Pass 0 to skip it.
        include_per_shot: Also fit a separate optimal gain per shot, which
            bounds what any scalar correction could achieve. Roughly doubles
            the global-fit work; see :func:`per_shot_best_gain`.

    Returns:
        The per-channel gain curves for this acquisition, plus the aliasing
        and time-delay checks that rule those two candidates out, and the
        per-shot upper bound when requested.
    """
    if gains is None:
        gains = gain_grid(DEFAULT_GAIN_LOW, DEFAULT_GAIN_HIGH, DEFAULT_GAIN_STEP)

    channels = list(PD_WAVELENGTHS_UM)
    wavelengths = [PD_WAVELENGTHS_UM[c] for c in channels]

    with h5py.File(file_path, 'r') as handle:
        sample_rate = float(handle.attrs['sample_rate'])
        n_shots = min(n_shots, handle[DRIVE_KEY].shape[0])
        drive = np.asarray(handle[DRIVE_KEY][:n_shots], dtype=np.float64)
        pd_raw = np.stack(
            [np.asarray(handle[c][:n_shots], dtype=np.float64) for c in channels],
            axis=1,
        )

    # One k=1 evaluation per shot, reused for every gain in the grid. This is
    # the whole reason a 101-gain sweep costs minutes rather than hours.
    unit = np.empty_like(drive)
    unit_vel = np.empty_like(drive)
    for i in range(n_shots):
        unit_vel[i], unit[i] = unit_gain_displacement(drive[i], sample_rate)

    sweep = sweep_gain(
        pd_raw,
        unit,
        wavelengths,
        gains,
        window_samples=window_samples,
        channel_names=channels,
        dataset=label,
        file=file_path.name,
        unit_velocity=unit_vel,
        sample_rate=sample_rate,
        delay_max_shift=delay_max_shift,
    )
    if include_per_shot:
        sweep.per_shot = per_shot_best_gain(
            pd_raw, unit, wavelengths, gains, channel_names=channels, dataset=label
        )
    return sweep


def run_gain_sweep(
    n_shots: int = 1000,
    gains: np.ndarray | None = None,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    output: Path | None = None,
    datasets: dict[str, str] | None = None,
    delay_max_shift: int = 8,
    include_per_shot: bool = True,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Sweep every configured acquisition and write the JSON artifact.

    Args:
        n_shots: Shots per dataset.
        gains: Candidate gains; defaults to the module's grid.
        window_samples: Window length for the windowed measure.
        output: Destination JSON path. Defaults to
            ``smi/analysis/data/gain_sweep_results.json``.
        datasets: Mapping of label to file name under ``data_dir``.
        delay_max_shift: Half-width of the delay check in samples.
        include_per_shot: Also fit an optimal gain per shot, bounding what any
            scalar correction could achieve.
        data_dir: Directory the dataset file names are relative to. Defaults
            to ``smi/analysis/data``.

    Returns:
        The results dictionary that was written to disk.
    """
    datasets = datasets or DEFAULT_DATASETS
    output = output or DEFAULT_RESULTS_PATH
    data_dir = data_dir or DATA_DIR
    if gains is None:
        gains = gain_grid(DEFAULT_GAIN_LOW, DEFAULT_GAIN_HIGH, DEFAULT_GAIN_STEP)

    results: dict[str, Any] = {
        # 1: first committed gain sweep. Bump when the schema changes shape,
        # as baseline_results.json does, so a stale reader fails loudly.
        'schema_version': 1,
        'config': {
            'n_shots': n_shots,
            'window_samples': window_samples,
            'gain_low': float(gains[0]),
            'gain_high': float(gains[-1]),
            'n_gains': int(np.asarray(gains).size),
            'live_k': LIVE_K,
            'legacy_k': LEGACY_K,
            'delay_max_shift': delay_max_shift,
            'include_per_shot': include_per_shot,
            'pd_wavelengths_um': dict(PD_WAVELENGTHS_UM),
        },
        'datasets': {},
    }
    for label, name in datasets.items():
        path = data_dir / name
        if not path.exists():
            logger.warning('skipping %s: %s not found', label, path)
            continue
        logger.info('sweeping gain on %s (%s)...', label, name)
        sweep = sweep_dataset(
            path,
            label,
            n_shots=n_shots,
            gains=gains,
            window_samples=window_samples,
            delay_max_shift=delay_max_shift,
            include_per_shot=include_per_shot,
        )
        results['datasets'][label] = sweep.to_dict()
        for channel in sweep.channels:
            curve = sweep.curves[channel]
            logger.info(
                '  %s %s (%.3f um): best k %.2f (ratio %.3f vs live %.3f), '
                'median R^2 %.3f at best vs %.3f at live, %.3f at legacy k=%.2f',
                label,
                channel,
                curve.wavelength_um,
                sweep.best_gain(channel),
                sweep.best_gain(channel) / LIVE_K,
                LIVE_K,
                sweep.best_r_squared(channel),
                sweep.r_squared_at(channel, LIVE_K),
                sweep.r_squared_at(channel, LEGACY_K),
                LEGACY_K,
            )
            if curve.peak_fringe_rate_hz is not None:
                logger.info(
                    '    ruled out: peak fringe rate %.1f kHz vs %.0f kHz '
                    'Nyquist (%.0fx headroom)',
                    curve.peak_fringe_rate_hz / 1e3,
                    (sweep.nyquist_hz or 0.0) / 1e3,
                    curve.fringe_rate_headroom or 0.0,
                )
            if curve.delay_median_r_squared is not None:
                assert curve.delay_shifts is not None
                pooled = DelayCurve(
                    shifts=np.asarray(curve.delay_shifts),
                    r_squared=np.asarray(curve.delay_median_r_squared),
                )
                logger.info(
                    '    delay: peak at %+d samples, %s (R^2 %.3f there vs '
                    '%.3f at zero)',
                    pooled.best_shift,
                    'INTERIOR -- a real delay'
                    if pooled.peak_is_interior
                    else 'not interior, so no delay',
                    float(pooled.r_squared.max()),
                    pooled.r_squared_at(0),
                )
            if sweep.per_shot is not None:
                # Logged next to the best-global-gain line, because the point
                # of this number is that it is an upper bound on that one.
                per_shot_gains = sweep.per_shot.best_gains(channel)
                logger.info(
                    '    caveat: per-shot gains median %.2f (IQR %.2f-%.2f), '
                    'median R^2 %.3f -- the ceiling for any scalar gain fix',
                    float(np.median(per_shot_gains)),
                    float(np.percentile(per_shot_gains, 25)),
                    float(np.percentile(per_shot_gains, 75)),
                    sweep.per_shot.median_r_squared(channel),
                )
        logger.info('  %s: %.1f s', label, sweep.runtime_s)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    logger.info('wrote %s', output)
    return results


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for regenerating the gain-sweep artifact.

    The defaults are the settings the committed artifact was produced with, so
    a bare ``python -m smi.analysis.gain_sweep`` reproduces it.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--n-shots',
        type=int,
        default=1000,
        help='Shots per dataset. This is method 1, sub-millisecond per fit, '
        "so 1000 costs ~9 min per acquisition -- do not copy method 2's 200, "
        'which exists for a 5.85 s/shot GPU decode.',
    )
    parser.add_argument('--gain-low', type=float, default=DEFAULT_GAIN_LOW)
    parser.add_argument(
        '--gain-high',
        type=float,
        default=DEFAULT_GAIN_HIGH,
        help='Top of the swept range. The default reaches the 2024 '
        "calibration's k=33.42 so the same curve adjudicates it.",
    )
    parser.add_argument('--gain-step', type=float, default=DEFAULT_GAIN_STEP)
    parser.add_argument(
        '--window-samples',
        type=int,
        default=DEFAULT_WINDOW_SAMPLES,
        help='Window length for the windowed measure. Pass 0 to report only '
        'the global fit, which is about twice as fast.',
    )
    parser.add_argument(
        '--delay-max-shift',
        type=int,
        default=8,
        help='Half-width in samples of the drive-to-photodiode delay check, '
        'swept at the winning gain. Pass 0 to skip it.',
    )
    parser.add_argument(
        '--no-per-shot',
        action='store_true',
        help='Skip the per-shot optimal-gain fit. That fit is the evidence '
        'that the residual is not a scale error, so the artifact is weaker '
        'without it.',
    )
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    run_gain_sweep(
        n_shots=args.n_shots,
        gains=gain_grid(args.gain_low, args.gain_high, args.gain_step),
        window_samples=args.window_samples or None,
        output=args.output,
        delay_max_shift=args.delay_max_shift,
        include_per_shot=not args.no_per_shot,
    )


if __name__ == '__main__':
    main()
