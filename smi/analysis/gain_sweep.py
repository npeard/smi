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
   :class:`CoilDriver` and recomputing the FFT for each of ~30 gains would
   cost ~30x more for an identical answer.
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
from smi.synthetic.coil_driver import CoilDriver

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_PATH = DATA_DIR / 'gain_sweep_results.json'

# The live constant, for ratio reporting. Read from the calibration file rather
# than copied, so this module cannot drift away from the value the training
# targets are actually built with.
LIVE_K = CalibrationParameters().k

# Sweep range and step. The range brackets the live 14.553 generously on both
# sides and reaches the 2024 calibration's 33.42, so that value is refuted by
# the same curve rather than by a separate run. A 0.25 step resolves a peak to
# well inside the ~5% the two acquisitions disagree by.
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
    """

    wavelength_um: float
    median_r_squared: list[float] = field(default_factory=list)
    median_r_squared_windowed: list[float] | None = None
    binned_r_squared: list[list[dict[str, float | int | str]]] = field(
        default_factory=list
    )
    binned_r_squared_windowed: list[list[dict[str, float | int | str]]] | None = None


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
        runtime_s: Wall-clock seconds spent on the sweep.
    """

    dataset: str
    file: str
    n_shots: int
    n_samples_per_shot: int
    window_samples: int | None
    gains: np.ndarray
    curves: dict[str, ChannelGainCurve] = field(default_factory=dict)
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
            channels[name] = entry
        return {
            'dataset': self.dataset,
            'file': self.file,
            'n_shots': self.n_shots,
            'n_samples_per_shot': self.n_samples_per_shot,
            'window_samples': self.window_samples,
            'live_k': LIVE_K,
            'legacy_k': LEGACY_K,
            'gains': [float(g) for g in self.gains],
            'channels': channels,
            'runtime_s': self.runtime_s,
        }


def sweep_gain(
    pd_signals: np.ndarray,
    unit_displacement: np.ndarray,
    wavelengths_um: Sequence[float],
    gains: np.ndarray,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    channel_names: Sequence[str] | None = None,
    dataset: str = 'in-memory',
    file: str = '<in-memory>',
) -> GainSweep:
    """Score the forward model at each gain and report median R^2 per channel.

    For every gain the displacement is ``gain * unit_displacement``, the
    method-1 closed-form fit runs per shot per channel, and the per-shot R^2
    values are reduced to a median (robust to the few shots whose fringe
    contrast is poor) and to excursion bins.

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

    Returns:
        The per-channel gain curves.

    Raises:
        ValueError: If the shot counts disagree, or the channel count does not
            match the wavelength list.
    """
    start = time.perf_counter()
    signals = np.asarray(pd_signals, dtype=np.float64)
    unit = np.asarray(unit_displacement, dtype=np.float64)
    wavelengths = np.asarray(wavelengths_um, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)

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

    return GainSweep(
        dataset=dataset,
        file=file,
        n_shots=int(n_shots),
        n_samples_per_shot=int(n_time),
        window_samples=window_samples,
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
) -> GainSweep:
    """Sweep the gain over a leading slice of one acquisition HDF5 file.

    N = 1000 by default. This is a method-1 measurement -- a few least-squares
    solves per shot -- so 10% of a 10k acquisition costs about a minute, while
    N = 200 left the extreme excursion bins at n = 14-72, too thin to carry
    the trend that separates a scale error from unmodeled drift. Method 2's
    N = 200 in :mod:`smi.analysis.baseline_fit` is a different budget for a
    different cost and must not be conflated with this one.

    Args:
        file_path: Path to the acquisition HDF5 file.
        label: Short label recorded in the result.
        n_shots: Shots to score, capped at the file's shot count.
        gains: Candidate gains. Defaults to :func:`gain_grid` over
            ``[DEFAULT_GAIN_LOW, DEFAULT_GAIN_HIGH]``, which brackets both the
            live constant and the 2024 calibration's value.
        window_samples: Window length for the windowed measure, or ``None``
            for the global fit only.

    Returns:
        The per-channel gain curves for this acquisition.
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
    # the whole reason the sweep is a minute rather than half an hour.
    unit = np.empty_like(drive)
    for i in range(n_shots):
        _, unit[i] = unit_gain_displacement(drive[i], sample_rate)

    return sweep_gain(
        pd_raw,
        unit,
        wavelengths,
        gains,
        window_samples=window_samples,
        channel_names=channels,
        dataset=label,
        file=file_path.name,
    )


def run_gain_sweep(
    n_shots: int = 1000,
    gains: np.ndarray | None = None,
    window_samples: int | None = DEFAULT_WINDOW_SAMPLES,
    output: Path | None = None,
    datasets: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Sweep every configured acquisition and write the JSON artifact.

    Args:
        n_shots: Shots per dataset.
        gains: Candidate gains; defaults to the module's grid.
        window_samples: Window length for the windowed measure.
        output: Destination JSON path. Defaults to
            ``smi/analysis/data/gain_sweep_results.json``.
        datasets: Mapping of label to file name under ``smi/analysis/data``.

    Returns:
        The results dictionary that was written to disk.
    """
    datasets = datasets or DEFAULT_DATASETS
    output = output or DEFAULT_RESULTS_PATH
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
            'pd_wavelengths_um': dict(PD_WAVELENGTHS_UM),
        },
        'datasets': {},
    }
    for label, name in datasets.items():
        path = DATA_DIR / name
        if not path.exists():
            logger.warning('skipping %s: %s not found', label, path)
            continue
        logger.info('sweeping gain on %s (%s)...', label, name)
        sweep = sweep_dataset(
            path, label, n_shots=n_shots, gains=gains, window_samples=window_samples
        )
        results['datasets'][label] = sweep.to_dict()
        for channel in sweep.channels:
            logger.info(
                '  %s %s (%.3f um): best k %.2f (ratio %.3f vs live %.3f), '
                'median R^2 %.3f at best vs %.3f at live, %.3f at legacy k=%.2f',
                label,
                channel,
                sweep.curves[channel].wavelength_um,
                sweep.best_gain(channel),
                sweep.best_gain(channel) / LIVE_K,
                LIVE_K,
                sweep.best_r_squared(channel),
                sweep.r_squared_at(channel, LIVE_K),
                sweep.r_squared_at(channel, LEGACY_K),
                LEGACY_K,
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
        help='Shots per dataset. This is method 1, about a millisecond per '
        "fit, so 1000 costs roughly a minute -- do not copy method 2's 200.",
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
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    run_gain_sweep(
        n_shots=args.n_shots,
        gains=gain_grid(args.gain_low, args.gain_high, args.gain_step),
        window_samples=args.window_samples or None,
        output=args.output,
    )


if __name__ == '__main__':
    main()
