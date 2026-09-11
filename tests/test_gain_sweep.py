"""Known-answer tests for the drive-to-displacement gain sweep.

The sweep's job is to answer "is the coil-driver gain constant ``k`` right?"
by scoring the Michelson forward model against the data at a range of gains.
That question has a checkable answer on synthetic data: generate fringes from
a displacement scaled by a *known* gain, and the sweep must pick that gain
back out. If it cannot do that, its verdict on the real acquisitions means
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from smi.analysis.baseline_fit import DRIVE_KEY, PD_WAVELENGTHS_UM
from smi.analysis.gain_sweep import (
    GainSweep,
    gain_grid,
    sweep_dataset,
    sweep_gain,
    unit_gain_displacement,
)
from smi.synthetic.calib_params import CalibrationParameters
from smi.synthetic.coil_driver import CoilDriver

SAMPLE_RATE = 488281.25
WAVELENGTHS = tuple(PD_WAVELENGTHS_UM.values())


def _band_limited_displacement(
    n_samples: int, seed: int, amplitude_um: float = 0.35
) -> np.ndarray:
    """Smooth multi-tone displacement in the band the speaker actually drives.

    Args:
        n_samples: Record length in samples.
        seed: Seed for the per-tone random phases.
        amplitude_um: Target RMS displacement (microns).

    Returns:
        Zero-mean displacement in microns, shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SAMPLE_RATE
    disp = np.zeros(n_samples)
    for freq, weight in ((120.0, 1.0), (310.0, 0.6), (700.0, 0.3)):
        disp += weight * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    disp -= disp.mean()
    return disp * (amplitude_um / disp.std())


def _fringes(
    displacement: np.ndarray, wavelength_um: float, phase: float
) -> np.ndarray:
    """Zero-mean Michelson fringes for a given displacement.

    Args:
        displacement: Mirror displacement (microns), shape ``(n,)``.
        wavelength_um: Laser wavelength (microns).
        phase: Interferometer phase offset (radians).

    Returns:
        Zero-mean fringe signal, shape ``(n,)``.
    """
    signal = np.cos(4.0 * np.pi / wavelength_um * displacement + phase)
    return signal - signal.mean()


def _synthetic_sweep_inputs(
    true_gain: float, n_shots: int = 6, n_samples: int = 4096, amplitude_um: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Fringes generated at ``true_gain`` times a unit-gain displacement.

    This mirrors exactly what the real sweep sees: a ``k=1`` displacement from
    the drive voltage, and photodiode signals whose fringes correspond to some
    unknown multiple of it.

    Args:
        true_gain: Gain the fringes were actually generated at.
        n_shots: Number of synthetic shots.
        n_samples: Samples per shot.
        amplitude_um: RMS of the unit-gain displacement (microns).

    Returns:
        Tuple of ``(pd_signals, unit_displacement)`` with shapes
        ``(n_shots, 3, n_samples)`` and ``(n_shots, n_samples)``.
    """
    rng = np.random.default_rng(4242)
    unit = np.stack(
        [
            _band_limited_displacement(
                n_samples, seed=100 + i, amplitude_um=amplitude_um
            )
            for i in range(n_shots)
        ]
    )
    signals = np.empty((n_shots, len(WAVELENGTHS), n_samples))
    for i in range(n_shots):
        for k, wavelength in enumerate(WAVELENGTHS):
            signals[i, k] = _fringes(
                true_gain * unit[i], wavelength, rng.uniform(0, 2 * np.pi)
            )
    return signals, unit


class TestUnitGainDisplacement:
    """The sweep must scale one ``k=1`` displacement, not refit per gain."""

    def test_scaling_the_unit_displacement_matches_a_rebuilt_driver(self) -> None:
        """The transfer function is linear in ``k``, and this pins that down.

        The whole sweep rests on it: if scaling were not exact, the reported
        best gain would be an artifact of the shortcut rather than a
        measurement.
        """
        rng = np.random.default_rng(1)
        drive = rng.standard_normal(2048)

        unit_vel, unit_disp = unit_gain_displacement(drive, SAMPLE_RATE)

        gain = 14.553
        driver = CoilDriver(CalibrationParameters(k=gain))
        vel, disp = driver.get_velocity_and_displacement(drive, SAMPLE_RATE)

        np.testing.assert_allclose(gain * unit_disp, disp, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(gain * unit_vel, vel, rtol=1e-12, atol=1e-12)

    def test_unit_displacement_is_the_live_displacement_divided_by_live_k(self) -> None:
        rng = np.random.default_rng(2)
        drive = rng.standard_normal(1024)
        live_k = CalibrationParameters().k

        _, unit_disp = unit_gain_displacement(drive, SAMPLE_RATE)
        _, live_disp = CoilDriver().get_velocity_and_displacement(drive, SAMPLE_RATE)

        np.testing.assert_allclose(
            unit_disp * live_k, live_disp, rtol=1e-12, atol=1e-12
        )


class TestGainGrid:
    """The grid is what resolves a peak, so its shape is part of the answer."""

    def test_spans_the_requested_range_inclusively(self) -> None:
        grid = gain_grid(10.0, 20.0, 0.25)

        assert grid[0] == pytest.approx(10.0)
        assert grid[-1] == pytest.approx(20.0)
        np.testing.assert_allclose(np.diff(grid), 0.25)

    def test_rejects_a_non_positive_step(self) -> None:
        with pytest.raises(ValueError, match='step must be positive'):
            gain_grid(10.0, 20.0, 0.0)

    def test_rejects_a_reversed_range(self) -> None:
        with pytest.raises(ValueError, match='high must exceed low'):
            gain_grid(20.0, 10.0, 0.5)


class TestSweepRecoversAKnownGain:
    """The property that makes the sweep's verdict on real data meaningful."""

    TRUE_GAIN = 12.0

    def test_recovers_the_gain_the_fringes_were_generated_at(self) -> None:
        signals, unit = _synthetic_sweep_inputs(self.TRUE_GAIN)

        sweep = sweep_gain(
            signals, unit, WAVELENGTHS, gain_grid(8.0, 16.0, 0.25), window_samples=None
        )

        for channel in sweep.channels:
            assert sweep.best_gain(channel) == pytest.approx(self.TRUE_GAIN, abs=0.25)

    def test_the_peak_is_sharp_not_a_plateau(self) -> None:
        """A flat curve would make "best gain" meaningless.

        The real finding is that median R^2 vs ``k`` is sharply peaked, so a
        sweep that reports a broad plateau on data with a known answer is not
        measuring what the finding claims.
        """
        signals, unit = _synthetic_sweep_inputs(self.TRUE_GAIN)

        sweep = sweep_gain(
            signals, unit, WAVELENGTHS, gain_grid(8.0, 16.0, 0.25), window_samples=None
        )

        channel = sweep.channels[0]
        peak = sweep.best_r_squared(channel)
        curve = np.asarray(sweep.median_r_squared(channel))
        gains = np.asarray(sweep.gains)
        far = np.abs(gains - self.TRUE_GAIN) > 1.5
        assert peak > 0.9
        assert curve[far].max() < 0.5 * peak

    def test_a_wrong_gain_scores_far_below_the_right_one(self) -> None:
        """The k=33.42 refutation has this shape, checked on known data."""
        signals, unit = _synthetic_sweep_inputs(self.TRUE_GAIN)

        sweep = sweep_gain(
            signals,
            unit,
            WAVELENGTHS,
            np.array([self.TRUE_GAIN, 3.0 * self.TRUE_GAIN]),
            window_samples=None,
        )

        channel = sweep.channels[0]
        right, wrong = sweep.median_r_squared(channel)
        assert right > 0.9
        assert wrong < 0.2

    def test_windowed_measure_is_reported_alongside(self) -> None:
        """Phase 2 made the windowed refit the honest measure.

        The sweep has to be able to report it, or the gain answer is quoted in
        a measure the rest of the branch has retracted.
        """
        signals, unit = _synthetic_sweep_inputs(self.TRUE_GAIN, n_samples=4096)

        sweep = sweep_gain(
            signals, unit, WAVELENGTHS, gain_grid(10.0, 14.0, 0.5), window_samples=1024
        )

        channel = sweep.channels[0]
        assert sweep.median_r_squared_windowed(channel) is not None
        assert sweep.best_gain(channel, windowed=True) == pytest.approx(
            self.TRUE_GAIN, abs=0.5
        )

    def test_windowed_columns_are_absent_when_windowing_is_off(self) -> None:
        signals, unit = _synthetic_sweep_inputs(self.TRUE_GAIN, n_shots=3)

        sweep = sweep_gain(
            signals, unit, WAVELENGTHS, np.array([12.0]), window_samples=None
        )

        assert sweep.median_r_squared_windowed(sweep.channels[0]) is None
        with pytest.raises(ValueError, match='windowed'):
            sweep.best_gain(sweep.channels[0], windowed=True)

    def test_recovers_a_different_known_gain_too(self) -> None:
        """One gain could be luck; two rules out a hardcoded answer."""
        signals, unit = _synthetic_sweep_inputs(9.5, n_shots=4)

        sweep = sweep_gain(
            signals, unit, WAVELENGTHS, gain_grid(7.0, 13.0, 0.25), window_samples=None
        )

        assert sweep.best_gain(sweep.channels[0]) == pytest.approx(9.5, abs=0.25)

    def test_rejects_mismatched_shot_counts(self) -> None:
        signals, unit = _synthetic_sweep_inputs(12.0, n_shots=4)

        with pytest.raises(ValueError, match='shots'):
            sweep_gain(signals, unit[:2], WAVELENGTHS, np.array([12.0]))

    def test_rejects_a_channel_count_mismatch(self) -> None:
        signals, unit = _synthetic_sweep_inputs(12.0, n_shots=3)

        with pytest.raises(ValueError, match='wavelengths'):
            sweep_gain(signals, unit, WAVELENGTHS[:2], np.array([12.0]))


class TestBinnedByExcursion:
    """The gain answer is binned by excursion because the trend is the point."""

    def test_bins_are_reported_per_gain_and_carry_their_n(self) -> None:
        signals, unit = _synthetic_sweep_inputs(12.0, n_shots=6, amplitude_um=0.8)

        sweep = sweep_gain(
            signals, unit, WAVELENGTHS, np.array([12.0]), window_samples=None
        )

        bins = sweep.binned_r_squared(sweep.channels[0])[0]
        assert [entry['ptp_um'] for entry in bins] == [
            '0-0.3',
            '0.3-1',
            '1-2',
            '2-3',
            '3+',
        ]
        assert sum(int(entry['n']) for entry in bins) == 6

    def test_excursion_is_measured_at_the_swept_gain_not_the_unit_gain(self) -> None:
        """A gain of 1 would put every shot in the smallest bin.

        The excursion a shot belongs to depends on the gain being scored, so
        binning against the unit-gain displacement would mislabel every bin.
        """
        signals, unit = _synthetic_sweep_inputs(12.0, n_shots=6, amplitude_um=0.5)

        sweep = sweep_gain(
            signals, unit, WAVELENGTHS, np.array([1.0, 4.0]), window_samples=None
        )

        at_unit, at_four = sweep.binned_r_squared(sweep.channels[0])
        smallest_bin = {entry['ptp_um']: int(entry['n']) for entry in at_unit}
        four = {entry['ptp_um']: int(entry['n']) for entry in at_four}
        assert smallest_bin['0-0.3'] == 0
        assert smallest_bin['0.3-1'] + smallest_bin['1-2'] == 6
        assert four['3+'] == 6


class TestSweepOverAnHdf5File:
    """The committed experiment reads an acquisition file, so that path counts."""

    def _write_dataset(self, path: Path, true_gain: float, n_shots: int = 5) -> None:
        """Write a small HDF5 acquisition whose fringes use a known gain.

        The drive voltage is chosen first, the unit-gain displacement derived
        from it through the real ``CoilDriver``, and the fringes built from
        ``true_gain`` times that -- so the file's own physics is consistent
        with the sweep's model and the answer is known.
        """
        rng = np.random.default_rng(77)
        n_samples = 2048
        drive = np.empty((n_shots, n_samples))
        signals = np.empty((n_shots, len(WAVELENGTHS), n_samples))
        for i in range(n_shots):
            tone = _band_limited_displacement(n_samples, seed=300 + i, amplitude_um=1.0)
            drive[i] = tone
            _, unit = unit_gain_displacement(tone, SAMPLE_RATE)
            for k, wavelength in enumerate(WAVELENGTHS):
                signals[i, k] = _fringes(
                    true_gain * unit, wavelength, rng.uniform(0, 2 * np.pi)
                )
        with h5py.File(path, 'w') as handle:
            handle.attrs['sample_rate'] = SAMPLE_RATE
            handle.create_dataset(DRIVE_KEY, data=drive)
            for k, channel in enumerate(PD_WAVELENGTHS_UM):
                handle.create_dataset(channel, data=signals[:, k])

    def test_sweeps_a_file_and_recovers_its_gain(self, tmp_path: Path) -> None:
        path = tmp_path / 'known_gain.h5'
        self._write_dataset(path, true_gain=11.0)

        sweep = sweep_dataset(
            path,
            'known',
            n_shots=5,
            gains=gain_grid(8.0, 14.0, 0.25),
            window_samples=None,
        )

        assert isinstance(sweep, GainSweep)
        assert sweep.n_shots == 5
        for channel in sweep.channels:
            assert sweep.best_gain(channel) == pytest.approx(11.0, abs=0.25)

    def test_serializes_to_a_json_ready_dict(self, tmp_path: Path) -> None:
        """The point of the phase is an artifact, so the dict must be complete."""
        path = tmp_path / 'known_gain.h5'
        self._write_dataset(path, true_gain=11.0)

        sweep = sweep_dataset(
            path,
            'known',
            n_shots=5,
            gains=gain_grid(10.0, 12.0, 0.5),
            window_samples=256,
        )
        payload = sweep.to_dict()

        # Round-trips through JSON, or it cannot be the artifact.
        restored = json.loads(json.dumps(payload))
        assert restored['dataset'] == 'known'
        assert restored['n_shots'] == 5
        assert restored['live_k'] == pytest.approx(CalibrationParameters().k)
        assert restored['gains'] == pytest.approx(list(sweep.gains))
        for channel in PD_WAVELENGTHS_UM:
            entry = restored['channels'][channel]
            assert len(entry['median_r_squared']) == len(sweep.gains)
            assert len(entry['median_r_squared_windowed']) == len(sweep.gains)
            assert entry['best_gain'] == pytest.approx(sweep.best_gain(channel))
            assert entry['best_gain_ratio_vs_live'] == pytest.approx(
                sweep.best_gain(channel) / CalibrationParameters().k
            )
