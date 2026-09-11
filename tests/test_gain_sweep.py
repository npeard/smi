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
    delay_sweep,
    fringe_rate_headroom,
    gain_grid,
    peak_fringe_rate_hz,
    per_shot_best_gain,
    run_gain_sweep,
    sweep_dataset,
    sweep_gain,
    unit_gain_displacement,
)
from smi.synthetic.calib_params import CalibrationParameters
from smi.synthetic.coil_driver import MS_PER_S, CoilDriver

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


class TestPeakFringeRate:
    """The aliasing check, which TASKS.md flags as informal and uncommitted."""

    def test_pure_tone_fringe_rate_matches_the_analytic_value(self) -> None:
        """A sinusoid's fringe rate is known in closed form.

        For ``d(t) = A sin(2 pi f t)`` the peak speed is ``2 pi A f`` and the
        Michelson phase is ``4*pi*d/lambda``, so the instantaneous fringe
        frequency peaks at ``2 * (2 pi A f) / lambda = 4 pi A f / lambda``.
        Anchoring against that makes the measured 7.8 / 9.7 kHz figures a
        measurement rather than a number read off a plot -- and it catches the
        two mistakes that matter, a dropped factor of 2 from the round trip
        and a um/s-versus-um/ms slip worth 1000x.
        """
        n = 16384
        freq_hz = 300.0
        amplitude_um = 1.5
        wavelength_um = 0.635
        t = np.arange(n) / SAMPLE_RATE
        # Velocity of the same tone, in um/ms to match the package convention.
        velocity = (
            amplitude_um
            * 2
            * np.pi
            * freq_hz
            * np.cos(2 * np.pi * freq_hz * t)
            / MS_PER_S
        )

        rate = peak_fringe_rate_hz(velocity, wavelength_um)

        expected = 4.0 * np.pi * amplitude_um * freq_hz / wavelength_um
        assert rate == pytest.approx(expected, rel=1e-3)

    def test_doubling_the_speed_doubles_the_fringe_rate(self) -> None:
        velocity = np.array([0.0, 0.5, -0.25])

        slow = peak_fringe_rate_hz(velocity, 0.635)
        fast = peak_fringe_rate_hz(2.0 * velocity, 0.635)

        assert fast == pytest.approx(2.0 * slow)

    def test_a_shorter_wavelength_crosses_more_fringes(self) -> None:
        """515 nm must report a higher rate than 635 nm for the same motion."""
        velocity = np.array([0.0, 0.4, -0.4])

        assert peak_fringe_rate_hz(velocity, 0.515) > peak_fringe_rate_hz(
            velocity, 0.635
        )

    def test_headroom_is_the_nyquist_ratio(self) -> None:
        headroom = fringe_rate_headroom(
            np.array([0.0, 0.5]), 0.635, sample_rate=488281.25
        )

        rate = peak_fringe_rate_hz(np.array([0.0, 0.5]), 0.635)
        assert headroom == pytest.approx(488281.25 / 2.0 / rate)


class TestDelaySweep:
    """The time-delay check, ruled out but not previously committed."""

    def test_an_injected_delay_shows_up_as_an_interior_peak(self) -> None:
        """The check only means something if it can detect a real delay.

        Ruling out a delay because the R^2-vs-tau curve has no interior peak
        is evidence only if an actual delay would produce one. Injecting a
        known shift and recovering it is that guarantee.
        """
        shift = 5
        # A large excursion is what makes 5 samples matter: the phase error a
        # misalignment produces is 4*pi*(d(t) - d(t - tau))/lambda, which
        # scales with the mirror speed. At the 0.35 um default the fringes are
        # slow enough that a 5-sample slip is nearly invisible, so a test
        # there would pass whether or not the sweep worked.
        disp = _band_limited_displacement(8192, seed=31, amplitude_um=4.0)
        signal = np.roll(_fringes(disp, 0.635, 0.4), shift)

        curve = delay_sweep(signal, disp, 0.635, max_shift=8)

        assert curve.best_shift == shift
        assert curve.r_squared_at(shift) > curve.r_squared_at(0) + 0.2

    def test_no_delay_leaves_the_peak_at_zero(self) -> None:
        disp = _band_limited_displacement(8192, seed=31, amplitude_um=1.2)
        signal = _fringes(disp, 0.635, 0.4)

        curve = delay_sweep(signal, disp, 0.635, max_shift=8)

        assert curve.best_shift == 0

    def test_shifts_span_the_requested_window(self) -> None:
        disp = _band_limited_displacement(2048, seed=31, amplitude_um=0.5)
        signal = _fringes(disp, 0.635, 0.0)

        curve = delay_sweep(signal, disp, 0.635, max_shift=3)

        assert list(curve.shifts) == [-3, -2, -1, 0, 1, 2, 3]

    def test_rejects_a_negative_window(self) -> None:
        disp = _band_limited_displacement(512, seed=1, amplitude_um=0.5)

        with pytest.raises(ValueError, match='max_shift'):
            delay_sweep(_fringes(disp, 0.635, 0.0), disp, 0.635, max_shift=-1)


class TestRuledOutChecksLandInTheArtifact:
    """A candidate is only "ruled out" if the artifact says so, with numbers."""

    def _write_dataset(self, path: Path, true_gain: float, n_shots: int = 4) -> None:
        """Write a small acquisition whose fringes use a known gain."""
        rng = np.random.default_rng(5)
        n_samples = 2048
        drive = np.empty((n_shots, n_samples))
        signals = np.empty((n_shots, len(WAVELENGTHS), n_samples))
        for i in range(n_shots):
            tone = _band_limited_displacement(n_samples, seed=400 + i, amplitude_um=1.0)
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

    def test_artifact_records_bandwidth_headroom_per_channel(
        self, tmp_path: Path
    ) -> None:
        """The aliasing check TASKS.md flags as uncommitted.

        Recorded per channel because the shortest wavelength crosses the most
        fringes per micron, so a single number would hide the binding case.
        """
        path = tmp_path / 'known.h5'
        self._write_dataset(path, true_gain=11.0)

        sweep = sweep_dataset(
            path, 'known', n_shots=4, gains=np.array([11.0]), window_samples=None
        )
        payload = json.loads(json.dumps(sweep.to_dict()))

        assert payload['nyquist_hz'] == pytest.approx(SAMPLE_RATE / 2.0)
        for channel in PD_WAVELENGTHS_UM:
            entry = payload['channels'][channel]
            assert entry['peak_fringe_rate_hz'] > 0.0
            assert entry['fringe_rate_headroom'] == pytest.approx(
                (SAMPLE_RATE / 2.0) / entry['peak_fringe_rate_hz']
            )
        # 515 nm crosses the most fringes, so it must have the least headroom.
        assert (
            payload['channels']['RP2_CH2']['fringe_rate_headroom']
            < payload['channels']['RP1_CH2']['fringe_rate_headroom']
        )

    def test_artifact_records_the_delay_curve_with_its_peak(
        self, tmp_path: Path
    ) -> None:
        """No interior peak is the evidence, so the whole curve is stored.

        Storing only "best shift = 0" would be an assertion; storing the curve
        lets a reader see it is flat rather than trust that it was.
        """
        path = tmp_path / 'known.h5'
        self._write_dataset(path, true_gain=11.0)

        sweep = sweep_dataset(
            path, 'known', n_shots=4, gains=np.array([11.0]), window_samples=None
        )
        payload = json.loads(json.dumps(sweep.to_dict()))

        for channel in PD_WAVELENGTHS_UM:
            delay = payload['channels'][channel]['delay_sweep']
            assert delay['shifts'][0] == -delay['shifts'][-1]
            assert len(delay['median_r_squared']) == len(delay['shifts'])
            # The synthetic file has no delay, so zero must win.
            assert delay['best_shift'] == 0


class TestPerShotBestGain:
    """The caveat that proves the residual is drift, not a scale error."""

    def test_per_shot_gain_beats_the_global_best_on_mixed_gains(self) -> None:
        """Per-shot fitting must actually be per shot.

        Half the shots are generated at one gain and half at another, so no
        single global gain can fit both. A per-shot optimum must beat the best
        global one here -- otherwise the "even a per-shot gain does not rescue
        the >3 um collapse" finding would be reporting a broken estimator
        rather than a property of the data.
        """
        low, high = 10.0, 14.0
        signals_a, unit_a = _synthetic_sweep_inputs(low, n_shots=4, n_samples=2048)
        signals_b, unit_b = _synthetic_sweep_inputs(high, n_shots=4, n_samples=2048)
        signals = np.concatenate([signals_a, signals_b])
        unit = np.concatenate([unit_a, unit_b])
        gains = gain_grid(8.0, 16.0, 0.25)

        per_shot = per_shot_best_gain(signals, unit, WAVELENGTHS, gains)
        global_sweep = sweep_gain(
            signals, unit, WAVELENGTHS, gains, window_samples=None
        )

        channel = per_shot.channels[0]
        assert per_shot.median_r_squared(channel) > global_sweep.best_r_squared(channel)

    def test_per_shot_gains_land_on_the_gain_each_shot_was_built_at(self) -> None:
        low, high = 10.0, 14.0
        signals_a, unit_a = _synthetic_sweep_inputs(low, n_shots=3, n_samples=2048)
        signals_b, unit_b = _synthetic_sweep_inputs(high, n_shots=3, n_samples=2048)
        signals = np.concatenate([signals_a, signals_b])
        unit = np.concatenate([unit_a, unit_b])

        per_shot = per_shot_best_gain(
            signals, unit, WAVELENGTHS, gain_grid(8.0, 16.0, 0.25)
        )

        fitted = np.asarray(per_shot.best_gains(per_shot.channels[0]))
        np.testing.assert_allclose(fitted[:3], low, atol=0.25)
        np.testing.assert_allclose(fitted[3:], high, atol=0.25)

    def test_per_shot_result_is_binned_and_serializable(self) -> None:
        signals, unit = _synthetic_sweep_inputs(12.0, n_shots=6, n_samples=2048)

        per_shot = per_shot_best_gain(
            signals, unit, WAVELENGTHS, gain_grid(10.0, 14.0, 0.5)
        )
        payload = json.loads(json.dumps(per_shot.to_dict()))

        for channel in PD_WAVELENGTHS_UM:
            entry = payload['channels'][channel]
            assert len(entry['best_gains']) == 6
            assert sum(int(b['n']) for b in entry['binned_r_squared']) == 6


class TestPerShotLandsInTheArtifact:
    """The caveat must be in the file, not only reachable from the API."""

    def test_run_writes_per_shot_gains_beside_the_global_sweep(
        self, tmp_path: Path
    ) -> None:
        """A caveat a reader cannot find is not a caveat.

        "A per-shot fitted gain does not rescue the collapse" has to sit in
        the same artifact as the best-global-gain claim it qualifies, or the
        two will be quoted apart.
        """
        rng = np.random.default_rng(9)
        n_shots, n_samples = 4, 1024
        drive = np.empty((n_shots, n_samples))
        signals = np.empty((n_shots, len(WAVELENGTHS), n_samples))
        for i in range(n_shots):
            tone = _band_limited_displacement(n_samples, seed=500 + i, amplitude_um=1.0)
            drive[i] = tone
            _, unit = unit_gain_displacement(tone, SAMPLE_RATE)
            for k, wavelength in enumerate(WAVELENGTHS):
                signals[i, k] = _fringes(11.0 * unit, wavelength, rng.uniform(0, 6.28))
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        with h5py.File(data_dir / 'known.h5', 'w') as handle:
            handle.attrs['sample_rate'] = SAMPLE_RATE
            handle.create_dataset(DRIVE_KEY, data=drive)
            for k, channel in enumerate(PD_WAVELENGTHS_UM):
                handle.create_dataset(channel, data=signals[:, k])

        output = tmp_path / 'out.json'
        results = run_gain_sweep(
            n_shots=n_shots,
            gains=gain_grid(10.0, 12.0, 0.5),
            window_samples=None,
            output=output,
            datasets={'known': 'known.h5'},
            data_dir=data_dir,
            delay_max_shift=2,
        )

        on_disk = json.loads(output.read_text(encoding='utf-8'))
        assert on_disk == results
        entry = on_disk['datasets']['known']
        assert entry['n_shots'] == n_shots
        per_shot = entry['per_shot_gain']
        assert per_shot['n_shots'] == n_shots
        for channel in PD_WAVELENGTHS_UM:
            gains = per_shot['channels'][channel]['best_gains']
            assert len(gains) == n_shots
            assert all(10.0 <= g <= 12.0 for g in gains)
