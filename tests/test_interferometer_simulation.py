#!/usr/bin/env python3
"""Tests for the interferometer simulation functionality."""

import numpy as np

from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.interferometers import InterferometerArray, MichelsonInterferometer
from smi.synthetic.waveform import Waveform


class TestInterferometerSimulation:
    """Test class for interferometer simulation functionality."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        # Create test interferometers with different wavelengths
        self.interferometer1 = MichelsonInterferometer(
            wavelength=0.633, phase=0
        )  # HeNe laser
        self.interferometer2 = MichelsonInterferometer(
            wavelength=0.515, phase=np.pi / 2
        )  # Green laser

        # Create an interferometer array
        self.interferometer_array = InterferometerArray(
            [self.interferometer1, self.interferometer2]
        )

        # Create a CoilDriver for testing
        self.coil_driver = CoilDriver()

        # Create a Waveform generator
        self.waveform = Waveform(start_freq=1, end_freq=1000)

        # Run a simulation to get test data
        self.time, self.signals, self.acq_voltage, self.displacement, self.velocity = (
            self.interferometer_array.sample_simulated(
                start_freq=1, end_freq=1000, coil_driver=self.coil_driver
            )
        )

    def test_buffer_size(self):
        """Test that all output arrays have the correct buffer size."""
        # All output arrays should have length equal to BUFFER_SIZE
        buffer_size = self.waveform.BUFFER_SIZE

        # Check time array
        assert len(self.time) <= buffer_size, (
            f'Time array exceeds buffer size: {len(self.time)} > {buffer_size}'
        )

        # Check signals
        for i, signal in enumerate(self.signals):
            assert len(signal) <= buffer_size, (
                f'Signal {i} exceeds buffer size: {len(signal)} > {buffer_size}'
            )

        # Check displacement and velocity
        assert len(self.displacement) <= buffer_size, (
            f'Displacement array exceeds buffer size: {len(self.displacement)} > '
            f'{buffer_size}'
        )
        assert len(self.velocity) <= buffer_size, (
            f'Velocity array exceeds buffer size: {len(self.velocity)} > {buffer_size}'
        )

        # Check that all arrays have the same length
        assert len(self.time) == len(self.displacement) == len(self.velocity), (
            f"Array lengths don't match: time={len(self.time)}, "
            f'displacement={len(self.displacement)}, velocity={len(self.velocity)}'
        )

        for i, signal in enumerate(self.signals):
            assert len(signal) == len(self.time), (
                f"Signal {i} length doesn't match time array: {len(signal)} != "
                f'{len(self.time)}'
            )

    def test_time_consistency(self):
        """Test that the time array is consistent with the acquisition sample rate."""
        # Get a new waveform to ensure we have the correct parameters
        waveform = Waveform(start_freq=1, end_freq=1000)

        # Calculate expected time step based on acquisition sample rate
        expected_dt = 1 / waveform.acq_sample_rate

        # Calculate actual time step from the time array
        actual_dt = self.time[1] - self.time[0] if len(self.time) > 1 else 0

        # Allow for small floating-point differences
        assert np.isclose(actual_dt, expected_dt, rtol=1e-10), (
            f"Time step doesn't match acquisition rate: {actual_dt} != {expected_dt}"
        )

        # Check that time array is evenly spaced
        if len(self.time) > 2:
            time_diffs = np.diff(self.time)
            assert np.allclose(time_diffs, time_diffs[0], rtol=1e-10), (
                'Time array is not evenly spaced'
            )

    def test_velocity_displacement_consistency(self):
        """Test that velocity and displacement are consistent with each other."""
        # In sample_simulated, velocity is calculated directly from voltage using
        # get_velocity rather than by differentiating displacement. These methods
        # can produce different results but should be correlated and have similar
        # magnitudes.

        # Calculate velocity from displacement using CoilDriver.derivative_displacement
        calculated_velocity = self.coil_driver.derivative_displacement(
            self.displacement, 1 / (self.time[1] - self.time[0])
        )

        # Check correlation between calculated velocity and velocity from
        # sample_simulated
        correlation = np.corrcoef(calculated_velocity, self.velocity)[0, 1]
        assert correlation > 0.99, (
            f'Low correlation between calculated and simulated velocity: {correlation}'
        )

        # Check that magnitudes are similar (mean absolute values within 20%)
        calc_vel_mag = np.mean(np.abs(calculated_velocity))
        sim_vel_mag = np.mean(np.abs(self.velocity))
        ratio = calc_vel_mag / sim_vel_mag if sim_vel_mag > 0 else float('inf')
        assert 0.99 < ratio < 1.01, (
            f'Velocity magnitudes differ too much: {calc_vel_mag} vs {sim_vel_mag}, '
            f'ratio={ratio}'
        )

        # Calculate displacement from velocity using CoilDriver.integrate_velocity
        calculated_displacement = self.coil_driver.integrate_velocity(
            self.velocity, 1 / (self.time[1] - self.time[0])
        )

        # Remove DC offset for comparison
        calculated_displacement = calculated_displacement - np.mean(
            calculated_displacement
        )
        displacement_normalized = self.displacement - np.mean(self.displacement)

        # Check correlation between calculated displacement and displacement from
        # sample_simulated
        correlation = np.corrcoef(calculated_displacement, displacement_normalized)[
            0, 1
        ]
        assert correlation > 0.99, (
            f'Low correlation between calculated and simulated displacement: '
            f'{correlation}'
        )

        # Check that magnitudes are similar (mean absolute values within 20%)
        calc_disp_mag = np.mean(np.abs(calculated_displacement))
        sim_disp_mag = np.mean(np.abs(displacement_normalized))
        ratio = calc_disp_mag / sim_disp_mag if sim_disp_mag > 0 else float('inf')
        assert 0.99 < ratio < 1.01, (
            f'Displacement magnitudes differ too much: {calc_disp_mag} vs '
            f'{sim_disp_mag}, ratio={ratio}'
        )

    def test_interferometer_signal_calculation(self):
        """Test that the interferometer signals match manual calculation."""
        # Manually calculate the interferometer signals
        manual_signals = []

        for interferometer in self.interferometer_array.interferometers:
            # Extract parameters
            wavelength = interferometer.wavelength
            phase = interferometer.phase

            # Calculate interference term
            interference = np.cos(
                2 * np.pi / wavelength * 2 * self.displacement + phase
            )

            # Calculate signal
            signal = interference

            # Remove DC offset to match the processing in get_simulated_buffer
            signal = signal - np.mean(signal)

            manual_signals.append(signal)

        # Compare manual signals with the signals from sample_simulated
        for i, (manual_signal, simulated_signal) in enumerate(
            zip(manual_signals, self.signals, strict=False)
        ):
            assert np.allclose(
                manual_signal, simulated_signal, rtol=1e-10, atol=1e-10
            ), f"Manually calculated signal {i} doesn't match simulated signal"

    def test_phase_shift(self):
        """Test that phase shift affects the interferometer signal correctly."""
        # Create two interferometers with the same wavelength but different phases
        interferometer1 = MichelsonInterferometer(wavelength=0.633, phase=0)
        interferometer2 = MichelsonInterferometer(wavelength=0.633, phase=np.pi / 2)

        # Create a simple displacement array
        displacement = np.linspace(0, 10, 1000)
        time = np.linspace(0, 1, 1000)

        # Get signals
        _, signal1, _, _ = interferometer1.get_simulated_buffer(displacement, time)
        _, signal2, _, _ = interferometer2.get_simulated_buffer(displacement, time)

        # For a 90-degree phase shift, the signals should be approximately in quadrature
        # Calculate correlation
        correlation = np.corrcoef(signal1, signal2)[0, 1]

        # For quadrature signals, correlation should be close to 0
        assert abs(correlation) < 0.1, (
            f'Signals with 90-degree phase shift have unexpected correlation: '
            f'{correlation}'
        )
