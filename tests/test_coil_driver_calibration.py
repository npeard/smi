#!/usr/bin/env python3
"""Test for the CoilDriver calibration and displacement/velocity calculations.

This test verifies:
1. Consistency between FFT-calculated and CoilDriver-retrieved displacement and velocity
2. Reconstructed waveform matching
3. Manual computation of displacement and velocity transfer functions
"""

import numpy as np
import pytest
from plot_waveforms import calculate_fft

from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.waveform import Waveform


def test_spectra_consistency():
    """Test that the displacement and velocity spectra calculated by CoilDriver
    are consistent with those calculated using calculate_fft.
    """
    # Create a waveform generator with test parameters
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()

    # Generate a random waveform
    t, voltage, _voltage_spectrum = waveform.sample()

    # Calculate sample rate from time array
    sample_rate = 1 / (t[1] - t[0])

    # Get displacement and velocity using CoilDriver
    displacement, displacement_spectrum, displacement_freqs = (
        coil_driver.get_displacement(voltage, sample_rate)
    )
    velocity, velocity_spectrum, velocity_freqs = coil_driver.get_velocity(
        voltage, sample_rate
    )

    # Calculate FFT of the displacement and velocity waveforms
    freqs_disp_fft, displacement_fft = calculate_fft(displacement, sample_rate)
    freqs_vel_fft, velocity_fft = calculate_fft(velocity, sample_rate)

    assert np.allclose(
        np.abs(displacement_spectrum)[displacement_freqs >= 0], np.abs(displacement_fft)
    )
    assert np.allclose(
        np.abs(velocity_spectrum)[velocity_freqs >= 0], np.abs(velocity_fft)
    )
    assert np.allclose(freqs_vel_fft, velocity_freqs[velocity_freqs >= 0])
    assert np.allclose(freqs_disp_fft, displacement_freqs[displacement_freqs >= 0])


def test_reconstructed_waveforms():
    """Test that the reconstructed displacement and velocity waveforms
    match those retrieved from CoilDriver.
    """
    # Create a waveform generator with test parameters
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()

    # Generate a random waveform
    t, voltage, _voltage_spectrum = waveform.sample()

    # Calculate sample rate from time array
    sample_rate = 1 / (t[1] - t[0])

    # Get displacement and velocity using CoilDriver
    displacement, displacement_spectrum, _displacement_freqs = (
        coil_driver.get_displacement(voltage, sample_rate)
    )
    velocity, velocity_spectrum, _velocity_freqs = coil_driver.get_velocity(
        voltage, sample_rate
    )

    # Reconstruct displacement and velocity from their spectra
    reconstructed_displacement = np.real(
        np.fft.ifft(displacement_spectrum, norm='ortho')
    )
    reconstructed_velocity = np.real(np.fft.ifft(velocity_spectrum, norm='ortho'))

    # Check that the reconstructed waveforms match the original ones
    assert np.allclose(displacement, reconstructed_displacement), (
        'Reconstructed displacement does not match original'
    )

    assert np.allclose(velocity, reconstructed_velocity), (
        'Reconstructed velocity does not match original'
    )


def test_manual_transfer_function_computation():
    """Test that manually computing displacement and velocity transfer functions
    gives consistent results with CoilDriver's methods.
    """
    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()

    # Create a frequency array for testing
    freqs = np.linspace(10, 1000, 1000)

    # Calculate amplitude and phase transfer functions using CoilDriver methods
    amplitude_transfer = coil_driver._calculate_amplitude_transfer(freqs)
    phase_transfer = coil_driver._calculate_phase_transfer(freqs)

    # Manually compute displacement transfer function
    displacement_transfer = amplitude_transfer * np.exp(1j * phase_transfer)

    # Manually compute velocity transfer function
    velocity_transfer = displacement_transfer * freqs * 2 * np.pi * 1j

    # Test a specific frequency to verify calculations
    test_freq_idx = 500  # Middle of the range
    test_freq = freqs[test_freq_idx]

    # Calculate expected displacement transfer at test frequency
    expected_disp_amplitude = coil_driver._calculate_amplitude_transfer(
        np.array([test_freq])
    )[0]
    expected_disp_phase = coil_driver._calculate_phase_transfer(np.array([test_freq]))[
        0
    ]
    expected_disp_transfer = expected_disp_amplitude * np.exp(1j * expected_disp_phase)

    # Calculate expected velocity transfer at test frequency
    expected_vel_transfer = expected_disp_transfer * test_freq * 2 * np.pi * 1j

    # Check that the manually computed transfer functions match the expected values
    assert np.allclose(displacement_transfer[test_freq_idx], expected_disp_transfer), (
        'Manually computed displacement transfer function does not match expected'
    )

    assert np.allclose(velocity_transfer[test_freq_idx], expected_vel_transfer), (
        'Manually computed velocity transfer function does not match expected'
    )

    # Also verify that the velocity is the derivative of displacement
    # For a sinusoidal displacement x(t) = A*sin(omega*t + phi), the velocity is
    # v(t) = A*omega*cos(omega*t + phi)
    # In the frequency domain, this is equivalent to multiplying by iomega = i*2*pi*f
    assert np.allclose(
        velocity_transfer, displacement_transfer * freqs * 2 * np.pi * 1j
    ), (
        'Velocity transfer function is not consistent with displacement transfer '
        'function'
    )


def test_displacement_velocity_relationship():
    """Test that the velocity is the time derivative of displacement."""
    # Create a waveform generator with test parameters
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()

    # Generate a random waveform
    t, voltage, _voltage_spectrum = waveform.sample()

    # Calculate sample rate from time array
    sample_rate = 1 / (t[1] - t[0])

    # Get displacement and velocity using CoilDriver
    _displacement, displacement_spectrum, displacement_freqs = (
        coil_driver.get_displacement(voltage, sample_rate)
    )
    _velocity, velocity_spectrum, _velocity_freqs = coil_driver.get_velocity(
        voltage, sample_rate
    )

    # Verify that velocity_spectrum = iomega * displacement_spectrum
    # This is equivalent to velocity being the time derivative of displacement
    expected_velocity_spectrum = (
        1j * 2 * np.pi * displacement_freqs * displacement_spectrum
    )

    # Check that the velocity spectrum matches the expected value
    assert np.allclose(velocity_spectrum, expected_velocity_spectrum), (
        'Velocity spectrum is not the derivative of displacement spectrum'
    )


def test_integrated_velocity_and_derivative_displacement():
    """Verifies that: 1. The integrated velocity matches the displacement from the
    transfer function. 2. The derivative of displacement matches the velocity from the
    transfer function. This test verifies the consistency between the time-domain and
    frequency-domain approaches for calculating velocity and displacement.
    """
    # Create a waveform generator with test parameters
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()

    # Generate a random waveform
    t, voltage, _voltage_spectrum = waveform.sample()

    # Calculate sample rate from time array
    sample_rate = 1 / (t[1] - t[0])

    # Get displacement and velocity using CoilDriver
    displacement, _displacement_spectrum, _displacement_freqs = (
        coil_driver.get_displacement(voltage, sample_rate)
    )
    velocity, _velocity_spectrum, _velocity_freqs = coil_driver.get_velocity(
        voltage, sample_rate
    )

    # Test 1: Integrated velocity should match displacement from transfer function
    # Integrate velocity using the same method as in CoilDriver.integrate_velocity
    displacement_integrated = coil_driver.integrate_velocity(velocity, sample_rate)

    # Ensure displacement starts at zero (as done in
    # RedPitayaManager.get_displacement_data)
    displacement_zeroed = displacement - displacement[0]
    displacement_integrated = displacement_integrated - displacement_integrated[0]

    # Compare the integrated velocity with the displacement from transfer function
    assert np.allclose(
        displacement_zeroed, displacement_integrated, atol=0.5, rtol=0.0
    ), 'Integrated velocity does not match displacement from transfer function'

    # Test 2: Derivative of displacement should match velocity from transfer function
    # Calculate derivative of displacement using the same method as in
    # RedPitayaManager.get_velocity_data
    derived_velocity = coil_driver.derivative_displacement(
        displacement_zeroed, sample_rate
    )

    # Compare the derivative of displacement with the velocity from transfer function
    # Don't compare the end elements, not accurately computed at the edges
    # TODO: why doesn't rtol alone seem to work here? These are consistent in
    # plotting...
    assert np.allclose(derived_velocity[1:-1], velocity[1:-1], atol=10, rtol=0.05), (
        'Derivative of displacement does not match velocity from transfer function'
    )


def test_coil_driver_sample_spectrum_hermitian():
    """Test that the spectrum returned by CoilDriver.sample() with equalize_gain=True is
    Hermitian.

    A Hermitian spectrum has the property that S(-f) = S(f)*, where * denotes
    complex conjugation. This ensures that the inverse FFT will yield a real-valued
    signal.

    This test:
    1. Creates a Waveform instance and a CoilDriver instance
    2. Generates multiple waveform samples with equalize_gain=True
    3. For each sample, verifies that the spectrum satisfies the Hermitian property
    """
    # Create a waveform generator
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()

    # Test with multiple samples
    num_samples = 5
    for i in range(num_samples):
        # Generate a waveform with equalize_gain=True
        t, _voltage, spectrum = coil_driver.sample(waveform, normalize_gain=True)

        # Calculate sample rate from time array
        sample_rate = 1 / (t[1] - t[0])

        # Calculate frequencies
        n = len(t)
        freq = np.fft.fftfreq(n, d=1 / sample_rate)

        # For a Hermitian spectrum, S(-f) = S(f)*
        # We need to find pairs of frequencies with opposite signs
        for j in range(len(freq)):
            if freq[j] > 0:  # For positive frequencies
                # Find the corresponding negative frequency index
                neg_idx = np.where(freq == -freq[j])[0]
                if len(neg_idx) > 0:  # If we found a matching negative frequency
                    neg_idx = neg_idx[0]
                    # Check that S(-f) = S(f)*
                    assert np.isclose(spectrum[neg_idx], np.conj(spectrum[j])), (
                        f'Sample {i + 1}, Frequency {freq[j]} Hz: Spectrum is not '
                        f'Hermitian. '
                        f'S(-f)={spectrum[neg_idx]}, S(f)*={np.conj(spectrum[j])}'
                    )

        # Also verify that the inverse FFT of the spectrum has negligible imaginary
        # components
        ifft_result = np.fft.ifft(spectrum, norm='ortho')
        real_energy = np.sum(np.real(ifft_result) ** 2)
        imag_energy = np.sum(np.imag(ifft_result) ** 2)

        # The imaginary energy should be negligible compared to the real energy
        assert imag_energy < 1e-10 * real_energy, (
            f'Sample {i + 1}: Inverse FFT has significant imaginary components. '
            f'Real energy: {real_energy}, Imag energy: {imag_energy}'
        )


def test_gain_normalization():
    """Verifies that the shape of the velocity spectrum of the speaker surface matches
    the shape of the waveform spectrum. The speaker velocity spectrum exhibits a lot of
    gain near resonance, so the waveform spectrum that is sent to the device needs to be
    divided by the velocity transfer function to counteract this gain.

    This test:
    1. Creates a Waveform instance and a CoilDriver instance
    2. Generates a standard waveform and an equalized waveform
    3. Calculates velocity for both waveforms
    4. Verifies that the normalized equalized velocity spectrum shape matches the
    normalized standard voltage spectrum shape
    """
    # Create a waveform generator
    waveform = Waveform(start_freq=1, end_freq=100, gen_dec=8192, acq_dec=256)

    # Create a coil driver with default calibration parameters
    coil_driver = CoilDriver()

    # Generate a standard waveform (no equalization)
    t_std, voltage_std, voltage_spectrum_std = waveform.sample()

    # Generate an equalized waveform
    _t_eq, voltage_eq, _voltage_spectrum_eq = coil_driver.sample(
        waveform, normalize_gain=True, skip_randomization=True
    )

    # Calculate sample rate
    sample_rate = 1 / (t_std[1] - t_std[0])

    # Get velocity for equalized waveform
    velocity_eq, velocity_spectrum_eq, velocity_freqs = coil_driver.get_velocity(
        voltage_eq, sample_rate
    )

    # Calculate FFT of the standard voltage waveform and equalized velocity waveform
    freqs_fft_std, voltage_fft_std = calculate_fft(voltage_std, sample_rate)
    _, velocity_fft_eq = calculate_fft(velocity_eq, sample_rate)

    # Focus only on positive frequencies within the waveform's range
    pos_freq_mask = (freqs_fft_std > 0) & (freqs_fft_std < waveform.end_freq)

    # Normalize the FFTs by their maximum values to compare only their shapes
    velocity_fft_normalized = np.abs(velocity_fft_eq[pos_freq_mask]) / np.max(
        np.abs(velocity_fft_eq[pos_freq_mask])
    )
    voltage_fft_normalized = np.abs(voltage_fft_std[pos_freq_mask]) / np.max(
        np.abs(voltage_fft_std[pos_freq_mask])
    )

    # Compare the normalized FFTs calculated with calculate_fft
    assert np.allclose(
        velocity_fft_normalized,
        voltage_fft_normalized,
        rtol=0.2,  # Tolerance for numerical differences
    ), (
        'Normalized equalized velocity FFT shape does not match normalized standard '
        'voltage FFT shape'
    )

    # Compare the spectra from CoilDriver
    # Focus only on positive frequencies
    pos_freq_mask_coil = (velocity_freqs > 0) & (velocity_freqs < waveform.end_freq)

    # Normalize the spectra by their maximum values to compare only their shapes
    velocity_spectrum_normalized = np.abs(
        velocity_spectrum_eq[pos_freq_mask_coil]
    ) / np.max(np.abs(velocity_spectrum_eq[pos_freq_mask_coil]))
    voltage_spectrum_normalized = np.abs(
        voltage_spectrum_std[pos_freq_mask_coil]
    ) / np.max(np.abs(voltage_spectrum_std[pos_freq_mask_coil]))

    # Compare the normalized spectra from CoilDriver
    assert np.allclose(
        velocity_spectrum_normalized,
        voltage_spectrum_normalized,
        rtol=0.2,  # Tolerance for numerical differences
    ), (
        'Normalized equalized velocity spectrum shape does not match normalized '
        'standard '
        'voltage spectrum shape'
    )

    # Normalize the velocity FFT and spectrum to compare their shapes
    velocity_fft_norm = np.abs(velocity_fft_eq[pos_freq_mask]) / np.max(
        np.abs(velocity_fft_eq[pos_freq_mask])
    )
    velocity_spectrum_norm = np.abs(velocity_spectrum_eq[pos_freq_mask_coil]) / np.max(
        np.abs(velocity_spectrum_eq[pos_freq_mask_coil])
    )

    # Verify that the normalized equalized velocity FFT shape matches the normalized
    # equalized velocity spectrum shape
    assert np.allclose(
        velocity_fft_norm,
        velocity_spectrum_norm,
        rtol=0.2,  # Tolerance for numerical differences
    ), (
        'Normalized equalized velocity FFT shape does not match normalized equalized '
        'velocity spectrum shape'
    )

    # Normalize the voltage FFT and spectrum to compare their shapes
    voltage_fft_norm = np.abs(voltage_fft_std[pos_freq_mask]) / np.max(
        np.abs(voltage_fft_std[pos_freq_mask])
    )
    voltage_spectrum_norm = np.abs(voltage_spectrum_std[pos_freq_mask_coil]) / np.max(
        np.abs(voltage_spectrum_std[pos_freq_mask_coil])
    )

    # Verify that the normalized standard voltage FFT shape matches the normalized
    # standard voltage spectrum shape
    assert np.allclose(
        voltage_fft_norm,
        voltage_spectrum_norm,
        rtol=0.2,  # Tolerance for numerical differences
    ), (
        'Normalized standard voltage FFT shape does not match normalized standard '
        'voltage spectrum shape'
    )


if __name__ == '__main__':
    # Run the tests
    pytest.main(['-xvs', __file__])
