#!/usr/bin/env python3
"""Test for the Waveform.sample() method.

This test compares the spectrum values returned by the Waveform.sample() method
with those calculated manually using the calculate_fft function.
"""

import logging

import numpy as np
import pytest
from plot_waveforms import calculate_fft

from smi.synthetic.waveform import Waveform

logger = logging.getLogger(__name__)


def test_waveform_sample_spectrum_consistency():
    """Test that the spectrum values returned by Waveform.sample() are consistent
    with those calculated by calculate_fft().

    This test:
    1. Creates a Waveform instance
    2. Calls sample() multiple times (at least 10)
    3. For each sample, compares the spectrum values returned by sample()
       with those calculated by calculate_fft()
    """
    # Run the test multiple times with different random waveforms
    num_tests = 10
    for i in range(num_tests):
        # Create a new waveform generator with slightly different frequency range
        # for each test
        test_start_freq = 10 + i
        test_end_freq = 1000 + i * 10
        waveform = Waveform(
            start_freq=test_start_freq,
            end_freq=test_end_freq,
            gen_dec=8192,
            acq_dec=256,
        )

        # Generate a random waveform
        t, voltage, voltage_spectrum = waveform.sample()

        # Calculate sample rate from time array
        sample_rate = 1 / (t[1] - t[0])

        # Calculate FFT of the voltage waveform using the calculate_fft function
        _freqs_fft, spectrum_fft = calculate_fft(voltage, sample_rate)

        # Extract magnitude and phase from the complex spectrum
        voltage_mag_fft = np.abs(spectrum_fft)
        voltage_mag_sample = np.abs(voltage_spectrum)[: len(voltage_mag_fft)]

        assert np.allclose(voltage_mag_sample, voltage_mag_fft, rtol=0.1), (
            f'Test {i + 1}: Spectrum magnitude mismatch'
        )

        logger.info(f'Completed test iteration {i + 1}/{num_tests}')


def test_waveform_sample_reproducibility():
    """Test that consecutive calls to sample() with the same parameters
    produce different random waveforms as expected.
    """
    # Create a waveform generator
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Generate two consecutive waveforms
    _, voltage1, _ = waveform.sample()
    _, voltage2, _ = waveform.sample()

    # They should be different (random generation)
    assert not np.allclose(voltage1, voltage2), (
        'Consecutive samples should be different'
    )


def test_waveform_frequency_range():
    """Test that the waveform spectrum is confined to the specified frequency range."""
    # Create a waveform generator with narrow frequency range
    start_freq = 100
    end_freq = 200
    waveform = Waveform(
        start_freq=start_freq, end_freq=end_freq, gen_dec=8192, acq_dec=256
    )

    # Generate a waveform
    _, voltage, voltage_spectrum = waveform.sample()

    # Calculate sample rate from time array
    sample_rate = 1 / (waveform.t[1] - waveform.t[0])

    # Calculate FFT of the voltage waveform using the calculate_fft function
    freqs_fft, _spectrum_fft = calculate_fft(voltage, sample_rate)

    # Check that significant spectral content is within the specified range
    # (allowing for some leakage)
    voltage_spectral_mod = np.abs(voltage_spectrum)[: len(freqs_fft)]
    threshold = 0.1 * np.max(voltage_spectral_mod)
    significant_freqs = freqs_fft[voltage_spectral_mod > threshold]

    # Check lower bound (with some tolerance)
    assert np.min(significant_freqs) >= start_freq * 0.9, (
        f'Significant frequency content found below start_freq: '
        f'{np.min(significant_freqs)} < {start_freq}'
    )

    # Check upper bound (with some tolerance)
    assert np.max(significant_freqs) <= end_freq * 1.1, (
        f'Significant frequency content found above end_freq: '
        f'{np.max(significant_freqs)} > {end_freq}'
    )


def test_waveform_output_range():
    """Test that the waveform output is properly normalized to the [-1, 1] range."""
    # Create a waveform generator
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Generate multiple waveforms and check their ranges
    for i in range(5):
        _, voltage, _ = waveform.sample()
        assert np.max(np.abs(voltage)) <= 1.0, (
            f'Test {i + 1}: Voltage exceeds normalized range [-1, 1]'
        )


def test_waveform_length_consistency():
    """Test that the time and voltage arrays have consistent lengths."""
    # Create a waveform generator
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Generate multiple waveforms and check their lengths
    for i in range(5):
        t, voltage, _voltage_spectrum = waveform.sample()
        assert len(t) == len(voltage), (
            f'Test {i + 1}: Time and voltage arrays have different lengths'
        )


def test_waveform_reconstruction():
    """Test that the reconstructed signal is close to the original signal."""
    # Create a waveform generator
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Generate a waveform
    _, voltage, voltage_spectrum = waveform.sample()

    # Convert to time domain
    reconstructed = np.real(np.fft.ifft(voltage_spectrum, norm='ortho'))
    reconstructed = np.fft.fftshift(reconstructed)

    # The reconstructed signal should be normalized
    assert np.max(np.abs(reconstructed)) <= 1.0, (
        'Reconstructed signal exceeds normalized range'
    )

    # The reconstructed signal should be close to the original signal
    assert np.allclose(voltage, reconstructed, rtol=0.1), 'Signal reconstruction failed'


def test_waveform_statistics():
    """Test that the Gaussian distributions overlaid on the histograms in
    plot_waveform_histograms accurately represent the statistical properties of the
    waveform samples.

    This test:
    1. Generates multiple waveform samples with phase randomization only
    2. Computes the expected Gaussian distributions based on the noise variance
    3. Compares the statistical properties of the samples with the
    expected distributions
    4. Verifies that the computed variance matches the empirical variance of the samples
    """
    # Create a waveform generator with test parameters
    start_freq = 100
    end_freq = 500
    num_samples = 20  # Reduced for test efficiency

    # Seeded: this compares an empirical variance over only 20 samples against
    # the analytic one at rtol=0.3, which is a wide enough tolerance to pass
    # most of the time and a narrow enough one to fail occasionally. Unseeded
    # it did exactly that, and an intermittently red suite is one people learn
    # to re-run rather than read.
    waveform = Waveform(
        start_freq=start_freq,
        end_freq=end_freq,
        gen_dec=8192,
        acq_dec=256,
        seed=20250910,
    )

    # Initialize arrays to store all voltage samples and reconstructed complex signals
    all_voltages = []
    all_complex_values = []

    # Store the last voltage_spectral_amp for variance calculation
    last_voltage_spectral_amp = None
    last_t = None

    # Generate multiple waveform samples with phase randomization only
    for _ in range(num_samples):
        # Generate a random waveform with phase randomization only
        t, voltage, voltage_spectrum = waveform.sample(randomize_phase_only=True)

        # Store for later use
        last_voltage_spectral_amp = voltage_spectrum
        last_t = t

        # Store the time-domain voltage values
        all_voltages.extend(voltage)

        # Use the complex spectrum directly
        complex_spectrum = voltage_spectrum

        # Convert to time domain without taking real part
        reconstructed_complex = np.fft.ifft(complex_spectrum, norm='ortho')
        reconstructed_complex = np.fft.fftshift(reconstructed_complex)

        # Store the complex-valued reconstructed signals
        all_complex_values.extend(reconstructed_complex)

    # Convert to numpy arrays
    all_voltages = np.array(all_voltages)
    all_complex_values = np.array(all_complex_values)

    # Compute variance for the noise distribution using the same formula as in
    # plot_waveform_histograms
    assert last_t is not None
    last_voltage_spectral_amp = np.abs(voltage_spectrum)
    noise_variance = (
        np.sum(last_voltage_spectral_amp**2)
        * len(last_t)
        * (last_t[1] - last_t[0]) ** 2
        / np.max(last_t)
        * (waveform.freq[1] - waveform.freq[0])
    )

    # Test 1: Check that the mean of the time-domain voltage values is close to zero
    assert np.isclose(np.mean(all_voltages), 0, atol=0.05), (
        f'Mean of voltage values ({np.mean(all_voltages)}) is not close to zero'
    )

    # Test 2: Check that the empirical variance of the time-domain voltage values is
    # close to the computed noise variance
    empirical_variance = np.var(all_voltages)
    assert np.isclose(empirical_variance, noise_variance, rtol=0.3), (
        f"Empirical variance ({empirical_variance}) doesn't match computed variance "
        f'({noise_variance})'
    )

    # Test 3: Check that the real and imaginary parts of the inverted spectrum have the
    # correct variance. The real variance should be as we computed above, while the
    # imaginary part variance should be near zero because the imaginary part should be
    # close to zero.
    real_parts = np.real(all_complex_values)
    imag_parts = np.imag(all_complex_values)

    real_variance = np.var(real_parts)
    imag_variance = np.var(imag_parts)
    expected_complex_variance = noise_variance

    assert np.isclose(real_variance, expected_complex_variance, rtol=0.3), (
        f"Real part variance ({real_variance}) doesn't match expected variance "
        f'({expected_complex_variance})'
    )
    assert np.isclose(imag_variance, 0, rtol=0.3), (
        f"Imaginary part variance ({imag_variance}) doesn't match expected variance "
        f'({0})'
    )

    # Test 4: Check that the real and imaginary parts are uncorrelated (for a circular
    # complex Gaussian)
    correlation = np.corrcoef(real_parts, imag_parts)[0, 1]
    assert abs(correlation) < 0.1, (
        f'Real and imaginary parts are correlated ({correlation}), should be '
        'uncorrelated'
    )

    # Test 5: Check the quartiles of the data against theoretical quartiles for a
    # Gaussian. For a standard normal distribution, the quartiles are
    # approximately -0.67, 0, and 0.67

    # Normalize the voltage values
    normalized_voltages = (all_voltages - np.mean(all_voltages)) / np.std(all_voltages)

    # Get the empirical quartiles
    q1, q2, q3 = np.percentile(normalized_voltages, [25, 50, 75])

    # Check against theoretical quartiles with some tolerance
    assert np.isclose(q1, -0.67, atol=0.1), (
        f'First quartile ({q1}) is not close to expected value (-0.67)'
    )
    assert np.isclose(q2, 0.0, atol=0.1), (
        f'Median ({q2}) is not close to expected value (0.0)'
    )
    assert np.isclose(q3, 0.67, atol=0.1), (
        f'Third quartile ({q3}) is not close to expected value (0.67)'
    )

    # Test 6: Check that the range of the data is reasonable for a Gaussian
    # For a large sample from a Gaussian, we expect most values to be within
    # +/-3 standard deviations
    normalized_max = np.max(np.abs(normalized_voltages))
    assert normalized_max < 5.0, (
        f'Maximum absolute normalized value ({normalized_max}) is too large for a '
        'Gaussian distribution'
    )


def test_waveform_ifft_real_valued():
    """Test that the inverse FFT of the spectrum has no imaginary components.
    This verifies that the Hermitian symmetry is properly maintained in the spectrum.

    This test:
    1. Creates a Waveform instance
    2. Generates multiple waveform samples
    3. For each sample, verifies that the inverse FFT of the spectrum has negligible
    imaginary components
    """
    # Create a waveform generator
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Test with multiple samples
    num_samples = 5
    for i in range(num_samples):
        # Generate a waveform
        _, _, spectrum = waveform.sample()

        # Compute the inverse FFT
        ifft_result = np.fft.ifft(spectrum, norm='ortho')

        # Calculate the ratio of imaginary to real energy
        real_energy = np.sum(np.real(ifft_result) ** 2)
        imag_energy = np.sum(np.imag(ifft_result) ** 2)

        # The imaginary energy should be negligible compared to the real energy
        # We'll use a threshold of 1e-10 times the real energy
        assert imag_energy < 1e-10 * real_energy, (
            f'Sample {i + 1}: Inverse FFT has significant imaginary components. '
            f'Real energy: {real_energy}, Imag energy: {imag_energy}'
        )


def test_waveform_spectrum_hermitian():
    """Test that the spectrum returned by Waveform.sample() is Hermitian.

    A Hermitian spectrum has the property that S(-f) = S(f)*, where * denotes
    complex conjugation. This ensures that the inverse FFT will yield a
    real-valued signal.

    This test:
    1. Creates a Waveform instance
    2. Generates multiple waveform samples
    3. For each sample, verifies that the spectrum satisfies the Hermitian property
    """
    # Create a waveform generator
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)

    # Test with multiple samples
    num_samples = 5
    for i in range(num_samples):
        # Generate a waveform
        _, _, spectrum = waveform.sample()

        # Get the frequency array
        freq = waveform.freq

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
                        f'Hermitian. S(-f)={spectrum[neg_idx]}, '
                        f'S(f)*={np.conj(spectrum[j])}'
                    )


if __name__ == '__main__':
    # Run the tests
    pytest.main(['-xvs', __file__])
