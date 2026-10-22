#!/usr/bin/env python3
"""Waveform Generator - A class for generating waveforms for Red Pitaya devices
This module provides a class that encapsulates functionality for generating
arbitrary waveforms with specific frequency characteristics.
"""

import numpy as np
import torch
from numpy.fft import fft, fftfreq, ifft
from numpy.random import default_rng

from ..redpitaya.redpitaya_config import RedPitayaConfig


class Waveform:
    """A class for generating waveforms with specific frequency characteristics for
    Red Pitaya devices.

    This class provides methods for:
    - Generating random waveforms within specific frequency ranges
    - Restricting waveforms to specific frequencies
    - Sampling waveforms for Red Pitaya output
    """

    # Constants from RedPitayaManager
    BUFFER_SIZE = RedPitayaConfig.BUFFER_SIZE
    SAMPLE_RATE_DEC1 = RedPitayaConfig.SAMPLE_RATE_DEC1

    def __init__(
        self,
        start_freq: float,
        end_freq: float,
        gen_dec: int = 8192,
        acq_dec: int = 256,
        allowed_freqs: list[float] | None = None,
        seed: int | None = None,
    ):
        """Initialize the Waveform generator.

        Args:
            start_freq: The lower bound of the valid frequency range (Hz)
            end_freq: The upper bound of the valid frequency range (Hz)
            gen_dec: Decimation for generation (default: 8192)
            acq_dec: Decimation for acquisition (default: 256)
            allowed_freqs: List of allowed frequencies to use (Hz), if None, all
            frequencies in range are allowed
        """
        self.start_freq = start_freq
        self.end_freq = end_freq
        self.gen_dec = gen_dec
        self.acq_dec = acq_dec
        self.allowed_freqs = allowed_freqs

        # Calculate sample rates
        self.gen_sample_rate = self.SAMPLE_RATE_DEC1 / self.gen_dec
        self.acq_sample_rate = self.SAMPLE_RATE_DEC1 / self.acq_dec

        # Calculate time array and frequency array
        self.burst_time = self.BUFFER_SIZE / self.gen_sample_rate
        self.t = np.linspace(0, self.burst_time, self.BUFFER_SIZE, endpoint=False)
        self.freq = fftfreq(self.BUFFER_SIZE, d=1 / self.gen_sample_rate)

        # Initialize spectrum and phases to None (will be randomized on first
        # sample). Annotated because the assignment alone infers `None`, and
        # then every later use -- ifft(self.spectrum), np.abs(self.spectrum) --
        # looks like a call against None to a type checker.
        self.spectrum: np.ndarray | None = None
        self.phase: np.ndarray | None = None

        # Initialize random number generator
        self.rng = default_rng(seed)

        # If allowed_freqs is not provided, use all frequencies in the range
        if self.allowed_freqs is None:
            # Calculate valid frequencies based on acquisition sample rate
            # Since our acquisition window size and resolution is fixed, we need to
            # choose generation frequencies that are compatible with that window
            # due to fundamental limitations of FFT (Nyquist and Rayleigh frequencies)
            self.valid_freqs = fftfreq(self.BUFFER_SIZE, d=1 / self.acq_sample_rate)
            # Filter to only include frequencies in the specified range
            self.valid_freqs = np.array(
                [
                    f
                    for f in self.valid_freqs
                    if start_freq <= abs(f) <= end_freq and f != 0
                ]
            )
        else:
            self.valid_freqs = np.array(self.allowed_freqs)

    def _randomize_spectrum(self):
        """Randomize the spectrum and phases for the waveform.
        This is called internally by sample() if needed.
        """
        # Only generate spectrum for non-negative frequencies (DC and positive)
        pos_freq_mask = self.freq >= 0
        pos_freqs = self.freq[pos_freq_mask]

        # Generate a random frequency spectrum between the start and end frequencies
        valid_freqs = self.valid_freqs[:, np.newaxis]
        valid_mask_pos = np.any(np.abs(pos_freqs - valid_freqs) <= 0.5, axis=0)

        # Generate spectrum only for positive frequencies
        spectrum_pos = self.rng.uniform(0.0, 1, len(pos_freqs))
        spectrum_pos = np.where(
            (pos_freqs >= self.start_freq)
            & (pos_freqs <= self.end_freq)
            & valid_mask_pos,
            spectrum_pos,
            0,
        )

        # Generate random Rayleigh distributed power spectrum for positive frequencies
        delta_f = self.freq[1] - self.freq[0] if len(self.freq) > 1 else 1.0
        rayleigh_spectrum_pos = self.rng.rayleigh(np.sqrt(spectrum_pos * delta_f))
        # See Phys. Rev. A 107, 042611 (2023) and https://doi.org/10.1016/0141-1187(84)90050-6
        # for why we use the Rayleigh distribution here
        # If we want Gaussian distributed a_n, b_n in a Fourier series for cosine and
        # sine components, each with variance_n = S(f_n)*\Delta f_n, then
        # c_n = sqrt(a_n**2 + b_n**2) is Rayleigh distributed with
        # variance_n = S(f_n)*\Delta f_n
        # rng.rayleigh(scale) expects a scale parameter = sqrt(variance) as input

        # Generate random phases for positive frequencies
        phi_pos = self.rng.uniform(0, 2 * np.pi, len(pos_freqs))

        # Create complex spectrum for positive frequencies
        spectrum_complex_pos = rayleigh_spectrum_pos * np.exp(1j * phi_pos)

        # Initialize full spectrum with zeros
        self.spectrum = np.zeros(len(self.freq), dtype=complex)

        # Set positive frequency components
        self.spectrum[pos_freq_mask] = spectrum_complex_pos

        # Apply Hermitian symmetry to get negative frequency components
        self.symmetrize_spectrum()

    def _randomize_phase(self):
        """Randomize only the phases for the waveform, keeping the spectrum amplitudes
        the same.
        This is called internally by sample() if randomize_phase_only=True.
        """
        # If no spectrum exists yet, we need to call _randomize_spectrum first
        if self.spectrum is None:
            self._randomize_spectrum()
            return

        # Only randomize phases for non-negative frequencies
        pos_freq_mask = self.freq >= 0

        # Get current amplitudes for all frequencies
        amplitudes = np.abs(self.spectrum)

        # Generate random phases only for non-negative frequencies
        phi_pos = self.rng.uniform(0, 2 * np.pi, int(np.sum(pos_freq_mask)))

        # Ensure DC component has zero phase if present
        if np.any(self.freq == 0):
            phi_pos[self.freq[pos_freq_mask] == 0] = 0

        # Create new spectrum with randomized phases for positive frequencies
        self.spectrum[pos_freq_mask] = amplitudes[pos_freq_mask] * np.exp(1j * phi_pos)

        # Apply Hermitian symmetry to get negative frequency components
        self.symmetrize_spectrum()

    def symmetrize_spectrum(self):
        """Enforce Hermitian symmetry on the spectrum to ensure real-valued signals
        after IFFT.
        This method uses a simple approach based on frequency sign to set negative
        frequency components as complex conjugates of their corresponding positive
        frequency components.
        """
        if self.spectrum is None:
            return

        # Create a new spectrum array with the same shape as the original
        symmetrized_spectrum = np.zeros_like(self.spectrum)

        # For evenly spaced FFT frequencies (which is the common case)
        # We can use a much simpler approach

        # Handle DC component (zero frequency) - ensure it's real
        zero_freq_idx = np.where(self.freq == 0)[0]
        if len(zero_freq_idx) > 0:
            symmetrized_spectrum[zero_freq_idx] = 0

        # Handle positive frequencies - keep as is
        pos_freq_idx = np.where(self.freq > 0)[0]
        if len(pos_freq_idx) > 0:
            symmetrized_spectrum[pos_freq_idx] = self.spectrum[pos_freq_idx]

        # For negative frequencies, find the corresponding positive frequency
        # and set to its complex conjugate
        for i, f in enumerate(self.freq):
            if f < 0:
                # Find the corresponding positive frequency index
                # For evenly spaced FFT frequencies, we can use a direct mapping
                pos_idx = np.argmin(np.abs(self.freq + f))
                if pos_idx != i:  # Make sure we don't use the same index
                    symmetrized_spectrum[i] = np.conj(self.spectrum[pos_idx])

        # Update the spectrum with the symmetrized version
        self.spectrum = symmetrized_spectrum

    def _random_single_tone_spectrum(self):
        """Generate a spectrum containing a single tone at a randomly selected
        frequency.
        This is called internally by sample() if random_single_tone=True.
        """
        # Create an empty spectrum
        spectrum = np.zeros(len(self.freq), dtype=complex)

        # Randomly select one of the valid frequencies
        if len(self.valid_freqs) > 0:
            positive_freqs = self.valid_freqs[self.valid_freqs > 0]
            selected_freq = self.rng.choice(positive_freqs)
        else:
            raise ValueError('No valid frequencies found')

        # Find the index in the frequency array closest to the selected frequency
        freq_idx = np.argmin(np.abs(self.freq - selected_freq))

        # Generate a random phase
        phi = self.rng.uniform(0, 2 * np.pi)

        # Set the amplitude at the selected frequency to 1 with the random phase
        spectrum[freq_idx] = 1.0 * np.exp(1j * phi)

        # Store the spectrum
        self.spectrum = spectrum

    def _init_torch_cache(self, device: torch.device) -> None:
        """Initialize cached torch tensors for generate_batch_torch.

        Precomputes frequency grid, valid-frequency mask, and Hermitian symmetry
        index mappings on the target device. Called once and cached.
        """
        N = self.BUFFER_SIZE
        freq_t = torch.fft.fftfreq(N, d=1.0 / self.acq_sample_rate, device=device)

        # Valid positive frequency mask
        valid_freqs_t = torch.tensor(
            self.valid_freqs, dtype=torch.float64, device=device
        )
        pos_mask = freq_t > 0
        pos_freqs = freq_t[pos_mask]

        # Match valid frequencies: for each positive freq, check if any valid freq
        # is within 0.5 Hz
        valid_match = torch.any(
            torch.abs(pos_freqs.unsqueeze(-1) - valid_freqs_t.unsqueeze(0)) <= 0.5,
            dim=-1,
        )
        # Combined mask: positive, in range, and matches a valid frequency
        valid_pos_full = torch.zeros(N, dtype=torch.bool, device=device)
        valid_pos_full[pos_mask] = (
            (pos_freqs >= self.start_freq) & (pos_freqs <= self.end_freq) & valid_match
        )

        # Hermitian symmetry: for standard FFT layout, negative freq at index k
        # maps to positive freq at index (N - k)
        neg_mask = freq_t < 0
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        conj_indices = N - neg_indices

        # Frequency resolution
        delta_f = float(self.acq_sample_rate / N)

        self._torch_cache = {
            'device': device,
            'freq': freq_t,
            'valid_pos_mask': valid_pos_full,
            'neg_indices': neg_indices,
            'conj_indices': conj_indices,
            'delta_f': delta_f,
            'N': N,
        }

    def generate_batch_torch(
        self, batch_size: int, device: torch.device | str = 'cpu'
    ) -> torch.Tensor:
        """Generate a batch of random displacement waveforms using PyTorch on GPU.

        Uses the same Rayleigh amplitude + uniform phase approach as the NumPy
        _randomize_spectrum method, but operates in batched mode on the target device.
        Reuses precomputed frequency grids and valid frequency masks from __init__.

        The generated waveforms have frequency content only within
        [start_freq, end_freq] at the acquisition sample rate, matching the
        real experiment's spectral properties.

        See Phys. Rev. A 107, 042611 (2023) and
        https://doi.org/10.1016/0141-1187(84)90050-6 for why we use the Rayleigh
        distribution for the spectral amplitudes.

        Args:
            batch_size: Number of waveforms to generate
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            Displacement tensor of shape [batch_size, BUFFER_SIZE]
        """
        device = torch.device(device) if isinstance(device, str) else device

        # Initialize or update torch cache if needed
        if not hasattr(self, '_torch_cache') or self._torch_cache['device'] != device:
            self._init_torch_cache(device)

        cache = self._torch_cache
        N = cache['N']
        valid_pos_mask = cache['valid_pos_mask']
        neg_indices = cache['neg_indices']
        conj_indices = cache['conj_indices']
        delta_f = cache['delta_f']

        # 1. Generate Rayleigh-distributed amplitudes for valid positive frequencies
        # Rayleigh(scale) = scale * sqrt(-2 * ln(U)), U ~ Uniform(0, 1)
        # scale = sqrt(S(f) * delta_f), with S(f) ~ Uniform(0, 1) for valid freqs
        u_amp = torch.rand(batch_size, N, device=device).clamp(min=1e-10)
        s_f = torch.rand(batch_size, N, device=device)  # power spectral density
        scale = torch.sqrt(s_f * delta_f)
        amplitudes = scale * torch.sqrt(-2.0 * torch.log(u_amp))
        # Zero out non-valid frequencies
        amplitudes[:, ~valid_pos_mask] = 0.0

        # 2. Random uniform phases in [0, 2*pi)
        phases = torch.rand(batch_size, N, device=device) * (2.0 * torch.pi)

        # 3. Build complex spectrum for positive frequencies
        spectrum = torch.zeros(batch_size, N, dtype=torch.cfloat, device=device)
        spectrum[:, valid_pos_mask] = (
            amplitudes[:, valid_pos_mask] * torch.exp(1j * phases[:, valid_pos_mask])
        ).to(torch.cfloat)

        # 4. Hermitian symmetry: spectrum[neg_idx] = conj(spectrum[N - neg_idx])
        spectrum[:, neg_indices] = spectrum[:, conj_indices].conj()

        # 5. DC and Nyquist to zero
        spectrum[:, 0] = 0.0
        if N % 2 == 0:
            spectrum[:, N // 2] = 0.0

        # 6. IFFT to time domain
        return torch.fft.ifft(spectrum, norm='ortho').real

    def sample(
        self,
        randomize_phase_only: bool = False,
        random_single_tone: bool = False,
        skip_randomization: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a sample waveform using the current configuration.
        Randomizes the spectrum and phases each time it's called.

        Args:
            randomize_phase_only: If True, only randomize the phases while keeping the
            spectrum amplitudes the same
            random_single_tone: If True, generate a single tone at a randomly selected
            valid frequency
            skip_randomization: If True, do not randomize the spectrum and phases. Used
            to repeatedly generate the same waveform during testing. Requires that
            sample() has been called at least once before, and we want to resample with
            the same spectrum and phases.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Array of time points
                - Amplitude points in time domain
                - Spectrum (complex amplitude)
        """
        # Generate appropriate spectrum based on parameters
        if random_single_tone:
            self._random_single_tone_spectrum()
        elif randomize_phase_only:
            self._randomize_phase()
        elif not skip_randomization or self.spectrum is None:
            # If skip_randomization is True, we will not randomize the spectrum and
            # phases. But, if self.spectrum is None, we need to call
            # _randomize_spectrum() to initialize it.
            self._randomize_spectrum()

        # Every branch above either sets self.spectrum or, in the
        # skip_randomization case, has already checked it is not None. The
        # assertion states that invariant for a reader and for the type
        # checker, which cannot follow it across the helper calls.
        assert self.spectrum is not None

        # Convert to time domain
        y = np.real(ifft(self.spectrum, norm='ortho'))
        y = np.fft.fftshift(y)

        # Recompute the spectrum and phase for use in other modules
        # First, shift back to match the original order
        y_unshifted = np.fft.ifftshift(y)

        # Compute the FFT to get the normalized spectrum
        normalized_spectrum = fft(y_unshifted, norm='ortho')

        return self.t, y, normalized_spectrum
