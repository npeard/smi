#!/usr/bin/env python3
"""Coil Driver - A class for calibrating and computing displacement and velocity
from voltage waveforms.

This module provides a class that encapsulates functionality for converting voltage
waveforms to displacement and velocity waveforms using calibration data.
"""

import numpy as np
import torch
from numpy.fft import fft, fftfreq, ifft

from .calib_params import CalibrationParameters
from .waveform import Waveform

# Create a default RNG instance for the module
_rng = np.random.default_rng()


class CoilDriver:
    """A class for calibrating and computing displacement and velocity from voltage
    waveforms.

    This class provides methods for:
    - Converting voltage waveforms to displacement waveforms
    - Converting voltage waveforms to velocity waveforms
    - Calculating transfer functions for the coil driver
    """

    def __init__(self, calibration_params: CalibrationParameters | None = None):
        """Initialize the CoilDriver with calibration parameters.

        Args:
            calibration_params: Calibration parameters for the coil driver.
                                If None, default parameters are used.
        """
        if calibration_params is None:
            self.params = CalibrationParameters()
        else:
            self.params = calibration_params

    def sample(
        self,
        waveform: Waveform,
        randomize_phase_only: bool = False,
        random_single_tone: bool = False,
        normalize_gain: bool = False,
        skip_randomization: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a sample waveform using the Waveform generator, with optional gain
        equalization.

        When normalize_gain is True, the spectrum is pre-compensated by dividing by the
        amplitude transfer function, so that when the displacement_spectrum is computed
        via multiplication, the spectral components are the same as originally generated
        by Waveform.sample().

        Args:
            waveform: Waveform generator instance
            randomize_phase_only: If True, only randomize the phases while keeping the
                spectrum amplitudes the same
            random_single_tone: If True, generate a single tone at a randomly selected
                valid frequency
            normalize_gain: If True, pre-compensate the spectrum to normalize the gain
                across frequencies
            skip_randomization: If True, do not randomize the spectrum and phases. Used
                to repeatedly generate the same waveform during testing. Requires that
                sample() has been called at least once before, and we want to resample
                with the same spectrum and phases.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Array of time points
                - Amplitude points in time domain
                - Spectrum (complex amplitude)
        """
        # If normalize_gain is False, simply return the output of Waveform.sample()
        if not normalize_gain:
            return waveform.sample(
                randomize_phase_only=randomize_phase_only,
                random_single_tone=random_single_tone,
            )

        # Generate a sample from the waveform
        t, _voltage, voltage_spectrum = waveform.sample(
            randomize_phase_only=randomize_phase_only,
            random_single_tone=random_single_tone,
            skip_randomization=skip_randomization,
        )

        # Calculate frequencies for the full spectrum
        sample_rate = 1 / (t[1] - t[0])
        n = len(t)
        freq = fftfreq(n, d=1 / sample_rate)

        # Get the complex transfer function
        transfer_function = self.get_transfer_function(freq)

        # Compute compensation for velocity transform (because velocity is the quantity
        # we care about, we'd like for its spectrum to be the same shape as the original
        # waveform (flat)
        velocity_transfer = transfer_function * 1j * 2 * np.pi * freq
        # We apply a scaling factor here so that we don't have to apply large amounts of
        # gain to the original spectrum. This is OK because ultimately we just want the
        # shape of the velocity spectrum to match the original waveform (flat) and we
        # don't care so much about the overall scaling as long as it is within the
        # hardware constraints and allows us to scan over multiple fringes
        scaling_factor = np.abs(velocity_transfer[freq == waveform.valid_freqs[0]])
        # The extra factor of 10 here is manually selected and seems to produce voltages
        # in a compatible range
        velocity_transfer /= scaling_factor * 10

        # Pre-compensate the spectrum by dividing by the complex transfer function
        nonzero_mask = velocity_transfer != 0
        normalized_spectrum = np.ones_like(voltage_spectrum)
        normalized_spectrum[nonzero_mask] = (
            voltage_spectrum[nonzero_mask] / velocity_transfer[nonzero_mask]
        )
        # Apply random uniform scaling to randomize total waveform power
        normalized_spectrum *= _rng.uniform(0, 1.0)

        # Convert to time domain
        normalized_voltage = np.real(ifft(normalized_spectrum, norm='ortho'))
        normalized_voltage = np.fft.fftshift(normalized_voltage)

        # Recompute the spectrum and phase for use in other modules
        # First, shift back to match the original order
        y_unshifted = np.fft.ifftshift(normalized_voltage)

        # Compute the FFT to get the normalized spectrum
        normalized_spectrum = fft(y_unshifted, norm='ortho')

        return t, normalized_voltage, normalized_spectrum

    def _calculate_amplitude_transfer(self, f: np.ndarray) -> np.ndarray:
        """Calculate the amplitude transfer function for the coil driver.

        Args:
            f: Frequencies at which to calculate the transfer function (Hz)

        Returns:
            Amplitude transfer function (microns/V)
        """
        return (self.params.k * self.params.f0**2) / np.sqrt(
            (self.params.f0**2 - f**2) ** 2
            + self.params.f0**2 * f**2 / self.params.Q**2
        )

    def _calculate_phase_transfer(self, f: np.ndarray) -> np.ndarray:
        """Calculate the phase transfer function for the coil driver.

        Args:
            f: Frequencies at which to calculate the transfer function (Hz)

        Returns:
            Phase transfer function (radians)
        """
        return (
            np.arctan2(self.params.f0 / self.params.Q * f, f**2 - self.params.f0**2)
            + self.params.c
        )

    def get_transfer_function(
        self, freq: np.ndarray, max_freq: float | None = None
    ) -> np.ndarray:
        """Calculate the complex transfer function for the given frequencies.

        Args:
            freq: Frequency array (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included

        Returns:
            Complex transfer function as a numpy array
        """
        # Calculate the amplitude transfer function
        amplitude_transfer = self._calculate_amplitude_transfer(np.abs(freq))

        # Apply frequency limit if specified
        if max_freq is not None:
            amplitude_transfer = np.where(
                np.abs(freq) <= max_freq, amplitude_transfer, 0
            )

        # Calculate the phase transfer function
        # For negative frequencies, we need to conjugate the phase
        phase_transfer = np.where(
            freq < 0,
            np.exp(-1j * self._calculate_phase_transfer(-freq)),
            np.exp(1j * self._calculate_phase_transfer(freq)),
        )

        # Return the complex transfer function
        return amplitude_transfer * phase_transfer

    def get_displacement_spectrum(
        self,
        voltage_waveform: np.ndarray,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the displacement spectrum from a voltage waveform using the
        calibration parameters. This is used to compute the displacement and velocity
        waveforms.

        Args:
            voltage_waveform: Voltage waveform in time domain (V)
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included

        Returns:
            Tuple containing:
            - Displacement spectrum in frequency domain
            - Frequency array (Hz)
        """
        # Calculate the spectrum of the voltage waveform
        voltage_spectrum = fft(voltage_waveform, norm='ortho')
        n = voltage_waveform.size
        sample_spacing = 1 / sample_rate
        freq = fftfreq(n, d=sample_spacing)  # units: cycles/s = Hz

        # Get the complex transfer function
        transfer_function = self.get_transfer_function(freq, max_freq)

        # Multiply by the transfer function
        displacement_spectrum = voltage_spectrum * transfer_function

        # Divide by 2 to account for the fact that we are using a two-sided spectrum
        displacement_spectrum /= 2

        return displacement_spectrum, freq

    def get_displacement(
        self,
        voltage_waveform: np.ndarray,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the displacement waveform from a voltage waveform using the
        calibration parameters.

        Args:
            voltage_waveform: Voltage waveform in time domain (V)
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included

        Returns:
            Tuple containing:
            - Displacement waveform in time domain (microns)
            - Displacement waveform in frequency domain
            - Frequency array (Hz)
        """
        displacement_spectrum, freq = self.get_displacement_spectrum(
            voltage_waveform, sample_rate, max_freq
        )

        # Convert back to time domain
        displacement_waveform = np.real(ifft(displacement_spectrum, norm='ortho'))

        return displacement_waveform, displacement_spectrum, freq

    def get_velocity(
        self,
        voltage_waveform: np.ndarray,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the velocity waveform from a voltage waveform using the calibration
        parameters.

        Args:
            voltage_waveform: Voltage waveform in time domain (V)
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz)
                      If None, all frequencies are included

        Returns:
            Tuple containing:
            - Velocity waveform in time domain (microns/s)
            - Velocity waveform in frequency domain
            - Frequency array (Hz)
        """
        displacement_spectrum, freq = self.get_displacement_spectrum(
            voltage_waveform, sample_rate, max_freq
        )

        # Calculate velocity spectrum by multiplying displacement spectrum by j*omega
        velocity_spectrum = displacement_spectrum * 1j * 2 * np.pi * freq

        # Convert back to time domain
        velocity_waveform = np.real(ifft(velocity_spectrum, norm='ortho'))

        return velocity_waveform, velocity_spectrum, freq

    def get_velocity_and_displacement(
        self,
        voltage_waveform: np.ndarray,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute velocity and displacement from one shared displacement spectrum.

        ``get_velocity`` and ``get_displacement`` each call
        ``get_displacement_spectrum`` internally, so calling both -- as the
        dataset does for every sample -- computes the same forward FFT and
        transfer-function product twice. This method computes it once.

        Args:
            voltage_waveform: Voltage waveform in time domain (V)
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include in the calculation (Hz).
                If None, all frequencies are included.

        Returns:
            Tuple of (velocity in microns/s, displacement in microns), both in
            the time domain and the same shape as ``voltage_waveform``.
        """
        displacement_spectrum, freq = self.get_displacement_spectrum(
            voltage_waveform, sample_rate, max_freq
        )
        velocity_spectrum = displacement_spectrum * 1j * 2 * np.pi * freq

        displacement = np.real(ifft(displacement_spectrum, norm='ortho'))
        velocity = np.real(ifft(velocity_spectrum, norm='ortho'))

        return velocity, displacement

    def get_transfer_function_torch(
        self, freq: torch.Tensor, max_freq: float | None = None
    ) -> torch.Tensor:
        """Torch port of :meth:`get_transfer_function`.

        Args:
            freq: 1-D real tensor of frequencies (Hz).
            max_freq: Maximum frequency to include (Hz). If None, all are kept.

        Returns:
            Complex transfer function with the same shape as ``freq``.
        """
        abs_freq = torch.abs(freq)
        f0 = self.params.f0
        q = self.params.Q

        amplitude = (self.params.k * f0**2) / torch.sqrt(
            (f0**2 - abs_freq**2) ** 2 + f0**2 * abs_freq**2 / q**2
        )
        if max_freq is not None:
            amplitude = torch.where(
                abs_freq <= max_freq, amplitude, torch.zeros_like(amplitude)
            )

        # Phase is evaluated at |f| and conjugated for negative frequencies,
        # matching the numpy branch exactly.
        phase = torch.atan2(f0 / q * abs_freq, abs_freq**2 - f0**2) + self.params.c
        signed_phase = torch.where(freq < 0, -phase, phase)

        return torch.polar(amplitude, signed_phase)

    def get_displacement_spectrum_torch(
        self,
        voltage_waveform: torch.Tensor,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched torch port of :meth:`get_displacement_spectrum`.

        Args:
            voltage_waveform: Real tensor of shape ``(batch, time)`` (or
                ``(time,)``) on any device.
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include (Hz). If None, all are kept.

        Returns:
            Tuple of (complex displacement spectrum with the same shape as the
            input, 1-D frequency tensor in Hz).
        """
        n = voltage_waveform.shape[-1]
        # Do the physics in float64/complex128 so the result matches the numpy
        # path to float32 tolerance even for float32 inputs.
        work = voltage_waveform.to(torch.float64)
        spectrum = torch.fft.fft(work, dim=-1, norm='ortho')
        freq = torch.fft.fftfreq(
            n, d=1.0 / sample_rate, device=voltage_waveform.device, dtype=torch.float64
        )
        transfer = self.get_transfer_function_torch(freq, max_freq)

        return spectrum * transfer / 2, freq

    def get_velocity_and_displacement_torch(
        self,
        voltage_waveform: torch.Tensor,
        sample_rate: float,
        max_freq: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched torch port of the velocity/displacement physics.

        Numerically equivalent to :meth:`get_velocity_and_displacement` (and
        hence to ``get_velocity``/``get_displacement``) to float32 tolerance,
        but operates on a whole ``(batch, time)`` tensor on-device via
        ``torch.fft``.

        Args:
            voltage_waveform: Real tensor of shape ``(batch, time)`` (or
                ``(time,)``) on any device.
            sample_rate: Sample rate of the waveform (Hz)
            max_freq: Maximum frequency to include (Hz). If None, all are kept.

        Returns:
            Tuple of (velocity in microns/s, displacement in microns), both real
            tensors with the input's shape, dtype and device.
        """
        displacement_spectrum, freq = self.get_displacement_spectrum_torch(
            voltage_waveform, sample_rate, max_freq
        )
        velocity_spectrum = displacement_spectrum * (2j * np.pi * freq)

        displacement = torch.fft.ifft(displacement_spectrum, dim=-1, norm='ortho').real
        velocity = torch.fft.ifft(velocity_spectrum, dim=-1, norm='ortho').real

        out_dtype = (
            voltage_waveform.dtype
            if voltage_waveform.is_floating_point()
            else torch.float32
        )
        return velocity.to(out_dtype), displacement.to(out_dtype)

    @staticmethod
    def integrate_velocity(
        velocity_waveform: np.ndarray | torch.Tensor, sample_rate: float
    ) -> np.ndarray | torch.Tensor:
        """Integrate velocity waveform to get displacement waveform using
        cumulative integration.

        Compatible with both NumPy arrays and PyTorch tensors.

        Args:
            velocity_waveform: Velocity waveform (microns/s)
                              Can be a NumPy array or PyTorch tensor
                              For PyTorch tensors, supports batch dimensions
                              [batch_size, channels, signal_length]
            sample_rate: Sample rate of the velocity waveform (Hz)

        Returns:
            Displacement waveform (microns) in the same format as input
        """
        # Check if input is a PyTorch tensor
        # More robust check using module name instead of attribute
        is_torch = 'torch' in str(type(velocity_waveform).__module__)
        # Time step
        dt = 1.0 / sample_rate

        if is_torch:
            # Handle batch dimensions if present
            if len(velocity_waveform.shape) > 1:
                _batch_size, _signal_length = velocity_waveform.shape

                displacement = torch.cumsum(velocity_waveform, dim=-1) * dt  # ty: ignore[no-matching-overload]  # narrowed to Tensor at runtime via is_torch
                # Shift displacement to start at zero
                displacement = displacement - displacement[:, 0].unsqueeze(-1)

            else:
                # Single waveform case
                displacement = torch.cumsum(velocity_waveform, dim=0) * dt  # ty: ignore[no-matching-overload]  # narrowed to Tensor at runtime via is_torch

                # Shift displacement to start at zero
                displacement = displacement - displacement[0]

            return displacement

        else:
            # NumPy implementation (original)
            # Simple cumulative integration (cumulative sum * dt)
            displacement = np.cumsum(velocity_waveform) * dt

            # Shift displacement to start at zero at the beginning of the trace
            return displacement - displacement[0]

    @staticmethod
    def derivative_displacement(
        displacement_waveform: np.ndarray | torch.Tensor, sample_rate: float
    ) -> np.ndarray | torch.Tensor:
        """Calculate the time derivative of a displacement waveform to get velocity.

        This method uses central differences to compute the derivative.
        Compatible with both NumPy arrays and PyTorch tensors.

        Args:
            displacement_waveform: Displacement waveform (microns)
                                  Can be a NumPy array or PyTorch tensor
                                  For PyTorch tensors, supports batch dimensions
                                  [batch_size, signal_length]
            sample_rate: Sample rate of the displacement waveform (Hz)

        Returns:
            Velocity waveform (microns/s) in the same format as input
        """
        # Check if input is a PyTorch tensor
        is_torch = 'torch' in str(type(displacement_waveform).__module__)

        # Time step
        dt = 1.0 / sample_rate

        if is_torch:
            # Handle batch dimensions if present
            if len(displacement_waveform.shape) > 1:
                _batch_size, _signal_length = displacement_waveform.shape
                velocity = torch.zeros_like(displacement_waveform)

                # First point (forward difference)
                velocity[:, 0] = (
                    displacement_waveform[:, 1] - displacement_waveform[:, 0]
                ) / dt

                # Middle points (central difference)
                velocity[:, 1:-1] = (
                    displacement_waveform[:, 2:] - displacement_waveform[:, :-2]
                ) / (2 * dt)

                # Last point (backward difference)
                velocity[:, -1] = (
                    displacement_waveform[:, -1] - displacement_waveform[:, -2]
                ) / dt
            else:
                velocity = torch.zeros_like(displacement_waveform)

                # First point (forward difference)
                velocity[0] = (displacement_waveform[1] - displacement_waveform[0]) / dt

                # Middle points (central difference)
                velocity[1:-1] = (
                    displacement_waveform[2:] - displacement_waveform[:-2]
                ) / (2 * dt)

                # Last point (backward difference)
                velocity[-1] = (
                    displacement_waveform[-1] - displacement_waveform[-2]
                ) / dt
        else:
            # NumPy implementation
            velocity = np.zeros_like(displacement_waveform)

            # First point (forward difference)
            velocity[0] = (displacement_waveform[1] - displacement_waveform[0]) / dt

            # Middle points (central difference)
            velocity[1:-1] = (
                displacement_waveform[2:] - displacement_waveform[:-2]
            ) / (2 * dt)

            # Last point (backward difference)
            velocity[-1] = (displacement_waveform[-1] - displacement_waveform[-2]) / dt

        return velocity
