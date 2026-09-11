#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch

from .coil_driver import CoilDriver
from .waveform import Waveform


class MichelsonInterferometer:
    """A class that simulates a Michelson interferometer for displacement measurements.

    This class models the optical interference pattern produced by a Michelson
    interferometer when one of its mirrors is displaced. It calculates the
    resulting interference signal based on the wavelength of light and the
    displacement of the mirror.

    Unit Conventions:
        - Wavelength: microns (umm)
        - Displacement: microns (umm)
        - Time: seconds (s)
        - Velocity: microns per second (umm/s)
        - Phase: radians

    Note:
        The interferometer model assumes that the displacement is measured in the
        same units as the wavelength (microns). The actual optical path difference
        is twice the mirror displacement due to the round trip of light.
    """

    def __init__(self, wavelength: float, phase: float):
        """Initialize a Michelson interferometer with specified parameters.

        Args:
            wavelength: The wavelength of light used in the interferometer in
                microns (umm).
            phase: Initial phase offset in radians, representing the random
                position offset of the interferometer mirrors.
        """
        self.wavelength = wavelength  # in microns
        self.phase = phase  # in radians, stands for random position offset

    def get_interferometer_output(
        self, displacement: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Calculate the interference signal for a given displacement array.

        This method implements the core interferometer physics, calculating the
        intensity of the interference pattern based on the mirror displacement.
        We only return the interference term, as if we had applied LayerNorm.

        Args:
            displacement: Array of mirror displacements in microns (umm).
                The displacement represents the physical movement of the mirror,
                not the optical path difference (which is twice the displacement).
                Can be either a NumPy array or a PyTorch tensor.

        Returns:
            Array of interference signal intensities corresponding to each
            displacement value, in the same format as the input (NumPy or PyTorch).
            The signal represents the detected light intensity at the
            interferometer output.
        """
        # Determine whether we're using NumPy or PyTorch based on input type
        if isinstance(displacement, torch.Tensor):
            # Use PyTorch operations for tensor inputs
            cos = torch.cos
            pi = torch.pi
        else:
            # Use NumPy operations for array inputs
            cos = np.cos
            pi = np.pi

        return cos(2 * pi / self.wavelength * 2 * displacement + self.phase)

    def get_simulated_buffer(
        self, displacement: np.ndarray, time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate simulated interferometer data including signal and velocity.

        This method calculates the interferometer output signal for the provided
        displacement data, calculates the corresponding velocity by differentiating
        the displacement, and removes the DC offset from the signal.

        Args:
            displacement: Array of mirror displacements in microns (umm).
            time: Array of time points in seconds (s) corresponding to the
            displacement data.

        Returns:
            Tuple containing:
            - time: The input time array (s).
            - signal: The interferometer output signal with DC offset removed (V).
            - displacement: The input displacement array (umm).
            - velocity: Calculated velocity array derived from displacement (umm/s).

        Note:
            Velocity is calculated by taking the first-order difference of the
            displacement and dividing by the time step. The first velocity value is
            duplicated to maintain the same array length as the displacement.
        """
        signal = self.get_interferometer_output(displacement)

        velocity = np.diff(displacement)
        velocity = np.insert(velocity, 0, velocity[0])
        velocity /= time[1] - time[0]

        # Remove DC offset
        signal = signal - np.mean(signal)

        return time, signal, displacement, velocity

    def plot_buffer(self, displacement: np.ndarray, time: np.ndarray):
        """Plot the interferometer signal, displacement, and velocity.

        This method generates a plot with two y-axes: one for the interferometer
        signal and one for both displacement and velocity. The plot shows the
        relationship between the interferometer output and the mirror motion.

        Args:
            displacement: Array of mirror displacements in microns (umm).
            time: Array of time points in seconds (s) corresponding to the
            displacement data.

        Note:
            - The left y-axis (blue) shows the interferometer signal in volts (V).
            - The right y-axis shows both displacement (red) in microns (umm) and
              velocity (green) in microns per second (umm/s), though only the
              displacement units are labeled on the axis.
            - The plot is displayed using matplotlib's plt.show() and will block
              execution until the plot window is closed.
        """
        time, signal, displacement, velocity = self.get_simulated_buffer(
            displacement, time
        )

        _fig, ax1 = plt.subplots(figsize=(18, 6))
        ax1.plot(time, signal, color='b')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal (V)', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(time, displacement, color='r')
        ax2.plot(time, velocity, color='g')
        ax2.set_ylabel('Displacement (umm)', color='r')
        ax2.tick_params('y', colors='r')

        plt.tight_layout()
        plt.show()


class InterferometerArray:
    """A class to manage an array of Michelson Interferometers.

    This class allows simulating multiple interferometers with the same
    displacement and time inputs, useful for multi-channel simulations.

    Args:
        interferometers: List of MichelsonInterferometer instances
    """

    def __init__(self, interferometers: list[MichelsonInterferometer]):
        self.interferometers = interferometers

    def get_simulated_buffer(
        self,
        displacement: np.ndarray | torch.Tensor,
        time: np.ndarray | torch.Tensor | None = None,
    ) -> (
        list[np.ndarray | torch.Tensor]
        | tuple[
            np.ndarray | torch.Tensor,
            list[np.ndarray | torch.Tensor],
            np.ndarray | torch.Tensor,
            np.ndarray | torch.Tensor | None,
        ]
    ):
        """Get simulated buffer data from all interferometers.

        This method feeds the same displacement and time data to all interferometers
        and collects their output signals. It works with both NumPy arrays and
        PyTorch tensors.

        Args:
            displacement: Displacement data array or tensor
            time: Time data array or tensor. If None, only interferometer outputs are
                calculated without velocity computation (PyTorch mode for loss
                functions)

        Returns:
            If time is provided (NumPy mode):
                Tuple containing:
                - time: Time data array
                - signals: List of signal arrays from each interferometer
                - displacement: Displacement data array
                - velocity: Velocity data calculated from displacement
            If time is None (PyTorch mode):
                List of signal tensors from each interferometer
        """
        # Check if we're in PyTorch mode (time=None)
        pytorch_mode = time is None

        # Get signals from each interferometer
        signals = []
        velocity = None

        if pytorch_mode:
            # PyTorch mode: just get interferometer outputs directly
            for interferometer in self.interferometers:
                signal = interferometer.get_interferometer_output(displacement)
                signals.append(signal)
            return signals
        else:
            # NumPy mode: use the full get_simulated_buffer method
            for interferometer in self.interferometers:
                # Get processed signal from each interferometer
                _, signal, _, interferometer_velocity = (
                    interferometer.get_simulated_buffer(displacement, time)
                )
                signals.append(signal)

                # Store velocity from the first interferometer (they should all be
                # the same)
                if velocity is None:
                    velocity = interferometer_velocity

            return time, signals, displacement, velocity

    def sample_simulated(
        self,
        start_freq: float = 1,
        end_freq: float = 1000,
        coil_driver: CoilDriver | None = None,
        randomize_phase_only: bool = False,
        random_single_tone: bool = False,
        normalize_gain: bool = True,
    ) -> tuple[
        np.ndarray, list[np.ndarray | torch.Tensor], np.ndarray, np.ndarray, np.ndarray
    ]:
        """Generate a simulated sample using CoilDriver to create displacement data.

        This method uses CoilDriver.sample() to generate a random voltage waveform and
        CoilDriver.get_displacement() to convert it to displacement, which is then fed
        to the interferometer array.

        Args:
            start_freq: The lower bound of the valid frequency range (Hz)
            end_freq: The upper bound of the valid frequency range (Hz)
            coil_driver: CoilDriver instance to use (creates a new one if None)
            randomize_phase_only: If True, only randomize phases while keeping spectrum
                amplitudes constant
            random_single_tone: If True, generate a single tone at a randomly selected
                frequency
            normalize_gain: If True, pre-compensate the spectrum to normalize gain
                across frequencies

        Returns:
            Tuple containing:
            - acq_time: Acquisition time data array
            - signals: List of signal arrays from each interferometer
            - acq_voltage: Acquisition voltage waveform
            - displacement: Displacement data array
            - velocity: Velocity data calculated from displacement
        """
        # Create a CoilDriver if not provided
        if coil_driver is None:
            coil_driver = CoilDriver()

        # Create a Waveform generator
        waveform = Waveform(start_freq=start_freq, end_freq=end_freq)

        # Generate a random voltage waveform using CoilDriver.sample()
        time, voltage, _voltage_spectrum = coil_driver.sample(
            waveform=waveform,
            randomize_phase_only=randomize_phase_only,
            random_single_tone=random_single_tone,
            normalize_gain=normalize_gain,
        )

        # Calculate generation sample rate from time array
        gen_sample_rate = 1 / (time[1] - time[0])

        # Get acquisition sample rate from the waveform
        acq_sample_rate = waveform.acq_sample_rate

        # Calculate the ratio between acquisition and generation sample rates
        # This determines how many samples we'll see in the acquisition window
        sample_rate_ratio = gen_sample_rate / acq_sample_rate

        # Calculate how many samples we would see at the acquisition rate
        # The acquisition buffer size is the same as the generation buffer size
        acq_buffer_size = waveform.BUFFER_SIZE

        # Calculate how many samples from the original waveform we'll actually see
        # due to the slower acquisition rate (aliasing effect)
        visible_samples = int(acq_buffer_size * sample_rate_ratio)

        # If visible_samples is greater than the length of the voltage array,
        # we'll see the whole waveform, otherwise we'll only see a portion
        visible_samples = min(visible_samples, len(voltage))

        # Create the "acquired" voltage waveform based on the sample rate ratio
        if sample_rate_ratio > 1:
            # If acquisition is slower than generation (typical case)
            # Take every nth sample where n is the integer ratio of the sample rates
            step = int(sample_rate_ratio)
            acq_voltage = voltage[:visible_samples:step]
            acq_time = time[:visible_samples:step]

            # Ensure we don't exceed the buffer size
            if len(acq_voltage) > acq_buffer_size:
                acq_voltage = acq_voltage[:acq_buffer_size]
                acq_time = acq_time[:acq_buffer_size]
        else:
            # If acquisition is faster than generation (sample_rate_ratio < 1)
            # We need to oversample - each generated voltage value will be measured
            # multiple times
            # Calculate how many times each sample should be repeated
            repeat_factor = int(1 / sample_rate_ratio)

            # Calculate how many original samples we can use while staying within
            # BUFFER_SIZE
            max_original_samples = acq_buffer_size // repeat_factor

            # Truncate the original arrays if needed
            if max_original_samples < len(voltage):
                voltage = voltage[:max_original_samples]
                time = time[:max_original_samples]

            # Create new arrays with the repeated values
            acq_voltage = np.repeat(voltage, repeat_factor)

            # Ensure we don't exceed the buffer size
            if len(acq_voltage) > acq_buffer_size:
                acq_voltage = acq_voltage[:acq_buffer_size]

            # Create a new time array with evenly spaced points at the acquisition rate
            total_time = time[-1] - time[0] if len(time) > 1 else 0
            num_acq_samples = len(acq_voltage)
            acq_time = np.linspace(
                time[0], time[0] + total_time, num_acq_samples, endpoint=False
            )

        # Convert voltage to displacement using CoilDriver.get_displacement()
        # but now using the acquisition sample rate
        displacement, _displacement_spectrum, _ = coil_driver.get_displacement(
            voltage_waveform=acq_voltage, sample_rate=acq_sample_rate
        )

        # Calculate velocity from displacement using the acquisition sample rate
        velocity, _, _ = coil_driver.get_velocity(acq_voltage, acq_sample_rate)

        # Feed the displacement and acquisition time to the interferometer array
        buffer_result = self.get_simulated_buffer(displacement, acq_time)
        assert isinstance(buffer_result, tuple)
        _, signals, _, _ = buffer_result

        return acq_time, signals, acq_voltage, displacement, velocity

    def plot_buffer(self, displacement: np.ndarray, time: np.ndarray):
        """Plot the signals from all interferometers along with displacement and
        velocity.

        Args:
            displacement: Displacement data array
            time: Time data array
        """
        result = self.get_simulated_buffer(displacement, time)
        assert isinstance(result, tuple)
        buffer_time, signals, buffer_displacement, velocity = result

        # Create a figure with subplots for each interferometer
        n_interferometers = len(self.interferometers)
        _fig, axes = plt.subplots(
            n_interferometers + 1,
            1,
            figsize=(18, 4 * (n_interferometers + 1)),
            sharex=True,
        )

        # Plot displacement and velocity in the top subplot
        axes[0].plot(buffer_time, buffer_displacement, color='r', label='Displacement')
        axes[0].set_ylabel('Displacement (umm)', color='r')
        axes[0].tick_params('y', colors='r')

        ax_twin = axes[0].twinx()
        ax_twin.plot(buffer_time, velocity, color='g', label='Velocity')
        ax_twin.set_ylabel('Velocity (umm/s)', color='g')
        ax_twin.tick_params('y', colors='g')

        # Add a legend to the top subplot
        lines1, labels1 = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Define a colormap for the interferometer signals
        colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))

        # Plot each interferometer signal in its own subplot
        for i, signal in enumerate(signals):
            # Get the wavelength in nanometers (convert from microns)
            wavelength_nm = self.interferometers[i].wavelength * 1000

            # Use a unique color from the colormap
            signal_color = colors[i]

            axes[i + 1].plot(buffer_time, signal, color=signal_color)
            axes[i + 1].set_ylabel(
                f'Signal {i + 1} ({wavelength_nm:.1f} nm) (V)', color=signal_color
            )
            axes[i + 1].tick_params('y', colors=signal_color)
            axes[i + 1].grid(True)

        # Set the x-label on the bottom subplot
        axes[-1].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()

    def plot_sample_simulated(self, **kwargs):
        """Generate a simulated sample and plot it.

        This is a convenience method that calls sample_simulated() and then plots the
        results.

        Args:
            **kwargs: Arguments to pass to sample_simulated()
        """
        time, signals, _acq_voltage, displacement, velocity = self.sample_simulated(
            **kwargs
        )

        # Create a figure with subplots for each interferometer
        n_interferometers = len(self.interferometers)
        _fig, axes = plt.subplots(
            n_interferometers + 1,
            1,
            figsize=(10, 3 * (n_interferometers + 1)),
            sharex=True,
        )

        # Plot displacement and velocity in the top subplot
        axes[0].plot(
            time, displacement, '.', color='r', label='Displacement'
        )  # Already in microns
        axes[0].set_ylabel('Displacement (umm)', color='r')
        axes[0].tick_params('y', colors='r')

        ax_twin = axes[0].twinx()
        ax_twin.plot(
            time, velocity, '.', color='g', label='Velocity'
        )  # Already in microns/s
        ax_twin.set_ylabel('Velocity (umm/s)', color='g')
        ax_twin.tick_params('y', colors='g')

        # Add a legend to the top subplot
        lines1, labels1 = axes[0].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Define a colormap for the interferometer signals
        colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))

        # Plot each interferometer signal in its own subplot
        for i, signal in enumerate(signals):
            # Get the wavelength in nanometers (convert from microns)
            wavelength_nm = self.interferometers[i].wavelength * 1000

            # Use a unique color from the colormap
            signal_color = colors[i]

            axes[i + 1].plot(time, signal, '.', color=signal_color)
            axes[i + 1].set_ylabel(
                f'Signal {i + 1} ({wavelength_nm:.1f} nm) (V)', color=signal_color
            )
            axes[i + 1].tick_params('y', colors=signal_color)
            axes[i + 1].grid(True)

        # Set the x-label on the bottom subplot
        axes[-1].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Create multiple interferometers with different parameters
    # Wavelengths in microns
    interferometer1 = MichelsonInterferometer(wavelength=0.633, phase=0)
    interferometer2 = MichelsonInterferometer(wavelength=0.515, phase=np.pi / 2)

    # Create an array of interferometers
    interferometer_array = InterferometerArray([interferometer1, interferometer2])

    # Method 1: Generate a simulated sample and get the data
    time, signals, acq_voltage, displacement, velocity = (
        interferometer_array.sample_simulated(start_freq=1, end_freq=1000)
    )

    # Method 2: Generate and plot in one step
    interferometer_array.plot_sample_simulated(
        start_freq=1, end_freq=1000, normalize_gain=True
    )
