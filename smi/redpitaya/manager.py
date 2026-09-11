#!/usr/bin/env python3
"""Red Pitaya Manager - A comprehensive class for managing Red Pitaya devices
This module provides a class that encapsulates functionality for data acquisition,
waveform generation, and device management for Red Pitaya devices.
"""

import contextlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

# Import our custom classes
try:
    # Try relative imports first (when used as a package)
    from ..synthetic.coil_driver import CoilDriver
    from ..synthetic.interferometers import MichelsonInterferometer
    from ..synthetic.waveform import Waveform
    from .redpitaya_config import RedPitayaConfig
    from .scpi import SCPI
except ImportError:
    # Fall back to absolute imports (when run as a script)
    from smi.redpitaya.redpitaya_config import RedPitayaConfig
    from smi.redpitaya.scpi import SCPI
    from smi.synthetic.coil_driver import CoilDriver
    from smi.synthetic.interferometers import MichelsonInterferometer
    from smi.synthetic.waveform import Waveform

logger = logging.getLogger(__name__)


class RedPitayaManager:
    """A comprehensive manager for Red Pitaya devices that encapsulates functionality
    for data acquisition, waveform generation, and device management.

    This class provides methods for:
    - Connecting to one or more Red Pitaya devices
    - Configuring and triggering scope acquisitions
    - Generating arbitrary waveforms
    - Saving acquired data
    - Plotting acquired data
    - Running multiple acquisitions with configurable delays
    - Daisy-chaining multiple devices
    """

    # Constants
    BUFFER_SIZE = RedPitayaConfig.BUFFER_SIZE
    SAMPLE_RATE_DEC1 = RedPitayaConfig.SAMPLE_RATE_DEC1

    def __init__(
        self,
        ip_addresses: str | list[str],
        data_save_path: str | None = None,
        blink_on_connect: bool = False,
    ):
        """Initialize the Red Pitaya Manager.

        Args:
            ip_addresses: IP address(es) of the Red Pitaya device(s)
            data_save_path: Path to save acquired data (default: None)
            blink_on_connect: Whether to blink LED 0 on successful connection
        """
        self.devices = []
        self.device_names = []

        # Connect to devices
        if isinstance(ip_addresses, str):
            ip_addresses = [ip_addresses]

        for i, ip in enumerate(ip_addresses):
            try:
                # Create SCPI instance
                device = SCPI(ip)
                self.devices.append(device)
                self.device_names.append(f'RP{i + 1}')
                logger.info(f'Connected to {ip} as {self.device_names[-1]}')

                # Blink LED 0 if requested for connectivity confirmation
                if blink_on_connect:
                    self.blink_led(device_idx=i, led_num=0, num_blinks=3, period=0.5)

            except Exception:
                logger.exception(f'Failed to connect to {ip}')

        # Set data save path
        if data_save_path:
            self.data_save_path = Path(data_save_path)
        else:
            # Default to a data directory in the project
            self.data_save_path = Path(__file__).parent.parent / 'analysis' / 'data'

        # Ensure the data directory exists
        self.data_save_path.mkdir(parents=True, exist_ok=True)

        # Initialize plotting
        self.fig = None
        self.axes = None
        self.plot_enabled = True

        # Initialize histogram data and figure
        self.hist_fig = None
        self.hist_axes = None
        self.histogram_data: dict[str, list | dict] = {
            'drive_voltage': [],
            'drive_voltage_spectrum': [],
            'photodiodes': {},
            'photodiode_spectra': {},
            'velocity': [],
            'velocity_spectrum': [],
            'displacement': [],
            'displacement_spectrum': [],
        }

        # Initialize device settings with defaults
        self.settings = {
            'gen_dec': 8192,
            'acq_dec': 256,
            'wave_form': 'ARBITRARY',
            'start_freq': 1,
            'end_freq': 1000,
            'trigger_source': 'NOW',
            'trigger_delay': 15 * 16384,
            # Empirical testing shows that waiting
            # at least 10 full buffers for triggering is sufficient to get rid of the
            # transient signal. No further delay than 20 full buffers is necessary.
            'channels_to_acquire': [
                1,
                2,
            ],  # Each device only has channels 1 and 2 by default
            'burst_mode': False,
            'burst_count': 1,
            'burst_period': None,
            'input4': False,  # Set to True for 4-input devices
        }

        # Initialize the coil driver for velocity and displacement calculations
        self.coil_driver = CoilDriver()

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self.close_all()

    def close_all(self):
        """Close all devices."""
        for device in self.devices:
            with contextlib.suppress(Exception):
                device.close()
        self.devices = []
        self.device_names = []

    def reset_all(self):
        """Reset generation and acquisition on all devices."""
        for device in self.devices:
            device.tx_txt('GEN:RST')
            device.tx_txt('ACQ:RST')

    def configure_daisy_chain(
        self, primary_idx: int = 0, secondary_indices: list[int] | None = None
    ):
        """Configure devices for daisy chain operation.

        Args:
            primary_idx: Index of the primary device
            secondary_indices: Indices of secondary devices
        """
        if not secondary_indices:
            secondary_indices = [
                i for i in range(len(self.devices)) if i != primary_idx
            ]

        if primary_idx >= len(self.devices) or any(
            idx >= len(self.devices) for idx in secondary_indices
        ):
            raise ValueError('Device index out of range')

        # Configure primary unit
        primary = self.devices[primary_idx]
        primary.tx_txt('DAISY:SYNC:TRig ON')
        primary.tx_txt('DAISY:SYNC:CLK ON')
        primary.tx_txt('DAISY:TRIG_O:SOUR ADC')
        primary.tx_txt('DIG:PIN LED5,1')  # LED Indicator

        logger.info(
            f'Primary device ({self.device_names[primary_idx]}) daisy chain configured:'
        )
        logger.debug(f'  Trig: {primary.txrx_txt("DAISY:SYNC:TRig?")}')
        logger.debug(f'  CLK: {primary.txrx_txt("DAISY:SYNC:CLK?")}')
        logger.debug(f'  Sour: {primary.txrx_txt("DAISY:TRIG_O:SOUR?")}')

        # Configure secondary units
        for idx in secondary_indices:
            secondary = self.devices[idx]
            secondary.tx_txt('DAISY:SYNC:TRig ON')
            secondary.tx_txt('DAISY:SYNC:CLK ON')
            secondary.tx_txt('DAISY:TRIG_O:SOUR ADC')
            secondary.tx_txt('DIG:PIN LED5,1')  # LED Indicator

            logger.info(
                f'Secondary device ({self.device_names[idx]}) daisy chain configured'
            )

    def configure_generation(
        self,
        device_idx: int = 0,
        channel: int = 1,
        wave_form: str | None = None,
        start_freq: int | None = None,
        end_freq: int | None = None,
        gen_dec: int | None = None,
        burst_mode: bool | None = None,
        burst_count: int | None = None,
        burst_period: int | None = None,
    ):
        """Configure signal generation on a device.

        Args:
            device_idx: Index of the device to configure
            channel: Output channel (1 or 2)
            wave_form: Waveform type (SINE, SQUARE, TRIANGLE, SAWU, SAWD, PWM,
                ARBITRARY, DC, DC_NEG)
            start_freq: Start frequency for frequency sweep
            end_freq: End frequency for frequency sweep
            gen_dec: Decimation for generation
            burst_mode: Enable/disable burst mode
            burst_count: Number of periods in one burst
            burst_period: Total time of one burst in uss
        """
        # Update settings with provided values
        if wave_form is not None:
            self.settings['wave_form'] = wave_form
        if start_freq is not None:
            self.settings['start_freq'] = start_freq
        if end_freq is not None:
            self.settings['end_freq'] = end_freq
        if gen_dec is not None:
            self.settings['gen_dec'] = gen_dec
        if burst_mode is not None:
            self.settings['burst_mode'] = burst_mode
        if burst_count is not None:
            self.settings['burst_count'] = burst_count
        if burst_period is not None:
            self.settings['burst_period'] = burst_period

        # Check if device index is valid
        if device_idx >= len(self.devices):
            raise ValueError(f'Device index {device_idx} out of range')

        device = self.devices[device_idx]

        # Reset generation
        device.tx_txt('GEN:RST')

        # Calculate parameters
        gen_smpl_rate = self.SAMPLE_RATE_DEC1 // self.settings['gen_dec']
        burst_time = self.BUFFER_SIZE / gen_smpl_rate
        freq = 1 / burst_time

        # If using arbitrary waveform, create it
        if self.settings['wave_form'] == 'ARBITRARY':
            # Use our Waveform class to generate a random waveform
            waveform_generator = Waveform(
                start_freq=self.settings['start_freq'],
                end_freq=self.settings['end_freq'],
                gen_dec=self.settings['gen_dec'],
                acq_dec=self.settings['acq_dec'],
                allowed_freqs=None,  # Use all valid frequencies in the range
            )

            # Generate waveform, passing through CoilDriver.sample() to
            # precompensate gain
            _, y, _ = self.coil_driver.sample(waveform_generator, normalize_gain=True)

            # Validate waveform amplitude
            if np.max(np.abs(y)) > 1.0:
                raise ValueError(
                    'Waveform amplitude exceeds 1.0. The waveform must '
                    'be normalized to [-1, 1] range.'
                )

            # Configure source with arbitrary waveform (amplitude fixed at 1.0)
            device.sour_set(
                channel,
                self.settings['wave_form'],
                1.0,  # Fixed amplitude of 1.0 - normalization handled by Waveform class
                freq,
                data=y,
                burst=self.settings['burst_mode'],
                ncyc=self.settings['burst_count'],
                period=self.settings['burst_period'],
            )
        else:
            # Configure source with standard waveform (amplitude fixed at 1.0)
            device.sour_set(
                channel,
                self.settings['wave_form'],
                1.0,  # Fixed amplitude of 1.0
                freq,
                burst=self.settings['burst_mode'],
                ncyc=self.settings['burst_count'],
                period=self.settings['burst_period'],
            )

        logger.info(
            f'Generation configured on {self.device_names[device_idx]}, ch {channel}'
        )

    def enable_output(self, device_idx: int = 0, channel: int = 1, enable: bool = True):
        """Enable or disable output on a device channel.

        Args:
            device_idx: Index of the device
            channel: Channel number (1 or 2)
            enable: True to enable, False to disable
        """
        if device_idx >= len(self.devices):
            raise ValueError(f'Device index {device_idx} out of range')

        device = self.devices[device_idx]

        if enable:
            device.tx_txt(f'OUTPUT{channel}:STATE ON')
            logger.info(
                f'Output enabled on {self.device_names[device_idx]}, channel {channel}'
            )
        else:
            device.tx_txt(f'OUTPUT{channel}:STATE OFF')
            logger.info(
                f'Output disabled on {self.device_names[device_idx]}, channel {channel}'
            )

    def trigger_generation(self, device_idx: int = 0, channel: int = 1):
        """Trigger generation on a device channel.

        Args:
            device_idx: Index of the device
            channel: Channel number (1 or 2)
        """
        if device_idx >= len(self.devices):
            raise ValueError(f'Device index {device_idx} out of range')

        device = self.devices[device_idx]
        device.tx_txt(f'SOUR{channel}:TRig:INT')
        logger.info(
            f'Generation triggered on {self.device_names[device_idx]}, ch {channel}'
        )

    def configure_acquisition(
        self, acq_dec: int | None = None, trigger_delay: int | None = None
    ):
        """Configure acquisition on all devices.

        Args:
            acq_dec: Decimation for acquisition
            trigger_delay: Trigger delay in samples
        """
        # Update settings with provided values
        if acq_dec is not None:
            self.settings['acq_dec'] = acq_dec
        if trigger_delay is not None:
            self.settings['trigger_delay'] = trigger_delay

        # Configure acquisition on all devices
        for i, device in enumerate(self.devices):
            device_name = self.device_names[i]
            device.tx_txt('ACQ:RST')
            device.acq_set(
                dec=self.settings['acq_dec'], trig_delay=self.settings['trigger_delay']
            )
            logger.info(f'Acquisition configured on {device_name}')

    def start_acquisition(self, device_idx: int = 0):
        """Start acquisition on all devices.

        For multiple devices, we start acquisition on secondary devices first,
        then on the primary device. This ensures proper triggering.

        Args:
            device_idx: Index of the device to use as primary
        """
        # Check if device index is valid
        if device_idx >= len(self.devices):
            raise ValueError(f'Device index {device_idx} out of range')

        # Get the primary device
        primary_device = self.devices[device_idx]
        primary_name = self.device_names[device_idx]

        # Identify secondary devices
        secondary_devices = []
        secondary_names = []
        for i, device in enumerate(self.devices):
            if i != device_idx:
                secondary_devices.append(device)
                secondary_names.append(self.device_names[i])

        # Start acquisition on secondary devices first
        for device, name in zip(secondary_devices, secondary_names, strict=False):
            device.tx_txt('ACQ:START')
            time.sleep(0.2)
            device.tx_txt('ACQ:TRig EXT_NE')
            logger.info(f'Acquisition started on {name} with trigger EXT_NE')

        # Start acquisition on primary device
        primary_device.tx_txt('ACQ:START')
        time.sleep(0.2)
        primary_device.tx_txt('ACQ:TRig CH1_PE')
        logger.info(f'Acquisition started on {primary_name} with trigger CH1_PE')

        # Store the primary device index for later use
        self._primary_device_idx = device_idx

    def get_acquisition_data(self, timeout: int = 5) -> dict[str, np.ndarray]:
        """Get acquisition data from all devices.

        This method waits for triggers and buffer fills on all devices,
        then retrieves the data.

        Args:
            timeout: Timeout in seconds for waiting for triggers

        Returns:
            Dictionary of acquired data
        """
        # Get the primary device (set in start_acquisition)
        device_idx = getattr(self, '_primary_device_idx', 0)
        primary_device = self.devices[device_idx]
        primary_name = self.device_names[device_idx]

        # Identify secondary devices
        secondary_devices = []
        secondary_names = []
        for i, device in enumerate(self.devices):
            if i != device_idx:
                secondary_devices.append(device)
                secondary_names.append(self.device_names[i])

        # Wait for primary device trigger with timeout
        logger.info(f'Waiting for {primary_name} trigger (timeout: {timeout}s)...')
        start_time = time.time()
        triggered = False

        while time.time() - start_time < timeout:
            primary_device.tx_txt('ACQ:TRig:STAT?')
            status = primary_device.rx_txt()
            logger.info(f'Trigger status: {status}')

            if status == 'TD':  # Triggered
                triggered = True
                break
            time.sleep(0.5)  # Check every half second

        if not triggered:
            logger.warning(f'Timeout waiting for {primary_name} trigger')
            # Try to recover by forcing a trigger
            primary_device.tx_txt('ACQ:TRig NOW')
            time.sleep(1)
        else:
            logger.info(f'Trigger detected on {primary_name}')

        # Wait for primary device buffer fill with timeout
        logger.info(
            f'Waiting for {primary_name} buffer to fill (timeout: {timeout}s)...'
        )
        start_time = time.time()
        filled = False

        while time.time() - start_time < timeout:
            primary_device.tx_txt('ACQ:TRig:FILL?')
            fill_status = primary_device.rx_txt()
            logger.debug(f'Fill status: {fill_status}')

            if fill_status == '1':  # Buffer filled
                filled = True
                break
            time.sleep(0.5)  # Check every half second

        if not filled:
            logger.warning(f'Timeout waiting for {primary_name} buffer to fill')
        else:
            logger.info(f'Buffer filled on {primary_name}')

        # Wait for secondary devices with timeout
        for device, name in zip(secondary_devices, secondary_names, strict=False):
            # Wait for trigger
            logger.info(f'Waiting for {name} trigger (timeout: {timeout}s)...')
            start_time = time.time()
            triggered = False

            while time.time() - start_time < timeout:
                device.tx_txt('ACQ:TRig:STAT?')
                status = device.rx_txt()
                logger.info(f'Trigger status: {status}')

                if status == 'TD':  # Triggered
                    triggered = True
                    break
                time.sleep(0.5)  # Check every half second

            if not triggered:
                logger.warning(f'Timeout waiting for {name} trigger')
                # Try to recover by forcing a trigger
                device.tx_txt('ACQ:TRig NOW')
                time.sleep(1)
            else:
                logger.info(f'Trigger detected on {name}')

            # Wait for buffer fill
            logger.info(f'Waiting for {name} buffer to fill (timeout: {timeout}s)...')
            start_time = time.time()
            filled = False

            while time.time() - start_time < timeout:
                device.tx_txt('ACQ:TRig:FILL?')
                fill_status = device.rx_txt()
                logger.debug(f'Fill status: {fill_status}')

                if fill_status == '1':  # Buffer filled
                    filled = True
                    break
                time.sleep(0.5)  # Check every half second

            if not filled:
                logger.warning(f'Timeout waiting for {name} buffer to fill')
            else:
                logger.info(f'Buffer filled on {name}')

        # Get data from all devices
        data = {}

        # Get data from primary device
        try:
            for chan in [1, 2]:
                if chan in self.settings['channels_to_acquire']:
                    try:
                        channel_data = np.array(
                            primary_device.acq_data(chan=chan, convert=True)
                        )
                        data[f'{primary_name}_CH{chan}'] = channel_data
                        logger.info(f'Acquired data from {primary_name} Channel {chan}')
                    except Exception:
                        logger.exception(
                            f'Error acquiring data from {primary_name}, channel {chan}'
                        )
        except Exception:
            logger.exception('Error during data acquisition from primary device')

        # Get data from secondary devices
        for device, name in zip(secondary_devices, secondary_names, strict=False):
            try:
                for chan in [1, 2]:
                    if chan in self.settings['channels_to_acquire']:
                        try:
                            channel_data = np.array(
                                device.acq_data(chan=chan, convert=True)
                            )
                            data[f'{name}_CH{chan}'] = channel_data
                            logger.info(
                                f'Successfully acquired data from {name} Channel {chan}'
                            )
                        except Exception:
                            logger.exception(
                                f'Error acquiring data from {name}, channel {chan}'
                            )
            except Exception:
                logger.exception(f'Error during data acquisition from {name}')

        return data

    def process_drive_voltage(
        self, speaker_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process speaker drive voltage data to calculate velocity and displacement.

        Args:
            speaker_data: Speaker voltage data

        Returns:
            Tuple containing:
            - velocity: Velocity data from transfer function
            - displacement: Displacement data from transfer function
            - velocity_spectrum: Velocity FFT
            - displacement_spectrum: Displacement FFT
            - frequencies: Frequency array
            - sample_rate: Calculated sample rate
        """
        # Calculate sample rate
        sample_rate = self.SAMPLE_RATE_DEC1 / self.settings['acq_dec']

        # Use the CoilDriver to calculate velocity and displacement from voltage
        velocity, velocity_spectrum, freq = self.coil_driver.get_velocity(
            speaker_data, sample_rate
        )
        displacement, displacement_spectrum, _ = self.coil_driver.get_displacement(
            speaker_data, sample_rate
        )

        return (
            velocity,
            displacement,
            velocity_spectrum,
            displacement_spectrum,
            freq,
            sample_rate,
        )

    def get_velocity_data(
        self, speaker_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get velocity data from speaker drive voltage.

        Args:
            speaker_data: Speaker voltage data

        Returns:
            Tuple containing:
            - velocity_tf: Velocity from transfer function
            - velocity_derivative: Velocity from derivative of displacement
            - velocity_spectrum: Velocity FFT
            - frequencies: Frequency array
        """
        # Get velocity and displacement data
        velocity_tf, displacement, velocity_spectrum, _, freq, sample_rate = (
            self.process_drive_voltage(speaker_data)
        )

        # Calculate velocity from derivative of displacement. derivative_displacement
        # accepts NumPy or Torch; this path passes a NumPy array, so the result is a
        # NumPy array (np.asarray is a no-op for an existing ndarray).
        velocity_derivative = np.asarray(
            self.coil_driver.derivative_displacement(displacement, sample_rate)
        )

        return velocity_tf, velocity_derivative, velocity_spectrum, freq

    def get_displacement_data(
        self, speaker_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get displacement data from speaker drive voltage.

        Args:
            speaker_data: Speaker voltage data

        Returns:
            Tuple containing:
            - displacement_tf: Displacement from transfer function
            - displacement_integrated: Displacement from integrated velocity
            - displacement_spectrum: Displacement FFT
            - frequencies: Frequency array
        """
        # Get velocity and displacement data
        velocity, displacement_tf, _, displacement_spectrum, freq, sample_rate = (
            self.process_drive_voltage(speaker_data)
        )

        # Integrate velocity to get displacement. integrate_velocity accepts NumPy or
        # Torch; this path passes a NumPy array, so the result is a NumPy array
        # (np.asarray is a no-op for an existing ndarray).
        displacement_integrated = np.asarray(
            self.coil_driver.integrate_velocity(velocity, sample_rate)
        )

        # Enforce that displacement starts at zero for each trace for easier comparison
        displacement_integrated -= displacement_integrated[0]
        displacement_tf -= displacement_tf[0]

        return displacement_tf, displacement_integrated, displacement_spectrum, freq

    def save_data(self, data: dict[str, np.ndarray], file_path: str | Path) -> str:
        """Save acquisition data to a file.

        Args:
            data: Dictionary of data to save
            file_path: Path to save the data to

        Returns:
            Path to the saved file
        """
        file_path = Path(file_path)

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we're creating or appending
        file_exists = file_path.exists()

        # Open file in appropriate mode
        mode = 'a' if file_exists else 'w'
        with h5py.File(file_path, mode) as f:
            # Store acquisition parameters as attributes (always update these)
            f.attrs['sample_rate'] = self.SAMPLE_RATE_DEC1 / self.settings['acq_dec']
            f.attrs['decimation'] = self.settings['acq_dec']
            f.attrs['buffer_size'] = self.BUFFER_SIZE

            # Only set creation_time when creating a new file
            if not file_exists:
                f.attrs['creation_time'] = datetime.now().isoformat()

            # Store any other relevant settings
            for key, value in self.settings.items():
                if isinstance(value, int | float | str | bool):
                    f.attrs[key] = value

            # Find Red Pitaya channel keys (the raw input signals)
            channel_keys = [
                key for key in data if key.startswith('RP') and '_CH' in key
            ]

            if not channel_keys:
                logger.warning('No Red Pitaya channel data found in acquisition')
                return str(file_path)

            # Get current dataset size if file exists
            if file_exists:
                # Get the first channel dataset to determine current size
                first_key = channel_keys[0]
                if first_key in f:
                    current_size = f[first_key].shape[0]
                else:
                    # Channel doesn't exist in file yet
                    current_size = 0
                    # Create datasets for each channel
                    for key in channel_keys:
                        channel_shape = data[key].shape
                        f.create_dataset(
                            key,
                            shape=(0, *channel_shape),
                            maxshape=(None, *channel_shape),
                            dtype='float32',
                            chunks=(1, *channel_shape),
                            compression='gzip',
                            compression_opts=4,
                        )
            else:
                # New file, start from 0
                current_size = 0
                # Create datasets for each channel
                for key in channel_keys:
                    channel_shape = data[key].shape
                    f.create_dataset(
                        key,
                        shape=(0, *channel_shape),
                        maxshape=(None, *channel_shape),
                        dtype='float32',
                        chunks=(1, *channel_shape),
                        compression='gzip',
                        compression_opts=4,
                    )

            # Resize datasets to accommodate new data
            for key in channel_keys:
                if key in data and key in f:
                    dataset = f[key]
                    new_size = current_size + 1  # Adding one sample
                    dataset.resize(new_size, axis=0)
                    # Add new data at the end
                    dataset[current_size] = data[key]

            # Update the number of samples attribute
            f.attrs['num_samples'] = current_size + 1

        logger.info(f'Data saved to {file_path}')
        return str(file_path)

    def setup_plot(self):
        """Set up the plot for visualization."""
        if not self.plot_enabled:
            return

        # Close any existing plot
        if self.fig is not None:
            plt.close(self.fig)

        # Determine number of rows based on number of channels
        num_rows = 2  # Default to 2 rows (primary device channels)

        # Add rows for secondary device if it exists
        if len(self.device_names) > 1:
            num_rows += 2  # Add 2 more rows for secondary device channels

        # Create a new figure with 3 columns: raw signals with velocity,
        # displacement comparison, and FFTs
        # Increase figure height for better spacing between subplots
        self.fig, self.axes = plt.subplots(num_rows, 3, figsize=(18, 12), sharex='col')

        # Ensure axes is always a 2D array even with a single row
        if num_rows == 1:
            self.axes = np.array([self.axes])

        # Set figure title
        self.fig.canvas.manager.set_window_title('Signal Analysis')

        # Add more space between subplots
        self.fig.subplots_adjust(hspace=0.4, wspace=0.5)

    def update_plot(
        self,
        data: dict[str, np.ndarray],
        vel_tf_data: np.ndarray | None = None,
        disp_derivative_data: np.ndarray | None = None,
        vel_fft: np.ndarray | None = None,  # noqa: ARG002
        disp_tf_data: np.ndarray | None = None,
        vel_integrated_data: np.ndarray | None = None,
        freqs: np.ndarray | None = None,
    ):
        """Update the plot with new data using a 3-column layout.

        Args:
            data: Dictionary of channel data
            vel_tf_data: Velocity data from transfer function
            disp_derivative_data: Velocity data from derivative of displacement
            vel_fft: Velocity FFT
            disp_tf_data: Displacement data from transfer function
            vel_integrated_data: Displacement data from integrated velocity
            freqs: Frequencies for FFT plot
        """
        if not self.plot_enabled or self.fig is None:
            return

        # First, remove any existing twin axes to start fresh
        for ax in self.fig.axes[:]:
            if ax not in self.axes.flatten():
                self.fig.delaxes(ax)

        # Clear all main axes
        for ax in self.axes.flatten():
            ax.clear()

        # Calculate time data
        acq_smpl_rate = self.SAMPLE_RATE_DEC1 // self.settings['acq_dec']
        time_data = np.linspace(
            0, (self.BUFFER_SIZE - 1) / acq_smpl_rate, self.BUFFER_SIZE
        )

        # Filter to only include raw channel data (not processed data)
        channel_data = {k: v for k, v in data.items() if '_CH' in k}

        # Define channel order with new naming convention
        channel_order = [
            f'{self.device_names[0]}_CH1',  # Speaker Drive Voltage (primary device CH1)
            f'{self.device_names[0]}_CH2',  # Primary device CH2
        ]

        # Add secondary device channels if they exist
        if len(self.device_names) > 1:
            channel_order.extend(
                [
                    f'{self.device_names[1]}_CH1',  # Secondary device CH1
                    f'{self.device_names[1]}_CH2',  # Secondary device CH2
                ]
            )

        # Colors for each channel
        colors = ['blue', 'red', 'green', 'purple']

        # Calculate FFT frequencies once
        if freqs is None and len(channel_data) > 0:
            # Get the length of any channel data
            any_data = next(iter(channel_data.values()))
            n = len(any_data)
            freqs = np.fft.fftfreq(n, 1 / acq_smpl_rate)

        # Only plot positive frequencies for FFT
        pos_idx = None
        pos_freqs = None
        if freqs is not None:
            pos_idx = freqs >= 0
            pos_freqs = freqs[pos_idx]

        # Create simulated Michelson signals if displacement data is available
        simulated_signals = {}
        if disp_tf_data is not None:
            # Displacement is already in microns, no conversion needed
            displacement_microns = disp_tf_data

            # Create interferometers for each photodiode wavelength
            # (wavelengths in microns)
            interferometer_635 = MichelsonInterferometer(wavelength=0.635, phase=0)
            interferometer_674 = MichelsonInterferometer(wavelength=0.6748, phase=0)
            interferometer_515 = MichelsonInterferometer(wavelength=0.515, phase=0)

            # Generate simulated signals
            _, simulated_signals['L635P5'], _, _ = (
                interferometer_635.get_simulated_buffer(
                    displacement=displacement_microns, time=time_data
                )
            )
            _, simulated_signals['HL6748MG'], _, _ = (
                interferometer_674.get_simulated_buffer(
                    displacement=displacement_microns, time=time_data
                )
            )
            _, simulated_signals['L515A1'], _, _ = (
                interferometer_515.get_simulated_buffer(
                    displacement=displacement_microns, time=time_data
                )
            )

            # Scale the simulated signals to match the amplitude of the real signals
            for key, signal in simulated_signals.items():
                # First normalize the simulated signal
                simulated_signals[key] = signal / np.max(np.abs(signal))

        # Plot each channel and its FFT
        for i, channel_name in enumerate(channel_order):
            if channel_name in channel_data and i < len(self.axes):
                # Get the channel data
                channel_values = channel_data[channel_name]

                # Create appropriate label
                if channel_name == f'{self.device_names[0]}_CH1':
                    label = 'Speaker Drive Voltage'
                    pd_type = None
                else:
                    # Extract device name and channel from the new format
                    device_name, channel = channel_name.split('_')
                    label = f'{device_name} {channel}'
                    if (device_name == 'RP1') and (channel == 'CH2'):
                        label = 'L635P5 PD'
                        pd_type = 'L635P5'
                    elif (device_name == 'RP2') and (channel == 'CH1'):
                        label = 'HL6748MG PD'
                        pd_type = 'HL6748MG'
                    elif (device_name == 'RP2') and (channel == 'CH2'):
                        label = 'L515A1 PD'
                        pd_type = 'L515A1'
                    else:
                        pd_type = None

                # Plot the raw data in the left column
                self.axes[i, 0].plot(
                    time_data,
                    channel_values,
                    color=colors[i % len(colors)],
                    label='Actual Signal',
                )
                self.axes[i, 0].set_ylabel(
                    'Amplitude (V)', color=colors[i % len(colors)]
                )
                self.axes[i, 0].tick_params(
                    axis='y', labelcolor=colors[i % len(colors)]
                )
                self.axes[i, 0].set_title(label)

                # Add simulated Michelson signal if this is a photodiode and we have
                # displacement data
                if pd_type in simulated_signals and disp_tf_data is not None:
                    # Scale the simulated signal to match the amplitude of the
                    # real signal
                    real_signal_amplitude = np.max(np.abs(channel_values))
                    scaled_simulated_signal = (
                        simulated_signals[pd_type] * real_signal_amplitude
                    )

                    # Use same color as real signal but with dashed line and alpha=0.5
                    self.axes[i, 0].plot(
                        time_data,
                        scaled_simulated_signal,
                        '--',
                        color=colors[i % len(colors)],
                        alpha=0.5,
                        label='Simulated Michelson Signal',
                    )
                    self.axes[i, 0].legend(loc='upper left', fontsize='small')

                # Add velocity twin axis for all plots in the first column
                if vel_tf_data is not None and disp_derivative_data is not None:
                    ax_vel = self.axes[i, 0].twinx()
                    # Plot velocity from transfer function
                    ax_vel.plot(
                        time_data, vel_tf_data, 'k-', alpha=0.7, label='Velocity'
                    )
                    # Plot velocity from derivative of displacement
                    ax_vel.plot(
                        time_data,
                        disp_derivative_data,
                        'k--',
                        alpha=0.7,
                        label='Derivative Displacement',
                    )
                    ax_vel.set_ylabel('Velocity (Microns/s)', color='black')
                    ax_vel.tick_params(axis='y', labelcolor='black')
                    ax_vel.legend(loc='upper right', fontsize='small')

                # Plot displacement data in the middle column
                if disp_tf_data is not None and vel_integrated_data is not None:
                    # Plot the raw voltage data on primary y-axis
                    self.axes[i, 1].plot(
                        time_data,
                        channel_values,
                        color=colors[i % len(colors)],
                        label='Actual Signal',
                    )
                    self.axes[i, 1].set_ylabel(
                        'Amplitude (V)', color=colors[i % len(colors)]
                    )
                    self.axes[i, 1].tick_params(
                        axis='y', labelcolor=colors[i % len(colors)]
                    )

                    # Add simulated Michelson signal if this is a photodiode
                    if pd_type in simulated_signals:
                        # Scale the simulated signal to match the amplitude of the
                        # real signal
                        real_signal_amplitude = np.max(np.abs(channel_values))
                        scaled_simulated_signal = (
                            simulated_signals[pd_type] * real_signal_amplitude
                        )

                        # Use same color as real signal, dashed and alpha=0.5
                        self.axes[i, 1].plot(
                            time_data,
                            scaled_simulated_signal,
                            '--',
                            color=colors[i % len(colors)],
                            alpha=0.5,
                            label='Simulated Michelson Signal',
                        )
                        self.axes[i, 1].legend(loc='upper left', fontsize='small')

                    # Create twin axis for displacement
                    ax_disp = self.axes[i, 1].twinx()

                    # Plot displacement from transfer function
                    ax_disp.plot(time_data, disp_tf_data, 'k-', label='Displacement')

                    # Plot integrated displacement on the same axis
                    ax_disp.plot(
                        time_data,
                        vel_integrated_data,
                        'k--',
                        label='Integrated Velocity',
                    )

                    # Add labels and legend
                    ax_disp.set_ylabel('Displacement (Microns)', color='black')
                    ax_disp.tick_params(axis='y', labelcolor='black')
                    ax_disp.legend(loc='upper right')

                    # Set title for displacement plot
                    self.axes[i, 1].set_title(f'{label} Displacement')
                    self.axes[i, 1].grid(True, alpha=0.3)

                # Plot the FFT in the right column
                if freqs is not None and pos_idx is not None:
                    # Calculate FFT
                    channel_fft_complex = fft(channel_values, norm='ortho')
                    channel_fft_mag = np.abs(channel_fft_complex)
                    channel_fft_phase = np.angle(channel_fft_complex, deg=True)

                    # Create twin axis for phase
                    ax_phase = self.axes[i, 2].twinx()

                    # Plot magnitude on primary y-axis (log scale)
                    self.axes[i, 2].semilogy(
                        pos_freqs,
                        channel_fft_mag[pos_idx],
                        color=colors[i % len(colors)],
                        label='Magnitude',
                    )
                    self.axes[i, 2].set_ylabel(
                        'Magnitude (log)', color=colors[i % len(colors)]
                    )
                    self.axes[i, 2].tick_params(
                        axis='y', labelcolor=colors[i % len(colors)]
                    )

                    # Plot phase on secondary y-axis with alpha=0.3
                    ax_phase.plot(
                        pos_freqs,
                        channel_fft_phase[pos_idx],
                        'r-',
                        label='Phase',
                        alpha=0.3,
                    )
                    ax_phase.set_ylabel('Phase (deg)', color='red')
                    ax_phase.tick_params(axis='y', labelcolor='red')
                    ax_phase.set_ylim(-180, 180)

                    # Set title for FFT plot
                    self.axes[i, 2].set_title(f'{label} FFT')

                    # Add grid to FFT plot
                    self.axes[i, 2].grid(True, which='both', ls='-', alpha=0.5)

        # Set common x-axis labels
        for i in range(len(self.axes)):
            # Only add x-axis label to bottom plots
            if i == len(self.axes) - 1:
                self.axes[i, 0].set_xlabel('Time (s)')
                self.axes[i, 1].set_xlabel('Time (s)')
                self.axes[i, 2].set_xlabel('Frequency (Hz)')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)  # Small pause to update the plot

    def show_plot(self, block: bool = False):
        """Show the plot.

        Args:
            block: Whether to block execution until the plot window is closed
        """
        if not self.plot_enabled or self.fig is None:
            return

        try:
            plt.figure(self.fig.number)
            plt.draw()
            if block:
                plt.show(block=True)
            else:
                plt.pause(0.01)  # Small pause to update plot
        except Exception:
            logger.exception('Error in show_plot')

    def setup_histograms(self):
        """Set up the histogram and spectrum plots for signal visualization."""
        if self.hist_fig is None:
            # Create figure with 4x3 subplots (histograms and spectra)
            self.hist_fig, self.hist_axes = plt.subplots(4, 3, figsize=(15, 12))
            self.hist_fig.canvas.manager.set_window_title(
                'Signal Histograms and Spectra'
            )
            # Add empty title to each subplot initially
            for ax in self.hist_axes.flatten():
                ax.set_title('No data yet')
                ax.grid(True, alpha=0.3)
            plt.tight_layout(pad=3.0)

    def update_histograms(
        self,
        data: dict[str, np.ndarray],
        vel_tf_data: np.ndarray | None = None,
        disp_tf_data: np.ndarray | None = None,
    ):
        """Update histograms and spectra with new data and display accumulated results.

        Args:
            data: Dictionary of channel data
            vel_tf_data: Velocity data from transfer function
            disp_tf_data: Displacement data from transfer function
        """
        if not self.plot_enabled or self.hist_fig is None:
            return

        # Make sure we're working with the histogram figure
        plt.figure(self.hist_fig.number)

        # Clear all axes
        for ax in self.hist_axes.flatten():
            ax.clear()

        # Calculate sample rate from the first channel data
        sample_rate = None
        if 'acq_dec' in self.settings:
            sample_rate = self.SAMPLE_RATE_DEC1 / self.settings['acq_dec']

        # Accumulate data for histograms and spectra
        # Drive voltage (Speaker)
        speaker_channel = f'{self.device_names[0]}_CH1'
        if speaker_channel in data:
            speaker_data = data[speaker_channel]
            self.histogram_data['drive_voltage'].append(speaker_data)

            # Calculate and accumulate spectrum
            if sample_rate is not None:
                # Calculate FFT
                speaker_fft_complex = fft(speaker_data, norm='ortho')
                speaker_fft_mag = np.abs(speaker_fft_complex)
                self.histogram_data['drive_voltage_spectrum'].append(speaker_fft_mag)

        # Photodiode signals
        for channel_name, channel_data in data.items():
            if '_CH' in channel_name and channel_name != speaker_channel:
                # Extract device name and channel
                device_name, channel = channel_name.split('_')

                # Determine photodiode type
                pd_type = None
                if (device_name == 'RP1') and (channel == 'CH2'):
                    pd_type = 'L635P5 PD'
                elif (device_name == 'RP2') and (channel == 'CH1'):
                    pd_type = 'HL6748MG PD'
                elif (device_name == 'RP2') and (channel == 'CH2'):
                    pd_type = 'L515A1 PD'

                if pd_type:
                    # These two entries are always dicts keyed by photodiode type;
                    # cast narrows the heterogeneous list|dict value type.
                    photodiodes = cast('dict', self.histogram_data['photodiodes'])
                    photodiode_spectra = cast(
                        'dict', self.histogram_data['photodiode_spectra']
                    )
                    # Accumulate time domain data
                    if pd_type not in photodiodes:
                        photodiodes[pd_type] = []
                    photodiodes[pd_type].append(channel_data)

                    # Calculate and accumulate spectrum
                    if sample_rate is not None:
                        # Calculate FFT
                        pd_fft_complex = fft(channel_data, norm='ortho')
                        pd_fft_mag = np.abs(pd_fft_complex)

                        if pd_type not in photodiode_spectra:
                            photodiode_spectra[pd_type] = []
                        photodiode_spectra[pd_type].append(pd_fft_mag)

        # Velocity and displacement data
        if vel_tf_data is not None:
            self.histogram_data['velocity'].append(vel_tf_data)

            # Calculate and accumulate spectrum
            if sample_rate is not None:
                # Calculate FFT
                vel_fft_complex = fft(vel_tf_data, norm='ortho')
                vel_fft_mag = np.abs(vel_fft_complex)
                self.histogram_data['velocity_spectrum'].append(vel_fft_mag)

        if disp_tf_data is not None:
            self.histogram_data['displacement'].append(disp_tf_data)

            # Calculate and accumulate spectrum
            if sample_rate is not None:
                # Calculate FFT
                disp_fft_complex = fft(disp_tf_data, norm='ortho')
                disp_fft_mag = np.abs(disp_fft_complex)
                self.histogram_data['displacement_spectrum'].append(disp_fft_mag)

        # Calculate FFT frequencies
        freqs = None
        pos_idx = None
        pos_freqs = None
        if sample_rate is not None and speaker_channel in data:
            n = len(data[speaker_channel])
            freqs = np.fft.fftfreq(n, 1 / sample_rate)
            # Only plot positive frequencies
            pos_idx = np.where(freqs > 0)
            pos_freqs = freqs[pos_idx]

        # Plot histograms and spectra
        # First row: Drive voltage, velocity, displacement histograms
        # Second row: Drive voltage, velocity, displacement spectra
        # Third row: PD signal histograms
        # Fourth row: PD signal spectra

        # Row 1: Histograms for drive voltage, velocity, displacement
        # Drive voltage histogram
        if self.histogram_data['drive_voltage']:
            drive_data = np.concatenate(self.histogram_data['drive_voltage'])

            # Calculate histogram
            hist, bin_edges = np.histogram(drive_data, bins=100)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            self.hist_axes[0, 0].hist(drive_data, bins=100, color='blue', alpha=0.7)

            # Calculate Gaussian overlay if spectrum data is available
            if (
                self.histogram_data['drive_voltage_spectrum']
                and sample_rate is not None
            ):
                drive_spectra = np.array(self.histogram_data['drive_voltage_spectrum'])
                avg_drive_spectrum = np.mean(drive_spectra, axis=0)

                # Calculate time step and total time
                if speaker_channel in data:
                    n = len(data[speaker_channel])
                    dt = 1 / sample_rate
                    total_time = n * dt
                    df = 1 / total_time

                    # Compute variance using the spectrum
                    drive_variance = (
                        np.sum(np.abs(avg_drive_spectrum) ** 2)
                        * n
                        * dt**2
                        / total_time
                        * df
                    )

                    # Create Gaussian curve
                    mean = np.mean(drive_data)
                    mean = 0
                    std_dev = np.sqrt(drive_variance)
                    gaussian = np.max(hist) * np.exp(
                        -0.5 * ((bin_centers - mean) / std_dev) ** 2
                    )

                    # Plot Gaussian overlay
                    self.hist_axes[0, 0].plot(
                        bin_centers,
                        gaussian,
                        'r-',
                        linewidth=2,
                        label=r'Gaussian ($\sigma$' + f'={std_dev:.4f})',
                    )
                    self.hist_axes[0, 0].legend(loc='upper right')

            self.hist_axes[0, 0].set_title('Speaker Drive Voltage')
            self.hist_axes[0, 0].set_xlabel('Amplitude (V)')
            self.hist_axes[0, 0].set_ylabel('Count')
            self.hist_axes[0, 0].grid(True, alpha=0.3)

        # Velocity histogram
        if self.histogram_data['velocity']:
            vel_data = np.concatenate(self.histogram_data['velocity'])

            # Calculate histogram
            hist, bin_edges = np.histogram(vel_data, bins=100)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            self.hist_axes[0, 1].hist(vel_data, bins=100, color='orange', alpha=0.7)

            # Calculate Gaussian overlay if spectrum data is available
            if self.histogram_data['velocity_spectrum'] and sample_rate is not None:
                vel_spectra = np.array(self.histogram_data['velocity_spectrum'])
                avg_vel_spectrum = np.mean(vel_spectra, axis=0)

                # Calculate time step and total time
                if speaker_channel in data:
                    n = len(data[speaker_channel])
                    dt = 1 / sample_rate
                    total_time = n * dt
                    df = 1 / total_time

                    # Compute variance using the spectrum
                    vel_variance = (
                        np.sum(np.abs(avg_vel_spectrum) ** 2)
                        * n
                        * dt**2
                        / total_time
                        * df
                    )

                    # Create Gaussian curve
                    mean = np.mean(vel_data)
                    mean = 0
                    std_dev = np.sqrt(vel_variance)
                    gaussian = np.max(hist) * np.exp(
                        -0.5 * ((bin_centers - mean) / std_dev) ** 2
                    )

                    # Plot Gaussian overlay
                    self.hist_axes[0, 1].plot(
                        bin_centers,
                        gaussian,
                        'r-',
                        linewidth=2,
                        label=r'Gaussian ($\sigma$=' + f'{std_dev:.4f})',
                    )
                    self.hist_axes[0, 1].legend(loc='upper right')

            self.hist_axes[0, 1].set_title('Velocity')
            self.hist_axes[0, 1].set_xlabel('Velocity (Microns/s)')
            self.hist_axes[0, 1].set_ylabel('Count')
            self.hist_axes[0, 1].grid(True, alpha=0.3)

        # Displacement histogram
        if self.histogram_data['displacement']:
            disp_data = np.concatenate(self.histogram_data['displacement'])

            # Calculate histogram
            hist, bin_edges = np.histogram(disp_data, bins=100)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            self.hist_axes[0, 2].hist(disp_data, bins=100, color='cyan', alpha=0.7)

            # Calculate Gaussian overlay if spectrum data is available
            if self.histogram_data['displacement_spectrum'] and sample_rate is not None:
                disp_spectra = np.array(self.histogram_data['displacement_spectrum'])
                avg_disp_spectrum = np.mean(disp_spectra, axis=0)

                # Calculate time step and total time
                if speaker_channel in data:
                    n = len(data[speaker_channel])
                    dt = 1 / sample_rate
                    total_time = n * dt
                    df = 1 / total_time

                    # Compute variance using the spectrum
                    disp_variance = (
                        np.sum(np.abs(avg_disp_spectrum) ** 2)
                        * n
                        * dt**2
                        / total_time
                        * df
                    )

                    # Create Gaussian curve
                    mean = np.mean(disp_data)
                    mean = 0
                    std_dev = np.sqrt(disp_variance)
                    gaussian = np.max(hist) * np.exp(
                        -0.5 * ((bin_centers - mean) / std_dev) ** 2
                    )

                    # Plot Gaussian overlay
                    self.hist_axes[0, 2].plot(
                        bin_centers,
                        gaussian,
                        'r-',
                        linewidth=2,
                        label=r'Gaussian ($\sigma$=' + f'{std_dev:.4f})',
                    )
                    self.hist_axes[0, 2].legend(loc='upper right')

            self.hist_axes[0, 2].set_title('Displacement')
            self.hist_axes[0, 2].set_xlabel('Displacement (Microns)')
            self.hist_axes[0, 2].set_ylabel('Count')
            self.hist_axes[0, 2].grid(True, alpha=0.3)

        # Row 2: Spectra for drive voltage, velocity, displacement
        if pos_freqs is not None:
            # Drive voltage spectrum
            if self.histogram_data['drive_voltage_spectrum']:
                drive_spectra = np.array(self.histogram_data['drive_voltage_spectrum'])
                avg_drive_spectrum = np.mean(drive_spectra, axis=0)
                self.hist_axes[1, 0].semilogy(
                    pos_freqs, avg_drive_spectrum[pos_idx], color='blue'
                )
                self.hist_axes[1, 0].set_title('Speaker Drive Voltage Spectrum')
                self.hist_axes[1, 0].set_xlabel('Frequency (Hz)')
                self.hist_axes[1, 0].set_ylabel('Magnitude')
                self.hist_axes[1, 0].grid(True, which='both', ls='-', alpha=0.5)
                self.hist_axes[1, 0].set_xlim(0, 2 * self.settings['end_freq'])

            # Velocity spectrum
            if self.histogram_data['velocity_spectrum']:
                vel_spectra = np.array(self.histogram_data['velocity_spectrum'])
                avg_vel_spectrum = np.mean(vel_spectra, axis=0)
                self.hist_axes[1, 1].semilogy(
                    pos_freqs, avg_vel_spectrum[pos_idx], color='orange'
                )
                self.hist_axes[1, 1].set_title('Velocity Spectrum')
                self.hist_axes[1, 1].set_xlabel('Frequency (Hz)')
                self.hist_axes[1, 1].set_ylabel('Magnitude')
                self.hist_axes[1, 1].grid(True, which='both', ls='-', alpha=0.5)
                self.hist_axes[1, 1].set_xlim(0, 2 * self.settings['end_freq'])

            # Displacement spectrum
            if self.histogram_data['displacement_spectrum']:
                disp_spectra = np.array(self.histogram_data['displacement_spectrum'])
                avg_disp_spectrum = np.mean(disp_spectra, axis=0)
                self.hist_axes[1, 2].semilogy(
                    pos_freqs, avg_disp_spectrum[pos_idx], color='cyan'
                )
                self.hist_axes[1, 2].set_title('Displacement Spectrum')
                self.hist_axes[1, 2].set_xlabel('Frequency (Hz)')
                self.hist_axes[1, 2].set_ylabel('Magnitude')
                self.hist_axes[1, 2].grid(True, which='both', ls='-', alpha=0.5)
                self.hist_axes[1, 2].set_xlim(0, 2 * self.settings['end_freq'])

        # Row 3: Photodiode histograms
        pd_positions = [(2, 0), (2, 1), (2, 2)]
        for i, (pd_type, pd_data_list) in enumerate(
            self.histogram_data['photodiodes'].items()
        ):
            if i < len(pd_positions) and pd_data_list:
                row, col = pd_positions[i]
                pd_data = np.concatenate(pd_data_list)
                self.hist_axes[row, col].hist(
                    pd_data,
                    bins=100,
                    color=['red', 'green', 'purple'][i % 3],
                    alpha=0.7,
                )
                self.hist_axes[row, col].set_title(f'{pd_type} Signal')
                self.hist_axes[row, col].set_xlabel('Amplitude (V)')
                self.hist_axes[row, col].set_ylabel('Count')
                self.hist_axes[row, col].grid(True, alpha=0.3)

        # Row 4: Photodiode spectra
        if pos_freqs is not None:
            pd_spectrum_positions = [(3, 0), (3, 1), (3, 2)]
            for i, (pd_type, pd_spectra_list) in enumerate(
                self.histogram_data['photodiode_spectra'].items()
            ):
                if i < len(pd_spectrum_positions) and pd_spectra_list:
                    row, col = pd_spectrum_positions[i]
                    pd_spectra = np.array(pd_spectra_list)
                    avg_pd_spectrum = np.mean(pd_spectra, axis=0)
                    self.hist_axes[row, col].semilogy(
                        pos_freqs,
                        avg_pd_spectrum[pos_idx],
                        color=['red', 'green', 'purple'][i % 3],
                    )
                    self.hist_axes[row, col].set_title(f'{pd_type} Spectrum')
                    self.hist_axes[row, col].set_xlabel('Frequency (Hz)')
                    self.hist_axes[row, col].set_ylabel('Magnitude')
                    self.hist_axes[row, col].grid(True, which='both', ls='-', alpha=0.5)

        # Update the figure
        plt.figure(self.hist_fig.number)
        plt.tight_layout()
        # Don't call plt.draw() or plt.pause() here - we'll do that in the
        # calling function

    def show_histograms(self, block: bool = False):
        """Show the histograms plot.

        Args:
            block: Whether to block execution until the plot window is closed
        """
        if self.hist_fig is None:
            return

        try:
            plt.figure(self.hist_fig.number)
            plt.draw()
            if block:
                plt.show(block=True)
            else:
                plt.pause(0.1)  # Increase pause time to ensure plot updates
        except Exception:
            logger.exception('Error in show_histograms')

    def run_one_shot(
        self,
        device_idx: int = 0,
        plot_data: bool = False,
        block_plot: bool = True,
        timeout: int = 5,
    ) -> dict[str, np.ndarray]:
        """Run a single acquisition cycle.

        Args:
            device_idx: Index of the device to use as primary
            plot_data: Whether to plot data
            block_plot: Whether to block on plot display
            timeout: Timeout in seconds for waiting for triggers

        Returns:
            Dictionary of acquired data
        """
        # Configure all devices
        logger.info(f'Configuring {len(self.devices)} device(s)...')

        # Configure generation on the primary device
        self.configure_generation(device_idx=device_idx)

        # Configure acquisition on all devices
        self.configure_acquisition()

        # Start acquisition on all devices
        self.start_acquisition(device_idx=device_idx)

        # Enable output on the primary device
        self.enable_output(device_idx=device_idx)

        # Trigger generation on the primary device
        self.trigger_generation(device_idx=device_idx)

        logger.info('ACQ start')

        # Get the data
        data = self.get_acquisition_data(timeout=timeout)

        # Process velocity and displacement data from Channel 1 of RP1 (speaker
        # drive voltage)
        speaker_data = data.get(f'{self.device_names[device_idx]}_CH1')
        vel_tf_data = None
        vel_derivative_data = None
        vel_fft = None
        disp_tf_data = None
        disp_integrated_data = None
        disp_fft = None
        freqs = None

        if speaker_data is not None:
            try:
                # Process velocity data (returns velocity from transfer function,
                # velocity from derivative, spectrum, freqs)
                vel_tf_data, vel_derivative_data, vel_fft, freqs = (
                    self.get_velocity_data(speaker_data)
                )

                # Process displacement data (returns displacement from transfer
                # function, integrated displacement, spectrum, freqs)
                disp_tf_data, disp_integrated_data, disp_fft, _ = (
                    self.get_displacement_data(speaker_data)
                )

                # Add data to the data dictionary
                data['Velocity_TF'] = vel_tf_data  # Velocity from transfer function
                data['Velocity_Derivative'] = (
                    vel_derivative_data  # Velocity from derivative of displacement
                )
                data['Velocity_FFT'] = vel_fft  # Velocity spectrum
                data['Displacement_TF'] = (
                    disp_tf_data  # Displacement from transfer function
                )
                data['Displacement_Integrated'] = (
                    disp_integrated_data  # Displacement from integrated velocity
                )
                data['Displacement_FFT'] = disp_fft  # Displacement spectrum
                data['Frequencies'] = freqs  # Frequency array
            except Exception:
                logger.exception('Error processing data')
        else:
            logger.warning(
                f'No speaker data found for {self.device_names[device_idx]}_CH1'
            )

        # Data is saved at the run_multiple_shots level if needed

        # Plot data if requested
        if plot_data:
            try:
                # Update the main plot first
                self.update_plot(
                    data,
                    vel_tf_data,
                    vel_derivative_data,
                    vel_fft,
                    disp_tf_data,
                    disp_integrated_data,
                    freqs,
                )

                # Update histograms
                self.update_histograms(data, vel_tf_data, disp_tf_data)

                # Handle the blocking behavior for the last plot only
                if block_plot:
                    plt.show(block=True)
                else:
                    # Just make sure both figures are displayed and updated
                    if self.fig is not None:
                        plt.figure(self.fig.number)
                        plt.draw()
                    if self.hist_fig is not None:
                        plt.figure(self.hist_fig.number)
                        plt.draw()
                    plt.pause(0.01)  # Small pause to update both plots
            except Exception:
                logger.exception('Error updating plots')

        return data

    def run_multiple_shots(
        self,
        num_shots: int,
        device_idx: int = 0,
        delay_between_shots: float = 0.75,
        plot_data: bool = False,
        keep_final_plot: bool = True,
        hdf5_file: str | None = None,
        timeout: int = 5,
    ) -> list[dict[str, np.ndarray]]:
        """Run multiple acquisition cycles.

        Args:
            num_shots: Number of shots to run
            device_idx: Index of the device to use as primary
            delay_between_shots: Delay between shots in seconds. We found a small
            decrease in training loss when increasing this from 0.5 to 1.0, so we
            choose 0.75 as a compromise.
            plot_data: Whether to plot data
            keep_final_plot: Whether to keep the final plot open for examination
            hdf5_file: Path to HDF5 file to save data incrementally
            timeout: Timeout in seconds for waiting for triggers

        Returns:
            List of dictionaries of acquired data
        """
        logger.info(f'Starting {num_shots} acquisition cycles')
        start_time = datetime.now()
        logger.info(f'Start time: {start_time.strftime("%H:%M:%S.%f")}')

        logger.info(f'delay_between_shots: {delay_between_shots}')

        all_data = []

        # Set up both plots once if plotting is enabled
        if plot_data:
            plt.ion()  # Turn on interactive mode first

            # Set up main plot
            self.setup_plot()
            if self.fig is not None:
                plt.figure(self.fig.number)
                plt.draw()
                plt.pause(0.01)

            # Set up histogram plot
            self.setup_histograms()
            if self.hist_fig is not None:
                plt.figure(self.hist_fig.number)
                plt.draw()
                plt.pause(0.01)

        try:
            for i in range(num_shots):
                logger.info(
                    f'Shot {i}/{num_shots} at {datetime.now().strftime("%H:%M:%S.%f")}'
                )

                try:
                    # For the last shot, set block_plot based on keep_final_plot
                    is_last_shot = i == num_shots - 1
                    should_block = is_last_shot and keep_final_plot

                    # Run one shot
                    shot_data = self.run_one_shot(
                        device_idx=device_idx,
                        plot_data=plot_data,
                        block_plot=should_block,  # Block on the last plot if requested
                        timeout=timeout,
                    )

                    # Save to HDF5 file if specified
                    if hdf5_file and shot_data:
                        try:
                            self.save_data(shot_data, hdf5_file)
                        except Exception:
                            logger.exception('Error saving data to HDF5 file')

                    all_data.append(shot_data)

                    # Wait between shots
                    if i < num_shots - 1:  # Don't wait after the last shot
                        time.sleep(delay_between_shots)

                except Exception:
                    logger.exception(f'Error in shot {i}')
                    # Continue with next shot instead of stopping
                    continue

        except KeyboardInterrupt:
            logger.info('Acquisition interrupted by user')

        finally:
            # Make sure to reset devices
            self.reset_all()

            # Turn off interactive mode if it was enabled
            if plot_data and not keep_final_plot:
                try:
                    plt.ioff()
                    if self.fig is not None:
                        plt.close(self.fig)  # Close the figure to clean up resources
                        self.fig = None
                        self.axes = None
                    if self.hist_fig is not None:
                        plt.close(self.hist_fig)  # Close the histogram figure
                        self.hist_fig = None
                        self.hist_axes = None
                        # Clear histogram data
                        self.histogram_data = {
                            'drive_voltage': [],
                            'photodiodes': {},
                            'displacement': [],
                            'velocity': [],
                        }
                except Exception:
                    logger.exception('Error cleaning up plots')

            end_time = datetime.now()
            logger.info(f'Completed {len(all_data)} acquisition cycles')
            logger.info(f'End time: {end_time.strftime("%H:%M:%S.%f")}')

        return all_data

    def blink_led(
        self,
        device_idx: int = 0,
        led_num: int = 0,
        num_blinks: int = 3,
        period: float = 1.0,
    ):
        """Blink an LED on the Red Pitaya device for connectivity troubleshooting.

        Args:
            device_idx: Index of the device to blink LED on
            led_num: LED number to blink (0-7)
            num_blinks: Number of times to blink
            period: Period of each blink in seconds
        """
        if len(self.devices) <= device_idx:
            logger.error(f'No device at index {device_idx}')
            return

        device = self.devices[device_idx]
        device_name = self.device_names[device_idx]

        logger.info(f'Blinking LED[{led_num}] on {device_name} {num_blinks} times')

        try:
            for _ in range(num_blinks):
                # Turn LED on
                device.tx_txt(f'DIG:PIN LED{led_num},1')
                time.sleep(period / 2.0)

                # Turn LED off
                device.tx_txt(f'DIG:PIN LED{led_num},0')
                time.sleep(period / 2.0)

            logger.info(f'Finished blinking LED on {device_name}')
        except Exception:
            logger.exception(f'Error blinking LED on {device_name}')
            logger.exception('Are you sure the SCPI server is running?')


# Example usage
if __name__ == '__main__':
    # # Connect to a single Red Pitaya device
    # rp_manager = RedPitayaManager("rp-f0c04a.local")

    # # Run a single acquisition
    # rp_manager.run_one_shot(
    #     store_data=False,
    #     plot_data=True
    # )

    # Connect to multiple Red Pitaya devices
    rp_manager = RedPitayaManager(
        ['rp-f0c04a.local', 'rp-f0c026.local'], blink_on_connect=True
    )

    # Configure daisy chain
    rp_manager.configure_daisy_chain()

    # Run multiple acquisitions
    rp_manager.run_multiple_shots(
        num_shots=10,
        delay_between_shots=1.0,
        plot_data=True,
        keep_final_plot=True,  # Keep the final plot open
    )
