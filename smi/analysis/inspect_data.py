#!/usr/bin/env python

import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from smi.analysis.datasets import VelocityDataset
from smi.redpitaya.redpitaya_config import RedPitayaConfig

logger = logging.getLogger(__name__)


def visualize_dataset(
    dataset_path: str | Path, max_samples: int = 10, batch_size: int = 1
):
    """Visualize samples from a dataset using VelocityDataset.

    Creates plots similar to the training.py visualization but without model
    predictions.
    Each plot includes:
    1. Velocity and displacement on dual y-axes
    2-4. Input signals from photodiode channels

    Args:
        dataset_path: Path to the HDF5 dataset file
        max_samples: Maximum number of samples to visualize
        batch_size: Batch size for the DataLoader
    """
    # Get sample rate from HDF5 file attributes if available
    sample_rate = None
    with h5py.File(dataset_path, 'r') as f:
        if 'sample_rate' in f.attrs:
            sample_rate = float(f.attrs['sample_rate'])
            logger.info(f'Using sample rate from HDF5 file: {sample_rate:.2f} Hz')
        else:
            # Default sample rate for Red Pitaya with decimation of 256
            sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
            logger.warning(f'Using default sample rate of {sample_rate:.2f} Hz')

        # Print dataset information
        logger.info(f'Dataset: {Path(dataset_path).name}')
        logger.info(f'Number of samples: {len(f[next(iter(f.keys()))])}')
        logger.info(f'Available channels: {list(f.keys())}')

    # Create dataset and dataloader
    dataset = VelocityDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Get samples from dataloader
    samples_processed = 0
    for batch_idx, (signals_t, velocity_t, displacement_t) in enumerate(dataloader):
        if samples_processed >= max_samples:
            break

        # Manually check around particular samples in the dataset
        if batch_idx <= 9990:
            pass

        # Convert tensors to numpy arrays
        signals = signals_t.numpy()
        velocity = velocity_t.numpy()
        displacement = displacement_t.numpy()

        # Get the number of samples in the batch and number of PD channels
        batch_size, num_channels, signal_length = signals.shape

        # Print shapes for debugging
        logger.debug(f'Batch {batch_idx + 1}:')
        logger.debug(f'Signals shape: {signals.shape}')
        logger.debug(f'Velocity shape: {velocity.shape}')
        logger.debug(f'Displacement shape: {displacement.shape}')

        # Plot each sample in the batch
        for i in range(min(batch_size, max_samples - samples_processed)):
            samples_processed += 1

            # Create a figure with subplots - one for velocities and up to 3 for signals
            _fig, axs = plt.subplots(1 + num_channels, 1, figsize=(10, 8), sharex=True)

            # If there's only one subplot, make it an array for consistent indexing
            if num_channels == 0:
                axs = [axs]

            # Create time array
            time = np.arange(signal_length) / sample_rate

            # Plot velocity on the primary y-axis
            axs[0].plot(time, velocity[i], label='Velocity', color='blue')
            axs[0].set_title('Velocity and Displacement')
            axs[0].set_ylabel('Velocity (umm/s)', color='blue')
            axs[0].tick_params(axis='y', labelcolor='blue')
            axs[0].grid(True, alpha=0.3)

            # Create a twin axis for displacement
            ax_twin = axs[0].twinx()
            ax_twin.plot(time, displacement[i], label='Displacement', color='green')
            ax_twin.set_ylabel('Displacement (umm)', color='green')
            ax_twin.tick_params(axis='y', labelcolor='green')

            # Ensure both legends are visible
            lines1, labels1 = axs[0].get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            # Plot each input signal channel
            for j in range(num_channels):
                channel_names = ['PD1 (RP1_CH2)', 'PD2 (RP2_CH1)', 'PD3 (RP2_CH2)']
                axs[j + 1].plot(time, signals[i, j, :], label=f'Channel {j + 1}')
                axs[j + 1].set_title(f'Input Signal - {channel_names[j]}')
                axs[j + 1].set_ylabel('Amplitude')
                axs[j + 1].grid(True)

            # Set x-axis label for the bottom subplot
            axs[-1].set_xlabel('Time (s)')

            # Add overall title
            plt.suptitle(f'Sample {samples_processed} from Dataset')
            plt.tight_layout()
            plt.show()


def plot_histograms(dataset_path: str | Path):
    """Plot histograms and spectra for each channel in the dataset.

    Creates histograms in the first row and FFT spectra in the second row for each
    channel.

    Args:
        dataset_path: Path to the HDF5 dataset file
    """
    # Get sample rate from HDF5 file attributes if available
    with h5py.File(dataset_path, 'r') as f:
        if 'sample_rate' in f.attrs:
            sample_rate = float(f.attrs['sample_rate'])
        else:
            # Default sample rate for Red Pitaya with decimation of 256
            sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256

        logger.info(f'Using sample rate: {sample_rate:.2f} Hz')
        # Create figure with subplots (2 rows, 4 columns)
        _fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Channel mapping and colors
        channel_info = {
            'RP1_CH1': {'label': 'Speaker Drive Voltage', 'color': 'blue', 'col': 0},
            'RP1_CH2': {'label': 'PD1 (635nm Red)', 'color': 'red', 'col': 1},
            'RP2_CH1': {'label': 'PD2 (675nm Deep Red)', 'color': 'darkred', 'col': 2},
            'RP2_CH2': {'label': 'PD3 (515nm Green)', 'color': 'green', 'col': 3},
        }

        # Calculate FFT frequencies
        first_key = next(iter(f.keys()))
        n = len(f[first_key][0, :])
        logger.info(f'Number of samples: {n}')
        freqs = np.fft.fftfreq(n, 1 / sample_rate)
        pos_idx = np.where(freqs > 0)
        pos_freqs = freqs[pos_idx]

        # Process each channel
        for key in f:
            if key in channel_info:
                info = channel_info[key]
                label = info['label']
                color = info['color']
                col = info['col']

                # Get data and flatten it if it's multi-dimensional
                data = f[key][:]
                data_flat = data.flatten()

                # Row 0: Plot histogram
                hist, bin_edges = np.histogram(data_flat, bins=100)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                axes[0, col].hist(data_flat, bins=100, color=color, alpha=0.7)

                # Calculate Gaussian overlay
                mean = np.mean(data_flat)
                std = np.std(data_flat)
                gaussian = np.max(hist) * np.exp(
                    -0.5 * ((bin_centers - mean) / std) ** 2
                )

                # Plot Gaussian overlay
                axes[0, col].plot(
                    bin_centers,
                    gaussian,
                    'r-',
                    linewidth=2,
                    label=f'Gaussian (sigma={std:.4f})',
                )

                axes[0, col].set_title(f'{label} Histogram')
                axes[0, col].set_xlabel('Amplitude (V)')
                axes[0, col].set_ylabel('Count')
                axes[0, col].grid(True, alpha=0.3)
                axes[0, col].legend(loc='upper right', fontsize='small')

                # Row 1: Plot spectrum
                # For FFT, we'll use the first sample if data has multiple samples
                if data.ndim > 1 and data.shape[0] > 0:
                    # Average the FFT of all samples
                    fft_mags = []
                    for i in range(min(data.shape[0], 10)):  # Use up to 10 samples
                        fft_complex = np.fft.fft(data[i], norm='ortho')
                        fft_mags.append(np.abs(fft_complex))
                    fft_mag = np.mean(fft_mags, axis=0)
                else:
                    fft_complex = np.fft.fft(data_flat, norm='ortho')
                    fft_mag = np.abs(fft_complex)

                axes[1, col].semilogy(pos_freqs, fft_mag[pos_idx], color=color)
                axes[1, col].set_title(f'{label} Spectrum')
                axes[1, col].set_xlabel('Frequency (Hz)')
                axes[1, col].set_ylabel('Magnitude')
                axes[1, col].grid(True, which='both', ls='-', alpha=0.5)

            # Set reasonable frequency limits for drive voltage spectrum
            max_freq = min(sample_rate / 2, 2000)  # Either Nyquist or 5kHz
            axes[1, 0].set_xlim(0, max_freq)

        # Set title and adjust layout
        plt.suptitle(f'Signal Analysis - {Path(dataset_path).name}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
        plt.show()


def analyze_dataset(dataset_path: str | Path):
    """Perform statistical analysis on a dataset.

    Calculates and displays statistics for each channel in the dataset.

    Args:
        dataset_path: Path to the HDF5 dataset file
    """
    with h5py.File(dataset_path, 'r') as f:
        logger.info(f'Dataset Analysis: {Path(dataset_path).name}')
        logger.info(f'Number of samples: {len(f[next(iter(f.keys()))])}')

        # Calculate statistics for each channel
        for key in f:
            if key == 'RP1_CH1':
                label = 'Speaker Drive Voltage'
            elif key == 'RP1_CH2':
                label = 'PD1 (635nm Red)'
            elif key == 'RP2_CH1':
                label = 'PD2 (675nm Deep Red)'
            elif key == 'RP2_CH2':
                label = 'PD3 (515nm Green)'
            else:
                label = key

            # Get data and calculate statistics
            data = f[key][:]
            mean = np.mean(data)
            std = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            range_val = max_val - min_val

            # Print statistics
            logger.info(f'{label} statistics:')
            logger.info(f'  - Mean: {mean:.4f}')
            logger.info(f'  - Std: {std:.4f}')
            logger.info(f'  - Min: {min_val:.4f}')
            logger.info(f'  - Max: {max_val:.4f}')
            logger.info(f'  - Range: {range_val:.4f}')

    # Plot histograms for the dataset
    plot_histograms(dataset_path)


if __name__ == '__main__':
    # Example usage
    # dataset_path = './analysis/data/free-space-all-amplitude-10000.h5'
    dataset_path = './analysis/data/free-space-synchro_10k.h5'

    # Visualize samples from the dataset
    visualize_dataset(dataset_path, max_samples=10)

    # Analyze dataset statistics and plot histograms
    # analyze_dataset(dataset_path)
