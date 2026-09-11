#!/usr/bin/env python

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from smi.redpitaya.manager import RedPitayaManager
from smi.redpitaya.redpitaya_config import RedPitayaConfig
from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.waveform import Waveform

logger = logging.getLogger(__name__)


def generate_dataset_from_rp(
    rp_manager: RedPitayaManager,
    output_dir: str | Path,
    num_samples: int,
    dataset_filename: str = 'dataset.h5',
) -> str:
    """Create a single dataset from Red Pitaya data.
    RedPitayaManager handles saving to HDF5 internally shot by shot, to protect against
    connection loss. See RedPitayaManager.save_data for more details.

    Args:
        rp_manager: RedPitayaManager instance to use for data acquisition
        output_dir: Directory to save the dataset
        num_samples: Total number of samples to acquire
        dataset_filename: Name of the output HDF5 file

    Returns:
        Path to the created dataset file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / dataset_filename

    # Create the single dataset
    logger.info(f'\nAcquiring {num_samples} samples into {dataset_path}...')
    _ = rp_manager.run_multiple_shots(num_shots=num_samples, hdf5_file=dataset_path)

    logger.info(f'Dataset creation complete: {dataset_path}')
    return str(dataset_path)


def generate_synthetic_test_data(
    num_samples: int,
    wavelengths_nm: list[float],
    batch_size: int = 64,
    start_freq: float = 1.0,
    end_freq: float = 1000.0,
    max_displacement_um: float = 5.0,
) -> DataLoader:
    """Generate synthetic test data and return as a DataLoader.

    Creates synthetic interferometer signals, velocity, and displacement
    data for model evaluation. Uses CPU tensors.

    Args:
        num_samples: Number of test samples to generate
        wavelengths_nm: List of interferometer wavelengths in nanometers
        batch_size: Batch size for the returned DataLoader
        start_freq: Lower bound of displacement spectrum frequency range (Hz)
        end_freq: Upper bound of displacement spectrum frequency range (Hz)
        max_displacement_um: Peak displacement amplitude in microns
        seed: Random seed for reproducibility

    Returns:
        DataLoader yielding (signals, velocity, displacement) tuples
    """
    torch.manual_seed(42)

    acq_sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
    waveform = Waveform(start_freq=start_freq, end_freq=end_freq)

    wavelengths_um = torch.tensor(
        [w / 1000.0 for w in wavelengths_nm], dtype=torch.float32
    ).view(1, -1, 1)

    num_channels = len(wavelengths_nm)

    # Generate all samples
    displacement = waveform.generate_batch_torch(num_samples, 'cpu')

    # Apply random amplitude scaling per sample
    scale = torch.rand(num_samples, 1) * max_displacement_um
    displacement = displacement * scale

    # Normalize each waveform to have unit max amplitude before scaling
    max_abs = displacement.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    displacement = displacement / max_abs * scale

    # Random interferometer phases per sample per channel
    random_phases = torch.rand(num_samples, num_channels, 1) * 2.0 * torch.pi

    # Compute interferometer signals
    signals = torch.cos(
        2.0 * torch.pi / wavelengths_um * 2.0 * displacement.unsqueeze(1)
        + random_phases
    )

    # Per-sample per-channel z-score normalization
    signals_mean = signals.mean(dim=-1, keepdim=True)
    signals_std = signals.std(dim=-1, keepdim=True).clamp(min=1e-8)
    signals = (signals - signals_mean) / signals_std

    # Compute velocity from displacement
    velocity = CoilDriver.derivative_displacement(displacement, acq_sample_rate)

    dataset = TensorDataset(signals, velocity, displacement)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
