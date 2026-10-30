#!/usr/bin/env python

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from smi.analysis.synthetic_lit_module import SyntheticLitModule
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
    *,
    drive_gain: float,
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
        drive_gain: Drive voltage scale relative to the rig's waveform
        seed: Random seed for reproducibility

    Returns:
        DataLoader yielding (signals, velocity, displacement) tuples, with
        velocity in um/ms and displacement in um
    """
    torch.manual_seed(42)

    acq_sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
    waveform = Waveform(start_freq=start_freq, end_freq=end_freq)

    wavelengths_um = torch.tensor(
        [w / 1000.0 for w in wavelengths_nm], dtype=torch.float32
    ).view(1, -1, 1)

    signals, velocity, displacement = SyntheticLitModule.generate_synthetic_batch(
        num_samples,
        torch.device('cpu'),
        waveform,
        wavelengths_um,
        CoilDriver(),
        drive_gain,
        acq_sample_rate,
    )

    dataset = TensorDataset(signals, velocity, displacement)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
