#!/usr/bin/env python

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from smi.redpitaya.manager import RedPitayaManager
from smi.synthetic.forward import SimConfig, simulate_batch

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
    num_samples: int, sim: dict[str, Any], batch_size: int, seed: int
) -> DataLoader:
    """Simulate a seeded CPU test set and return it as a DataLoader.

    Args:
        num_samples: Number of test shots to simulate.
        sim: Every ``SimConfig`` field by name (e.g. a checkpoint's ``sim``
            hyperparameter), so test data matches the training simulator.
        batch_size: Batch size for the returned DataLoader.
        seed: Seed of the CPU generator.

    Returns:
        DataLoader yielding (signals, velocity, displacement) tuples, with
        velocity in um/ms and displacement in um
    """
    gen = torch.Generator().manual_seed(seed)
    batch = simulate_batch(SimConfig(**sim), num_samples, torch.device('cpu'), gen)
    dataset = TensorDataset(batch.signals, batch.velocity, batch.displacement)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
