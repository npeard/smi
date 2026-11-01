#!/usr/bin/env python

import logging
from pathlib import Path

from smi.redpitaya.manager import RedPitayaManager

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
