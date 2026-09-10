#!/usr/bin/env python

import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from smi.redpitaya.redpitaya_config import RedPitayaConfig
from smi.synthetic.coil_driver import CoilDriver

logger = logging.getLogger(__name__)


class VelocityDataset(Dataset):
    """Dataset class for velocity prediction from photodiode signals.

    This dataset is designed to work with HDF5 files containing Red Pitaya channel data
    with the following structure:
    - RP1_CH1: Speaker drive voltage (from which velocity is computed)
    - RP1_CH2, RP2_CH1, RP2_CH2: Photodiode signals from interferometers

    The dataset returns RAW photodiode signals. Input/output normalization is no
    longer done here: it is baked into the model via the Model wrapper
    (analysis/models/base.py), using version-controlled per-feature statistics
    (analysis/features/normalization.py). The rest of the pipeline works in raw
    physical units.

    The dataset returns:
    - signals: Raw photodiode signals (1-3 channels based on num_pd_channels)
    - velocity: Velocity computed from speaker drive voltage
    - displacement: Displacement computed by integrating velocity

    For standard models that only need velocity, the displacement can be ignored in the
    training loop.

    Args:
        file_path: Path to HDF5 file
        num_pd_channels: Number of photodiode channels to use (1-3)
        cache_size: Number of items to cache in memory (0 for no caching)
    """

    def __init__(
        self, file_path: str | Path, num_pd_channels: int = 3, cache_size: int = 0
    ):
        self.file_path = file_path
        self.num_pd_channels = min(max(1, num_pd_channels), 3)  # Ensure between 1 and 3
        self.cache_size = cache_size
        self.sample_rate = None  # Will be set in open_hdf5

        # Channel keys for photodiode signals
        self.pd_channel_keys = ['RP1_CH2', 'RP2_CH1', 'RP2_CH2'][: self.num_pd_channels]

        # Channel key for speaker drive voltage
        self.voltage_key = 'RP1_CH1'

        # Initialize cache
        self._cache = {}
        self._cache_keys = []

        # Initialize CoilDriver for velocity calculation
        self.coil_driver = CoilDriver()

        # Validate file and get dataset info
        with h5py.File(self.file_path, 'r') as f:
            self._validate_file_structure(f)
            self.length = self._get_dataset_length(f)

        self.opened_flag = False

    def _validate_file_structure(self, f: h5py.File) -> None:
        """Validate the HDF5 file has required datasets."""
        if self.voltage_key not in f:
            raise ValueError(f"Voltage key '{self.voltage_key}' not found in file")

        for key in self.pd_channel_keys:
            if key not in f:
                raise ValueError(f"Photodiode channel key '{key}' not found in file")

    def _get_dataset_length(self, f: h5py.File) -> int:
        """Get the length of the dataset."""
        return len(f[self.voltage_key])

    def open_hdf5(self):
        """Open HDF5 file for reading.

        This is done lazily to support multiprocessing in DataLoader.
        Also sets the sample_rate from file attributes if available.
        """
        if not self.opened_flag:
            self.h5_file = h5py.File(self.file_path, 'r')
            self.voltage_data = self.h5_file[self.voltage_key]
            self.pd_data = [self.h5_file[key] for key in self.pd_channel_keys]

            # Get sample rate from file attributes if available
            if 'sample_rate' in self.h5_file.attrs:
                self.sample_rate = float(self.h5_file.attrs['sample_rate'])
            else:
                # Default sample rate for Red Pitaya with decimation of 256
                self.sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
                logger.warning(
                    f'Using default sample rate of {self.sample_rate:.2f} Hz'
                )

            self.opened_flag = True

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.

        Returns:
            Tuple of (signals, velocity, displacement) where:
            - signals: Tensor of shape (num_pd_channels, signal_length) containing
              photodiode signals
            - velocity: Tensor of shape (signal_length,) containing velocity
            - displacement: Tensor of shape (signal_length,) containing displacement
              computed by integrating velocity
        """
        # Check cache first
        if idx in self._cache:
            signals, velocity, displacement = self._cache[idx]
        else:
            # Lazy loading of HDF5 file
            self.open_hdf5()

            # Load photodiode signals
            pd_signals = [self.pd_data[i][idx] for i in range(len(self.pd_data))]

            # Load voltage and compute velocity
            voltage = self.voltage_data[idx]
            # One shared displacement spectrum feeds both outputs. Calling
            # get_velocity and get_displacement separately would compute the
            # same forward FFT and transfer-function product twice per sample.
            velocity, displacement = self.coil_driver.get_velocity_and_displacement(
                voltage, self.sample_rate
            )

            # Stack photodiode signals into a single tensor
            signals = np.stack(pd_signals)

            # Add to cache
            if self.cache_size > 0:
                self._add_to_cache(idx, (signals, velocity, displacement))

        # Convert to PyTorch tensors (raw signals; the model normalizes internally)
        signals_tensor = torch.FloatTensor(signals)
        velocity_tensor = torch.FloatTensor(velocity)
        displacement_tensor = torch.FloatTensor(displacement)

        return signals_tensor, velocity_tensor, displacement_tensor

    def _add_to_cache(
        self, key: int, value: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Add an item to the cache, maintaining cache size limit."""
        if self.cache_size == 0:
            return

        if len(self._cache) >= self.cache_size:
            # Remove oldest item if cache is full
            oldest_key = self._cache_keys[0]
            del self._cache[oldest_key]
            self._cache_keys.pop(0)

        self._cache[key] = value
        self._cache_keys.append(key)


class GPUResidentDataset(Dataset):
    """The whole dataset held as tensors on one device, indexed by slicing.

    An opt-in alternative to reading and transforming samples one at a time.
    ``__getitem__`` is a device-side index into pre-materialized tensors, so no
    HDF5 read, no FFT and no host-to-device copy happens during training.

    **This is not a speedup on the current workload.** It was benchmarked
    (``smi.analysis.benchmark_loading``) against the DataLoader path on an RTX
    3090 Ti with the production-size TCN and came out 1.7% faster over the full
    10k-shot file -- because training is GPU bound, not loader bound. Use it
    when you want to remove worker processes from the picture (debugging,
    determinism, Windows worker-spawn overhead) or when the model is cheap
    enough that the loader would matter, not because it makes training faster.

    Memory: 4 bytes * signal_length * (num_pd_channels + 2) per shot. For the
    10k x 16384 files with 3 channels that is 3.28 GB. Verify it fits before
    constructing; :meth:`fits_on_device` does that check.

    Args:
        file_path: Path to the HDF5 file.
        device: Device to hold the tensors on. Passed explicitly -- this class
            never guesses a device.
        num_pd_channels: Number of photodiode channels to load (1-3).
        max_shots: If set, load only the first ``max_shots`` shots.
    """

    #: Shots per batched physics call during construction. See __init__.
    PHYSICS_CHUNK = 512

    def __init__(
        self,
        file_path: str | Path,
        device: torch.device | str,
        num_pd_channels: int = 3,
        max_shots: int | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.device = torch.device(device)
        self.num_pd_channels = min(max(1, num_pd_channels), 3)
        pd_keys = ['RP1_CH2', 'RP2_CH1', 'RP2_CH2'][: self.num_pd_channels]

        coil_driver = CoilDriver()
        with h5py.File(self.file_path, 'r') as f:
            for key in [*pd_keys, 'RP1_CH1']:
                if key not in f:
                    raise ValueError(f"Channel key '{key}' not found in file")
            total = len(f['RP1_CH1'])
            n = total if max_shots is None else min(max_shots, total)
            if 'sample_rate' in f.attrs:
                self.sample_rate = float(f.attrs['sample_rate'])
            else:
                self.sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
                logger.warning('Using default sample rate of %.2f Hz', self.sample_rate)
            signals = np.stack(
                [np.asarray(f[key][:n], dtype=np.float32) for key in pd_keys], axis=1
            )
            voltage = np.asarray(f['RP1_CH1'][:n], dtype=np.float32)

        # Do the drive-voltage physics once for the whole file, batched, rather
        # than once per sample per epoch. On a CUDA device this runs on-device
        # via torch.fft; on CPU the same torch code path is used, so the two
        # agree bit-for-bit within a run.
        #
        # Chunked, because the transform's intermediates are complex128: a
        # single 10000 x 16384 call would transiently need ~10 GB of workspace
        # on top of the 3.3 GB it is producing, and would OOM on anything but
        # an empty 24 GB card. The chunk size is a workspace/launch-overhead
        # tradeoff, not a correctness knob -- the result is identical.
        self.signals = torch.from_numpy(signals).to(self.device)
        self.velocity = torch.empty(
            (n, signals.shape[-1]), dtype=torch.float32, device=self.device
        )
        self.displacement = torch.empty_like(self.velocity)
        for start in range(0, n, self.PHYSICS_CHUNK):
            stop = min(start + self.PHYSICS_CHUNK, n)
            chunk = torch.from_numpy(voltage[start:stop]).to(self.device)
            velocity, displacement = coil_driver.get_velocity_and_displacement_torch(
                chunk, self.sample_rate
            )
            self.velocity[start:stop] = velocity
            self.displacement[start:stop] = displacement
        self.length = n
        logger.info(
            'GPUResidentDataset: %d shots resident on %s (%.2f GB)',
            n,
            self.device,
            self.resident_bytes() / 1e9,
        )

    @staticmethod
    def required_bytes(
        file_path: str | Path, num_pd_channels: int = 3, max_shots: int | None = None
    ) -> int:
        """Bytes the resident tensors will occupy for this file."""
        with h5py.File(file_path, 'r') as f:
            total, signal_length = f['RP1_CH1'].shape
        n = total if max_shots is None else min(max_shots, total)
        channels = min(max(1, num_pd_channels), 3) + 2  # + velocity + displacement
        return 4 * int(n) * int(signal_length) * channels

    @classmethod
    def fits_on_device(
        cls,
        file_path: str | Path,
        device: torch.device | str,
        num_pd_channels: int = 3,
        max_shots: int | None = None,
        headroom_bytes: int = 3 << 30,
    ) -> bool:
        """Whether the dataset plus working headroom fits on ``device``.

        ``headroom_bytes`` defaults to 3 GB, which is what the benchmarked
        production TCN needs for parameters, activations and cuDNN workspace at
        batch 32. A model with larger activations needs a larger value.
        """
        device = torch.device(device)
        if device.type != 'cuda':
            return True  # host RAM is not tracked here; only VRAM is scarce
        free, _total = torch.cuda.mem_get_info(device)
        return cls.required_bytes(file_path, num_pd_channels, max_shots) + int(
            headroom_bytes
        ) <= int(free)

    def resident_bytes(self) -> int:
        """Bytes currently held by this dataset's tensors."""
        return sum(
            t.element_size() * t.numel()
            for t in (self.signals, self.velocity, self.displacement)
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (signals, velocity, displacement) views for one shot."""
        return self.signals[idx], self.velocity[idx], self.displacement[idx]


def get_data_loaders(
    dataset_path: str,
    split_ratios: tuple[int, int, int] = (80, 10, 10),
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for training, validation, and testing from a single dataset.

    The train/val/test split is always deterministic with a hardcoded seed of 42,
    ensuring the same split every time for reproducible evaluation.

    Args:
        dataset_path: Path to the single HDF5 dataset file
        split_ratios: Tuple of (train, val, test) percentages that sum to 100
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes for data loading
        **dataset_kwargs: Additional arguments to pass to the dataset class

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Validate split ratios
    if sum(split_ratios) != 100:
        raise ValueError(f'Split ratios must sum to 100, got {sum(split_ratios)}')

    # Create the full dataset to get its length
    full_dataset = VelocityDataset(dataset_path, **dataset_kwargs)
    total_length = len(full_dataset)

    # Calculate split sizes
    train_size = int(total_length * split_ratios[0] / 100)
    val_size = int(total_length * split_ratios[1] / 100)
    test_size = total_length - train_size - val_size  # Remainder goes to test

    # Create random split with hardcoded seed for deterministic splits
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    logger.info(
        f'Dataset split - Total: {total_length}, Train: {train_size}, '
        f'Val: {val_size}, Test: {test_size}'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
