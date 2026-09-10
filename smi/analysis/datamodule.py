#!/usr/bin/env python
"""Lightning DataModule for velocity/displacement prediction.

Encapsulates the train/val/test split and DataLoader construction that used to
live as boilerplate in ``datasets.get_data_loaders`` and ``TrainingInterface``.
The split is deterministic (fixed seed) for reproducible evaluation.
"""

import logging

import lightning as lightning_module
import torch
from torch.utils.data import DataLoader, random_split

from smi.analysis.datasets import GPUResidentDataset, VelocityDataset

logger = logging.getLogger(__name__)


class VelocityDataModule(lightning_module.LightningDataModule):
    """DataModule wrapping a single HDF5 file.

    By default samples are read and transformed one at a time by
    :class:`VelocityDataset` behind a torch DataLoader. Passing
    ``preload_device`` instead materializes the whole file as tensors on that
    device (:class:`GPUResidentDataset`) and serves batches by indexing them.

    On the measured workload the resident mode is *not* a speedup: training is
    GPU bound, and it came out 1.7% ahead of an 8-worker DataLoader on an RTX
    3090 Ti over the full 10k-shot file. See
    ``smi.analysis.benchmark_loading``. It is offered because it removes worker
    processes and the host-to-device copy, which matters for determinism and
    for cheaper models, not because it makes this model train faster.

    Args:
        dataset_path: Path to the HDF5 dataset file.
        split_ratios: (train, val, test) percentages summing to 100.
        batch_size: Batch size for all dataloaders.
        num_workers: Worker processes for data loading. Ignored (forced to 0)
            when ``preload_device`` is set, since the data is already tensors.
        seed: Seed for the deterministic split.
        preload_device: Device to hold the whole dataset on, e.g. ``'cuda:0'``.
            ``None`` (the default) keeps the existing per-sample DataLoader
            path, so this argument never changes behaviour unless asked for.
            If the data plus ``preload_headroom_bytes`` does not fit the named
            CUDA device, the DataModule logs a warning and falls back to the
            DataLoader path rather than OOMing mid-training.
        preload_headroom_bytes: VRAM to leave free for the model, activations
            and cuDNN workspace when deciding whether the data fits. The 3 GB
            default is what the production-size TCN needs at batch 32.
        **dataset_kwargs: Forwarded to the dataset (e.g. ``num_pd_channels``;
            ``cache_size`` applies only to the DataLoader path).
    """

    def __init__(
        self,
        dataset_path: str,
        split_ratios: tuple[int, int, int] = (80, 10, 10),
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        preload_device: torch.device | str | None = None,
        preload_headroom_bytes: int = 3 << 30,
        **dataset_kwargs: object,
    ) -> None:
        super().__init__()
        if sum(split_ratios) != 100:
            raise ValueError(f'Split ratios must sum to 100, got {sum(split_ratios)}')
        self.dataset_path = dataset_path
        self.split_ratios = split_ratios
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.preload_device = (
            None if preload_device is None else torch.device(preload_device)
        )
        self.preload_headroom_bytes = preload_headroom_bytes
        self.dataset_kwargs = dataset_kwargs
        # Set in setup(): False until we know the data actually fit.
        self.preloaded = False
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _build_full_dataset(self) -> object:
        """Build the resident dataset if requested and it fits, else the lazy one."""
        if self.preload_device is None:
            return VelocityDataset(self.dataset_path, **self.dataset_kwargs)

        num_pd_channels = int(self.dataset_kwargs.get('num_pd_channels', 3))
        if not GPUResidentDataset.fits_on_device(
            self.dataset_path,
            self.preload_device,
            num_pd_channels=num_pd_channels,
            headroom_bytes=self.preload_headroom_bytes,
        ):
            needed = GPUResidentDataset.required_bytes(
                self.dataset_path, num_pd_channels
            )
            logger.warning(
                'preload_device=%s requested but the dataset needs %.2f GB plus '
                '%.2f GB headroom and does not fit; falling back to the '
                'DataLoader path.',
                self.preload_device,
                needed / 1e9,
                self.preload_headroom_bytes / 1e9,
            )
            return VelocityDataset(self.dataset_path, **self.dataset_kwargs)

        self.preloaded = True
        return GPUResidentDataset(
            self.dataset_path,
            device=self.preload_device,
            num_pd_channels=num_pd_channels,
        )

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Create the dataset and the deterministic train/val/test split.

        ``stage`` is part of the LightningDataModule API; the same split is built
        for every stage, so it is unused here.
        """
        if self.train_dataset is not None:
            return

        full_dataset = self._build_full_dataset()
        total = len(full_dataset)
        train_size = int(total * self.split_ratios[0] / 100)
        val_size = int(total * self.split_ratios[1] / 100)
        test_size = total - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        logger.info(
            'Dataset split - Total: %d, Train: %d, Val: %d, Test: %d',
            total,
            train_size,
            val_size,
            test_size,
        )

    def _loader(self, dataset: object, *, shuffle: bool) -> DataLoader:
        """Build a DataLoader with the shared settings."""
        if self.preloaded:
            # The samples are already tensors on the target device. Worker
            # processes cannot touch a CUDA tensor and pinning is meaningless,
            # so both are off; the collate is a plain stack of device views.
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=False,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training dataloader."""
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader (unshuffled)."""
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test dataloader (unshuffled)."""
        return self._loader(self.test_dataset, shuffle=False)
