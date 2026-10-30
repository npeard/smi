#!/usr/bin/env python

import logging
from typing import Any, override

import torch
from torch import nn
from torch.utils.data import Dataset

from smi.analysis.lit_module import LitModule
from smi.analysis.models.base import Model
from smi.synthetic.forward import SimConfig, normalization_stats, simulate_batch

logger = logging.getLogger(__name__)

# Seeded CPU sample behind the baked normalization, fixed so a checkpoint
# rebuilds the same buffers. Measured seed-to-seed spread over 5 seeds: input
# std < 1%, velocity std 4.6% (the per-shot U(0, 1) drive power is heavy-tailed),
# which is ample for a fixed affine scale; 0.13 s (Michelson) to 0.5 s (C = 0.9).
NORM_STATS_SHOTS = 256
NORM_STATS_SEED = 0


def random_crop(length: int | None, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One random contiguous window per shot, shared by every tensor.

    Signals ``[B, C, L]`` and targets ``[B, L]`` are cropped at the same
    positions, so they stay aligned.

    Args:
        length: Window length; ``None`` returns the tensors unchanged.
        *tensors: ``[B, ..., L]`` tensors sharing ``B`` and ``L``.

    Returns:
        The cropped tensors, in order.
    """
    full = tensors[0].shape[-1]
    if length is None:
        return tensors
    if not 0 < length <= full:
        raise ValueError(f'train_crop_length {length} must lie in (0, {full}]')
    device = tensors[0].device
    start = torch.randint(0, full - length + 1, (tensors[0].shape[0],), device=device)
    index = start[:, None] + torch.arange(length, device=device)
    cropped = []
    for t in tensors:
        idx = index.view(t.shape[0], *([1] * (t.dim() - 2)), length)
        cropped.append(t.gather(-1, idx.expand(*t.shape[:-1], length)))
    return tuple(cropped)


class SyntheticIndexDataset(Dataset):
    """Dummy dataset that returns indices, for use with SyntheticLitModule.

    The actual data is generated on-device in the LightningModule's training_step.
    This dataset only provides the correct number of iterations per epoch.
    """

    def __init__(self, length: int):
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> int:
        return idx


class SyntheticLitModule(LitModule):
    """Lightning Module that simulates its training data on-device each step.

    Each training/validation/test step draws a fresh batch from
    :func:`smi.synthetic.forward.simulate_batch` (drive voltage through the
    coil, then fringe and detector per the ``SimConfig``) and feeds it to the
    parent class's loss computation. The model's input/output normalization is
    fixed per simulator config and baked into the ``Model`` wrapper, so the
    data reaching a causal network carries no whole-shot statistics.
    """

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
        training_hparams: dict | None = None,
        data_hparams: dict | None = None,
        steps_per_epoch: int = 1000,
        *,
        sim: dict[str, Any],
        train_crop_length: int | None,
    ):
        """Initialize SyntheticLitModule with hyperparameter configuration.

        Args:
        model_hparams: Hyperparameters for the model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        training_hparams: Hyperparameters for training configuration
        data_hparams: Hyperparameters for data configuration
        steps_per_epoch: Number of training batches per epoch
        sim: Every ``SimConfig`` field by name (the YAML ``synthetic.sim``
            block); kept as a plain dict so it is saved with the checkpoint
        train_crop_length: Length of the random window each training and
            validation shot is cut to after simulating the full shot (the
            trajectory spectrum needs the full length); ``None`` keeps the
            full shot. Evaluation never crops.
        """
        # _make_model runs inside LitModule.__init__ and needs the config.
        self.sim_config = SimConfig(**sim)
        if len(self.sim_config.wavelengths_um) != model_hparams['in_channels']:
            raise ValueError(
                f'{len(self.sim_config.wavelengths_um)} wavelengths but '
                f'in_channels={model_hparams["in_channels"]}'
            )
        super().__init__(
            model_hparams=model_hparams,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=loss_hparams,
            training_hparams=training_hparams,
            data_hparams=data_hparams,
        )
        self.steps_per_epoch = steps_per_epoch
        self.train_crop_length = train_crop_length
        self.save_hyperparameters(ignore=['model'])
        logger.info(f'SyntheticLitModule sim config {self.sim_config.hash()}')

    @override
    def _make_model(self, inner: nn.Module) -> Model:
        """Wrap with the simulator config's fixed normalization stats."""
        if self.target != 'velocity':
            raise ValueError(
                'synthetic normalization scales the velocity target, '
                f'not {self.target!r}'
            )
        return Model(
            inner,
            **normalization_stats(self.sim_config, NORM_STATS_SHOTS, NORM_STATS_SEED),
        )

    def _generate_synthetic_batch(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        """Simulate ``(signals, velocity, displacement)``, cropped, on ``device``."""
        batch = simulate_batch(self.sim_config, batch_size, device, None)
        return random_crop(
            self.train_crop_length, batch.signals, batch.velocity, batch.displacement
        )

    @override
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Generate synthetic data on-device and delegate to parent training_step.

        Args:
            batch: Tensor of indices from SyntheticIndexDataset (batch size only)
            batch_idx: Index of current batch
        """
        batch_size = len(batch)
        synthetic_batch = self._generate_synthetic_batch(batch_size, self.device)
        return super().training_step(synthetic_batch, batch_idx)

    @override
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Generate synthetic data on-device and delegate to parent validation_step.

        Args:
            batch: Tensor of indices from SyntheticIndexDataset (batch size only)
            batch_idx: Index of current batch
        """
        batch_size = len(batch)
        synthetic_batch = self._generate_synthetic_batch(batch_size, self.device)
        super().validation_step(synthetic_batch, batch_idx)

    @override
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Generate synthetic data on-device and delegate to parent test_step.

        Args:
            batch: Tensor of indices from SyntheticIndexDataset (batch size only)
            batch_idx: Index of current batch
        """
        batch_size = len(batch)
        synthetic_batch = self._generate_synthetic_batch(batch_size, self.device)
        super().test_step(synthetic_batch, batch_idx)

    @override
    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic data on-device and delegate to parent predict_step."""
        batch_size = len(batch)
        synthetic_batch = self._generate_synthetic_batch(batch_size, self.device)
        return super().predict_step(synthetic_batch, batch_idx)

    @override
    def on_validation_epoch_start(self) -> None:
        """Set a fixed random seed for reproducible validation data."""
        torch.manual_seed(42)

    @override
    def on_test_epoch_start(self) -> None:
        """Fixed seed distinct from validation, so test data is never val data."""
        torch.manual_seed(4242)
