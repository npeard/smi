#!/usr/bin/env python

import contextlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as lightning_module
import torch
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from smi.analysis.datamodule import VelocityDataModule
from smi.analysis.lit_module import LitModule
from smi.analysis.synthetic_lit_module import SyntheticIndexDataset, SyntheticLitModule

logger = logging.getLogger(__name__)

SIGNED_RMSE_KEY = 'val/velocity_rmse_um_per_ms'
SIGN_INVARIANT_RMSE_KEY = 'val/velocity_rmse_sign_invariant_um_per_ms'

# Floor on ceil(patience_epochs / check_val_every_n_epoch), the number of
# validation checks patience spans: fewer and a noisy mid-schedule dip can pass
# for a plateau.
MIN_PATIENCE_CHECKS = 5


def convergence_monitor(loss_config: dict[str, Any]) -> str:
    """Validation metric a run must plateau on, decided by the loss.

    A sign-invariant (Michelson) run never drives the signed RMSE down, so
    monitoring it would never converge.

    Args:
        loss_config: The ``loss`` section; ``sign_invariant`` is required.

    Returns:
        The validation metric key to monitor.
    """
    if loss_config['sign_invariant']:
        return SIGN_INVARIANT_RMSE_KEY
    return SIGNED_RMSE_KEY


class RelativeEarlyStopping(EarlyStopping):
    """Early stopping whose improvement threshold is relative to the best value.

    Lightning's ``min_delta`` is absolute, which makes one patience setting
    mean different things at different RMSE scales (noise-free ceilings are an
    order of magnitude below noisy runs). Here a value improves only when it
    beats the best by more than ``min_rel_delta * best``. Minimizing only,
    since the monitored quantity is an RMSE.
    """

    def __init__(
        self, monitor: str, min_rel_delta: float, patience: int, **kwargs: Any
    ):
        super().__init__(monitor=monitor, patience=patience, mode='min', **kwargs)
        self.min_rel_delta = min_rel_delta

    def _evaluate_stopping_criteria(
        self, current: torch.Tensor
    ) -> tuple[bool, str | None]:
        # Lightning stores min_delta pre-negated for 'min' mode and compares
        # monitor_op(current - min_delta, best); the first check has best=inf.
        best = float(self.best_score)
        self.min_delta = -self.min_rel_delta * best if math.isfinite(best) else 0.0
        return super()._evaluate_stopping_criteria(current)


@dataclass
class TrainingConfig:
    """Configuration class for a single training run.

    Hyperparameter search no longer expands list-valued YAML fields into a
    Cartesian product here. Instead, :mod:`smi.analysis.tune_search` interprets
    list-valued fields as a Ray Tune search space and samples a single config
    per trial, which is then turned into a :class:`TrainingConfig`.
    """

    model_config: dict[str, Any]
    training_config: dict[str, Any]
    data_config: dict[str, Any]
    loss_config: dict[str, Any]
    synthetic_config: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, config_path: str) -> 'TrainingConfig':
        """Load a single training configuration from a YAML file.

        List-valued fields (if any) are passed through verbatim; this method no
        longer performs grid expansion. To run a hyperparameter search over
        list-valued fields, use :func:`smi.analysis.tune_search.run_search`.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            A single :class:`TrainingConfig`.
        """
        with Path(config_path).open() as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model_config=config_dict['model'],
            training_config=config_dict['training'],
            loss_config=config_dict['loss'],
            data_config=config_dict['data'],
            synthetic_config=config_dict.get('synthetic'),
        )


class TrainingInterface:
    """Main trainer class for managing model training."""

    CHECKPOINT_DIR = Path(__file__).parent / 'models' / 'checkpoints'

    def __init__(
        self,
        config: TrainingConfig | None = None,
        experiment_name: str | None = None,
        *,
        extra_callbacks: list[Any] | None = None,
    ):
        """Initialize the training interface.

        Args:
            config: Training configuration (None for checkpoint evaluation mode)
            experiment_name: Name for logging and checkpointing
            extra_callbacks: Additional Lightning callbacks to attach to the
                trainer (e.g. Ray Tune's report callback). Defaults to none.
        """
        self.config = config
        self.checkpoint_dir = str(self.CHECKPOINT_DIR)

        # Only setup training components if config is provided
        if config is not None:
            self.experiment_name = experiment_name or config.model_config['type']

            # Create checkpoint directory
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

            # Setup data
            self.setup_data()

            # Create Lightning module
            self.lightning_module = self.create_lightning_module()

            # Setup training
            self.trainer = self.setup_trainer(extra_callbacks=extra_callbacks)

        # Check what version of PyTorch is installed
        logger.info(f'PyTorch version: {torch.__version__}')

        # Check the current CUDA version being used
        logger.info(f'CUDA version: {torch.version.cuda}')

        # Guard on is_available() (not just a CUDA build): on a CUDA-build torch
        # without a GPU/driver -- e.g. CPU-only CI or a Ray worker with no GPU --
        # touching torch.cuda eagerly initializes CUDA and raises "Found no NVIDIA
        # driver". is_available() returns False gracefully in that case.
        if torch.cuda.is_available():
            # Check if CUDA is available and if so, print the device name
            logger.info(f'Device name: {torch.cuda.get_device_properties("cuda").name}')

            # Check if FlashAttention is available
            logger.info(
                f'FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}'
            )

    def _is_synthetic(self) -> bool:
        """Check if synthetic training mode is enabled."""
        return (
            self.config.synthetic_config is not None
            and self.config.synthetic_config.get('use_synthetic_training', False)
        )

    def setup_data(
        self,
        dataset_path: str | None = None,
        batch_size: int | None = None,
        split_ratios: tuple[int, int, int] | None = None,
        num_workers: int | None = None,
    ):
        """Setup data loaders.

        For synthetic training mode, creates dummy dataloaders that provide
        batch indices (actual data is generated on-device in the LitModule).
        For real data, loads HDF5 datasets as before.

        Args:
            dataset_path: Path to dataset file (if None, uses config)
            batch_size: Batch size (if None, uses config)
            split_ratios: Train/val/test split ratios (if None, uses config)
            num_workers: Number of data loader workers (if None, uses config)
        """
        if batch_size is None:
            batch_size = self.config.training_config['batch_size']

        # Synthetic training: create dummy dataloaders (no DataModule).
        if self._is_synthetic():
            self.datamodule = None
            syn = self.config.synthetic_config
            steps = syn['steps_per_epoch']
            val_steps = syn.get('val_steps', steps // 5)

            train_ds = SyntheticIndexDataset(steps * batch_size)
            val_ds = SyntheticIndexDataset(val_steps * batch_size)

            self.train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True
            )
            self.val_loader = DataLoader(val_ds, batch_size=batch_size)
            self.test_loader = self.val_loader
            return

        # Real data: build a Lightning DataModule over the HDF5 dataset.
        base_dir = Path(__file__).parent.parent

        def resolve_path(data_dir: str, filename: str | None = None) -> str:
            """Resolve path relative to project root."""
            data_dir = str(data_dir).lstrip('./')
            abs_dir = base_dir / data_dir
            Path(abs_dir).mkdir(parents=True, exist_ok=True)
            return str(abs_dir / filename) if filename else str(abs_dir)

        if dataset_path is None:
            data_dir = self.config.data_config['data_dir']
            dataset_path = resolve_path(
                data_dir, self.config.data_config['dataset_file']
            )

        if split_ratios is None:
            split_ratios = self.config.data_config['split_ratios']

        if num_workers is None:
            num_workers = self.config.data_config['num_workers']

        self.datamodule = VelocityDataModule(
            dataset_path=dataset_path,
            split_ratios=tuple(split_ratios),
            batch_size=batch_size,
            num_workers=num_workers,
            num_pd_channels=self.config.data_config.get('num_pd_channels', 3),
        )

    def create_lightning_module(self) -> LitModule:
        """Create lightning module based on model type.

        Derives in_channels from data configuration:
        - Synthetic: len(wavelengths_nm)
        - Real data: num_pd_channels (default 3)
        """
        # Derive in_channels from data, not model config
        if self._is_synthetic():
            sim = self.config.synthetic_config['sim']
            self.config.model_config['in_channels'] = len(sim['wavelengths_um'])
        else:
            self.config.model_config['in_channels'] = self.config.data_config.get(
                'num_pd_channels', 3
            )

        # Common optimizer hyperparameters
        optimizer_hparams = {
            'name': self.config.training_config['optimizer'],
            # TODO: why is lr a string?
            'lr': float(self.config.training_config['learning_rate']),
            'momentum': self.config.training_config['momentum'],
            'weight_decay': float(self.config.training_config['weight_decay']),
        }

        # Common scheduler hyperparameters
        warmup_epochs = self.config.training_config['warmup_epochs']
        T_0 = self.config.training_config['T_0']
        T_mult = self.config.training_config['T_mult']

        scheduler_hparams = {
            'warmup_epochs': warmup_epochs,
            'T_0': T_0,
            'T_mult': T_mult,
            'eta_min': self.config.training_config['eta_min'],
        }

        common_kwargs = dict(
            model_hparams=self.config.model_config,
            optimizer_hparams=optimizer_hparams,
            scheduler_hparams=scheduler_hparams,
            loss_hparams=self.config.loss_config,
            training_hparams=self.config.training_config,
            data_hparams=self.config.data_config,
        )

        if self._is_synthetic():
            syn = self.config.synthetic_config
            return SyntheticLitModule(
                **common_kwargs,
                steps_per_epoch=syn['steps_per_epoch'],
                sim=syn['sim'],
                train_crop_length=syn['train_crop_length'],
                restoring_checkpoint=False,
            )

        return LitModule(**common_kwargs)

    def setup_trainer(
        self, *, extra_callbacks: list[Any] | None = None
    ) -> lightning_module.Trainer:
        """Setup Lightning trainer with callbacks and loggers.

        Args:
            extra_callbacks: Additional callbacks appended after the default
                logging callbacks (e.g. Ray Tune's report callback).
        """
        # Required, no default: with patience it fixes how many checks a plateau
        # needs, so it changes what "converged" means.
        check_val_every_n_epoch = self.config.training_config['check_val_every_n_epoch']
        callbacks = [self._convergence_callback(check_val_every_n_epoch)]
        # Best-by-monitor on every path, logging or not: a converged run sits up
        # to ``patience`` epochs past its best, so final weights would bias the
        # reported ceiling. Same metric as the convergence monitor by design.
        checkpoint_prefix = ''
        # Add WandB logger if configured
        if self.config.training_config['use_logging']:
            loggers = [
                WandbLogger(
                    project=self.config.training_config['wandb_project'],
                    name=self.experiment_name,
                    save_dir=self.checkpoint_dir,
                )
            ]
            callbacks.append(LearningRateMonitor())
            checkpoint_prefix = f'{loggers[0].experiment.id}_'
        else:
            loggers = []
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=Path(self.checkpoint_dir) / self.experiment_name,
            filename=checkpoint_prefix + '{epoch}',
            monitor=self.convergence_callback.monitor,
            mode='min',
            save_top_k=1,
        )
        callbacks.append(self.checkpoint_callback)

        # Append any caller-supplied callbacks (e.g. Ray Tune's reporter).
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        # Get accelerator and device settings from config
        accelerator = self.config.training_config['accelerator']
        devices = self.config.training_config['devices']
        # Required, no default: precision changes results, so it must be stated.
        precision = self.config.training_config['precision']

        # Convert devices to proper type if it's a string
        if isinstance(devices, str):
            with contextlib.suppress(ValueError):
                devices = int(devices)

        return lightning_module.Trainer(
            max_epochs=self.config.training_config['max_epochs'],
            callbacks=callbacks,
            logger=loggers,
            check_val_every_n_epoch=check_val_every_n_epoch,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
        )

    def _convergence_callback(
        self, check_val_every_n_epoch: int
    ) -> RelativeEarlyStopping:
        """Build the plateau-stopping callback from ``training.convergence``.

        ``training.max_epochs`` stays the single epoch cap; the convergence
        block only says when to stop sooner.

        Args:
            check_val_every_n_epoch: Validation cadence, to convert the
                epoch-denominated patience into validation checks.

        Raises:
            ValueError: If the configured monitor does not match the loss, or
                patience spans fewer than MIN_PATIENCE_CHECKS validation checks.
        """
        convergence = self.config.training_config['convergence']
        monitor = convergence['monitor']
        expected = convergence_monitor(self.config.loss_config)
        if monitor != expected:
            raise ValueError(
                f'training.convergence.monitor is {monitor!r} but '
                f'loss.sign_invariant={self.config.loss_config["sign_invariant"]} '
                f'requires {expected!r}'
            )
        checks = math.ceil(convergence['patience_epochs'] / check_val_every_n_epoch)
        if checks < MIN_PATIENCE_CHECKS:
            raise ValueError(
                f'patience_epochs={convergence["patience_epochs"]} with '
                f'check_val_every_n_epoch={check_val_every_n_epoch} is {checks} '
                f'validation checks; a plateau needs patience to span at least '
                f'{MIN_PATIENCE_CHECKS} validation checks'
            )
        self.convergence_callback = RelativeEarlyStopping(
            monitor=monitor,
            min_rel_delta=float(convergence['min_rel_delta']),
            patience=checks,
        )
        return self.convergence_callback

    @property
    def converged(self) -> bool:
        """Whether the last :meth:`train` stopped on a plateau (not max_epochs).

        False for a run that hit ``training.max_epochs`` or whose monitored
        metric went non-finite. Pass it to ``evaluate(..., converged=...)``.
        """
        return (
            self.convergence_callback.stopping_reason
            == EarlyStoppingReason.PATIENCE_EXHAUSTED
        )

    @property
    def best_model_path(self) -> str:
        """Checkpoint with the lowest monitored value (empty if none was finite)."""
        return self.checkpoint_callback.best_model_path

    @property
    def best_monitor(self) -> float:
        """Lowest monitored validation value seen (NaN if none was finite)."""
        score = self.checkpoint_callback.best_model_score
        return math.nan if score is None else float(score)

    @property
    def final_monitor(self) -> float:
        """Monitored validation value at the last validation check."""
        return float(self.trainer.callback_metrics[self.convergence_callback.monitor])

    @property
    def epochs_run(self) -> int:
        """Epochs the last :meth:`train` completed."""
        return self.trainer.current_epoch

    def train(self) -> bool:
        """Train the model.

        Returns:
            :attr:`converged` for this run.
        """
        # A second fit must not report the first run's stop reason.
        self.convergence_callback.stopping_reason = EarlyStoppingReason.NOT_STOPPED
        self.convergence_callback.wait_count = 0
        self.convergence_callback.best_score = torch.tensor(math.inf)
        self.convergence_callback.stopped_epoch = 0
        # Real data uses the DataModule; synthetic uses index dataloaders.
        if getattr(self, 'datamodule', None) is not None:
            self.trainer.fit(self.lightning_module, datamodule=self.datamodule)
        else:
            self.trainer.fit(
                self.lightning_module,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader,
            )
        return self.converged

    def test(self):
        """Test the model using best checkpoint."""
        # Lightning always installs a default ModelCheckpoint, so its presence
        # does not mean a best checkpoint exists (it has none without a monitor
        # or when logging is off); test the in-memory weights in that case.
        has_best = any(
            isinstance(cb, ModelCheckpoint) and cb.best_model_path
            for cb in self.trainer.callbacks
        )
        ckpt_path = 'best' if has_best else None

        if getattr(self, 'datamodule', None) is not None:
            self.trainer.test(
                self.lightning_module, datamodule=self.datamodule, ckpt_path=ckpt_path
            )
        elif hasattr(self, 'test_loader'):
            self.trainer.test(
                self.lightning_module, dataloaders=self.test_loader, ckpt_path=ckpt_path
            )
