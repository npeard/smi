#!/usr/bin/env python

import logging
import sys
from typing import Any, cast, override

import lightning as lightning_module
import torch
from torch import nn, optim

# Conditional import for Muon optimizer
try:
    from muon import (  # ty: ignore[unresolved-import]  # optional dependency
        Muon,
        MuonWithAuxAdam,
        SingleDeviceMuonWithAuxAdam,
    )

    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    Muon: type | None = None
    MuonWithAuxAdam: type | None = None
    SingleDeviceMuonWithAuxAdam: type | None = None

from smi.analysis.features import default_registry
from smi.analysis.features.normalization import NORMALIZATION_STATS
from smi.analysis.models.base import Model
from smi.analysis.models.factory import create_model
from smi.analysis.models.vicreg import VicRegLoss
from smi.redpitaya.redpitaya_config import RedPitayaConfig
from smi.synthetic.coil_driver import CoilDriver

logger = logging.getLogger(__name__)


class LitModule(lightning_module.LightningModule):
    """Lightning Module for fusing multiple sensor channels into velocity predictions.
    It supports both static and dynamic loss weighting between velocity and displacement
    loss terms for improved training stability.
    """

    def __init__(
        self,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
        training_hparams: dict | None = None,
        data_hparams: dict | None = None,
    ):
        """Initialize LitModule with hyperparameter configuration.

        Args:
        model_hparams: Hyperparameters for the model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        training_hparams: Hyperparameters for training configuration (includes target)
        data_hparams: Hyperparameters for data configuration (for logging only)
        """
        super().__init__()
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.loss_hparams = loss_hparams
        self.training_hparams = training_hparams
        self.data_hparams = data_hparams

        # Determine what the model should target (needed before wrapping, since
        # the normalization wrapper scales the output by the target's stats).
        if training_hparams and 'target' in training_hparams:
            self.target = training_hparams['target']
            if self.target not in ['velocity', 'displacement']:
                raise ValueError(
                    f"target must be 'velocity' or 'displacement', got '{self.target}'"
                )
        else:
            # Default to velocity for backward compatibility
            self.target = 'velocity'
            logger.warning(
                "No 'target' specified in training config, defaulting to 'velocity'"
            )

        # Build the inner network and wrap it so input/output normalization is
        # baked into buffers; the wrapper's forward stays in raw physical units.
        inner = create_model(model_hparams)
        self._inner_model = inner
        self.model = self._make_model(inner)

        # Torch compilation (requires PyTorch 2+, Python < 3.12, and GPU)
        torch_major = int(torch.__version__.split('.')[0])
        if (
            torch_major >= 2
            and sys.version_info < (3, 12)
            and torch.cuda.is_available()
        ):
            self.model = torch.compile(self.model)
            # Note: torch.compile may add prefixes to parameter names, which can cause
            # issues with state_dict loading. See notebooks/prediction.py for an example
            # of how to handle this.

        # Initialize dynamic loss weights if enabled
        if 'dynamic' not in self.loss_hparams:
            raise KeyError("'dynamic' key must be specified in loss_hparams config")
        self.dynamic_weighting = self.loss_hparams['dynamic']

        if self.dynamic_weighting:
            # Initialize learnable weight parameters for homoscedastic uncertainty
            # Start with reasonable initial values (e.g., 1.0)
            self.log_weight_velocity = nn.Parameter(torch.tensor(0.0))  # log(1.0) = 0.0
            self.log_weight_displacement = nn.Parameter(
                torch.tensor(0.0)
            )  # log(1.0) = 0.0

        torch.set_float32_matmul_precision('high')

        # VICReg: only instantiate when weight > 0 and the inner model supports
        # encode() (the wrapper delegates encode() to the inner model).
        self._vicreg_weight = float(loss_hparams.get('vicreg_weight', 0.0))
        if self._vicreg_weight > 0 and hasattr(self._inner_model, 'encode'):
            self.vicreg_loss = VicRegLoss(
                sim_coeff=float(loss_hparams.get('vicreg_sim_coeff', 25.0)),
                std_coeff=float(loss_hparams.get('vicreg_std_coeff', 25.0)),
                cov_coeff=float(loss_hparams.get('vicreg_cov_coeff', 1.0)),
            )
        else:
            self.vicreg_loss = None

        # Include model size metrics in logged hyperparameters
        total_params = getattr(self._inner_model, 'total_params', None)
        receptive_field = getattr(self._inner_model, 'receptive_field', None)
        if total_params is not None:
            model_hparams = {**(model_hparams or {}), 'total_params': int(total_params)}
        if receptive_field is not None:
            model_hparams = {
                **(model_hparams or {}),
                'receptive_field': int(receptive_field),
            }
        self.model_hparams = model_hparams
        self.save_hyperparameters(ignore=['model'])

    def _make_model(self, inner: nn.Module) -> Model:
        """Wrap the inner network with baked input/output normalization.

        Uses the version-controlled per-feature stats: the first ``in_channels``
        photodiode input features scale the inputs and the target feature scales
        the output. Subclasses with a different input scale (e.g. on-device
        synthetic data) may override this.
        """
        in_channels = self.model_hparams['in_channels']
        input_names = default_registry.input_features()[:in_channels]
        return Model.from_registry_stats(
            inner, input_names, self.target, NORMALIZATION_STATS
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        All models share a unified seq2seq interface: given input of shape
        [batch_size, num_channels, signal_length] they return
        [batch_size, signal_length].

        Args:
            x: Input tensor of shape [batch_size, num_channels, signal_length]

        Returns:
            Tensor of shape [batch_size, signal_length] containing model predictions
            (velocity or displacement depending on self.target)
        """
        with torch.set_grad_enabled(self.training):
            # All models return [batch_size, 1, signal_length]
            output = self.model(x)
        return output.squeeze(1)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        # Make a copy to avoid modifying the original hparams dict
        optimizer_hparams = self.optimizer_hparams.copy()

        # Configure optimizer
        optimizer_name = optimizer_hparams.pop('name')
        if optimizer_name == 'Adam':
            # Discard momentum hyperparameter for Adam optimizer
            _ = optimizer_hparams.pop('momentum', None)
            optimizer = optim.AdamW(self.parameters(), **optimizer_hparams)
        elif optimizer_name == 'SGD':
            # Discard weight_decay hyperparameter for SGD optimizer
            _ = optimizer_hparams.pop('weight_decay', None)
            optimizer = optim.SGD(self.parameters(), **optimizer_hparams)
        elif optimizer_name == 'Muon':
            if not MUON_AVAILABLE:
                raise ImportError(
                    'Muon optimizer requires the muon-optimizer library. '
                    'Install it with: pip install muon-optimizer'
                )
            # Discard momentum hyperparameter for Muon optimizer
            _ = optimizer_hparams.pop('momentum', None)

            # Muon uses MuonWithAuxAdam which applies:
            # - Muon to hidden layer weights (parameters with ndim >= 2)
            # - AdamW to other parameters (biases, norms, etc.)

            # Separate parameters by dimensionality
            # For models with body/head structure, handle all model parameters
            hidden_weights = [p for p in self.model.parameters() if p.ndim >= 2]
            other_params = [p for p in self.model.parameters() if p.ndim < 2]

            # If dynamic weighting is enabled, add the dynamic weight parameters
            # These should be optimized with AdamW (not Muon) since they're scalars
            if self.dynamic_weighting:
                other_params.extend(
                    [self.log_weight_velocity, self.log_weight_displacement]
                )

            # Get hyperparameters
            lr = optimizer_hparams.pop('lr')
            weight_decay = float(optimizer_hparams.pop('weight_decay'))

            # Configure parameter groups
            # Muon for hidden weights, AdamW for biases/norms
            if len(hidden_weights) > 0:
                logger.info('Muon optimizer has found hidden weights')
            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=True,
                    lr=lr,
                    weight_decay=weight_decay,
                ),
                dict(
                    params=other_params,
                    use_muon=False,
                    lr=lr * 0.1,  # Typically use lower LR for non-Muon params
                    betas=(0.9, 0.95),
                    weight_decay=weight_decay,
                ),
            ]

            # Use SingleDeviceMuonWithAuxAdam for single GPU/CPU training
            # Use MuonWithAuxAdam for distributed training
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # Guarded by the MUON_AVAILABLE check above, so not None here.
                optimizer = MuonWithAuxAdam(  # ty: ignore[call-non-callable]
                    param_groups, **optimizer_hparams
                )
            else:
                # Guarded by the MUON_AVAILABLE check above, so not None here.
                optimizer = SingleDeviceMuonWithAuxAdam(  # ty: ignore[call-non-callable]
                    param_groups, **optimizer_hparams
                )
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')

        # Configure multi-stage scheduler: linear warmup then cosine annealing
        warmup_epochs = self.scheduler_hparams['warmup_epochs']
        eta_min = self.scheduler_hparams['eta_min']
        T_0 = self.scheduler_hparams['T_0']
        T_mult = self.scheduler_hparams['T_mult']

        # LambdaLR for linear warmup
        def warmup_lambda(epoch: int) -> float:
            if warmup_epochs == 0:
                return 1.0
            return float(epoch + 1) / float(warmup_epochs)

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=float(eta_min)
        )

        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}

    def loss_function(
        self,
        prediction: torch.Tensor,
        velocity_target: torch.Tensor,
        displacement_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Custom loss function that returns a dictionary of loss components.

        Supports two modes based on self.target:
        - 'velocity': Model predicts velocity, displacement loss is auxiliary
          (physics-informed)
        - 'displacement': Model predicts displacement, velocity loss is not used

        Args:
            prediction: Model predictions (velocity or displacement based on
                self.target)
            velocity_target: Ground truth velocity
            displacement_target: Ground truth displacement

        Returns:
            Dictionary of loss components with keys representing loss types
            and values representing the corresponding loss tensors.
        """
        sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256

        if self.target == 'velocity':
            # Model predicts velocity
            velocity_hat = prediction
            # Auxiliary loss: displacement (physics-informed, derived from velocity)
            displacement_hat = CoilDriver.integrate_velocity(velocity_hat, sample_rate)
            # Shift displacements to start at 0 for fair comparison
            displacement_hat -= displacement_hat[:, 0:1]
            displacement_target -= displacement_target[:, 0:1]
        elif self.target == 'displacement':
            # Model predicts displacement
            displacement_hat = prediction
            # Shift displacements to start at 0 for fair comparison
            displacement_hat -= displacement_hat[:, 0:1]
            displacement_target -= displacement_target[:, 0:1]
            # Auxiliary loss: velocity (physics-informed, derived from displacement)
            velocity_hat = CoilDriver.derivative_displacement(
                displacement_hat, sample_rate
            )
        else:
            raise ValueError(f'Unknown target: {self.target}')

        # Convert velocity to um/ms instead of um/s stored in Dataset
        velocity_hat *= 1e-3
        velocity_target *= 1e-3

        # Velocity Loss
        velocity_loss = nn.MSELoss()(velocity_hat, velocity_target)

        # Displacement Loss
        displacement_loss = nn.MSELoss()(displacement_hat, displacement_target)

        # Create loss dictionary
        loss_dict = {
            'velocity': velocity_loss,
            'displacement': displacement_loss,
            'total_unweighted': velocity_loss + displacement_loss,
        }

        # Weighted total loss
        if self.dynamic_weighting:
            weight_velocity = torch.exp(self.log_weight_velocity)
            weight_displacement = torch.exp(self.log_weight_displacement)

            loss_dict['total'] = (
                (1.0 / weight_velocity**2) * velocity_loss
                + (1.0 / weight_displacement**2) * displacement_loss
                + torch.log(weight_velocity * weight_displacement)
            )
            loss_dict['weight_velocity'] = weight_velocity
            loss_dict['weight_displacement'] = weight_displacement
        else:
            loss_dict['total'] = (
                self.loss_hparams['velocity_loss_weight'] * velocity_loss
                + self.loss_hparams['displacement_loss_weight'] * displacement_loss
            )

        return loss_dict

    def _compute_vicreg_loss(self, signals: torch.Tensor) -> torch.Tensor | None:
        """Compute pairwise VICReg loss over siamese encoder outputs.

        Returns None (no-op) when VICReg is disabled or the model has no
        ``encode`` method, so callers can guard cheaply with ``is not None``.

        Args:
            signals: [batch, in_channels, seq_len]

        Returns:
            Scalar VICReg loss, or None.
        """
        if self.vicreg_loss is None:
            return None
        _, channel_features = self.model.encode(signals)
        vicreg = torch.tensor(0.0, device=signals.device)
        n = len(channel_features)
        for i in range(n):
            for j in range(i + 1, n):
                # Pool over time: [batch, embed_dim, seq_len] -> [batch, embed_dim]
                fi = channel_features[i].mean(dim=-1)
                fj = channel_features[j].mean(dim=-1)
                vicreg = vicreg + self.vicreg_loss(fi, fj)
        return vicreg

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch

        Returns:
            Total loss value for backpropagation
        """
        signals, velocity_target, displacement_target = batch
        prediction = self(signals)  # Model prediction (velocity or displacement)

        loss_dict = self.loss_function(prediction, velocity_target, displacement_target)

        vicreg = self._compute_vicreg_loss(signals)
        if vicreg is not None:
            loss_dict['vicreg'] = vicreg
            loss_dict['total'] = loss_dict['total'] + self._vicreg_weight * vicreg

        # Log each loss component with train_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'train/{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                on_epoch=True,
            )

        # Return the total loss for backpropagation
        return loss_dict['total']

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch
        """
        signals, velocity_target, displacement_target = batch
        prediction = self(signals)  # Model prediction (velocity or displacement)

        loss_dict = self.loss_function(prediction, velocity_target, displacement_target)

        vicreg = self._compute_vicreg_loss(signals)
        if vicreg is not None:
            loss_dict['vicreg'] = vicreg
            loss_dict['total'] = loss_dict['total'] + self._vicreg_weight * vicreg

        # Log each loss component with val_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'val/{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                sync_dist=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )

    @override
    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Test step for model.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch
        """
        signals, velocity_target, displacement_target = batch
        prediction = self(signals)  # Model prediction (velocity or displacement)

        loss_dict = self.loss_function(prediction, velocity_target, displacement_target)

        vicreg = self._compute_vicreg_loss(signals)
        if vicreg is not None:
            loss_dict['vicreg'] = vicreg
            loss_dict['total'] = loss_dict['total'] + self._vicreg_weight * vicreg

        # Log each loss component with test_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'test/{loss_name}_loss',
                loss_value,
                sync_dist=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )

    @override
    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prediction step for model. Return all relevant quantities for plotting.

        Args:
            batch: Tuple of (signals, velocity_target, displacement_target)
            batch_idx: Index of current batch
            dataloader_idx: Index of dataloader

        Returns:
            Tuple of (velocity_hat, velocity_target, displacement_hat,
                        displacement_target, signals)
        """
        signals, velocity_target, displacement_target = batch

        prediction = self(signals)  # Model prediction (velocity or displacement)
        sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256

        # Derive both velocity and displacement for plotting, regardless of target
        if self.target == 'velocity':
            velocity_hat = prediction
            # CoilDriver returns ndarray | Tensor; the tensor input path always
            # yields a Tensor here.
            displacement_hat = cast(
                torch.Tensor,
                CoilDriver.integrate_velocity(velocity_hat, sample_rate=sample_rate),
            )
        else:  # self.target == 'displacement'
            displacement_hat = prediction
            velocity_hat = cast(
                torch.Tensor,
                CoilDriver.derivative_displacement(displacement_hat, sample_rate),
            )

        # Shift all displacements to start at zero for easy comparison
        displacement_hat -= displacement_hat[:, 0:1]
        displacement_target -= displacement_target[:, 0:1]

        return (
            velocity_hat,
            velocity_target,
            displacement_hat,
            displacement_target,
            signals,
        )
