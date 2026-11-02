#!/usr/bin/env python

import logging
from typing import Any, cast, override

import lightning as lightning_module
import torch
from torch import nn, optim

# Conditional import for Muon optimizer
try:
    from muon import Muon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam  # ty: ignore[unresolved-import]  # optional dependency

    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    Muon: type | None = None
    MuonWithAuxAdam: type | None = None
    SingleDeviceMuonWithAuxAdam: type | None = None

from smi.core.kinematics import derivative_displacement, integrate_velocity
from smi.core.metrics import apply_best_sign, best_sign, velocity_rmse
from smi.data.features import default_registry
from smi.data.features.normalization import NORMALIZATION_STATS
from smi.learning.models.base import Model, SequenceModel
from smi.learning.models.factory import create_model

logger = logging.getLogger(__name__)


class LitModule(lightning_module.LightningModule):
    """Lightning Module for fusing multiple sensor channels into velocity predictions.
    It supports both static and dynamic loss weighting between velocity and displacement
    loss terms for improved training stability.
    """

    def __init__(
        self,
        training_hparams: dict,
        sample_rate: float,
        model_hparams: dict | None = None,
        optimizer_hparams: dict | None = None,
        scheduler_hparams: dict | None = None,
        loss_hparams: dict | None = None,
        data_hparams: dict | None = None,
    ):
        """Initialize LitModule with hyperparameter configuration.

        Args:
        training_hparams: Hyperparameters for training configuration; ``target``
            is required (no default)
        sample_rate: Sample rate (Hz) of the data the module sees; the loss
            integrates and differentiates at it
        model_hparams: Hyperparameters for the model
        optimizer_hparams: Hyperparameters for the optimizer
        scheduler_hparams: Hyperparameters for the learning rate scheduler
        loss_hparams: Hyperparameters for the loss function
        data_hparams: Hyperparameters for data configuration (for logging only)
        """
        super().__init__()
        self.model_hparams = model_hparams
        self.optimizer_hparams = optimizer_hparams
        self.scheduler_hparams = scheduler_hparams
        self.loss_hparams = loss_hparams
        self.training_hparams = training_hparams
        self.data_hparams = data_hparams
        self.sample_rate = sample_rate

        # Determine what the model should target (needed before wrapping, since
        # the normalization wrapper scales the output by the target's stats).
        self.target = training_hparams['target']
        if self.target not in ['velocity', 'displacement']:
            raise ValueError(
                f"target must be 'velocity' or 'displacement', got '{self.target}'"
            )

        # Build the inner network and wrap it so input/output normalization is
        # baked into buffers; the wrapper's forward stays in raw physical units.
        inner = create_model(model_hparams)
        self._inner_model = inner
        self.model = self._make_model(inner)

        # Initialize dynamic loss weights if enabled
        if 'dynamic' not in self.loss_hparams:
            raise KeyError("'dynamic' key must be specified in loss_hparams config")
        self.dynamic_weighting = self.loss_hparams['dynamic']
        # Required, no default: whether the loss forgives a per-shot global sign.
        self.sign_invariant = bool(self.loss_hparams['sign_invariant'])

        if self.dynamic_weighting:
            # Initialize learnable weight parameters for homoscedastic uncertainty
            # Start with reasonable initial values (e.g., 1.0)
            self.log_weight_velocity = nn.Parameter(torch.tensor(0.0))  # log(1.0) = 0.0
            self.log_weight_displacement = nn.Parameter(
                torch.tensor(0.0)
            )  # log(1.0) = 0.0

        torch.set_float32_matmul_precision('high')

        # Include model size metrics in logged hyperparameters
        total_params = getattr(self._inner_model, 'total_params', None)
        receptive_field = inner.receptive_field
        if total_params is not None:
            model_hparams = {**(model_hparams or {}), 'total_params': int(total_params)}
        if receptive_field is not None:
            model_hparams = {
                **(model_hparams or {}),
                'receptive_field': int(receptive_field),
            }
        self.model_hparams = model_hparams
        self.save_hyperparameters(ignore=['model'])

    @property
    def inner_model(self) -> SequenceModel:
        """The bare architecture inside the normalizing wrapper."""
        return self._inner_model

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

    def velocity_from_output(self, prediction: torch.Tensor) -> torch.Tensor:
        """Recover predicted velocity (um/ms) from the model's output.

        When the model targets displacement the velocity is differentiated
        back out, using the same conversion ``loss_function`` and
        ``predict_step`` use, so the logged RMSE describes exactly the
        quantity those report. Public because ``evaluate`` scores a
        checkpoint's output through it.

        Args:
            prediction: Model output, velocity or displacement per
                ``self.target``.

        Returns:
            Predicted velocity in um/ms.
        """
        if self.target == 'velocity':
            return prediction
        return cast(torch.Tensor, derivative_displacement(prediction, self.sample_rate))

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
        sample_rate = self.sample_rate

        if self.target == 'velocity':
            # Model predicts velocity
            velocity_hat = prediction
            # Auxiliary loss: displacement (physics-informed, derived from velocity)
            displacement_hat = integrate_velocity(velocity_hat, sample_rate)
            # Shift displacements to start at 0 for fair comparison
            displacement_hat = displacement_hat - displacement_hat[:, :1]
            displacement_target = displacement_target - displacement_target[:, :1]
        elif self.target == 'displacement':
            # Model predicts displacement
            displacement_hat = prediction
            # Shift displacements to start at 0 for fair comparison
            displacement_hat = displacement_hat - displacement_hat[:, :1]
            displacement_target = displacement_target - displacement_target[:, :1]
            # Auxiliary loss: velocity (physics-informed, derived from displacement)
            velocity_hat = derivative_displacement(displacement_hat, sample_rate)
        else:
            raise ValueError(f'Unknown target: {self.target}')

        if self.sign_invariant:
            trained = (
                (velocity_hat, velocity_target)
                if self.target == 'velocity'
                else (displacement_hat, displacement_target)
            )
            velocity_hat, displacement_hat = apply_best_sign(
                velocity_hat, displacement_hat, *trained
            )

        # No unit conversion here. Velocity is um/ms and displacement is um as
        # stored, converted once at the CoilDriver boundary, so the logged
        # velocity loss is the MSE of exactly the quantities predict_step
        # reports. A rescale at this site is what previously made the training
        # log and the reported baseline differ by 1000x.

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

        # Log each loss component with train_ prefix, without prog_bar: a
        # progress-bar value is read back with .item() every batch, a
        # host-device sync on every step.
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train/{loss_name}_loss', loss_value, on_epoch=True)

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

        # Log each loss component with val_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'val/{loss_name}_loss',
                loss_value,
                prog_bar=(loss_name == 'total'),  # Only show total loss in progress bar
                sync_dist=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )

        # RMSE in stored units, so a run is directly comparable to the
        # analytic baseline without a separate evaluation pass. With
        # loss.sign_invariant=True, val/velocity_loss is sign-invariant but this
        # signed RMSE deliberately is not: its collapse near std(v) is the
        # visible symptom of a Michelson model that cannot see direction.
        velocity_hat = self.velocity_from_output(prediction)
        self.log(
            'val/velocity_rmse_um_per_ms',
            velocity_rmse(velocity_hat, velocity_target),
            sync_dist=torch.distributed.is_available()
            and torch.distributed.is_initialized(),
        )
        # Logged beside the signed RMSE so a sign collapse is visible in the
        # curves: signed stays near std(v) while this one keeps falling.
        self.log(
            'val/velocity_rmse_sign_invariant_um_per_ms',
            velocity_rmse(
                best_sign(velocity_hat, velocity_target) * velocity_hat, velocity_target
            ),
            sync_dist=torch.distributed.is_available()
            and torch.distributed.is_initialized(),
        )
        # The value of both RMSEs above for a zero output, reduced over the
        # epoch the same way, so a run stalled at the zero-output plateau can be
        # told from a converged one (TrainingInterface.converged).
        self.log(
            'val/velocity_rmse_zero_predictor_um_per_ms',
            velocity_rmse(torch.zeros_like(velocity_hat), velocity_target),
            sync_dist=torch.distributed.is_available()
            and torch.distributed.is_initialized(),
        )
        if self.target == 'displacement':
            self._log_displacement_rmse(prediction, displacement_target)

    def _log_displacement_rmse(
        self, displacement_hat: torch.Tensor, displacement_target: torch.Tensor
    ) -> None:
        """Log a displacement target's own monitor keys, start offsets removed.

        A displacement-target run is monitored on these, not on its velocity
        (see ``training_interface.convergence_monitor``). Same three variants
        and reduction as the velocity keys; ``velocity_rmse`` is a plain RMSE,
        here in um.
        """
        d_hat = displacement_hat - displacement_hat[:, :1]
        d = displacement_target - displacement_target[:, :1]
        sync = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.log('val/displacement_rmse_um', velocity_rmse(d_hat, d), sync_dist=sync)
        self.log(
            'val/displacement_rmse_sign_invariant_um',
            velocity_rmse(best_sign(d_hat, d) * d_hat, d),
            sync_dist=sync,
        )
        self.log(
            'val/displacement_rmse_zero_predictor_um',
            velocity_rmse(torch.zeros_like(d), d),
            sync_dist=sync,
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

        # Log each loss component with test_ prefix
        for loss_name, loss_value in loss_dict.items():
            self.log(
                f'test/{loss_name}_loss',
                loss_value,
                sync_dist=torch.distributed.is_available()
                and torch.distributed.is_initialized(),
            )

        self.log(
            'test/velocity_rmse_um_per_ms',
            velocity_rmse(self.velocity_from_output(prediction), velocity_target),
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
        sample_rate = self.sample_rate

        # Derive both velocity and displacement for plotting, regardless of target
        if self.target == 'velocity':
            velocity_hat = prediction
            # CoilDriver returns ndarray | Tensor; the tensor input path always
            # yields a Tensor here.
            displacement_hat = cast(
                torch.Tensor, integrate_velocity(velocity_hat, sample_rate=sample_rate)
            )
        else:  # self.target == 'displacement'
            displacement_hat = prediction
            velocity_hat = cast(
                torch.Tensor, derivative_displacement(displacement_hat, sample_rate)
            )

        # Shift all displacements to start at zero for easy comparison
        displacement_hat = displacement_hat - displacement_hat[:, :1]
        displacement_target = displacement_target - displacement_target[:, :1]

        return (
            velocity_hat,
            velocity_target,
            displacement_hat,
            displacement_target,
            signals,
        )
