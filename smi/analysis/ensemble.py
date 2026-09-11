#!/usr/bin/env python

"""vmap ensemble of same-architecture models for within-architecture sweeps.

This module trains ``K`` models of the *same* architecture simultaneously on a
single shared minibatch using ``torch.func``. It is the inner-level
parallelization strategy of the search pipeline (see
``docs/hyperparameter-search-pipeline-design.md`` section 4, component 4): a
maximally GPU-efficient way to fan out the within-architecture axis (seed/init,
loss weights, dropout, and -- as a stretch -- per-member learning rate) because
all members share one CUDA context, one data generation, and one set of batched
kernel launches.

Design summary:

- Each member is a full ``Model``-wrapped network built by :func:`create_model`
  with a *different seed*, so initializations differ. Loss weights and dropout
  can also vary per member; the architecture is identical across members.
- ``stack_module_state`` produces new stacked parameter/buffer tensors with a
  leading ``K`` dim. Those stacked tensors -- *not* the original ``K`` modules --
  must be the optimizer leaves for gradients to flow, so the stacked params are
  registered as trainable :class:`torch.nn.Parameter` leaves (inside an
  :class:`torch.nn.ParameterDict`) and the stacked buffers as registered
  buffers.
- A meta-device ``base`` module (a deep copy of member 0 moved to ``meta``)
  carries the architecture; the forward is
  ``functional_call(base, (params, buffers), (x,))`` vmapped over the leading
  ``K`` dim with the shared batch broadcast (``in_dims=(0, 0, None)``).
"""

import copy
import logging
from typing import Any, override

import lightning as lightning_module
import torch
from torch import Tensor, nn
from torch.func import functional_call, stack_module_state

from smi.analysis.models.base import Model
from smi.analysis.models.factory import create_model
from smi.analysis.synthetic_lit_module import SyntheticLitModule
from smi.redpitaya.redpitaya_config import RedPitayaConfig
from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.waveform import Waveform

logger = logging.getLogger(__name__)


def physics_loss(
    prediction: Tensor,
    velocity_target: Tensor,
    displacement_target: Tensor,
    target: str,
    velocity_loss_weight: float,
    displacement_loss_weight: float,
) -> dict[str, Tensor]:
    """Physics-informed velocity/displacement loss for a single member.

    This is the static, ``self``-free form of
    :meth:`smi.analysis.lit_module.LitModule.loss_function` (static loss
    weighting only -- dynamic homoscedastic weighting is not used by the
    ensemble). It is reused per member so the ensemble matches the project's
    training objective exactly.

    Args:
        prediction: Model prediction ``[B, L]`` (velocity or displacement
            depending on ``target``).
        velocity_target: Ground-truth velocity ``[B, L]``.
        displacement_target: Ground-truth displacement ``[B, L]``.
        target: Either ``'velocity'`` or ``'displacement'``.
        velocity_loss_weight: Weight on the velocity MSE term.
        displacement_loss_weight: Weight on the displacement MSE term.

    Returns:
        Dict with keys ``'velocity'``, ``'displacement'``, ``'total_unweighted'``
        and ``'total'`` (the weighted sum used for backprop).
    """
    sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256

    if target == 'velocity':
        velocity_hat = prediction
        displacement_hat = CoilDriver.integrate_velocity(velocity_hat, sample_rate)
        displacement_hat = displacement_hat - displacement_hat[:, 0:1]
        displacement_target = displacement_target - displacement_target[:, 0:1]
    elif target == 'displacement':
        displacement_hat = prediction
        displacement_hat = displacement_hat - displacement_hat[:, 0:1]
        displacement_target = displacement_target - displacement_target[:, 0:1]
        velocity_hat = CoilDriver.derivative_displacement(displacement_hat, sample_rate)
    else:
        raise ValueError(f'Unknown target: {target}')

    # Convert velocity to um/ms instead of um/s
    velocity_hat = velocity_hat * 1e-3
    velocity_target = velocity_target * 1e-3

    velocity_loss = nn.functional.mse_loss(velocity_hat, velocity_target)
    displacement_loss = nn.functional.mse_loss(displacement_hat, displacement_target)

    total = (
        velocity_loss_weight * velocity_loss
        + displacement_loss_weight * displacement_loss
    )
    return {
        'velocity': velocity_loss,
        'displacement': displacement_loss,
        'total_unweighted': velocity_loss + displacement_loss,
        'total': total,
    }


def _set_dropout(module: nn.Module, p: float) -> None:
    """Set the probability of every dropout layer in ``module`` to ``p``."""
    for sub in module.modules():
        if isinstance(sub, nn.modules.dropout._DropoutNd):
            sub.p = p


def _sanitize(name: str) -> str:
    """Map a dotted parameter name to a valid ``ParameterDict`` key.

    ``ParameterDict`` keys may not contain ``.`` (it would be parsed as nested
    module access), so dots are replaced with a sentinel that round-trips via
    :func:`_desanitize`.
    """
    return name.replace('.', '__dot__')


def _desanitize(name: str) -> str:
    """Inverse of :func:`_sanitize`."""
    return name.replace('__dot__', '.')


def build_member(
    model_hparams: dict[str, Any],
    seed: int,
    dropout: float | None = None,
    in_channels: int | None = None,
) -> Model:
    """Build one ``Model``-wrapped member with a deterministic seed.

    Args:
        model_hparams: Hyperparameters passed to :func:`create_model`. The
            architecture must be identical across all members of an ensemble.
        seed: Seed used to make this member's initialization distinct.
        dropout: If given, overrides the dropout probability of every dropout
            layer in the inner network (lets dropout vary per member).
        in_channels: Number of input channels for the identity normalization
            wrapper; defaults to ``model_hparams['in_channels']``.

    Returns:
        A ``Model``-wrapped network with identity (synthetic-scale)
        normalization, matching the synthetic training path.
    """
    torch.manual_seed(seed)
    inner = create_model(model_hparams)
    if inner is None:
        raise ValueError(f'create_model returned None for hparams: {model_hparams}')
    if dropout is not None:
        _set_dropout(inner, dropout)
    channels = in_channels if in_channels is not None else model_hparams['in_channels']
    return Model.identity(inner, channels)


class EnsembleModule(lightning_module.LightningModule):
    """Lightning module training ``K`` same-architecture members on a shared batch.

    All members consume the *same* minibatch each step. The vmapped forward
    returns ``[K, B, L]``; a per-member physics loss is computed by looping over
    the (small) ``K`` dimension and summed for a single backward pass. Per-member
    train/val losses are logged and the best (lowest-loss) member is tracked.

    The MUST-HAVE optimizer is a single shared-learning-rate ``Adam`` over the
    stacked parameters. Per-member learning rate is implemented as a stretch goal
    via Lightning manual optimization (``per_member_lr`` argument): a manual Adam
    update whose step is scaled by a per-member lr vector broadcast over the
    leading ``K`` dimension.

    Args:
        model_hparams: Architecture hyperparameters (shared by all members).
        seeds: Per-member seeds; its length defines ``K``.
        target: ``'velocity'`` or ``'displacement'`` -- the model's prediction.
        lr: Shared learning rate (used when ``per_member_lr`` is ``None``).
        velocity_loss_weights: Per-member velocity loss weight, or a single float
            broadcast to all members. Defaults to ``1.0`` each.
        displacement_loss_weights: Per-member displacement loss weight, or a
            single float broadcast to all members. Defaults to ``1.0`` each.
        dropouts: Per-member dropout override, or a single float, or ``None`` to
            leave each member's architecture default.
        per_member_lr: Optional per-member learning rate list (length ``K``).
            When provided, manual optimization with a per-member-scaled Adam
            update is used instead of the shared-lr optimizer.
    """

    def __init__(
        self,
        model_hparams: dict[str, Any],
        seeds: list[int],
        target: str = 'velocity',
        lr: float = 1e-3,
        velocity_loss_weights: list[float] | float = 1.0,
        displacement_loss_weights: list[float] | float = 1.0,
        dropouts: list[float] | float | None = None,
        per_member_lr: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.model_hparams = model_hparams
        self.seeds = list(seeds)
        self.num_members = len(self.seeds)
        if self.num_members == 0:
            raise ValueError('seeds must be non-empty (defines ensemble size K)')
        self.target = target
        self.lr = lr

        self.velocity_loss_weights = self._broadcast(
            velocity_loss_weights, self.num_members
        )
        self.displacement_loss_weights = self._broadcast(
            displacement_loss_weights, self.num_members
        )
        if dropouts is None:
            member_dropouts: list[float | None] = [None] * self.num_members
        else:
            member_dropouts = self._broadcast(dropouts, self.num_members)

        if per_member_lr is not None:
            if len(per_member_lr) != self.num_members:
                raise ValueError('per_member_lr must have length K')
            self.automatic_optimization = False
            # Coerce to float: YAML scientific notation like ``1e-3`` (no decimal
            # point) parses as a string, which would break the manual update.
            per_member_lr = [float(x) for x in per_member_lr]
        self.per_member_lr = per_member_lr

        # Build K distinct members of the same architecture.
        members = [
            build_member(model_hparams, seed=self.seeds[i], dropout=member_dropouts[i])
            for i in range(self.num_members)
        ]

        # Stack module state: NEW stacked tensors with a leading K dim. These
        # stacked tensors -- not the original modules -- must be the optimizer
        # leaves, so register them as trainable Parameters / buffers here.
        params, buffers = stack_module_state(members)

        # A meta-device copy of one member carries the architecture for
        # functional_call (no real storage; the stacked params/buffers supply it).
        # Assign via object.__setattr__ so nn.Module does NOT register it as a
        # child module -- otherwise Lightning's `.to(device)` during `fit` would
        # try to move its meta tensors ("Cannot copy out of meta tensor"). The
        # base never needs real storage; functional_call substitutes the stacked
        # params/buffers (which are registered and moved normally).
        object.__setattr__(self, 'base', copy.deepcopy(members[0]).to('meta'))

        self._stacked_params = nn.ParameterDict(
            {
                _sanitize(name): nn.Parameter(tensor.clone())
                for name, tensor in params.items()
            }
        )
        # Buffers are identical across members for LayerNorm-based models, but we
        # keep the stacked (leading-K) form for a uniform functional_call.
        self._buffer_names: list[str] = []
        for name, tensor in buffers.items():
            key = _sanitize(name)
            self.register_buffer(key, tensor.clone())
            self._buffer_names.append(name)

        # Register per-member loss-weight tensors as buffers (move with .to()).
        # float() each weight: YAML ``1e-3``-style values may arrive as strings.
        self.register_buffer(
            'velocity_weight_vec',
            torch.tensor(
                [float(w) for w in self.velocity_loss_weights], dtype=torch.float32
            ),
        )
        self.register_buffer(
            'displacement_weight_vec',
            torch.tensor(
                [float(w) for w in self.displacement_loss_weights], dtype=torch.float32
            ),
        )

        self.best_member_idx: int | None = None
        self.best_val_loss: float = float('inf')

    @staticmethod
    def _broadcast(value: list[Any] | Any, k: int) -> list[Any]:
        """Broadcast a scalar to length ``k``, or validate a length-``k`` list."""
        if isinstance(value, (list, tuple)):
            if len(value) != k:
                raise ValueError(f'Expected length {k}, got {len(value)}')
            return list(value)
        return [value] * k

    def _params_dict(self) -> dict[str, Tensor]:
        """Reconstruct the stacked-parameter dict keyed by original dotted names."""
        return {
            _desanitize(key): tensor for key, tensor in self._stacked_params.items()
        }

    def _buffers_dict(self) -> dict[str, Tensor]:
        """Reconstruct the stacked-buffer dict keyed by original dotted names."""
        return {name: getattr(self, _sanitize(name)) for name in self._buffer_names}

    @override
    def forward(self, signals: Tensor) -> Tensor:
        """Run all K members over one shared batch.

        Args:
            signals: Shared input ``[B, C, L]`` fed to every member.

        Returns:
            Predictions ``[K, B, L]`` (the ``Model`` wrapper's ``[B, 1, L]``
            output is squeezed on the channel dim).
        """
        params = self._params_dict()
        buffers = self._buffers_dict()

        def fmodel(p: dict[str, Tensor], b: dict[str, Tensor], x: Tensor) -> Tensor:
            return functional_call(self.base, (p, b), (x,))

        # in_dims=(0, 0, None): map over the leading K dim of params/buffers,
        # broadcast the shared batch to every member.
        out = torch.vmap(fmodel, in_dims=(0, 0, None))(params, buffers, signals)
        # out: [K, B, 1, L] -> [K, B, L]
        return out.squeeze(2)

    def _per_member_loss(
        self, batch: tuple[Tensor, Tensor, Tensor]
    ) -> tuple[Tensor, list[dict[str, Tensor]]]:
        """Compute the stacked per-member total loss and per-member loss dicts.

        Args:
            batch: Shared ``(signals, velocity, displacement)``.

        Returns:
            ``(totals, loss_dicts)`` where ``totals`` is a ``[K]`` tensor of
            per-member weighted totals and ``loss_dicts`` holds the full loss
            dict for each member (for logging).
        """
        signals, velocity_target, displacement_target = batch
        preds = self(signals)  # [K, B, L]

        loss_dicts: list[dict[str, Tensor]] = []
        totals = []
        for i in range(self.num_members):
            # Clone targets per member: physics_loss mutates copies internally but
            # the baseline shift must not leak across members.
            loss_dict = physics_loss(
                preds[i],
                velocity_target.clone(),
                displacement_target.clone(),
                target=self.target,
                velocity_loss_weight=float(self.velocity_weight_vec[i]),
                displacement_loss_weight=float(self.displacement_weight_vec[i]),
            )
            loss_dicts.append(loss_dict)
            totals.append(loss_dict['total'])
        return torch.stack(totals), loss_dicts

    def _log_members(self, stage: str, loss_dicts: list[dict[str, Tensor]]) -> None:
        """Log per-member loss components under ``{stage}/member{i}/...`` keys."""
        for i, loss_dict in enumerate(loss_dicts):
            for name, value in loss_dict.items():
                self.log(f'{stage}/member{i}/{name}_loss', value)

    @override
    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        """Train all members on the shared batch with a single summed backward.

        Args:
            batch: Shared ``(signals, velocity, displacement)``.
            batch_idx: Lightning batch index.

        Returns:
            The summed-over-members total loss (for automatic optimization).
        """
        totals, loss_dicts = self._per_member_loss(batch)
        summed = totals.sum()

        self._log_members('train', loss_dicts)
        self.log('train/total_loss', summed, prog_bar=True)
        best = int(torch.argmin(totals).item())
        self.log('train/best_member', float(best))

        if self.per_member_lr is None:
            return summed

        # Stretch goal: manual per-member-lr update.
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(summed)
        self._manual_per_member_step()
        return summed

    @override
    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> None:
        """Evaluate all members on the shared batch and track the best member."""
        totals, loss_dicts = self._per_member_loss(batch)
        self._log_members('val', loss_dicts)
        self.log('val/total_loss', totals.sum(), prog_bar=True)

        member_best = int(torch.argmin(totals).item())
        member_best_loss = float(totals[member_best].item())
        if member_best_loss < self.best_val_loss:
            self.best_val_loss = member_best_loss
            self.best_member_idx = member_best
        self.log('val/best_member', float(member_best))
        self.log('val/best_member_loss', member_best_loss)
        # Report the best member's UNWEIGHTED total under the same key the
        # single-model search trials use, so a Ray Tune / ASHA search can compare
        # an ensemble trial against single-model trials on one metric.
        self.log(
            'val/total_unweighted_loss', loss_dicts[member_best]['total_unweighted']
        )

    @override
    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        """Evaluate all members on the shared batch (test stage logging)."""
        totals, loss_dicts = self._per_member_loss(batch)
        self._log_members('test', loss_dicts)
        self.log('test/total_loss', totals.sum())

    @torch.no_grad()
    def _manual_per_member_step(self) -> None:
        """Apply a per-member-lr Adam step on the stacked parameters.

        Uses a hand-rolled Adam (decoupled per-member lr) because ``torch.optim``
        applies one lr per param group, not one per leading-K slice. The per-member
        lr vector is broadcast over the leading ``K`` dim of every stacked param.
        State (first/second moments, step count) is kept in ``self._adam_state``.
        """
        if not hasattr(self, '_adam_state'):
            self._adam_state = {
                key: {
                    'step': 0,
                    'exp_avg': torch.zeros_like(p),
                    'exp_avg_sq': torch.zeros_like(p),
                }
                for key, p in self._stacked_params.items()
            }
        lr_vec = torch.tensor(
            self.per_member_lr, dtype=torch.float32, device=self.device
        )
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for key, p in self._stacked_params.items():
            if p.grad is None:
                continue
            state = self._adam_state[key]
            state['step'] += 1
            t = state['step']
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            bias_c1 = 1 - beta1**t
            bias_c2 = 1 - beta2**t
            denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
            step_dir = (exp_avg / bias_c1) / denom  # [K, ...]
            # Broadcast per-member lr over the leading K dim.
            shape = [self.num_members] + [1] * (p.dim() - 1)
            p.add_(step_dir * lr_vec.view(shape), alpha=-1.0)

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure a single shared-lr Adam over the stacked parameters.

        When ``per_member_lr`` is set, the returned optimizer is still an Adam
        (so Lightning has a handle) but the actual update is performed manually in
        :meth:`_manual_per_member_step`; its lr here is irrelevant.
        """
        return torch.optim.Adam(self._stacked_params.parameters(), lr=self.lr)

    def best_member_model(self, member_idx: int | None = None) -> Model:
        """Export one member as a standalone ``Model`` by slicing the stack.

        Args:
            member_idx: Member to export. Defaults to the tracked best member
                (or member 0 if none has been selected yet).

        Returns:
            A fresh ``Model`` (same architecture) whose weights are member
            ``member_idx``'s slice of the stacked parameters/buffers.
        """
        if member_idx is None:
            member_idx = self.best_member_idx if self.best_member_idx is not None else 0
        if not 0 <= member_idx < self.num_members:
            raise IndexError(f'member_idx {member_idx} out of range')

        member = build_member(self.model_hparams, seed=self.seeds[member_idx])
        state = member.state_dict()
        # Overwrite with the trained slice for this member.
        for key, tensor in self._stacked_params.items():
            state[_desanitize(key)] = tensor[member_idx].detach().clone()
        for name in self._buffer_names:
            stacked = getattr(self, _sanitize(name))
            state[name] = stacked[member_idx].detach().clone()
        member.load_state_dict(state)
        return member


class SyntheticEnsembleModule(EnsembleModule):
    """``EnsembleModule`` that generates one shared synthetic batch on-device.

    This is the inner level of the search pipeline's synthetic-data workflow: it
    mirrors :class:`smi.analysis.synthetic_lit_module.SyntheticLitModule` (same
    on-device Rayleigh-spectrum generation via :meth:`generate_synthetic_batch`)
    but feeds the SINGLE generated batch to all ``K`` members each step (the
    ``in_dims=(0, 0, None)`` broadcast in the parent), so the cheap on-GPU data is
    generated once and shared. The dataloader only supplies an iteration count
    (use :class:`SyntheticIndexDataset`); the batch content is generated here.

    Args:
        model_hparams: Shared architecture hyperparameters; ``in_channels`` must
            equal ``len(wavelengths_nm)``.
        seeds: Per-member seeds (length defines ``K``).
        wavelengths_nm: Interferometer wavelengths in nanometers (one per channel).
        start_freq: Lower bound of the displacement spectrum (Hz).
        end_freq: Upper bound of the displacement spectrum (Hz).
        max_displacement_um: Peak displacement amplitude in microns.
        **ensemble_kwargs: Forwarded to :class:`EnsembleModule` (``target``, ``lr``,
            ``velocity_loss_weights``, ``displacement_loss_weights``, ``dropouts``,
            ``per_member_lr``).
    """

    def __init__(
        self,
        model_hparams: dict[str, Any],
        seeds: list[int],
        *,
        wavelengths_nm: list[float],
        start_freq: float = 1.0,
        end_freq: float = 1000.0,
        max_displacement_um: float = 5.0,
        **ensemble_kwargs: Any,
    ) -> None:
        if len(wavelengths_nm) != model_hparams['in_channels']:
            raise ValueError(
                f'len(wavelengths_nm)={len(wavelengths_nm)} must equal '
                f'in_channels={model_hparams["in_channels"]}'
            )
        super().__init__(model_hparams, seeds, **ensemble_kwargs)
        self.max_displacement_um = max_displacement_um
        self.acq_sample_rate = RedPitayaConfig.SAMPLE_RATE_DEC1 / 256
        self.waveform = Waveform(start_freq=start_freq, end_freq=end_freq)
        self.register_buffer(
            'wavelengths_um',
            torch.tensor(
                [w / 1000.0 for w in wavelengths_nm], dtype=torch.float32
            ).view(1, -1, 1),
        )

    def _shared_batch(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Generate one synthetic batch (shared across all members) on-device."""
        return SyntheticLitModule.generate_synthetic_batch(
            len(batch),
            self.device,
            self.waveform,
            self.wavelengths_um,
            self.max_displacement_um,
            self.acq_sample_rate,
        )

    @override
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Generate a shared synthetic batch, then run the ensemble training step."""
        return super().training_step(self._shared_batch(batch), batch_idx)

    @override
    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        """Generate a shared synthetic batch, then run the ensemble validation step."""
        super().validation_step(self._shared_batch(batch), batch_idx)

    @override
    def on_validation_epoch_start(self) -> None:
        """Fix the seed so validation batches are reproducible across epochs."""
        torch.manual_seed(42)
