"""Lightning module training a :class:`~smi.learning.flow.FlowMatching` sampler.

A sibling of :class:`~smi.learning.synthetic_lit_module.SyntheticLitModule`
rather than a mode of it: that class inherits ``LitModule``'s regression
contract -- a ``Model`` wrapper with a scalar output scale, the
velocity/displacement loss, sign invariance, and an ``in_channels`` check
against the wavelengths -- none of which holds for a velocity field whose
input is target, signals and time and whose output may be several channels.
A flow mode would branch every step. What is shared, the simulator, crop and
held-out replay, is reused from that module.

Tasks:

* ``'denoise'`` (Q3.3): cond = signals through the fiber, x1 = the free-space
  signals of the same shot. Fiber draws follow every motion, phase and
  visibility draw in ``simulate_batch``, so one seed with ``fiber=None`` and
  with the fiber gives the same trajectory (pinned in ``tests/test_flow.py``).
* ``'posterior'`` (Q3.4): cond = signals, x1 = velocity.

Both sides are scaled by fixed affine stats from a seeded sample, so the field
works in unit-variance coordinates; :meth:`sample` returns raw units.
"""

import dataclasses
from typing import Any, override

import lightning
import torch
from torch import Tensor

from smi.learning.flow import FlowMatching
from smi.learning.models.factory import create_model
from smi.learning.synthetic_lit_module import (
    NORM_STATS_SEED,
    NORM_STATS_SHOTS,
    VAL_SEED,
    HeldOutReplay,
    random_crop,
)
from smi.theory.forward import SimConfig, detect, simulate_batch, simulate_optics

TASKS = ('denoise', 'posterior')


def paired_batch(
    cfg: SimConfig, task: str, batch_size: int, generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """``(x1, cond)`` for ``task``, simulated on ``generator``'s device.

    ``denoise`` runs the optics twice from the same generator state, once with
    ``cfg.fiber`` and once without, then gives both the same post-fiber draws
    (phase/contrast/amplitude drift, DC offset, noise): the halves differ by
    the fiber alone. The generator ends where the fiber run left it.
    """
    if task not in TASKS:
        raise ValueError(f'task must be one of {TASKS}, not {task!r}')
    device = generator.device
    if task == 'posterior':
        batch = simulate_batch(cfg, batch_size, device, generator)
        return batch.velocity[:, None], batch.signals
    if cfg.fiber is None:
        raise ValueError('denoise needs a fiber in the sim config')
    if cfg.fiber.pbs_channel:
        raise ValueError('denoise needs matching channels; a PBS fiber adds C')
    state = generator.get_state()
    free = simulate_optics(
        dataclasses.replace(cfg, fiber=None), batch_size, device, generator
    )
    generator.set_state(state)
    fiber = simulate_optics(cfg, batch_size, device, generator)
    detector_state = generator.get_state()
    x1 = detect(cfg, free, device, generator).signals
    generator.set_state(detector_state)
    return x1, detect(cfg, fiber, device, generator).signals


class FlowLitModule(lightning.LightningModule):
    """Simulate paired batches each step and fit the flow-matching loss."""

    def __init__(
        self,
        model_hparams: dict[str, Any],
        training_hparams: dict[str, Any],
        steps_per_epoch: int,
        *,
        sim: dict[str, Any],
        task: str,
        n_time_features: int,
        train_crop_length: int | None,
        fit_norm_stats: bool,
        encoder_channels: int = 0,
        aux_weight: float = 0.0,
    ):
        """Build the velocity network from ``model_hparams`` for ``task``.

        Args:
            model_hparams: Backbone config (factory ``type`` plus fields);
                ``in_channels`` and ``out_channels`` are overwritten.
            training_hparams: Needs ``learning_rate``, ``eta_min`` and
                ``max_epochs`` (Adam with cosine decay).
            steps_per_epoch: Training batches per epoch.
            sim: Every ``SimConfig`` field by name, fiber included.
            task: ``'denoise'`` or ``'posterior'``.
            n_time_features: Sin/cos pairs embedding t.
            train_crop_length: Training/validation window; None = full shot.
            fit_norm_stats: True to fit the affine stats on NORM_STATS_SHOTS
                seeded shots; False when restoring a checkpoint, whose state
                dict carries them, so one shot only sizes the buffers.
            encoder_channels: E, the conditioning encoder's output channels
                (same backbone family, cond in), or 0 for no encoder. Defaulted
                only because the pre-encoder checkpoints' hparams lack it.
            aux_weight: The encoder's regression-loss weight (0 without one).
        """
        super().__init__()
        self.save_hyperparameters()
        self.sim_config = SimConfig.from_dict(sim)
        self.task = task
        self.training_hparams = training_hparams
        self.steps_per_epoch = steps_per_epoch
        self.train_crop_length = train_crop_length

        x1, cond = paired_batch(
            self.sim_config,
            task,
            NORM_STATS_SHOTS if fit_norm_stats else 1,
            torch.Generator().manual_seed(NORM_STATS_SEED),
        )
        for name, t in (('x1', x1), ('cond', cond)):
            self.register_buffer(f'{name}_mean', t.mean(dim=(0, 2))[None, :, None])
            self.register_buffer(f'{name}_std', t.std(dim=(0, 2))[None, :, None])
        hparams = {
            **model_hparams,
            'in_channels': x1.shape[1]
            + cond.shape[1]
            + encoder_channels
            + 2 * n_time_features,
            'out_channels': x1.shape[1],
        }
        encoder = None
        if encoder_channels:
            encoder = create_model(
                {
                    **model_hparams,
                    'in_channels': cond.shape[1],
                    'out_channels': encoder_channels,
                }
            )
        self.flow = FlowMatching(
            create_model(hparams), x1.shape[1], n_time_features, encoder, aux_weight
        )
        self._held_out_replay = HeldOutReplay()

    def _normalized(self, x1: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        return (
            (x1 - self.x1_mean) / self.x1_std,
            (cond - self.cond_mean) / self.cond_std,
        )

    def _batch(self, batch_size: int, generator: torch.Generator) -> tuple[Tensor, ...]:
        x1, cond = paired_batch(self.sim_config, self.task, batch_size, generator)
        x1, cond = random_crop(self.train_crop_length, x1, cond, generator=generator)
        return self._normalized(x1, cond)

    @override
    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # A fresh seed per step from the global RNG, because the pair needs a
        # generator it can rewind; the global stream stays seeded per run.
        seed = int(torch.randint(2**62, (1,)).item())
        generator = torch.Generator(self.device).manual_seed(seed)
        flow, aux = self.flow.loss(*self._batch(len(batch), generator), None)
        self.log('train/flow_loss', flow, on_epoch=True)
        self.log('train/aux_loss', aux, on_epoch=True)
        return flow + aux

    @override
    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        generator = self._val_generator
        x1, cond = self._held_out_replay(
            generator,
            (len(batch), self.train_crop_length, self.device),
            lambda: self._batch(len(batch), generator),
        )
        flow, aux = self.flow.loss(x1, cond, generator)
        self.log('val/flow_loss', flow, on_epoch=True)
        self.log('val/aux_loss', aux, on_epoch=True)

    @override
    def on_validation_epoch_start(self) -> None:
        self._val_generator = torch.Generator(self.device).manual_seed(VAL_SEED)

    def sample(
        self,
        cond: Tensor,
        n_samples: int,
        n_steps: int,
        generator: torch.Generator | None,
    ) -> Tensor:
        """``[S, B, C_x, L]`` draws of x1 given raw ``cond``, in raw units."""
        cond = (cond - self.cond_mean) / self.cond_std
        draws = self.flow.sample(cond, n_samples, n_steps, generator)
        return draws * self.x1_std + self.x1_mean

    @override
    def configure_optimizers(self) -> dict[str, Any]:
        hp = self.training_hparams
        optimizer = torch.optim.Adam(self.parameters(), lr=hp['learning_rate'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp['max_epochs'], eta_min=hp['eta_min']
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
