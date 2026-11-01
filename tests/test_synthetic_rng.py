"""On-the-fly training streams: fresh every epoch and seed, fixed validation.

The training batches come from the global RNG (seeded by the grid cell), so
any reseed of it from a validation hook replays the same batches every epoch
and erases the seed axis. These tests drive a real Lightning loop, sanity
check included, because the bug lived in the hook ordering.
"""

import dataclasses
from collections import defaultdict

import lightning
import pytest
import torch
from torch.utils.data import DataLoader

from smi.analysis.ensemble import SyntheticEnsembleModule
from smi.analysis.synthetic_lit_module import (
    VAL_SEED,
    SyntheticIndexDataset,
    SyntheticLitModule,
)
from tests._devices import devices
from tests._sims import WAVELENGTHS_UM, michelson_sim

CROP = 256
BATCH = 2
EPOCHS = 3


def _sim() -> dict:
    return dataclasses.asdict(michelson_sim())


def _tcn() -> dict:
    return dict(
        type='TCN',
        sequence_length=CROP,
        in_channels=len(WAVELENGTHS_UM),
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=3,
        temporal_channels=[4, 4],
        dilation_base=2,
        dropout=0.0,
    )


def _lit_module() -> SyntheticLitModule:
    return SyntheticLitModule(
        model_hparams=_tcn(),
        optimizer_hparams={'name': 'Adam', 'lr': 1e-3, 'weight_decay': 0.0},
        scheduler_hparams={'warmup_epochs': 0, 'eta_min': 0.0, 'T_0': 10, 'T_mult': 1},
        loss_hparams={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 1.0,
            'sign_invariant': False,
        },
        training_hparams={'target': 'velocity'},
        data_hparams={},
        steps_per_epoch=2,
        sim=_sim(),
        train_crop_length=CROP,
        restoring_checkpoint=False,
    )


def _ensemble_module() -> SyntheticEnsembleModule:
    return SyntheticEnsembleModule(
        _tcn(), [0, 1], sim=_sim(), target='velocity', sign_invariant=False
    )


def _record(module, method: str) -> dict:
    """Wrap ``module.<method>`` to log ``(stage, epoch) -> [signals]``."""
    seen = defaultdict(list)
    inner = getattr(module, method)

    def wrapped(*args, **kwargs):
        out = inner(*args, **kwargs)
        trainer = module.trainer
        stage = 'sanity' if trainer.sanity_checking else trainer.state.stage.value
        seen[stage, trainer.current_epoch].append(out[0].detach().cpu().clone())
        return out

    setattr(module, method, wrapped)
    return seen


def _trainer(device: str) -> lightning.Trainer:
    return lightning.Trainer(
        max_epochs=EPOCHS,
        accelerator=device,
        devices=1,
        precision='32-true',
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=2,
    )


def _fit(make, method: str, seed: int, device: str) -> dict:
    lightning.seed_everything(seed)
    module = make()
    seen = _record(module, method)
    train = DataLoader(SyntheticIndexDataset(2 * BATCH), batch_size=BATCH, shuffle=True)
    val = DataLoader(SyntheticIndexDataset(2 * BATCH), batch_size=BATCH)
    _trainer(device).fit(module, train, val)
    return seen


def _train_key(seen: dict, epoch: int) -> tuple:
    return next(k for k in seen if k[1] == epoch and 'train' in k[0])


def _val_key(seen: dict, epoch: int) -> tuple:
    return next(k for k in seen if k[1] == epoch and 'valid' in k[0])


MODULES = [
    pytest.param(_lit_module, '_generate_synthetic_batch', id='lit'),
    pytest.param(_ensemble_module, '_shared_batch', id='ensemble'),
]


@pytest.mark.parametrize('device', devices())
@pytest.mark.parametrize(('make', 'method'), MODULES)
def test_training_batches_fresh_each_epoch_and_validation_fixed(make, method, device):
    seen = _fit(make, method, 0, device)
    firsts = [seen[_train_key(seen, e)][0] for e in range(EPOCHS)]
    for i in range(EPOCHS):
        for j in range(i + 1, EPOCHS):
            assert not torch.equal(firsts[i], firsts[j]), (i, j)
    vals = [seen[_val_key(seen, e)] for e in range(EPOCHS)]
    for later in vals[1:]:
        assert len(later) == len(vals[0])
        for a, b in zip(vals[0], later, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('device', devices())
@pytest.mark.parametrize(('make', 'method'), MODULES)
def test_grid_seeds_draw_different_training_data(make, method, device):
    a = _fit(make, method, 0, device)
    b = _fit(make, method, 1, device)
    for epoch in range(EPOCHS):
        assert not torch.equal(
            a[_train_key(a, epoch)][0], b[_train_key(b, epoch)][0]
        ), epoch


def test_test_stage_uses_full_shots_and_differs_from_validation():
    module = _lit_module()
    seen = _record(module, '_generate_synthetic_batch')
    loader = DataLoader(SyntheticIndexDataset(BATCH), batch_size=BATCH)
    trainer = _trainer('cpu')
    trainer.validate(module, loader, verbose=False)
    trainer.test(module, loader, verbose=False)
    (val,) = next(v for k, v in seen.items() if 'valid' in k[0])
    (test,) = next(v for k, v in seen.items() if 'test' in k[0])
    assert val.shape[-1] == CROP
    assert test.shape[-1] > CROP
    assert not torch.equal(val, test[..., :CROP])


@pytest.mark.parametrize(('make', 'method'), MODULES)
def test_no_global_reseed_in_construction_or_eval_hooks(make, method, monkeypatch):
    """CPU-runnable guard (CI has no MPS) for the reseed bug the MPS tests catch.

    Reseeding the global RNG anywhere in a module's construction or its
    validation/test hooks replays the same training batches every epoch.
    """

    def forbidden(*_args, **_kwargs):
        raise AssertionError('global torch RNG reseeded')

    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    monkeypatch.setattr(torch.random, 'manual_seed', forbidden)
    module = make()
    loader = DataLoader(SyntheticIndexDataset(BATCH), batch_size=BATCH)
    trainer = _trainer('cpu')
    trainer.validate(module, loader, verbose=False)
    if isinstance(module, SyntheticLitModule):
        trainer.test(module, loader, verbose=False)


def test_replayed_held_out_batches_are_the_simulated_ones():
    """A sanity-check-sized prefix is cached, then a longer epoch extends it.

    Replay must leave the generator where simulation would have, or the first
    uncached batch after a cached prefix would repeat the stream's start.
    """

    def epoch(module, n_batches: int) -> list[torch.Tensor]:
        gen = torch.Generator().manual_seed(VAL_SEED)
        return [
            module._generate_synthetic_batch(BATCH, gen, CROP)[0]
            for _ in range(n_batches)
        ]

    replaying = _lit_module()
    epoch(replaying, 2)
    replayed = epoch(replaying, 3)
    simulated = epoch(_lit_module(), 3)
    for a, b in zip(replayed, simulated, strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(replayed[2], replayed[0])
