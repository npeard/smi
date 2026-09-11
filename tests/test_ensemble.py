#!/usr/bin/env python3
"""CPU tests for the vmap ensemble (smi.analysis.ensemble).

Tiny K, short sequence_length, in_channels=3. Every test runs on CPU and
exercises the same code path used on GPU.
"""

import torch
from torch.func import functional_call

from smi.analysis.ensemble import EnsembleModule, physics_loss
from smi.analysis.synthetic_lit_module import SyntheticLitModule
from smi.synthetic.waveform import Waveform

K = 3
IN_CHANNELS = 3
SEQ_LEN = 64
BATCH = 4


def _tcn_hparams() -> dict:
    return {
        'type': 'TCN',
        'sequence_length': SEQ_LEN,
        'in_channels': IN_CHANNELS,
        'activation': 'GELU',
        'use_layer_norm': True,
        'use_weight_norm': False,
        'kernel_size': 3,
        'temporal_channels': [8, 8],
        'dilation_base': 2,
        'dropout': 0.0,
    }


def _make_ensemble(**kwargs) -> EnsembleModule:
    defaults = dict(
        model_hparams=_tcn_hparams(), seeds=[0, 1, 2], target='velocity', lr=1e-3
    )
    defaults.update(kwargs)
    return EnsembleModule(**defaults)


def test_forward_shape():
    """Shared-input forward returns [K, B, L]."""
    ens = _make_ensemble()
    x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN)
    out = ens(x)
    assert out.shape == (K, BATCH, SEQ_LEN), out.shape


def test_vmap_equals_looped_forward():
    """Vmapped ensemble forward equals the looped per-member functional forward."""
    ens = _make_ensemble()
    x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN)

    # Eval mode so dropout (p=0 here anyway) is fully deterministic.
    ens.eval()
    vmapped = ens(x)  # [K, B, L]

    # Looped reference: call the same meta base functionally, one member at a
    # time, slicing the stacked params/buffers.
    params = ens._params_dict()
    buffers = ens._buffers_dict()
    looped = []
    for i in range(ens.num_members):
        p_i = {name: t[i] for name, t in params.items()}
        b_i = {name: t[i] for name, t in buffers.items()}
        out_i = functional_call(ens.base, (p_i, b_i), (x,))  # [B, 1, L]
        looped.append(out_i.squeeze(1))
    looped_t = torch.stack(looped)  # [K, B, L]

    assert torch.allclose(vmapped, looped_t, atol=1e-5, rtol=1e-4), (
        (vmapped - looped_t).abs().max().item()
    )


def test_members_have_distinct_inits():
    """Different seeds yield different parameter values across members."""
    ens = _make_ensemble()
    # At least one parameter must differ across members (distinct seeds/inits).
    # Biases are typically zero-initialized and identical, so scan all params.
    differs = any(
        not torch.allclose(tensor[0], tensor[1])
        for tensor in ens._stacked_params.values()
    )
    assert differs, 'all member parameters are identical across seeds'


def test_training_step_finite_and_optimizes():
    """A few optimizer steps run and produce finite, non-increasing loss."""
    ens = _make_ensemble()
    ens.train()
    opt = ens.configure_optimizers()

    x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN)
    velocity = torch.randn(BATCH, SEQ_LEN)
    displacement = torch.randn(BATCH, SEQ_LEN)
    batch = (x, velocity, displacement)

    losses = []
    for _ in range(5):
        opt.zero_grad()
        totals, _ = ens._per_member_loss(batch)
        loss = totals.sum()
        assert torch.isfinite(loss)
        loss.backward()
        # Gradients must flow to the stacked params (the optimizer leaves).
        assert any(p.grad is not None for p in ens._stacked_params.parameters())
        opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(loss_value)) for loss_value in losses)
    assert losses[-1] <= losses[0] + 1e-6


def test_per_member_lr_distinct_updates():
    """With per-member lr, members receive different effective updates."""
    lrs = [0.0, 1e-2, 1e-2]
    ens = _make_ensemble(per_member_lr=lrs)
    assert ens.automatic_optimization is False

    # Snapshot params before.
    before = {k: v.detach().clone() for k, v in ens._stacked_params.items()}

    x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN)
    batch = (x, torch.randn(BATCH, SEQ_LEN), torch.randn(BATCH, SEQ_LEN))

    totals, _ = ens._per_member_loss(batch)
    summed = totals.sum()
    summed.backward()
    ens._manual_per_member_step()

    # Member 0 (lr=0) must be unchanged; member 1 (lr>0) must change.
    for key, p in ens._stacked_params.items():
        if p[0].numel() > 1:
            assert torch.allclose(p[0], before[key][0]), 'lr=0 member changed'
            assert not torch.allclose(p[1], before[key][1]), 'lr>0 member unchanged'
            return
    raise AssertionError('no multi-element parameter found')


def test_best_member_model_matches_slice():
    """Exported best member is a standalone Model whose weights match the slice."""
    ens = _make_ensemble()
    ens.eval()
    model = ens.best_member_model(member_idx=1)

    x = torch.randn(BATCH, IN_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        standalone = model(x).squeeze(1)  # [B, L]
        ensemble_out = ens(x)[1]  # member 1, [B, L]
    assert torch.allclose(standalone, ensemble_out, atol=1e-5, rtol=1e-4)


def test_physics_loss_matches_litmodule_logic():
    """physics_loss reproduces the expected static-weighted total."""
    pred = torch.randn(BATCH, SEQ_LEN)
    vel = torch.randn(BATCH, SEQ_LEN)
    disp = torch.randn(BATCH, SEQ_LEN)
    out = physics_loss(
        pred,
        vel,
        disp,
        target='velocity',
        velocity_loss_weight=2.0,
        displacement_loss_weight=3.0,
    )
    expected = 2.0 * out['velocity'] + 3.0 * out['displacement']
    assert torch.allclose(out['total'], expected)


def test_synthetic_batch_sharing():
    """generate_synthetic_batch produces one shared batch usable by all members."""
    waveform = Waveform(start_freq=100.0, end_freq=1000.0)
    wavelengths_um = torch.tensor([0.635, 0.6748, 0.515]).view(1, -1, 1)
    signals, velocity, displacement = SyntheticLitModule.generate_synthetic_batch(
        batch_size=BATCH,
        device=torch.device('cpu'),
        waveform=waveform,
        wavelengths_um=wavelengths_um,
        max_displacement_um=5.0,
        acq_sample_rate=Waveform.SAMPLE_RATE_DEC1 / 256,
    )
    assert signals.shape[0] == BATCH
    assert signals.shape[1] == IN_CHANNELS
    assert signals.shape[-1] == velocity.shape[-1] == displacement.shape[-1]
    assert torch.isfinite(signals).all()

    # One batch, shared across an ensemble whose seq_len matches the batch.
    hparams = _tcn_hparams()
    hparams['sequence_length'] = signals.shape[-1]
    ens = EnsembleModule(model_hparams=hparams, seeds=[0, 1, 2])
    out = ens(signals)
    assert out.shape == (K, BATCH, signals.shape[-1])
