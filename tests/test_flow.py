"""Flow matching: the paired simulation, the sampler on a known posterior, shapes."""

import dataclasses
from collections.abc import Iterator
from unittest.mock import Mock

import lightning
import pytest
import torch
from torch import nn

from smi.experiments import flow_eval
from smi.experiments.flow_eval import arm_cell, training_identity
from smi.experiments.grids import grid as get_grid
from smi.experiments.grids.q3_fiber import GRID as Q3_FIBER
from smi.experiments.runner import expand_grid
from smi.learning import flow_lit_module
from smi.learning.configs import model_config
from smi.learning.flow import FlowMatching
from smi.learning.flow_lit_module import FlowLitModule, paired_batch
from smi.theory.forward import SimConfig, simulate_batch

DENOISE = 'flow/denoise-unettransformer-offline-44k'


def _fiber_sim() -> SimConfig:
    syn = model_config(DENOISE).synthetic_config
    assert syn is not None
    return SimConfig.from_dict(syn['sim'])


def test_fiber_pair_shares_trajectory():
    """Same seed, fiber on and off: identical motion, different signals."""
    cfg = _fiber_sim()
    free_cfg = dataclasses.replace(cfg, fiber=None)
    cpu = torch.device('cpu')
    free = simulate_batch(free_cfg, 2, cpu, torch.Generator().manual_seed(7))
    fiber = simulate_batch(cfg, 2, cpu, torch.Generator().manual_seed(7))
    assert torch.equal(free.velocity, fiber.velocity)
    assert torch.equal(free.displacement, fiber.displacement)
    assert not torch.allclose(free.signals, fiber.signals)
    # paired_batch's fiber half is that fiber run.
    _, cond = paired_batch(cfg, 'denoise', 2, torch.Generator().manual_seed(7))
    assert torch.equal(cond, fiber.signals)


def test_pair_shares_post_fiber_draws():
    """Additive DC offset and noise are the same realization in both halves.

    Both are added after the fiber, so each half minus its noise-free twin is
    exactly the shared offset plus noise; separate draws would differ.
    """
    clean = dataclasses.replace(
        _fiber_sim(),
        snr_db=None,
        dc_offset_range=(0.0, 0.0),
        amplitude_drift_rel=0.0,
        phase_drift_rad_per_sqrt_s=0.0,
        contrast_drift_rel=0.0,
        pd_bandwidth_hz=None,
        adc_bits=None,
    )
    noisy = dataclasses.replace(clean, snr_db=20.0, dc_offset_range=(-0.1, 0.1))
    x1, cond = paired_batch(noisy, 'denoise', 2, torch.Generator().manual_seed(3))
    x1_0, cond_0 = paired_batch(clean, 'denoise', 2, torch.Generator().manual_seed(3))
    assert not torch.allclose(x1, x1_0)
    torch.testing.assert_close(x1 - x1_0, cond - cond_0)


@pytest.mark.parametrize(
    ('name', 'arm'),
    [
        (DENOISE, 'fiber'),
        ('flow/posterior-free-unettransformer-global-44k', 'nominal'),
        ('flow/posterior-fiber-unettransformer-global-44k', 'fiber'),
    ],
)
def test_flow_sims_are_q3_fiber_arms(name, arm):
    """Each flow trains on the sim of the q3_fiber arm it is scored on."""
    syn = model_config(name).synthetic_config
    assert syn is not None
    (cell,) = [
        c for c in expand_grid(Q3_FIBER) if c.family == 'tcan-44k' and c.sim_name == arm
    ]
    assert SimConfig.from_dict(syn['sim']).hash() == cell.sim.hash()


def _pointwise(in_channels: int) -> nn.Module:
    """An MLP on ``[B, C, 1]``, enough for a 1-D posterior.

    Linear rather than 1x1 Conv1d: 50x faster per step on CPU at L = 1.
    """
    return nn.Sequential(
        nn.Flatten(1),
        nn.Linear(in_channels, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 1),
        nn.Unflatten(1, (1, 1)),
    )


def test_gaussian_posterior_moments():
    """X ~ N(0, 1), y = x + N(0, s^2): p(x | y) = N(y / (1 + s^2), s^2 / (1 + s^2))."""
    torch.manual_seed(0)
    noise_var = 0.25
    n_time = 4
    flow = FlowMatching(_pointwise(2 + 2 * n_time), 1, n_time, None, 0.0)
    opt = torch.optim.Adam(flow.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000, 1e-5)
    for _ in range(3000):
        x = torch.randn(512, 1, 1)
        y = x + noise_var**0.5 * torch.randn_like(x)
        loss = sum(flow.loss(x, y, None))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    y = torch.tensor([-1.5, 0.0, 1.0]).reshape(3, 1, 1)
    draws = flow.sample(y, 20000, 20, torch.Generator().manual_seed(1))
    assert draws.shape == (20000, 3, 1, 1)
    assert draws.dtype == torch.float32
    mean = draws.mean(dim=0).flatten()
    var = draws.var(dim=0).flatten()
    want_var = noise_var / (1 + noise_var)
    want_mean = y.flatten() / (1 + noise_var)
    # 5% of the posterior std for the mean (5% of 0 is no tolerance at y = 0).
    assert torch.allclose(mean, want_mean, atol=0.05 * want_var**0.5)
    assert torch.allclose(var, torch.full_like(var, want_var), rtol=0.05)


@pytest.mark.parametrize('task', ['denoise', 'posterior'])
def test_lit_module_shapes(task):
    """The registered backbone trains a step and samples [S, B, C_x, L], float32."""
    cfg = model_config(DENOISE)
    syn = cfg.synthetic_config
    assert syn is not None
    module = FlowLitModule(
        cfg.model_config,
        cfg.training_config,
        steps_per_epoch=1,
        sim=syn['sim'],
        task=task,
        n_time_features=syn['flow']['n_time_features'],
        train_crop_length=256,
        fit_norm_stats=True,
    )
    loss = module.training_step(torch.arange(2), 0)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()

    n_channels = len(syn['sim']['wavelengths_um'])
    cond = torch.randn(2, n_channels, 128)
    draws = module.sample(cond, 3, 2, torch.Generator().manual_seed(0))
    x_channels = n_channels if task == 'denoise' else 1
    assert draws.shape == (3, 2, x_channels, 128)
    assert draws.dtype == torch.float32


def test_validation_loss_is_identical_across_passes(monkeypatch: pytest.MonkeyPatch):
    """Two validation passes on one module draw the same batches, x_0 and t.

    The global RNG moves between them, as training would move it, so only a
    fixed validation generator keeps the monitored loss noise-free.
    """
    cfg = model_config(DENOISE)
    syn = cfg.synthetic_config
    assert syn is not None
    module = FlowLitModule(
        cfg.model_config,
        cfg.training_config,
        steps_per_epoch=1,
        sim=syn['sim'],
        task='posterior',
        n_time_features=syn['flow']['n_time_features'],
        train_crop_length=128,
        fit_norm_stats=True,
    ).eval()
    logged: list[float] = []
    monkeypatch.setattr(
        module,
        'log',
        lambda k, v, **_: logged.append(v.item()) if k == 'val/flow_loss' else None,
    )
    with torch.no_grad():
        for _ in range(2):
            module.on_validation_epoch_start()
            module.validation_step(torch.arange(2), 0)
            torch.rand(10)
    assert logged[0] == logged[1]


def test_checkpoint_restores_stats_without_refitting(tmp_path, monkeypatch):
    """Loading simulates one shot to size the buffers, then takes the saved stats."""
    cfg = model_config(DENOISE)
    syn = cfg.synthetic_config
    assert syn is not None
    module = FlowLitModule(
        cfg.model_config,
        cfg.training_config,
        steps_per_epoch=1,
        sim=syn['sim'],
        task='posterior',
        n_time_features=syn['flow']['n_time_features'],
        train_crop_length=128,
        fit_norm_stats=True,
    )
    path = tmp_path / 'flow.ckpt'
    torch.save(
        {
            'state_dict': module.state_dict(),
            'hyper_parameters': dict(module.hparams),
            'pytorch-lightning_version': lightning.__version__,
        },
        path,
    )
    sizes = []

    def counted(cfg, task, batch_size, generator):
        sizes.append(batch_size)
        return paired_batch(cfg, task, batch_size, generator)

    monkeypatch.setattr(flow_lit_module, 'paired_batch', counted)
    loaded = FlowLitModule.load_from_checkpoint(path, fit_norm_stats=False)
    assert sizes == [1]
    for name in ('x1_mean', 'x1_std', 'cond_mean', 'cond_std'):
        assert torch.equal(getattr(loaded, name), getattr(module, name))


def test_ensemble_members_share_the_training_budget():
    """q1_channels trains its nominal TCAN-44k on another budget: not a member."""
    ref = arm_cell(Q3_FIBER, 'nominal')
    want = training_identity(ref, Q3_FIBER)
    for name, same in (('q1_r1_seeds', True), ('q1_channels', False)):
        grid = get_grid(name)
        cells = [
            c
            for c in expand_grid(grid)
            if c.family == 'tcan-44k' and c.sim.hash() == ref.sim.hash()
        ]
        assert cells
        assert all((training_identity(c, grid) == want) is same for c in cells)


def test_denoised_estimator_nests_draw_counts(monkeypatch: pytest.MonkeyPatch):
    """Estimators sharing a cache average prefixes of one set of draws."""
    calls = []

    def fake_chunks(*args: object) -> Iterator[torch.Tensor]:
        calls.append(args)
        yield from torch.arange(4.0).reshape(4, 1, 1, 1).split(2)

    monkeypatch.setattr(flow_eval, '_draw_chunks', fake_chunks)
    monkeypatch.setattr(flow_eval, '_check_wavelengths', lambda *_: None)
    flow = Mock(device=torch.device('cpu'))
    flow.parameters.return_value = []
    regressor = Mock(name='r', model_type='r', total_params=0, device='cpu')
    regressor.predict_velocity = lambda x, _wl: x
    sampling = flow_eval.Sampling(n_samples=4, n_steps=2, per_call=2, seed=0)
    cache: dict[str, dict[int, torch.Tensor]] = {}
    signals = torch.zeros(1, 1, 1)
    means = [
        flow_eval.DenoisedEstimator(flow, regressor, sampling, n, (2, 4), cache)
        .predict_velocity(signals, ())
        .item()
        for n in (2, 4)
    ]
    assert means == [0.5, 1.5]
    assert len(calls) == 1


class _ConvNet(nn.Module):
    """Two conv layers plus a broadcast global mean: sees the whole sequence."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(in_channels, 32, 9, padding=4),
            nn.SiLU(),
            nn.Conv1d(32, 32, 9, padding=16, dilation=4),
            nn.SiLU(),
        )
        self.head = nn.Conv1d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.local(x)
        return self.head(torch.cat([h, h.mean(-1, keepdim=True).expand_as(h)], 1))


def _shortcut_toy(n: int, length: int, generator: torch.Generator | None):
    """x1 = a * ramp (one parameter); cond carries a only as a noisy frequency.

    x_t reveals a for most t, so the flow loss is nearly minimized without
    reading cond -- the shortcut the trained posterior flows took.
    """
    pos = torch.linspace(-1, 1, length)
    a = torch.rand(n, 1, 1, generator=generator) * 2 - 1
    phase = torch.rand(n, 1, 1, generator=generator) * 2 * torch.pi
    cond = torch.sin(torch.pi * (24 + 8 * a) * (pos + 1) + phase)
    return a * pos, cond + 0.5 * torch.randn(n, 1, length, generator=generator)


def _fit_toy(encoder_channels: int, length: int) -> tuple[float, float]:
    """(zeroed-cond sensitivity, draw RMSE / truth RMS) after a short fit."""
    torch.manual_seed(0)
    n_time = 4
    encoder = _ConvNet(1, encoder_channels) if encoder_channels else None
    flow = FlowMatching(
        _ConvNet(2 + encoder_channels + 2 * n_time, 1), 1, n_time, encoder, 1.0
    )
    opt = torch.optim.Adam(flow.parameters(), lr=3e-3)
    for _ in range(600):
        loss = sum(flow.loss(*_shortcut_toy(64, length, None), None))
        opt.zero_grad()
        loss.backward()
        opt.step()
    x1, cond = _shortcut_toy(32, length, torch.Generator().manual_seed(5))
    real, zeroed = (
        flow.sample(c, 4, 16, torch.Generator().manual_seed(1))
        for c in (cond, torch.zeros_like(cond))
    )
    rms = real.pow(2).mean().sqrt()
    sensitivity = float((real - zeroed).pow(2).mean().sqrt() / rms)
    return sensitivity, float(
        (real - x1).pow(2).mean().sqrt() / x1.pow(2).mean().sqrt()
    )


def test_encoder_makes_the_flow_read_a_shortcut_conditioning():
    """Without the encoder the field ignores cond and samples the prior."""
    length = 256
    plain = _fit_toy(0, length)
    encoded = _fit_toy(4, length)
    assert plain[0] < 0.05
    assert encoded[0] > 0.3
    assert encoded[1] < plain[1]


def _posterior_module(encoder_channels: int) -> FlowLitModule:
    cfg = model_config(DENOISE)
    syn = cfg.synthetic_config
    assert syn is not None
    return FlowLitModule(
        cfg.model_config,
        cfg.training_config,
        steps_per_epoch=1,
        sim=syn['sim'],
        task='posterior',
        n_time_features=syn['flow']['n_time_features'],
        train_crop_length=128,
        fit_norm_stats=False,
        encoder_channels=encoder_channels,
        aux_weight=1.0,
    )


def test_encoder_estimator_is_the_aux_head_in_raw_units():
    """The audit's regressor is exactly the encoder channel the aux loss fits."""
    module = _posterior_module(2).eval()
    wl = tuple(module.sim_config.wavelengths_um)
    signals = torch.randn(2, len(wl), 128)
    est = flow_eval.EncoderEstimator(module, 'enc')
    pred = est.predict_velocity(signals, wl)
    cond = (signals - module.cond_mean) / module.cond_std
    encoder = module.flow.encoder
    assert encoder is not None
    with torch.no_grad():
        raw = encoder(cond)[:, 0] * module.x1_std[:, 0] + module.x1_mean[:, 0]
    assert pred.shape == (2, 128)
    torch.testing.assert_close(pred, raw)
    with pytest.raises(ValueError, match='no conditioning encoder'):
        flow_eval.EncoderEstimator(_posterior_module(0), 'none')


def test_high_frequency_fraction_separates_smooth_from_noise():
    """Below 1% for the truth itself; most of white noise's power is above."""
    t = torch.linspace(0, 1, 4096)
    smooth = torch.sin(2 * torch.pi * 3 * t)
    assert flow_eval.high_frequency_fraction(smooth, smooth) <= 0.01
    noise = torch.randn(4096, generator=torch.Generator().manual_seed(0))
    assert flow_eval.high_frequency_fraction(noise, smooth) > 0.9
