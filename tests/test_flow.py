"""Flow matching: the paired simulation, the sampler on a known posterior, shapes."""

import dataclasses

import pytest
import torch
from torch import nn

from smi.experiments.grids.q3_fiber import GRID as Q3_FIBER
from smi.experiments.runner import expand_grid
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
    flow = FlowMatching(_pointwise(2 + 2 * n_time), 1, n_time)
    opt = torch.optim.Adam(flow.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3000, 1e-5)
    for _ in range(3000):
        x = torch.randn(512, 1, 1)
        y = x + noise_var**0.5 * torch.randn_like(x)
        loss = flow.loss(x, y)
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
