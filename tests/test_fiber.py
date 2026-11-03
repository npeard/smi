"""Multimode-fiber model (R7): unitarity, reciprocity, spectral and PBS laws."""

import dataclasses
import math

import pytest
import torch

from smi.theory import fiber
from smi.theory.fiber import FiberConfig, simulate_fiber
from smi.theory.forward import (
    SimConfig,
    derived_generator,
    simulate_batch,
    smi_excess_phase,
)
from tests._sims import SAMPLE_RATE, michelson_sim

CPU = torch.device('cpu')


def _fiber(**kw) -> FiberConfig:
    base = {
        'n_modes': 4,
        'drift_rad_per_sqrt_s': 0.0,
        'path_drift_um_per_sqrt_s': 0.0,
        'dispersion_per_um': 0.0,
        'spectral_corr_width_um': 0.01,
        'monitor_wavelengths_um': (),
        'pbs_channel': False,
    }
    return FiberConfig(**{**base, **kw})


def _unitaries(cfg, wavelengths, batch=4, knots=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return fiber.speckle_unitaries(cfg, wavelengths, batch, knots, 1e-3, CPU, gen)


def test_transmission_is_unitary():
    u = _unitaries(_fiber(drift_rad_per_sqrt_s=5.0), (0.635, 0.515))
    eye = torch.eye(u.shape[-1], dtype=u.dtype)
    torch.testing.assert_close(u @ u.mH, eye.expand_as(u), atol=1e-5, rtol=0)


def test_round_trip_is_symmetric():
    """Reciprocity: the round trip T^T T equals its own transpose."""
    r = fiber.round_trip(_unitaries(_fiber(), (0.635,)))
    torch.testing.assert_close(r, r.transpose(-1, -2), atol=1e-5, rtol=0)


def _kappa_corr(dlam_over_w: float) -> float:
    cfg = _fiber(n_modes=2, spectral_corr_width_um=0.01)
    lam = (0.635, 0.635 + dlam_over_w * 0.01)
    gen = torch.Generator().manual_seed(1)
    draw = simulate_fiber(cfg, lam, 2000, 1, SAMPLE_RATE, CPU, gen)
    k1, k2 = draw.kappa[:, 0, 0], draw.kappa[:, 1, 0]
    num = (k1 * k2.conj()).mean().abs()
    return float(num / ((k1.abs() ** 2).mean() * (k2.abs() ** 2).mean()).sqrt())


def test_spectral_decorrelation():
    assert _kappa_corr(10.0) < 0.1
    assert _kappa_corr(0.02) > 0.9


def test_single_mode_perp_power_anticorrelates_exactly():
    """N = 1, lossless: |kappa|^2 = 1 - P_perp, so the correlation is -1."""
    gen = torch.Generator().manual_seed(2)
    draw = simulate_fiber(_fiber(n_modes=1), (0.635,), 500, 1, SAMPLE_RATE, CPU, gen)
    a, b = draw.kappa.abs().flatten() ** 2, draw.p_perp.flatten()
    # float32 matrix_exp of a norm ~ pi sqrt(2) generator: ~1e-5 rounding.
    torch.testing.assert_close(a + b, torch.ones_like(a), atol=1e-4, rtol=0)
    assert torch.corrcoef(torch.stack([a, b]))[0, 1] < -0.9999


def _sim(fib, seed=0, **kw):
    cfg = dataclasses.replace(michelson_sim(), feedback_c=0.2, alpha=3.0, fiber=fib)
    cfg = dataclasses.replace(cfg, **kw)
    return cfg, simulate_batch(cfg, 2, CPU, torch.Generator().manual_seed(seed))


def test_none_fiber_is_bitwise_unchanged_and_hash_unmoved():
    base = dataclasses.replace(michelson_sim(), feedback_c=0.2, alpha=3.0)
    _, b = _sim(None)
    a = simulate_batch(base, 2, CPU, torch.Generator().manual_seed(0))
    assert torch.equal(a.signals, b.signals)
    assert 'fiber' not in base.identity_fields()
    assert base.hash() == dataclasses.replace(base, fiber=None).hash()
    assert base.hash() != dataclasses.replace(base, fiber=_fiber()).hash()


def test_fiber_config_round_trips_through_dict():
    cfg, _ = _sim(_fiber())
    assert SimConfig.from_dict(dataclasses.asdict(cfg)) == cfg


def test_monitor_channel_ignores_target():
    """Same draws, different motion: only the target channels change."""
    lam = michelson_sim().wavelengths_um
    fib = _fiber(monitor_wavelengths_um=(lam[1],), path_drift_um_per_sqrt_s=1.0)
    cfg, b = _sim(fib)
    gen = torch.Generator().manual_seed(0)
    faster = dataclasses.replace(cfg, drive_gain=2 * cfg.drive_gain)
    ref = simulate_batch(faster, 2, CPU, gen)
    torch.testing.assert_close(ref.signals[:, 1], b.signals[:, 1])
    assert not torch.allclose(ref.signals[:, 0], b.signals[:, 0])


def test_pbs_channel_appends_perp_power():
    _, b = _sim(_fiber(pbs_channel=True))
    n_ch = len(michelson_sim().wavelengths_um)
    assert b.signals.shape[1] == 2 * n_ch
    perp = b.signals[:, n_ch:]
    assert perp.min() >= 0
    assert perp.max() <= 1


def test_monitor_must_be_a_channel():
    with pytest.raises(ValueError, match='not channels'):
        _sim(_fiber(monitor_wavelengths_um=(1.55,)))


def test_fiber_phase_scale():
    """Round-trip fiber phase per um of path: 4 pi n(lambda) / lambda."""
    fib = _fiber()
    assert fib.fiber_phase_scale(0.5) == pytest.approx(4 * math.pi / 0.5)
    disp = dataclasses.replace(fib, dispersion_per_um=0.1)
    assert disp.fiber_phase_scale(0.5) == pytest.approx(4 * math.pi * 0.95 / 0.5)


def test_interpolated_powers_stay_lossless():
    """Between unitary knots, |kappa|^2 + P_perp is renormalized onto <= 1."""
    gen = torch.Generator().manual_seed(3)
    fib = _fiber(drift_rad_per_sqrt_s=300.0)
    draw = simulate_fiber(fib, (0.635,), 8, 4096, SAMPLE_RATE, CPU, gen)
    assert float((draw.kappa.abs() ** 2 + draw.p_perp).max()) <= 1 + 1e-6


def test_tensor_feedback_must_stay_weak():
    with pytest.raises(ValueError, match='below 1'):
        smi_excess_phase(torch.zeros(3), torch.tensor([0.2, 1.0, 0.1]), 3.0)


def test_derived_generator_is_deterministic():
    """The off-device fiber stream is a function of the caller's seed alone."""

    def draws(seed):
        parent = torch.Generator().manual_seed(seed)
        return torch.randn(4, generator=derived_generator(parent, CPU))

    assert torch.equal(draws(0), draws(0))
    assert not torch.equal(draws(0), draws(1))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='needs MPS')
def test_fiber_runs_with_an_mps_generator():
    cfg = dataclasses.replace(
        michelson_sim(), feedback_c=0.2, alpha=3.0, fiber=_fiber()
    )
    mps = torch.device('mps')
    a = simulate_batch(cfg, 2, mps, torch.Generator(mps).manual_seed(0))
    b = simulate_batch(cfg, 2, mps, torch.Generator(mps).manual_seed(0))
    assert torch.equal(a.signals.cpu(), b.signals.cpu())
