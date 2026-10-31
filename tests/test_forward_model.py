"""The synthetic forward model: physics, noise knobs, and causal-safe stats."""

import dataclasses
import math

import pytest
import torch

from smi.analysis.synthetic_lit_module import SyntheticLitModule
from smi.synthetic import forward
from smi.synthetic.forward import (
    NOMINAL_DRIVE_GAIN,
    SMI_NEWTON_ITERATIONS,
    SimConfig,
    fringe,
    pd_lowpass,
    simulate_batch,
    smi_excess_phase,
)
from tests._devices import devices
from tests._dtype_guard import no_float64
from tests._sims import SAMPLE_RATE, WAVELENGTHS_UM, michelson_sim

CPU = torch.device('cpu')
BATCH = 8


def _noise_free(feedback_c: float = 0.0, alpha: float = 0.0) -> SimConfig:
    return dataclasses.replace(michelson_sim(), feedback_c=feedback_c, alpha=alpha)


def _simulate(cfg: SimConfig, seed: int = 0, batch: int = BATCH, device=CPU):
    gen = torch.Generator().manual_seed(seed)
    with no_float64():
        return simulate_batch(cfg, batch, torch.device(device), gen)


@pytest.mark.parametrize('device', devices())
def test_noise_free_michelson_matches_cosine(device):
    out = _simulate(_noise_free(), device=device)
    assert out.signals.device.type == torch.device(device).type
    lam = torch.tensor(WAVELENGTHS_UM, device=out.signals.device).view(1, -1, 1)
    phase0 = out.optical_phase - 4 * math.pi * out.displacement[:, None] / lam
    # The per-channel offset phi_k is constant within a shot.
    torch.testing.assert_close(
        phase0, phase0[..., :1].expand_as(phase0), atol=2e-3, rtol=0
    )
    torch.testing.assert_close(
        out.signals, torch.cos(out.optical_phase), atol=1e-5, rtol=0
    )


# (field overrides that switch one knob on) -- the off value is noise_free's.
KNOBS = {
    'feedback_c': {'feedback_c': 0.5, 'alpha': 3.0},
    'snr_db': {'snr_db': 20.0},
    'shot_noise': {'snr_db': 20.0, 'shot_noise': True},
    'visibility_range': {'visibility_range': (0.5, 0.9)},
    'dc_offset_range': {'dc_offset_range': (-0.2, 0.3)},
    'amplitude_drift_rel': {'amplitude_drift_rel': 0.1},
    'phase_drift_rad_per_sqrt_s': {'phase_drift_rad_per_sqrt_s': 5.0},
    'contrast_drift_rel': {'contrast_drift_rel': 0.1},
    'pd_bandwidth_hz': {'pd_bandwidth_hz': 20e3},
    'adc_bits': {'adc_bits': 6, 'adc_full_scale': 1.0},
}


@pytest.mark.parametrize('knob', sorted(KNOBS))
def test_each_knob_off_is_identity(knob):
    base_cfg = _noise_free()
    on_cfg = dataclasses.replace(base_cfg, **KNOBS[knob])
    off_cfg = dataclasses.replace(
        on_cfg, **{k: getattr(base_cfg, k) for k in KNOBS[knob]}
    )
    base, on, off = _simulate(base_cfg), _simulate(on_cfg), _simulate(off_cfg)
    for field in dataclasses.fields(base):
        assert torch.equal(getattr(off, field.name), getattr(base, field.name))
    assert not torch.equal(on.signals, base.signals), 'the knob must do something'
    # A knob never shifts the random stream that draws the motion.
    for name in ('voltage', 'velocity', 'displacement'):
        assert torch.equal(getattr(on, name), getattr(base, name))
    # Only phase drift moves the optical phase (hence phi_k stays put too).
    if knob != 'phase_drift_rad_per_sqrt_s':
        assert torch.equal(on.optical_phase, base.optical_phase)


def test_phase_drift_does_not_change_visibility_draw():
    cfg = dataclasses.replace(_noise_free(), visibility_range=(0.5, 0.9))
    # A negligible drift leaves the signal equal only if V is the same draw.
    tiny = dataclasses.replace(cfg, phase_drift_rad_per_sqrt_s=1e-6)
    torch.testing.assert_close(
        _simulate(cfg).signals, _simulate(tiny).signals, atol=1e-3, rtol=0
    )


@pytest.mark.parametrize('snr_db', [0.0, 10.0, 20.0, 40.0])
def test_snr_matches_setting(snr_db):
    clean = _simulate(_noise_free(), batch=32).signals
    noisy = _simulate(
        dataclasses.replace(_noise_free(), snr_db=snr_db), batch=32
    ).signals
    measured = 10 * torch.log10(clean.pow(2).mean() / (noisy - clean).pow(2).mean())
    assert abs(float(measured) - snr_db) < 0.5


def test_adc_levels():
    cfg = dataclasses.replace(
        _noise_free(), snr_db=10.0, adc_bits=4, adc_full_scale=1.0
    )
    signals = _simulate(cfg).signals
    for ch in range(signals.shape[1]):
        assert signals[:, ch].unique().numel() <= 16
    # Two's-complement range: the top code is one step below full scale.
    assert float(signals.max()) <= 1.0 - 1.0 / 8 + 1e-6
    assert float(signals.min()) >= -1.0 - 1e-6


def test_lowpass_minus_3db():
    cutoff = 20e3
    n = torch.arange(16384, dtype=torch.float32)
    tone = torch.cos(2 * math.pi * cutoff / SAMPLE_RATE * n)
    with no_float64():
        out = pd_lowpass(tone[None], cutoff, SAMPLE_RATE)[0]
    # Steady state only: project the second half onto the tone's quadratures.
    half = slice(8192, None)
    arg = 2 * math.pi * cutoff / SAMPLE_RATE * n[half]
    amp = 2 * torch.hypot(
        (out[half] * torch.cos(arg)).mean(), (out[half] * torch.sin(arg)).mean()
    )
    gain_db = 20 * math.log10(float(amp))
    assert abs(gain_db + 3.0) < 0.3


def test_lowpass_is_causal():
    x = torch.randn(2, 3, 4096)
    y = x.clone()
    y[..., 2000:] += 1.0
    with no_float64():
        a, b = pd_lowpass(x, 5e3, SAMPLE_RATE), pd_lowpass(y, 5e3, SAMPLE_RATE)
    torch.testing.assert_close(a[..., :2000], b[..., :2000], atol=1e-5, rtol=0)
    assert not torch.allclose(a[..., 2000:], b[..., 2000:])


def test_smi_c_zero_is_michelson():
    phase = torch.linspace(-50.0, 50.0, 10001)
    assert torch.equal(fringe(phase, 0.0, 3.0), torch.cos(phase))
    out = _simulate(_noise_free(feedback_c=0.0, alpha=5.0))
    assert torch.equal(out.signals, _simulate(_noise_free()).signals)


@pytest.mark.parametrize('feedback_c', [0.05, 0.5, 0.9])
@pytest.mark.parametrize('alpha', [3.0, 5.0])
def test_smi_newton_converges(feedback_c, alpha):
    phase = torch.linspace(-200.0, 200.0, 400001)
    with no_float64():
        delta, psi = smi_excess_phase(phase, feedback_c, alpha)
    residual = delta + feedback_c * torch.sin(psi + delta)
    assert float(residual.abs().max()) < 1e-6
    assert SMI_NEWTON_ITERATIONS <= 8


def test_smi_rejects_moderate_feedback():
    with pytest.raises(ValueError, match='moderate'):
        fringe(torch.zeros(4), 1.0, 3.0)
    with pytest.raises(ValueError, match='moderate'):
        _noise_free(feedback_c=1.2)


def _skew(phase: torch.Tensor, feedback_c: float, alpha: float) -> float:
    slope = torch.diff(fringe(phase, feedback_c, alpha))
    return float(slope.pow(3).mean() / slope.pow(2).mean().pow(1.5))


def test_smi_fringe_asymmetry_tracks_sign():
    """Sign convention: d increasing (v > 0) -> phase increasing -> skew < 0.

    For v > 0 the SMI fringe (alpha > 0) rises slowly and falls steeply, so the
    third moment of dI/dt over whole fringes is negative; v < 0 mirrors it.
    """
    # 20 whole fringes at a constant rate, sampled finely.
    ramp = torch.linspace(0.0, 20 * 2 * math.pi, 200001)
    for alpha in (3.0, 5.0):
        assert _skew(ramp, 0.5, alpha) < -0.1
        assert _skew(-ramp, 0.5, alpha) > 0.1
    assert abs(_skew(ramp, 0.0, 0.0)) < 1e-4
    assert abs(_skew(ramp, 0.5, 0.0)) < 1e-4


def test_michelson_global_sign_symmetry():
    """(d, phi) and (-d, -phi) give identical signals: spec Q3's global sign.

    The symmetry survives SMI feedback when alpha = 0 and breaks for alpha != 0,
    so only C > 0 together with alpha != 0 makes direction observable.
    """
    out = _simulate(_noise_free())
    phase = out.optical_phase
    torch.testing.assert_close(fringe(-phase, 0.0, 0.0), out.signals, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        fringe(-phase, 0.5, 0.0), fringe(phase, 0.5, 0.0), atol=1e-5, rtol=0
    )
    assert (fringe(-phase, 0.5, 3.0) - fringe(phase, 0.5, 3.0)).abs().max() > 0.1


def test_all_float32():
    every_knob: dict = {}
    for overrides in KNOBS.values():
        every_knob.update(overrides)
    out = _simulate(dataclasses.replace(_noise_free(), **every_knob))
    for field in dataclasses.fields(out):
        assert getattr(out, field.name).dtype == torch.float32


def test_deterministic_given_generator():
    cfg = dataclasses.replace(
        _noise_free(feedback_c=0.3, alpha=3.0), snr_db=20.0, pd_bandwidth_hz=50e3
    )
    a, b, c = _simulate(cfg, seed=1), _simulate(cfg, seed=1), _simulate(cfg, seed=2)
    for field in dataclasses.fields(a):
        assert torch.equal(getattr(a, field.name), getattr(b, field.name))
    assert not torch.equal(a.signals, c.signals)


def test_hash_is_stable_and_field_sensitive():
    assert _noise_free().hash() == _noise_free().hash()
    assert (
        _noise_free().hash() != dataclasses.replace(_noise_free(), snr_db=20.0).hash()
    )


def test_hash_ignores_int_vs_float_spelling():
    float_cfg = _noise_free()
    int_cfg = SimConfig.noise_free(
        wavelengths_um=list(WAVELENGTHS_UM),
        sample_rate=SAMPLE_RATE,
        start_freq_hz=1,
        end_freq_hz=1000,
        drive_gain=NOMINAL_DRIVE_GAIN,
        feedback_c=0,
        alpha=0,
    )
    assert int_cfg == float_cfg
    assert int_cfg.hash() == float_cfg.hash()
    ints = dataclasses.replace(float_cfg, snr_db=20, visibility_range=[1, 1])
    floats = dataclasses.replace(float_cfg, snr_db=20.0, visibility_range=(1.0, 1.0))
    assert ints.hash() == floats.hash()


def test_full_model_causal_with_baked_normalization():
    seq_len = 1024
    module = SyntheticLitModule(
        model_hparams=dict(
            type='TCN',
            sequence_length=seq_len,
            in_channels=len(WAVELENGTHS_UM),
            activation='GELU',
            use_layer_norm=True,
            use_weight_norm=False,
            kernel_size=3,
            temporal_channels=[8, 8, 8],
            dilation_base=2,
            dropout=0.0,
        ),
        optimizer_hparams={'name': 'Adam', 'lr': 1e-3, 'weight_decay': 0.0},
        scheduler_hparams={},
        loss_hparams={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 1.0,
            'sign_invariant': False,
        },
        training_hparams={'target': 'velocity'},
        data_hparams={},
        steps_per_epoch=1,
        train_crop_length=None,
        sim=dataclasses.asdict(_noise_free()),
        restoring_checkpoint=False,
    )
    model = module.model.eval()
    # Baked stats, not identity: velocity std is the synthetic motion scale.
    assert not torch.allclose(model.output_std, torch.ones(1))
    assert not torch.allclose(model.input_std, torch.ones(1))

    x = _simulate(_noise_free(), batch=2).signals[..., :seq_len]
    t0 = 600
    y = x.clone()
    y[..., t0:] += 0.5
    with torch.no_grad():
        a, b = model(x), model(y)
    torch.testing.assert_close(a[..., :t0], b[..., :t0], atol=1e-6, rtol=0)
    assert not torch.allclose(a[..., t0:], b[..., t0:])


@pytest.mark.parametrize('device', devices())
def test_cached_simulation_is_bitwise_the_uncached_one(device):
    """Warm caches (coil transfers, wavelengths, low-pass kernel) change nothing.

    Clearing them first makes the first batch build every tensor afresh, the
    uncached path; the second, same seed, reads them all back.
    """
    cfg = dataclasses.replace(
        _noise_free(feedback_c=0.3, alpha=3.0), snr_db=20.0, pd_bandwidth_hz=50e3
    )
    for cache in (forward.drive, forward._lowpass_kernel, forward._wavelengths_tensor):
        cache.cache_clear()
    cold, warm = (
        simulate_batch(
            cfg, 4, torch.device(device), torch.Generator(device).manual_seed(3)
        )
        for _ in range(2)
    )
    for field in dataclasses.fields(cold):
        assert torch.equal(getattr(cold, field.name), getattr(warm, field.name))


def test_nominal_phase_drift_rate_is_its_fringes_converted():
    # Pinned to the rate every shipped grid's sim hashes were computed under:
    # defining it from the fringe count must not move it by one ulp.
    assert forward.NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S == 35.31967896870391
    assert forward.rad_per_sqrt_s_to_fringes(
        forward.NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S
    ) == pytest.approx(forward.NOMINAL_PHASE_DRIFT_FRINGES)
