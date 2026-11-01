"""Synthetic motion must come from the rig's drive path through the coil."""

import math
import types

import numpy as np
import pytest
import torch

import smi.synthetic.coil_driver as coil_module
from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.forward import (
    MAX_SPEED_UM_PER_MS,
    NOMINAL_DRIVE_GAIN,
    sample_trajectory,
)
from smi.synthetic.waveform import Waveform
from tests._dtype_guard import no_float64

SAMPLE_RATE = Waveform.SAMPLE_RATE_DEC1 / 256
CPU = torch.device('cpu')
# Published median per-shot true velocity RMS, 0.373 and 0.377 um/ms for the
# two acquisitions (report/report.typ sec 5.1), um/ms.
TARGET_MEDIAN_RMS = 0.375


def _waveform() -> Waveform:
    return Waveform(start_freq=1.0, end_freq=1000.0)


def _shots(gain: float, n: int = 256) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    with no_float64():
        _, velocity, _ = sample_trajectory(
            n, _waveform(), CoilDriver(), gain, SAMPLE_RATE, CPU, gen
        )
    return velocity


def test_precompensation_matches_numpy_sample(monkeypatch: pytest.MonkeyPatch):
    """Torch port equals CoilDriver.sample(normalize_gain=True), same spectrum."""
    # sample() draws its random power scale from a module-level generator; pin
    # it to 1 so only the pre-compensation is compared.
    monkeypatch.setattr(
        coil_module, '_rng', types.SimpleNamespace(uniform=lambda _lo, _hi: 1.0)
    )
    coil, wf = CoilDriver(), _waveform()
    _, expected, _ = coil.sample(wf, normalize_gain=True, skip_randomization=True)
    # skip_randomization reuses the spectrum the call above fixed on ``wf``.
    raw = torch.from_numpy(np.fft.ifftshift(wf.sample(skip_randomization=True)[1]))
    with no_float64():
        got = coil.precompensate_torch(
            raw.float()[None], wf.gen_sample_rate, float(wf.valid_freqs[0])
        )
    np.testing.assert_allclose(
        got[0].numpy(), expected, atol=1e-4 * np.abs(expected).max()
    )


def test_trajectory_matches_numpy_coil_path():
    coil = CoilDriver()
    gen = torch.Generator().manual_seed(0)
    with no_float64():
        voltage, velocity, displacement = sample_trajectory(
            1, _waveform(), coil, NOMINAL_DRIVE_GAIN, SAMPLE_RATE, CPU, gen
        )
    v_np = voltage[0].numpy().astype(np.float64)
    expected_vel, _, _ = coil.get_velocity(v_np, SAMPLE_RATE)
    expected_disp, _, _ = coil.get_displacement(v_np, SAMPLE_RATE)
    np.testing.assert_allclose(
        velocity[0].numpy(), expected_vel, atol=1e-4 * np.abs(expected_vel).max()
    )
    np.testing.assert_allclose(
        displacement[0].numpy(), expected_disp, atol=1e-4 * np.abs(expected_disp).max()
    )


def test_nominal_gain_gives_published_velocity_scale():
    velocity = _shots(NOMINAL_DRIVE_GAIN)
    median_rms = velocity.pow(2).mean(dim=-1).sqrt().median().item()
    assert median_rms == pytest.approx(TARGET_MEDIAN_RMS, rel=0.01)


def test_velocity_range_within_inverse_fit_limit():
    velocity = _shots(4 * NOMINAL_DRIVE_GAIN)
    p999 = torch.quantile(velocity.abs().flatten(), 0.999).item()
    assert p999 < MAX_SPEED_UM_PER_MS, f'p99.9 |v| = {p999:.2f} um/ms'
    # The limit is that percentile x1.5 rounded up, not an arbitrary ceiling:
    # a stale limit far above it would let inverse_fit search needlessly wide.
    assert math.ceil(1.5 * p999) >= MAX_SPEED_UM_PER_MS


def test_off_grid_min_freq_raises_clear_error():
    coil, wf = (
        CoilDriver(),
        Waveform(start_freq=1.0, end_freq=1000.0, allowed_freqs=[37.5]),
    )
    match = 'not on the frequency grid'
    with pytest.raises(ValueError, match=match):
        coil.sample(wf, normalize_gain=True)
    with pytest.raises(ValueError, match=match):
        sample_trajectory(1, wf, coil, 1.0, SAMPLE_RATE, CPU, None)
