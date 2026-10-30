"""Synthetic motion must come from a drive voltage through the coil."""

import numpy as np
import torch

from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.forward import MAX_DRIVE_V, MAX_SPEED_UM_PER_MS, sample_trajectory
from smi.synthetic.waveform import Waveform
from tests._dtype_guard import no_float64

SAMPLE_RATE = Waveform.SAMPLE_RATE_DEC1 / 256
CPU = torch.device('cpu')


def _waveform() -> Waveform:
    return Waveform(start_freq=100.0, end_freq=1000.0)


def test_trajectory_matches_numpy_coil_path():
    coil = CoilDriver()
    gen = torch.Generator().manual_seed(0)
    with no_float64():
        voltage, velocity, displacement = sample_trajectory(
            1, _waveform(), coil, MAX_DRIVE_V, SAMPLE_RATE, CPU, gen
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


def test_velocity_range_within_inverse_fit_limit():
    gen = torch.Generator().manual_seed(0)
    _, velocity, _ = sample_trajectory(
        256, _waveform(), CoilDriver(), MAX_DRIVE_V, SAMPLE_RATE, CPU, gen
    )
    p999 = torch.quantile(velocity.abs().flatten(), 0.999).item()
    assert p999 < MAX_SPEED_UM_PER_MS, f'p99.9 |v| = {p999:.2f} um/ms'
