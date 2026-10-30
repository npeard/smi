"""Synthetic forward model: drive-voltage trajectories through the coil."""

import torch

from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.waveform import Waveform

# The rig's drive voltage is bounded by 1.0 V: RedPitayaManager rejects any
# waveform whose peak exceeds 1.0 (smi/redpitaya/manager.py:279-283) and sends
# it at a fixed generator amplitude of 1.0 (manager.py:286-290), so the
# Waveform's peak-normalized output is the physical drive voltage.
MAX_DRIVE_V = 1.0

# Evaluator limit for baseline_fit.inverse_fit(max_speed_um_per_ms=...).
# Measured 99.9th percentile |v| over 256 shots (generator seed 0, MAX_DRIVE_V,
# Waveform 100-1000 Hz) is 30.2 um/ms (1-1000 Hz: 28.2); times a margin of 1.5
# is 45.3, rounded up. Reproduce with
# tests/test_synthetic_trajectory.py::test_velocity_range_within_inverse_fit_limit.
MAX_SPEED_UM_PER_MS = 46.0


def sample_trajectory(
    batch_size: int,
    waveform: Waveform,
    coil: CoilDriver,
    max_drive_v: float,
    sample_rate: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw drive voltages and push them through the coil transfer function.

    Each shot's voltage is peak-normalized, run through ``coil``, then scaled by
    a per-shot amplitude ``U(0, max_drive_v)``. The transfer function is linear,
    so scaling after it equals scaling the voltage first and keeps the
    transfer-function output reusable at unit drive.

    Args:
        batch_size: Number of shots.
        waveform: Source of the random drive-voltage spectrum.
        coil: Speaker transfer function.
        max_drive_v: Peak drive voltage ceiling (V).
        sample_rate: Acquisition sample rate (Hz).
        device: Target device.
        generator: Optional generator living on ``device``.

    Returns:
        ``(voltage [V], velocity [um/ms], displacement [um])``, each ``[B, L]``.
    """
    voltage = waveform.generate_batch_torch(batch_size, device, generator)
    voltage = voltage / voltage.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    velocity, displacement = coil.get_velocity_and_displacement_torch(
        voltage, sample_rate
    )
    amp = torch.rand(batch_size, 1, device=device, generator=generator) * max_drive_v
    return voltage * amp, velocity * amp, displacement * amp
