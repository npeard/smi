"""Synthetic forward model: rig-faithful drive-voltage trajectories."""

import torch

from smi.synthetic.coil_driver import CoilDriver
from smi.synthetic.waveform import Waveform

# Scale of the drive voltage relative to the rig's pre-compensated waveform
# (CoilDriver.sample(normalize_gain=True), driven by RedPitayaManager at
# generator amplitude 1.0, manager.py:286-290). The code chain alone does not
# reproduce the rig's velocity scale (it depends strongly on the Waveform
# range and is ~350x too large at the rig default 1-1000 Hz), so this is set
# so the MEDIAN per-shot velocity std over 256 seeded shots (Waveform
# 1-1000 Hz, generator seed 0; 153 um/ms at drive_gain=1) is 0.44 um/ms, the
# typical per-shot velocity std already published in report/report.typ
# sec 4.1 (no acquisition file was opened). The gain is therefore partly a
# compensation for the generator-vs-acquisition time-base mismatch in the rig
# path (see sample_trajectory), not a purely physical analog gain.
NOMINAL_DRIVE_GAIN = 0.002879

# Evaluator limit for baseline_fit.inverse_fit(max_speed_um_per_ms=...):
# measured 99.9th percentile |v| at drive_gain = 4 x NOMINAL_DRIVE_GAIN (the
# top of the sweep) over the same 256 shots, times a margin of 1.5, rounded up.
# Measured 12.75 um/ms, x1.5 = 19.13, rounded up.
MAX_SPEED_UM_PER_MS = 20.0


def sample_trajectory(
    batch_size: int,
    waveform: Waveform,
    coil: CoilDriver,
    drive_gain: float,
    sample_rate: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw rig-style drive voltages and push them through the coil.

    Mirrors ``CoilDriver.sample(normalize_gain=True)``: the random waveform
    spectrum is pre-compensated by the velocity transfer function and scaled by
    a per-shot U(0, 1) power draw, then multiplied by ``drive_gain`` before the
    coil transfer function.

    Args:
        batch_size: Number of shots.
        waveform: Source of the random (velocity-shaped) spectrum.
        coil: Speaker transfer function.
        drive_gain: Voltage scale relative to the rig's pre-compensated
            waveform (``NOMINAL_DRIVE_GAIN`` is the calibrated value).
        sample_rate: Acquisition sample rate (Hz).
        device: Target device.
        generator: Optional generator living on ``device``.

    Returns:
        ``(voltage, velocity [um/ms], displacement [um])``, each ``[B, L]``.
    """
    raw = waveform.generate_batch_torch(batch_size, device, generator)
    # The rig pre-compensates on the Waveform's own time base (the generator
    # rate, via t in CoilDriver.sample), but the coil transfer is applied to the
    # acquired record at ``sample_rate``; the mismatch is part of the rig path.
    voltage = coil.precompensate_torch(
        raw, waveform.gen_sample_rate, float(waveform.valid_freqs[0])
    )
    amp = torch.rand(batch_size, 1, device=device, generator=generator) * drive_gain
    voltage = voltage * amp
    velocity, displacement = coil.get_velocity_and_displacement_torch(
        voltage, sample_rate
    )
    return voltage, velocity, displacement
