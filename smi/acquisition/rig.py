"""The physical rig: which boards, how they are triggered, and the board constants."""

from stemlab import SCPI, STEMLAB_125_14_SPEC, BoardSpec, RedPitayaManager

from smi.core.acquisition import AcquisitionConfig

# Board 0 drives the coil and is the daisy-chain primary; board 1 follows it.
HOSTS = ('rp-f0c04a.local', 'rp-f0c026.local')
PRIMARY = 0
# Trigger delay is in samples after the trigger. Empirically, waiting 10 full
# buffers is enough to clear the start-up transient; 20 is never needed.
TRIGGER_DELAY_SAMPLES = 15 * STEMLAB_125_14_SPEC.buffer_size
# Wait per trigger/fill on each board, and the pause between shots. Raising the
# pause from 0.5 s to 1.0 s lowered training loss slightly; 0.75 s is the compromise.
SHOT_TIMEOUT_S = 5.0
SHOT_DELAY_S = 0.75


def acquisition_config(spec: BoardSpec) -> AcquisitionConfig:
    """The smi-side description of a stemlab board model."""
    return AcquisitionConfig(
        sample_rate_dec1_hz=spec.base_sample_rate_hz,
        buffer_size=spec.buffer_size,
        adc_bits=spec.adc_bits,
        input_full_scale_v=spec.input_full_scale_v,
    )


def connect(hosts: tuple[str, ...]) -> RedPitayaManager:
    """Open SCPI connections to ``hosts``, which are STEMlab 125-14 boards."""
    return RedPitayaManager(hosts, SCPI, STEMLAB_125_14_SPEC)
