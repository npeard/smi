from dataclasses import dataclass


@dataclass
class RedPitayaConfig:
    """Configuration for Red Pitaya devices."""

    BUFFER_SIZE: int = 16384  # Number of samples in buffer
    SAMPLE_RATE_DEC1: float = 125e6  # Sample rate for decimation=1 in Samples/s (Hz)
    MODEL_NO: None | int = None
    # STEMlab 125-14 board spec: 14-bit fast analog inputs.
    ADC_BITS: int = 14
    # Input range on the LV jumper (+/-1 V; HV is +/-20 V). The acquisition
    # code never selects a gain (manager.py acq_set passes none, after ACQ:RST
    # restores the LV default), so LV is assumed, not read from the rig.
    INPUT_FULL_SCALE_LV_V: float = 1.0
