from dataclasses import dataclass


@dataclass
class RedPitayaConfig:
    """Configuration for Red Pitaya devices."""

    BUFFER_SIZE: int = 16384  # Number of samples in buffer
    SAMPLE_RATE_DEC1: float = 125e6  # Sample rate for decimation=1 in Samples/s (Hz)
    MODEL_NO: None | int = None
