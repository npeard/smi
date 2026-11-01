"""Hardware constants of a Red Pitaya board."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoardSpec:
    """Fixed properties of one board model.

    Attributes:
        base_sample_rate_hz: ADC/DAC sample rate at decimation 1.
        buffer_size: Samples in one acquisition or generation buffer.
        adc_bits: ADC resolution.
        input_full_scale_v: Input range on LV gain, +/- this many volts.
    """

    base_sample_rate_hz: float
    buffer_size: int
    adc_bits: int
    input_full_scale_v: float


STEMLAB_125_14_SPEC = BoardSpec(
    base_sample_rate_hz=125e6, buffer_size=16384, adc_bits=14, input_full_scale_v=1.0
)
