"""Description of the acquisition hardware that theory, data and simulator code
take as a required argument.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionConfig:
    """Fixed properties of an acquisition board. No defaults, by design: a
    call site names the rig it means rather than inheriting one.

    Attributes:
        sample_rate_dec1_hz: Sample rate at decimation 1 (Hz).
        buffer_size: Samples per buffer.
        adc_bits: ADC resolution.
        input_full_scale_v: Input range, +/- this many volts.
    """

    sample_rate_dec1_hz: float
    buffer_size: int
    adc_bits: int
    input_full_scale_v: float


# STEMlab 125-14: 14-bit fast inputs. LV jumper (+/-1 V; HV is +/-20 V) is
# assumed, not read from the rig: acquisition never selects a gain (manager.py
# acq_set passes none, after ACQ:RST restores the LV default).
STEMLAB_125_14 = AcquisitionConfig(
    sample_rate_dec1_hz=125e6, buffer_size=16384, adc_bits=14, input_full_scale_v=1.0
)

# Names configuration files (YAML, CLI flags) may use for a board.
BOARDS = {'STEMLAB_125_14': STEMLAB_125_14}

# RedPitayaManager acquires at acq_dec = 256 (manager.py settings), and every
# file written before the sample_rate attribute existed was acquired so.
NOMINAL_ACQ_DECIMATION = 256


def parse_acquisition(
    value: str | Mapping[str, float] | AcquisitionConfig,
) -> AcquisitionConfig:
    """Board from its config spelling: a BOARDS name, a field dict, or itself.

    YAML names a board; checkpoints and frozen sets store ``asdict()`` output.
    Called where those enter the program, so downstream types stay strict.
    """
    if isinstance(value, AcquisitionConfig):
        return value
    if isinstance(value, str):
        return BOARDS[value]
    return AcquisitionConfig(**value)
