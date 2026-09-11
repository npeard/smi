from dataclasses import dataclass

# @dataclass
# class CalibrationParameters:
#     """Calibration parameters for the coil driver.

#     Attributes:
#         f0: Resonant frequency in Hz
#         Q: Quality factor
#         k: Gain factor
#         c: Phase offset
#         speaker_part_number: Optional part number of the speaker
#     """

#     f0: float = 257.20857316296724
#     Q: float = 15.804110908084784
#     k: float = 33.42493417407945
#     c: float = -3.208233068626455
#     speaker_part_number: str | None = None


@dataclass
class CalibrationParameters:
    """Calibration parameters for the coil driver.

    Attributes:
        f0: Resonant frequency in Hz
        Q: Quality factor
        k: Gain factor
        c: Phase offset
        speaker_part_number: Optional part number of the speaker
    """

    f0: float = 258.936
    Q: float = 16.98
    k: float = (
        14.553  # This value is correct. Incorrect fringe counts in 2024 calibration?
    )
    c: float = -3.220
    speaker_part_number: str | None = None
