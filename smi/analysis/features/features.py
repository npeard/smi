"""Built-in feature definitions.

Importing this module registers every feature on the
:data:`~smi.analysis.features.registry.default_registry`.

Inputs are the photodiode channels (identity on the raw array). Targets are the
velocity and displacement waveforms derived from the speaker drive voltage
(``RP1_CH1``) via the :class:`CoilDriver` physics model. Feature names are
stable identifiers used as keys in the version-controlled ``normalization.py``.
"""

from collections.abc import Callable

import numpy as np

from smi.synthetic.coil_driver import CoilDriver

from .registry import RawShot, register_feature

# Single shared CoilDriver (default calibration) for target computation.
_coil_driver = CoilDriver()

# Map stable input-feature names to their raw photodiode channel keys. Order
# matches VelocityDataset.pd_channel_keys so input_features()[:N] selects the
# first N photodiode channels consistently.
PD_FEATURE_TO_CHANNEL = {
    'pd_RP1_CH2': 'RP1_CH2',
    'pd_RP2_CH1': 'RP2_CH1',
    'pd_RP2_CH2': 'RP2_CH2',
}


def _make_pd_feature(channel: str) -> Callable[[RawShot], np.ndarray]:
    """Build an identity compute function for a photodiode channel."""

    def compute(raw: RawShot) -> np.ndarray:
        return np.asarray(raw[channel], dtype=np.float32)

    return compute


# Register the photodiode input features.
for _feat_name, _channel in PD_FEATURE_TO_CHANNEL.items():
    register_feature(_feat_name, 'input')(_make_pd_feature(_channel))


@register_feature('velocity', 'target')
def velocity(raw: RawShot) -> np.ndarray:
    """Velocity waveform (microns/s) from the drive voltage RP1_CH1."""
    voltage = np.asarray(raw['RP1_CH1'])
    sample_rate = float(raw['sample_rate'])
    return _coil_driver.get_velocity(voltage, sample_rate)[0]


@register_feature('displacement', 'target')
def displacement(raw: RawShot) -> np.ndarray:
    """Displacement waveform (microns) from the drive voltage RP1_CH1."""
    voltage = np.asarray(raw['RP1_CH1'])
    sample_rate = float(raw['sample_rate'])
    return _coil_driver.get_displacement(voltage, sample_rate)[0]
