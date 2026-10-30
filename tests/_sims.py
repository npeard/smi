"""Simulator configs and fixtures shared by the synthetic-pipeline tests.

Registered as a pytest plugin by ``conftest.py``, which makes ``small_chunks``
available to every test module.
"""

import pytest

from smi.synthetic import frozen
from smi.synthetic.forward import NOMINAL_DRIVE_GAIN, SimConfig

SAMPLE_RATE = 125e6 / 256
WAVELENGTHS_UM = (0.635, 0.6748, 0.515)
#: Frozen-set chunk size for tests: real chunks are 256 shots of 16384
#: samples, and the logic under test is chunk-size free.
SMALL_CHUNK_SHOTS = 4


def michelson_sim() -> SimConfig:
    """Noise-free Michelson (C = 0, alpha = 0) on the rig's three channels."""
    return SimConfig.noise_free(
        wavelengths_um=WAVELENGTHS_UM,
        sample_rate=SAMPLE_RATE,
        start_freq_hz=1.0,
        end_freq_hz=1000.0,
        drive_gain=NOMINAL_DRIVE_GAIN,
        feedback_c=0.0,
        alpha=0.0,
    )


@pytest.fixture
def small_chunks(monkeypatch):
    """Shrink frozen-set chunks to ``SMALL_CHUNK_SHOTS``."""
    monkeypatch.setattr(frozen, 'CHUNK_SHOTS', SMALL_CHUNK_SHOTS)
