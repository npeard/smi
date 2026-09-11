"""Tests for the feature registry and built-in feature definitions."""

import numpy as np

from smi.analysis.features import default_registry
from smi.analysis.features.features import PD_FEATURE_TO_CHANNEL
from smi.synthetic.coil_driver import CoilDriver

SIGNAL_LENGTH = 4096
SAMPLE_RATE = 488281.25


def _make_raw_shot(seed: int = 0) -> dict:
    """Build a synthetic raw shot dict with all four channels."""
    rng = np.random.default_rng(seed)
    return {
        'RP1_CH1': rng.standard_normal(SIGNAL_LENGTH).astype(np.float32),
        'RP1_CH2': rng.standard_normal(SIGNAL_LENGTH).astype(np.float32),
        'RP2_CH1': rng.standard_normal(SIGNAL_LENGTH).astype(np.float32),
        'RP2_CH2': rng.standard_normal(SIGNAL_LENGTH).astype(np.float32),
        'sample_rate': SAMPLE_RATE,
    }


def test_registered_feature_names():
    """Input and target features are registered with the expected names."""
    assert default_registry.input_features() == [
        'pd_RP1_CH2',
        'pd_RP2_CH1',
        'pd_RP2_CH2',
    ]
    assert default_registry.target_features() == ['velocity', 'displacement']


def test_compute_shapes_and_dtype():
    """compute() returns one float32 array per requested feature."""
    raw = _make_raw_shot()
    names = default_registry.input_features() + default_registry.target_features()
    out = default_registry.compute(raw, names)
    assert set(out) == set(names)
    for arr in out.values():
        assert arr.shape == (SIGNAL_LENGTH,)
        assert arr.dtype == np.float32


def test_input_features_are_identity():
    """Each photodiode input feature returns its raw channel verbatim."""
    raw = _make_raw_shot()
    out = default_registry.compute(raw, default_registry.input_features())
    for feat_name, channel in PD_FEATURE_TO_CHANNEL.items():
        np.testing.assert_array_equal(out[feat_name], raw[channel])


def test_targets_match_coil_driver():
    """velocity/displacement targets match a direct CoilDriver computation."""
    raw = _make_raw_shot(seed=1)
    out = default_registry.compute(raw, ['velocity', 'displacement'])
    cd = CoilDriver()
    expected_v = cd.get_velocity(raw['RP1_CH1'], SAMPLE_RATE)[0]
    expected_d = cd.get_displacement(raw['RP1_CH1'], SAMPLE_RATE)[0]
    np.testing.assert_allclose(out['velocity'], expected_v, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out['displacement'], expected_d, rtol=1e-5, atol=1e-5)
