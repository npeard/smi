"""The HDF5 schema: key constants match real files; write/read round-trips."""

import h5py
import numpy as np
import pytest

from smi.data import schema
from smi.data.frozen import write_frozen_set
from tests._sims import michelson_sim


@pytest.mark.usefixtures('small_chunks')
def test_key_constants_match_frozen_set(tmp_path):
    # A frozen set is written by an independent code path, so it pins the
    # constants to the strings real files carry.
    path = write_frozen_set(michelson_sim(), 0, 2, tmp_path / 'f.h5')
    with h5py.File(path, 'r') as f:
        keys = set(f)
        assert schema.SAMPLE_RATE_ATTR in f.attrs
        assert schema.NUM_SAMPLES_ATTR in f.attrs
    assert set(schema.CHANNEL_KEYS) <= keys
    assert {
        schema.VELOCITY_KEY,
        schema.DISPLACEMENT_KEY,
        schema.OPTICAL_PHASE_KEY,
    } <= keys


def test_round_trip_preserves_arrays_and_attrs(tmp_path):
    rng = np.random.default_rng(0)
    shots = [
        {k: rng.standard_normal(32).astype(np.float32) for k in schema.CHANNEL_KEYS}
        for _ in range(3)
    ]
    attrs = {'sample_rate': 1234.5, 'decimation': 256, 'note': 'x'}
    path = tmp_path / 'a.h5'
    for i, shot in enumerate(shots):
        # Reopened in append mode, as the acquisition loop does.
        with h5py.File(path, 'w' if i == 0 else 'a') as f:
            schema.write_shot(f, shot, attrs)

    arrays, got = schema.read_shots(path)
    for key in schema.CHANNEL_KEYS:
        np.testing.assert_array_equal(arrays[key], np.stack([s[key] for s in shots]))
    assert got['sample_rate'] == 1234.5
    assert got['decimation'] == 256
    assert got['note'] == 'x'
    assert got[schema.NUM_SAMPLES_ATTR] == 3

    sub, _ = schema.read_shots(path, [schema.DRIVE_KEY], start=1, stop=3)
    assert list(sub) == [schema.DRIVE_KEY]
    np.testing.assert_array_equal(sub[schema.DRIVE_KEY][0], shots[1][schema.DRIVE_KEY])


def test_sample_rate_falls_back_only_when_attr_missing():
    assert schema.read_sample_rate({'sample_rate': 100.0}) == 100.0
    assert schema.read_sample_rate({}) == schema.LEGACY_SAMPLE_RATE_HZ == 488281.25
    assert pytest.approx(125e6 / 256) == schema.LEGACY_SAMPLE_RATE_HZ
