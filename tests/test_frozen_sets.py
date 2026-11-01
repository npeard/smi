"""Frozen synthetic test sets: byte-stable, hash-keyed, loader-compatible."""

import dataclasses
import json

import h5py
import numpy as np
import pytest
import torch

from smi.analysis.datasets import VelocityDataset
from smi.synthetic import frozen
from smi.synthetic.forward import NOMINAL_DRIVE_GAIN, SIMULATOR_VERSION, SimConfig
from smi.synthetic.frozen import (
    frozen_set_path,
    load_or_create,
    pd_keys,
    write_frozen_set,
)
from tests._sims import SAMPLE_RATE, WAVELENGTHS_UM

pytestmark = pytest.mark.usefixtures('small_chunks')


def _cfg(wavelengths=WAVELENGTHS_UM, feedback_c=0.5, alpha=3.0) -> SimConfig:
    return SimConfig(
        wavelengths_um=wavelengths,
        sample_rate=SAMPLE_RATE,
        start_freq_hz=1.0,
        end_freq_hz=1000.0,
        drive_gain=NOMINAL_DRIVE_GAIN,
        feedback_c=feedback_c,
        alpha=alpha,
        snr_db=30.0,
        shot_noise=True,
        visibility_range=(0.6, 1.0),
        dc_offset_range=(-0.1, 0.1),
        amplitude_drift_rel=0.05,
        phase_drift_rad_per_sqrt_s=0.1,
        contrast_drift_rel=0.05,
        pd_bandwidth_hz=2000.0,
        adc_bits=12,
        adc_full_scale=2.0,
    )


def _read(path) -> dict[str, np.ndarray]:
    with h5py.File(path, 'r') as f:
        return {k: f[k][:] for k in f}


def test_same_seed_identical_bytes(tmp_path):
    cfg = _cfg()
    a = write_frozen_set(cfg, 7, 5, tmp_path / 'a.h5')
    b = write_frozen_set(cfg, 7, 5, tmp_path / 'b.h5')
    assert a.read_bytes() == b.read_bytes()
    other = _read(write_frozen_set(cfg, 8, 5, tmp_path / 'c.h5'))
    assert not np.array_equal(_read(a)['velocity'], other['velocity'])


def test_prefix_independent_of_n_shots(tmp_path):
    cfg = _cfg()
    small = _read(write_frozen_set(cfg, 3, 3, tmp_path / 's.h5'))
    large = _read(write_frozen_set(cfg, 3, 10, tmp_path / 'l.h5'))
    for name, value in small.items():
        np.testing.assert_array_equal(value, large[name][:3])


def test_hash_changes_with_any_field(tmp_path):
    cfg = _cfg()
    perturbed = {
        'wavelengths_um': (0.635, 0.6748, 0.5),
        'sample_rate': SAMPLE_RATE * 2,
        'start_freq_hz': 2.0,
        'end_freq_hz': 900.0,
        'drive_gain': 0.003,
        'feedback_c': 0.4,
        'alpha': 2.0,
        'snr_db': 20.0,
        'shot_noise': False,
        'visibility_range': (0.5, 1.0),
        'dc_offset_range': (-0.2, 0.1),
        'amplitude_drift_rel': 0.06,
        'phase_drift_rad_per_sqrt_s': 0.2,
        'contrast_drift_rel': 0.06,
        'pd_bandwidth_hz': 1500.0,
        'adc_bits': 10,
        'adc_full_scale': 3.0,
    }
    assert set(perturbed) == {f.name for f in dataclasses.fields(SimConfig)}
    paths = {frozen_set_path(cfg, 0, 4, tmp_path)}
    for name, value in perturbed.items():
        # Bypass validation: only the identity of the config is under test.
        other = dataclasses.replace(cfg)
        object.__setattr__(other, name, value)
        paths.add(frozen_set_path(other, 0, 4, tmp_path))
    assert len(paths) == len(perturbed) + 1
    assert len(frozen_set_path(cfg, 1, 4, tmp_path).name) == len(
        frozen_set_path(cfg, 0, 4, tmp_path).name
    )
    assert frozen_set_path(cfg, 1, 4, tmp_path) != frozen_set_path(cfg, 0, 4, tmp_path)
    assert frozen_set_path(cfg, 0, 5, tmp_path) != frozen_set_path(cfg, 0, 4, tmp_path)


def test_schema_attrs_and_layout(tmp_path):
    cfg = _cfg()
    path = write_frozen_set(cfg, 11, 5, tmp_path / 'x.h5')
    with h5py.File(path, 'r') as f:
        assert f.attrs['sample_rate'] == SAMPLE_RATE
        assert f.attrs['seed'] == 11
        assert f.attrs['simulator_version'] == SIMULATOR_VERSION
        np.testing.assert_array_equal(f.attrs['pd_wavelengths_um'], WAVELENGTHS_UM)
        assert SimConfig(**json.loads(f.attrs['sim_config'])) == cfg
        length = f['RP1_CH1'].shape[1]
        for key in ('RP1_CH1', 'RP1_CH2', 'RP2_CH1', 'RP2_CH2', 'velocity'):
            ds = f[key]
            assert ds.shape == (5, length)
            assert ds.dtype == np.float32
            assert ds.chunks == (1, length)
            assert ds.compression == 'gzip'
            assert ds.compression_opts == 4
        assert f['optical_phase'].shape == (5, 3, length)
    assert load_or_create(cfg, 11, 5, tmp_path) == frozen_set_path(cfg, 11, 5, tmp_path)
    assert frozen_set_path(cfg, 11, 5, tmp_path).read_bytes() == path.read_bytes()


def test_pd_keys():
    assert pd_keys(WAVELENGTHS_UM) == ['RP1_CH2', 'RP2_CH1', 'RP2_CH2']
    assert pd_keys((0.635, 0.675, 0.515)) == ['RP1_CH2', 'RP2_CH1', 'RP2_CH2']
    assert pd_keys((0.635, 0.515, 0.6748)) == ['PD_0', 'PD_1', 'PD_2']
    assert pd_keys((0.635, 0.6748)) == ['PD_0', 'PD_1']


def test_non_rig_channels_use_generic_keys(tmp_path):
    path = write_frozen_set(_cfg(wavelengths=(0.635, 0.6748)), 0, 2, tmp_path / 'g.h5')
    data = _read(path)
    assert {'PD_0', 'PD_1', 'RP1_CH1'} <= set(data)
    assert data['optical_phase'].shape[1] == 2


def test_existing_loader_reads_frozen_set(tmp_path):
    path = write_frozen_set(_cfg(), 5, 3, tmp_path / 'r.h5')
    stored = _read(path)
    dataset = VelocityDataset(path, num_pd_channels=3)
    assert len(dataset) == 3
    for i in range(3):
        signals, velocity, displacement = dataset[i]
        np.testing.assert_array_equal(
            signals.numpy(),
            np.stack([stored[k][i] for k in ('RP1_CH2', 'RP2_CH1', 'RP2_CH2')]),
        )
        scale = float(np.abs(stored['velocity'][i]).max())
        torch.testing.assert_close(
            velocity, torch.from_numpy(stored['velocity'][i]), atol=1e-5 * scale, rtol=0
        )
        scale = float(np.abs(stored['displacement'][i]).max())
        torch.testing.assert_close(
            displacement,
            torch.from_numpy(stored['displacement'][i]),
            atol=1e-5 * scale,
            rtol=0,
        )


def test_load_or_create_regenerates_after_chunk_change(tmp_path, monkeypatch):
    cfg = _cfg()
    path = load_or_create(cfg, 2, 5, tmp_path)
    before = _read(path)
    assert load_or_create(cfg, 2, 5, tmp_path) == path
    monkeypatch.setattr(frozen, 'CHUNK_SHOTS', 2)
    load_or_create(cfg, 2, 5, tmp_path)
    with h5py.File(path, 'r') as f:
        assert f.attrs['chunk_shots'] == 2
    assert not np.array_equal(_read(path)['velocity'], before['velocity'])
    assert not list(tmp_path.glob('*.part'))


def test_load_or_create_regenerates_after_simulator_version_change(
    tmp_path, monkeypatch
):
    cfg = _cfg()
    path = load_or_create(cfg, 2, 5, tmp_path)
    monkeypatch.setattr(frozen, 'SIMULATOR_VERSION', SIMULATOR_VERSION + 1)
    assert load_or_create(cfg, 2, 5, tmp_path) == path
    with h5py.File(path, 'r') as f:
        assert f.attrs['simulator_version'] == SIMULATOR_VERSION + 1
