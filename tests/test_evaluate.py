"""One evaluator: metrics, wavelength selection, checkpoints, and the oracle."""

import dataclasses
import json

import h5py
import numpy as np
import pytest
import torch

from smi.analysis.datasets import VelocityDataset
from smi.analysis.evaluate import (
    CheckpointEstimator,
    InverseFitEstimator,
    evaluate,
    evaluate_to_artifact,
)
from smi.analysis.metrics import sign_invariant_rmse
from smi.analysis.results_artifact import read_results
from smi.analysis.synthetic_lit_module import SyntheticLitModule
from smi.synthetic import frozen
from smi.synthetic.forward import NOMINAL_DRIVE_GAIN, SimConfig
from smi.synthetic.frozen import write_frozen_set

SAMPLE_RATE = 125e6 / 256
RIG = (0.635, 0.6748, 0.515)


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(frozen, 'CHUNK_SHOTS', 4)


def _michelson() -> SimConfig:
    return SimConfig.noise_free(
        wavelengths_um=RIG,
        sample_rate=SAMPLE_RATE,
        start_freq_hz=1.0,
        end_freq_hz=1000.0,
        drive_gain=NOMINAL_DRIVE_GAIN,
        feedback_c=0.0,
        alpha=0.0,
    )


class _Truth:
    """Estimator that returns the file's velocity (times ``scale``)."""

    name = model_type = 'truth'
    causal = True
    receptive_field = 1
    total_params = 0
    device = 'cpu'

    def __init__(self, path, scale=1.0):
        self.config = {}
        with h5py.File(path, 'r') as f:
            self.velocity = torch.from_numpy(f['velocity'][:])
        self.scale = scale
        self.seen = []

    def predict_velocity(self, signals, _wavelengths_um):
        self.seen.append(signals)
        n = signals.shape[0]
        out = self.scale * self.velocity[self._at : self._at + n]
        self._at += n
        return out

    _at = 0


@pytest.fixture(scope='module')
def michelson_set(tmp_path_factory):
    path = tmp_path_factory.mktemp('frozen') / 'michelson.h5'
    # Module scope runs before the function-scoped autouse monkeypatch.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(frozen, 'CHUNK_SHOTS', 4)
        return write_frozen_set(_michelson(), 3, 4, path)


def test_sign_invariant_rmse_per_shot():
    target = torch.tensor([[1.0, -2.0, 3.0], [1.0, 1.0, 1.0]])
    pred = torch.stack([-target[0], target[1] + 1.0])
    out = sign_invariant_rmse(pred, target)
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0]))


def test_perfect_estimator_scores_zero(michelson_set):
    result = evaluate(_Truth(michelson_set), michelson_set, RIG, None)
    for key, summary in result.record.metrics.items():
        assert summary['max'] == 0.0, key
    assert result.record.converged
    assert result.record.causal
    assert result.record.sim_config_hash == _michelson().hash()


def test_sign_flip_invariant_but_signed_differs(michelson_set):
    result = evaluate(_Truth(michelson_set, scale=-1.0), michelson_set, RIG, None)
    m = result.record.metrics
    assert m['velocity_rmse_sign_invariant']['max'] == 0.0
    # Flipped prediction: signed error is 2|v|, so NRMSE is 2 * rms(v) / std(v).
    assert m['velocity_nrmse']['min'] > 1.9
    assert m['velocity_nrmse_sign_invariant']['max'] == 0.0
    for table in result.record.binned.values():
        assert len(table) == 10
        assert all(row['rmse_sign_invariant'] == 0.0 for row in table if row['n'])
        assert any(row['rmse'] > 0.0 for row in table if row['n'])


def test_bins_partition_every_sample(michelson_set):
    result = evaluate(_Truth(michelson_set), michelson_set, RIG, 3)
    n_samples = 3 * 16384
    for table in result.record.binned.values():
        assert sum(row['n'] for row in table) == n_samples
    # Quantile bins hold equal counts up to ties.
    counts = [r['n'] for r in result.record.binned['abs_velocity']]
    assert max(counts) - min(counts) < 0.02 * n_samples


def test_wavelength_subset_reorders_and_selects(michelson_set):
    est = _Truth(michelson_set)
    evaluate(est, michelson_set, (0.515, 0.635), 2)
    with h5py.File(michelson_set, 'r') as f:
        expected = np.stack([f['RP2_CH2'][:2], f['RP1_CH2'][:2]], axis=1)
    np.testing.assert_array_equal(est.seen[0].numpy(), expected)


def _real_schema_file(path, length=2048):
    rng = np.random.default_rng(0)
    t = np.arange(length) / 1e4
    with h5py.File(path, 'w') as f:
        f['RP1_CH1'] = np.stack([np.sin(2 * np.pi * 50 * t)] * 2).astype('float32')
        for key in ('RP1_CH2', 'RP2_CH1', 'RP2_CH2'):
            f[key] = rng.normal(size=(2, length)).astype('float32')
    return path


def test_subset_by_wavelength_on_real_schema(tmp_path):
    path = _real_schema_file(tmp_path / 'acq.h5')

    class Spy(_Truth):
        def __init__(self):
            self.seen = []
            self.config = {}

        def predict_velocity(self, signals, _wavelengths_um):
            self.seen.append(signals)
            return torch.zeros(signals.shape[0], signals.shape[2])

    est = Spy()
    result = evaluate(est, path, (0.515,), None)
    with h5py.File(path, 'r') as f:
        torch.testing.assert_close(est.seen[0][:, 0], torch.from_numpy(f['RP2_CH2'][:]))
    assert result.record.wavelengths_um == [0.515]
    assert result.record.sim_config_hash is None


def test_velocity_dataset_selects_channels_by_wavelength(tmp_path):
    path = _real_schema_file(tmp_path / 'acq.h5')
    ds = VelocityDataset(path, wavelengths_um=(0.515, 0.635))
    assert ds.pd_channel_keys == ['RP2_CH2', 'RP1_CH2']


def test_checkpoint_estimator_uses_baked_normalization_and_checks_channels(
    tmp_path, michelson_set
):
    module = SyntheticLitModule(
        model_hparams=dict(
            type='TCN',
            sequence_length=1024,
            in_channels=3,
            activation='GELU',
            use_layer_norm=True,
            use_weight_norm=False,
            kernel_size=3,
            temporal_channels=[8, 8],
            dilation_base=2,
            dropout=0.0,
        ),
        optimizer_hparams={'name': 'Adam', 'lr': 1e-3, 'weight_decay': 0.0},
        scheduler_hparams={},
        loss_hparams={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 1.0,
        },
        training_hparams={'target': 'velocity'},
        data_hparams={},
        steps_per_epoch=1,
        sim=dataclasses.asdict(_michelson()),
    )
    ckpt = tmp_path / 'm.ckpt'
    torch.save(
        {'hyper_parameters': dict(module.hparams), 'state_dict': module.state_dict()},
        ckpt,
    )
    est = CheckpointEstimator(ckpt, 'cpu')
    assert est.causal
    assert est.receptive_field == module._inner_model.receptive_field
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        expected = module.eval().model(x)[:, 0]
    torch.testing.assert_close(est.predict_velocity(x, RIG), expected)
    with pytest.raises(ValueError, match='trained on'):
        est.predict_velocity(x[:, :2], RIG[:2])
    with pytest.raises(ValueError, match='trained on'):
        est.predict_velocity(x, RIG[::-1])

    out = tmp_path / 'results.json'
    evaluate_to_artifact(est, michelson_set, RIG, 2, out)
    evaluate_to_artifact(est, michelson_set, RIG, 2, out)
    records = read_results(out)
    assert len(records) == 2
    assert records[0].estimator.startswith('TCN:')
    assert json.loads(out.read_text())['schema_version'] == 1


def test_inverse_fit_noise_free_oracle(tmp_path):
    path = write_frozen_set(_michelson(), 7, ORACLE_SHOTS, tmp_path / 'oracle.h5')
    est = InverseFitEstimator.for_dataset(path, None)
    result = evaluate(est, path, RIG, None)
    nrmse = result.record.metrics['velocity_nrmse_sign_invariant']['mean']
    assert not result.record.causal
    assert nrmse < ORACLE_NRMSE_BOUND


# 16 shots took more than two minutes on CPU (about 8 s per shot), so 4.
ORACLE_SHOTS = 4
# Measured mean sign-invariant velocity NRMSE 0.280 (signed 1.692: the decoder
# returns an arbitrary sign per shot) on 4 shots, seed 7; pinned at 1.5x.
ORACLE_NRMSE_BOUND = 0.42
