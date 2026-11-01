"""One evaluator: metrics, wavelength selection, checkpoints, and the oracle."""

import dataclasses
import json
import shutil

import h5py
import numpy as np
import pytest
import torch

from smi.analysis import synthetic_lit_module
from smi.analysis.evaluate import CheckpointEstimator, InverseFitEstimator, evaluate
from smi.analysis.results_artifact import (
    RunRecord,
    append_results,
    read_results,
    write_results,
)
from smi.analysis.synthetic_lit_module import SyntheticLitModule
from smi.core.kinematics import derivative_displacement, integrate_velocity
from smi.core.metrics import sign_invariant_rmse
from smi.data import frozen
from smi.data.datasets import VelocityDataset, select_pd_keys
from smi.data.frozen import pd_keys, write_frozen_set
from smi.theory.forward import SIMULATOR_VERSION
from tests._sims import SAMPLE_RATE, SMALL_CHUNK_SHOTS, michelson_sim
from tests._sims import WAVELENGTHS_UM as RIG

# What a runner passes from training; the oracle's convention is (True, None).
TRAINED = {'converged': True, 'train_seed': 3}


pytestmark = pytest.mark.usefixtures('small_chunks')


class _Truth:
    """Estimator that returns the file's velocity (times ``scale``)."""

    name = model_type = 'truth'
    causal = True
    receptive_field = 1
    total_params = 0
    device = 'cpu'

    def __init__(self, path, scale=1.0, offset=0.0):
        self.config = {}
        self.offset = offset
        with h5py.File(path, 'r') as f:
            self.velocity = torch.from_numpy(f['velocity'][:])
        self.scale = scale
        self.seen = []

    def predict_velocity(self, signals, _wavelengths_um):
        self.seen.append(signals)
        n = signals.shape[0]
        out = self.scale * self.velocity[self._at : self._at + n] + self.offset
        self._at += n
        return out

    _at = 0


@pytest.fixture(scope='module')
def michelson_set(tmp_path_factory):
    path = tmp_path_factory.mktemp('frozen') / 'michelson.h5'
    # Module scope runs before the function-scoped autouse monkeypatch.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(frozen, 'CHUNK_SHOTS', SMALL_CHUNK_SHOTS)
        return write_frozen_set(michelson_sim(), 3, 4, path)


def test_sign_invariant_rmse_per_shot():
    target = torch.tensor([[1.0, -2.0, 3.0], [1.0, 1.0, 1.0]])
    pred = torch.stack([-target[0], target[1] + 1.0])
    out = sign_invariant_rmse(pred, target)
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0]))


VELOCITY_METRICS = (
    'velocity_rmse',
    'velocity_rmse_sign_invariant',
    'velocity_nrmse',
    'velocity_nrmse_sign_invariant',
)


def test_perfect_estimator_scores_zero(michelson_set):
    result = evaluate(_Truth(michelson_set), michelson_set, RIG, None, **TRAINED)
    for key in VELOCITY_METRICS:
        assert result.record.metrics[key]['max'] == 0.0, key
    assert result.record.converged
    assert result.record.causal
    assert result.record.sim_config_hash == michelson_sim().hash()
    assert result.record.simulator_version == SIMULATOR_VERSION


def test_sign_flip_invariant_but_signed_differs(michelson_set):
    result = evaluate(
        _Truth(michelson_set, scale=-1.0), michelson_set, RIG, None, **TRAINED
    )
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
    result = evaluate(_Truth(michelson_set), michelson_set, RIG, 3, **TRAINED)
    n_samples = 3 * 16384
    for table in result.record.binned.values():
        assert sum(row['n'] for row in table) == n_samples
    # Quantile bins hold equal counts up to ties.
    counts = [r['n'] for r in result.record.binned['abs_velocity']]
    assert max(counts) - min(counts) < 0.02 * n_samples


def test_wavelength_subset_reorders_and_selects(michelson_set):
    est = _Truth(michelson_set)
    evaluate(est, michelson_set, (0.515, 0.635), 2, **TRAINED)
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
    result = evaluate(est, path, (0.515,), None, **TRAINED)
    with h5py.File(path, 'r') as f:
        torch.testing.assert_close(est.seen[0][:, 0], torch.from_numpy(f['RP2_CH2'][:]))
    assert result.record.wavelengths_um == [0.515]
    assert result.record.sim_config_hash is None
    assert result.record.simulator_version is None


def test_velocity_dataset_selects_channels_by_wavelength(tmp_path):
    path = _real_schema_file(tmp_path / 'acq.h5')
    ds = VelocityDataset(path, wavelengths_um=(0.515, 0.635))
    assert ds.pd_channel_keys == ['RP2_CH2', 'RP1_CH2']


def _tcn_module(target):
    """A tiny TCN on the Michelson sim, predicting ``target``."""
    velocity_weight = 1.0 if target == 'velocity' else 0.0
    return SyntheticLitModule(
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
            'velocity_loss_weight': velocity_weight,
            'displacement_loss_weight': 1.0,
            'sign_invariant': False,
        },
        training_hparams={'target': target},
        data_hparams={},
        steps_per_epoch=1,
        train_crop_length=None,
        sim=dataclasses.asdict(michelson_sim()),
        restoring_checkpoint=False,
    )


def test_checkpoint_estimator_uses_baked_normalization_and_checks_channels(
    tmp_path, michelson_set, monkeypatch
):
    module = _tcn_module('velocity')
    ckpt = tmp_path / 'm.ckpt'
    torch.save(
        {'hyper_parameters': dict(module.hparams), 'state_dict': module.state_dict()},
        ckpt,
    )

    # Restoring overwrites the buffers, so the 256-shot stats are not re-run.
    def refuse(*_args):
        pytest.fail('normalization_stats ran on restore')

    monkeypatch.setattr(synthetic_lit_module, 'normalization_stats', refuse)
    est = CheckpointEstimator(ckpt, 'cpu')
    assert est.causal
    assert est.receptive_field == module.inner_model.receptive_field
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        expected = module.eval().model(x)[:, 0]
    torch.testing.assert_close(est.predict_velocity(x, RIG), expected)
    with pytest.raises(ValueError, match='trained on'):
        est.predict_velocity(x[:, :2], RIG[:2])
    with pytest.raises(ValueError, match='trained on'):
        est.predict_velocity(x, RIG[::-1])

    out = tmp_path / 'results.json'
    for _ in range(2):
        append_results(out, [evaluate(est, michelson_set, RIG, 2, **TRAINED).record])
    records = read_results(out)
    assert len(records) == 2
    assert records[0].estimator.startswith('TCN:')
    assert json.loads(out.read_text())['schema_version'] == 1


def test_displacement_checkpoint_is_differentiated_before_scoring(tmp_path):
    module = _tcn_module('displacement')
    ckpt = tmp_path / 'd.ckpt'
    torch.save(
        {'hyper_parameters': dict(module.hparams), 'state_dict': module.state_dict()},
        ckpt,
    )
    est = CheckpointEstimator(ckpt, 'cpu')
    x = torch.randn(2, 3, 512)
    with torch.no_grad():
        displacement = module.eval().model(x)[:, 0]
    torch.testing.assert_close(
        est.predict_velocity(x, RIG), derivative_displacement(displacement, SAMPLE_RATE)
    )


def test_recurrent_checkpoint_records_unbounded_receptive_field(
    tmp_path, michelson_set
):
    module = SyntheticLitModule(
        model_hparams=dict(
            type='LSTM',
            sequence_length=1024,
            in_channels=3,
            hidden_size=4,
            num_layers=1,
            dropout=0.0,
            bidirectional=True,
            receptive_field='unbounded',
        ),
        optimizer_hparams={'name': 'Adam', 'lr': 1e-3, 'weight_decay': 0.0},
        scheduler_hparams={},
        loss_hparams={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 1.0,
            'sign_invariant': False,
        },
        training_hparams={'target': 'velocity'},
        data_hparams={},
        steps_per_epoch=1,
        train_crop_length=None,
        sim=dataclasses.asdict(michelson_sim()),
        restoring_checkpoint=False,
    )
    ckpt = tmp_path / 'lstm.ckpt'
    torch.save(
        {'hyper_parameters': dict(module.hparams), 'state_dict': module.state_dict()},
        ckpt,
    )
    out = tmp_path / 'results.json'
    result = evaluate(
        CheckpointEstimator(ckpt, 'cpu'), michelson_set, RIG, 2, **TRAINED
    )
    append_results(out, [result.record])
    (record,) = read_results(out)
    assert record.receptive_field is None
    assert (
        json.loads(out.read_text())['datasets'][record.dataset][0]['receptive_field']
        is None
    )


def _per_shot_nrmse_reference(pred, target):
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=-1))
    return rmse / np.std(target, axis=-1)


def test_scaled_prediction_has_exact_nrmse(michelson_set):
    result = evaluate(
        _Truth(michelson_set, scale=0.5), michelson_set, RIG, None, **TRAINED
    )
    with h5py.File(michelson_set, 'r') as f:
        v = f['velocity'][:].astype(np.float64)
    expected = 0.5 * np.sqrt(np.mean(v**2, axis=-1)) / np.std(v, axis=-1)
    np.testing.assert_allclose(result.per_shot['velocity_nrmse'], expected, rtol=1e-4)
    np.testing.assert_allclose(
        result.per_shot['velocity_nrmse_sign_invariant'], expected, rtol=1e-4
    )
    np.testing.assert_allclose(
        result.per_shot['velocity_nrmse'],
        _per_shot_nrmse_reference(0.5 * v, v),
        rtol=1e-4,
    )


def test_sign_invariance_is_per_shot_not_per_sample():
    target = torch.linspace(1.0, 2.0, 8).repeat(3, 1)
    flips = torch.ones(8)
    flips[4:] = -1.0
    # Shot 0 flips mid-shot, shot 1 is negated whole, shot 2 is untouched.
    pred = torch.stack([flips * target[0], -target[1], target[2]])
    out = sign_invariant_rmse(pred, target)
    # Negating the pred wins on the smaller first half: error 2t there, 0 after.
    mid_shot = torch.sqrt(torch.sum((2 * target[0, :4]) ** 2) / 8)
    # A per-sample sign choice would give 0 for shot 0; a per-shot one cannot.
    torch.testing.assert_close(
        out, torch.stack([mid_shot, torch.tensor(0.0), torch.tensor(0.0)])
    )
    assert out[0] > 0


def test_rig_wavelength_spelling_resolves_to_stored_channel(tmp_path):
    path = _real_schema_file(tmp_path / 'acq.h5')
    with h5py.File(path, 'r') as f:
        # The rig stores 0.675 for the laser the checkpoints spell 0.6748.
        assert select_pd_keys(f, (0.6748,)) == select_pd_keys(f, (0.675,))
        with pytest.raises(ValueError, match='matches'):
            select_pd_keys(f, (0.5,))


def test_unknown_wavelength_raises_through_evaluate(tmp_path):
    path = _real_schema_file(tmp_path / 'acq.h5')

    class Never(_Truth):
        def __init__(self):
            self.config = {}

    with pytest.raises(ValueError, match='matches'):
        evaluate(Never(), path, (0.5,), None, **TRAINED)


def test_zero_std_shot_raises_naming_the_shot(tmp_path):
    path = tmp_path / 'flat.h5'
    rng = np.random.default_rng(0)
    velocity = rng.normal(size=(3, 64)).astype('float32')
    velocity[1] = 0.25
    with h5py.File(path, 'w') as f:
        f.attrs['sample_rate'] = SAMPLE_RATE
        f.attrs['pd_wavelengths_um'] = np.asarray(RIG[:1])
        f['velocity'] = velocity
        f['displacement'] = np.cumsum(velocity, axis=1)
        for key in pd_keys(RIG[:1]):
            f[key] = rng.normal(size=(3, 64)).astype('float32')
    with pytest.raises(ValueError, match='shot 1'):
        evaluate(_Truth(path), path, RIG[:1], None, **TRAINED)


def _stored_phase_file(path):
    """Three-wavelength file whose optical phase differs per channel."""
    rng = np.random.default_rng(0)
    shots, length = 2, 32
    velocity = rng.normal(size=(shots, length)).astype('float32')
    phase = np.zeros((shots, 3, length))
    phase[:, 0] = np.pi / 2  # 0.635 um channel: farthest from an extremum
    phase[:, 1] = np.pi / 2
    phase[:, 2] = 3 * np.pi + 0.3  # 0.515 um channel: distance 0.3 to 3 pi
    with h5py.File(path, 'w') as f:
        f.attrs['sample_rate'] = SAMPLE_RATE
        f.attrs['pd_wavelengths_um'] = np.asarray(RIG)
        f['velocity'] = velocity
        f['displacement'] = np.cumsum(velocity, axis=1)
        f['optical_phase'] = phase
        for key in pd_keys(RIG):
            f[key] = rng.normal(size=(shots, length)).astype('float32')
    return path


def test_fringe_distance_bin_uses_shortest_wavelength_phase_column(tmp_path):
    path = _stored_phase_file(tmp_path / 'phase.h5')
    result = evaluate(_Truth(path, offset=1.0), path, RIG, None, **TRAINED)
    table = result.record.binned['fringe_distance']
    # 0.3 rad falls in [pi/20 * 1, pi/20 * 2) = [0.157, 0.314): bin 1.
    assert [row['n'] for row in table] == [0, 2 * 32] + [0] * 8
    assert table[1]['rmse'] == pytest.approx(1.0, rel=1e-5)
    # The same channel asked for alone is still found by wavelength.
    only = evaluate(_Truth(path, offset=1.0), path, (0.515,), None, **TRAINED)
    assert only.record.binned['fringe_distance'][1]['n'] == 2 * 32


def _integrable_file(path, offset_um):
    """File whose displacement is exactly ``integrate_velocity`` of its velocity.

    A frozen set's velocity is the spectral derivative of its displacement, so
    integrating it back carries a small discretization error; here the two
    agree to rounding, which gives the displacement metrics a known answer.
    ``offset_um`` is a per-shot start offset the metric must ignore.
    """
    rng = np.random.default_rng(1)
    shots, length = 3, 256
    velocity = np.cumsum(rng.normal(size=(shots, length)), axis=1).astype('float32')
    displacement = (
        integrate_velocity(torch.from_numpy(velocity).double(), SAMPLE_RATE).numpy()
        + np.asarray(offset_um)[:, None]
    )
    with h5py.File(path, 'w') as f:
        f.attrs['sample_rate'] = SAMPLE_RATE
        f.attrs['pd_wavelengths_um'] = np.asarray(RIG[:1])
        f['velocity'] = velocity
        f['displacement'] = displacement
        for key in pd_keys(RIG[:1]):
            f[key] = rng.normal(size=(shots, length)).astype('float32')
    return path, displacement - displacement[:, :1]


def test_perfect_velocity_scores_zero_displacement_rmse(tmp_path):
    path, _ = _integrable_file(tmp_path / 'int.h5', [5.0, -2.0, 0.0])
    result = evaluate(_Truth(path), path, RIG[:1], None, **TRAINED)
    for key in ('displacement_rmse', 'displacement_rmse_sign_invariant'):
        assert result.record.metrics[key]['max'] < 1e-9, key


def test_constant_velocity_bias_gives_the_ramp_rms(tmp_path):
    path, _ = _integrable_file(tmp_path / 'int.h5', [5.0, -2.0, 0.0])
    bias = 0.5
    result = evaluate(_Truth(path, offset=bias), path, RIG[:1], None, **TRAINED)
    # Integrated bias is the ramp b * dt * k, k = 0..L-1, once offsets go.
    dt_ms = 1e3 / SAMPLE_RATE
    ramp_rms = bias * dt_ms * np.sqrt(np.mean(np.arange(256.0) ** 2))
    np.testing.assert_allclose(
        result.per_shot['displacement_rmse'], ramp_rms, rtol=1e-5
    )


def test_sign_flip_scores_zero_sign_invariant_displacement(tmp_path):
    path, centered = _integrable_file(tmp_path / 'int.h5', [5.0, -2.0, 0.0])
    result = evaluate(_Truth(path, scale=-1.0), path, RIG[:1], None, **TRAINED)
    assert result.record.metrics['displacement_rmse_sign_invariant']['max'] < 1e-9
    np.testing.assert_allclose(
        result.per_shot['displacement_rmse'],
        2 * np.sqrt(np.mean(centered**2, axis=1)),
        rtol=1e-5,
    )


def test_displacement_sign_invariant_is_the_per_shot_min_over_signs(tmp_path):
    # A small negative scale plus a bias: velocity and displacement can then
    # prefer different signs, which is where reusing the velocity's sign would
    # score above the signed value.
    path, _ = _integrable_file(tmp_path / 'int.h5', [5.0, -2.0, 0.0])
    plus = evaluate(_Truth(path, -0.1, -1.0), path, RIG[:1], None, **TRAINED)
    minus = evaluate(_Truth(path, 0.1, 1.0), path, RIG[:1], None, **TRAINED)
    signed = plus.per_shot['displacement_rmse']
    invariant = plus.per_shot['displacement_rmse_sign_invariant']
    assert np.all(invariant <= signed)
    np.testing.assert_allclose(
        invariant, np.minimum(signed, minus.per_shot['displacement_rmse']), rtol=1e-12
    )


def test_record_labels_training_and_test_seeds_and_finiteness(tmp_path, michelson_set):
    est = _Truth(michelson_set, scale=float('nan'))
    out = tmp_path / 'r.json'
    result = evaluate(est, michelson_set, RIG, 2, converged=False, train_seed=11)
    append_results(out, [result.record])
    entry = next(iter(json.loads(out.read_text())['datasets'].values()))[0]
    assert entry['train_seed'] == 11
    assert entry['test_seed'] == 3
    assert entry['converged'] is False
    assert entry['finite_predictions'] is False
    assert entry['causal'] is True
    (record,) = read_results(out)
    assert record.train_seed == 11

    ok = evaluate(
        _Truth(michelson_set), michelson_set, RIG, 2, converged=True, train_seed=None
    )
    assert ok.record.finite_predictions
    assert ok.record.train_seed is None
    assert ok.record.test_seed == 3


def test_file_without_a_seed_records_no_test_seed(tmp_path, michelson_set):
    path = tmp_path / 'noseed.h5'
    shutil.copyfile(michelson_set, path)
    with h5py.File(path, 'r+') as f:
        del f.attrs['seed']
    result = evaluate(_Truth(path), path, RIG, 2, converged=True, train_seed=None)
    assert result.record.test_seed is None


def test_records_without_new_fields_still_read_and_serialize_causal(tmp_path):
    old = RunRecord(
        model_type='TCN',
        config={},
        total_params=1,
        receptive_field=1,
        causal=False,
        seed=0,
        device='cpu',
        dataset='d',
        metrics={},
    )
    path = tmp_path / 'old.json'
    write_results(path, [old])
    entry = json.loads(path.read_text())['datasets']['d'][0]
    assert entry['causal'] is False
    assert read_results(path)[0].train_seed is None


def test_inverse_fit_noise_free_oracle(tmp_path):
    path = write_frozen_set(michelson_sim(), 7, ORACLE_SHOTS, tmp_path / 'oracle.h5')
    est = InverseFitEstimator.for_dataset(path, None)
    result = evaluate(est, path, RIG, None, converged=True, train_seed=None)
    nrmse = result.record.metrics['velocity_nrmse_sign_invariant']['mean']
    assert not result.record.causal
    assert nrmse < ORACLE_NRMSE_BOUND


# 16 shots took more than two minutes on CPU (about 8 s per shot), so 4.
ORACLE_SHOTS = 4
# Measured mean sign-invariant velocity NRMSE 0.280 (signed 1.692: the decoder
# returns an arbitrary sign per shot) on 4 shots, seed 7; pinned at 1.5x.
ORACLE_NRMSE_BOUND = 0.42
