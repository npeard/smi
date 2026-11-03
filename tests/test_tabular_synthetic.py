"""Tests for the tabular models on synthetic frozen sets."""

import dataclasses

import h5py
import numpy as np
import pytest

from smi.data.datasets import select_pd_keys
from smi.data.frozen import RIG_WAVELENGTHS_UM
from smi.estimators.tabular import predict_trace, windowize
from smi.experiments import tabular_synthetic
from smi.experiments.evaluate import evaluate
from smi.experiments.grid import EvalSet, Grid, Sim
from smi.experiments.grids import grid
from smi.experiments.results_artifact import read_results
from smi.experiments.runner import build_sims

pytestmark = pytest.mark.usefixtures('small_chunks')

WAVELENGTHS = RIG_WAVELENGTHS_UM[:2]
WINDOW = 9


class _Mean:
    """Window predictor returning the mean of its features."""

    def predict(self, x):
        return x.mean(axis=1)


def test_predict_trace_matches_overlap_average():
    rng = np.random.default_rng(0)
    signals = rng.standard_normal((2, 40))
    trace = predict_trace(_Mean(), signals, WINDOW)
    assert trace.shape == (40,)
    x, _ = windowize(signals, np.zeros(40), WINDOW, 1)
    values = x.mean(axis=1)
    expected = np.zeros(40)
    counts = np.zeros(40)
    for start, value in enumerate(values):
        expected[start : start + WINDOW] += value
        counts[start : start + WINDOW] += 1
    np.testing.assert_allclose(trace, expected / counts)


def _tiny_grid() -> Grid:
    base = grid('q3_s1_families')
    return dataclasses.replace(
        base,
        channel_subsets=(WAVELENGTHS,),
        sims=(Sim('mich', 'noise_free', feedback_c=0.0, alpha=0.0),),
        test_set=EvalSet(seed=1000, n_shots=2),
    )


def _run(tmp_path, monkeypatch, out):
    monkeypatch.setattr(tabular_synthetic, 'grid', lambda _name: _tiny_grid())
    return tabular_synthetic.run_tabular(
        'tiny',
        ('ridge', 'lightgbm'),
        WINDOW,
        out,
        n_train_shots=4,
        train_seed=0,
        frozen_root=tmp_path / 'frozen',
        lock_path=tmp_path / 'lock',
    )


def test_records_match_network_metric_keys_and_are_deterministic(tmp_path, monkeypatch):
    records = _run(tmp_path, monkeypatch, tmp_path / 'a.json')
    assert [r.estimator for r in records] == [f'ridge-w{WINDOW}', f'lightgbm-w{WINDOW}']
    network_keys = {
        'velocity_rmse',
        'velocity_rmse_sign_invariant',
        'velocity_nrmse',
        'velocity_nrmse_sign_invariant',
        'displacement_rmse',
        'displacement_rmse_sign_invariant',
    }
    for r in records:
        assert set(r.metrics) == network_keys
        assert r.test_seed == 1000
        assert r.train_seed == 0
        assert r.cell_key.endswith('__mich__seed0')
    assert read_results(tmp_path / 'a.json') == records
    # A second call on the same artifact skips the recorded cells.
    assert _run(tmp_path, monkeypatch, tmp_path / 'a.json') == []
    again = _run(tmp_path, monkeypatch, tmp_path / 'b.json')
    assert [r.metrics for r in again] == [r.metrics for r in records]


def test_perfect_window_predictor_scores_zero_through_evaluate(tmp_path):
    """A window model that reads the truth scores RMSE 0 through evaluate()."""
    sim = build_sims(_tiny_grid().sims[0], WAVELENGTHS)[0][1]
    dataset = tabular_synthetic.load_or_create(sim, 1000, 2, tmp_path)
    # Overwrite the 635 nm channel with the true velocity, so a window-1 model
    # that returns its only feature is exact.
    with h5py.File(dataset, 'r+') as f:
        key = select_pd_keys(f, WAVELENGTHS[:1])[0]
        f[key][...] = f['velocity'][...]
    est = tabular_synthetic.TabularEstimator('id', 'ridge', _First(), 1, {})
    metrics = evaluate(est, dataset, WAVELENGTHS[:1], 2, True, 0).record.metrics
    assert metrics['velocity_rmse']['mean'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['velocity_nrmse']['mean'] == pytest.approx(0.0, abs=1e-6)


class _First:
    def predict(self, x):
        return x[:, 0]
