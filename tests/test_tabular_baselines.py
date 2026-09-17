"""Tests for the tabular (non-torch) baselines."""

import numpy as np
import pytest

from smi.analysis.baselines.tabular import (
    LIGHTGBM_AVAILABLE,
    fit_lightgbm,
    fit_ridge,
    predict_windows,
    windowize,
)


def test_windowize_shapes():
    signals = np.random.default_rng(0).standard_normal((3, 2048))
    targets = np.random.default_rng(1).standard_normal(2048)
    x, y = windowize(signals, targets, window=256, stride=64)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 3 * 256


def test_windowize_preserves_channel_major_layout():
    """Feature k*window + j must be channel k at offset j, or the fit is noise."""
    signals = np.arange(2 * 16, dtype=float).reshape(2, 16)
    targets = np.zeros(16)
    x, _ = windowize(signals, targets, window=4, stride=4)
    # First window, second channel: samples 16..19 of the flat arange.
    np.testing.assert_array_equal(x[0, 4:8], [16.0, 17.0, 18.0, 19.0])


def test_windowize_labels_the_window_centre():
    signals = np.zeros((1, 16))
    targets = np.arange(16, dtype=float)
    _, y = windowize(signals, targets, window=4, stride=4)
    np.testing.assert_array_equal(y, [2.0, 6.0, 10.0, 14.0])


def test_windowize_rejects_a_window_longer_than_the_signal():
    with pytest.raises(ValueError, match='exceeds signal length'):
        windowize(np.zeros((2, 8)), np.zeros(8), window=16, stride=1)


def test_ridge_recovers_a_linear_relationship():
    """A linear target must be fit near-exactly; otherwise the path is wrong."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 12))
    y = x @ rng.standard_normal(12)
    model = fit_ridge(x, y, alpha=1e-6)
    assert np.sqrt(np.mean((predict_windows(model, x) - y) ** 2)) < 1e-3


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason='lightgbm not installed')
def test_lightgbm_learns_a_nonlinear_relationship():
    """Trees must beat predicting the mean on a target no linear fit captures."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2048, 4))
    y = np.sign(x[:, 0]) * np.abs(x[:, 1])
    model = fit_lightgbm(x, y, {'num_boost_round': 60, 'min_data_in_leaf': 5})
    rmse = np.sqrt(np.mean((predict_windows(model, x) - y) ** 2))
    assert rmse < 0.5 * np.std(y)


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason='lightgbm not installed')
def test_lightgbm_and_ridge_share_a_prediction_shape():
    """predict_windows must flatten both to [n_samples] so callers can't branch."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 6))
    y = x @ rng.standard_normal(6)
    booster = fit_lightgbm(x, y, {'num_boost_round': 5})
    ridge = fit_ridge(x, y, alpha=1.0)
    assert predict_windows(booster, x).shape == (128,)
    assert predict_windows(ridge, x).shape == (128,)


def test_torch_and_the_tabular_stack_coexist():
    """The OpenMP hazard CLAUDE.md documents, asserted rather than assumed.

    Two OpenMP runtimes in one process abort at np.corrcoef. This test fails
    loudly if a future resolve reintroduces that, instead of the suite dying
    mid-run somewhere unrelated.
    """
    import torch  # noqa: PLC0415 -- import order is the thing under test

    from smi.analysis.baselines import tabular  # noqa: F401, PLC0415

    rng = np.random.default_rng(0)
    corr = np.corrcoef(rng.standard_normal(4096), rng.standard_normal(4096))[0, 1]
    assert np.isfinite(corr)
    assert torch.randn(8, 8).sum().isfinite()
