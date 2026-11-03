#!/usr/bin/env python3

"""Tabular baselines: gradient-boosted trees and ridge regression.

Both consume the same windowed view of a shot that the windowed torch models
do -- ``window`` samples of every channel flattened into one feature vector,
predicting the target at the window's centre -- so their error numbers sit in
the same table as the networks' without a caveat about a different task.
"""

from typing import Any, Protocol

import numpy as np
from sklearn.linear_model import Ridge

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised only in a stripped env
    # LightGBM is in the `simple` pixi feature. An env built without it should
    # skip the boosted-tree baseline, not fail to import the module that also
    # holds the ridge baseline and the shared windowing.
    LIGHTGBM_AVAILABLE = False

#: Whether the tabular baselines are causal: no, since :func:`windowize` labels
#: each window at its centre, so the prediction uses the half window after it.
CAUSAL = False

#: Window length, matched to the Barland receptive field.
WINDOW = 91
#: Hop between *training* windows. Dense strides would make a 16384-long shot
#: into 16k rows; 64 keeps the design matrix tractable without changing what
#: the model is asked to predict. Evaluation is always stride 1
#: (:func:`predict_trace`): at stride s the reconstructed trace is piecewise
#: constant over s samples, a discretization floor on the RMSE independent of
#: the model (0.052 um/ms at s=64, against a TCN at 0.112).
FIT_STRIDE = 64
RIDGE_ALPHA = 1.0
LIGHTGBM_PARAMS: dict[str, Any] = {
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 200,
    'verbose': -1,
    # Same fit on rerun: no row/column sampling races across threads.
    'deterministic': True,
    'force_row_wise': True,
}


def trace_span(window: int) -> int:
    """Two-sided samples one :func:`predict_trace` output depends on.

    A sample is covered by every window starting up to ``window - 1`` before
    it, and each of those reads up to ``window - 1`` after it, so the span is
    ``2 * window - 1`` -- the honest receptive field of a stride-1 trace.
    """
    return 2 * window - 1


class SupportsPredict(Protocol):
    """Anything with a scikit-learn-style ``predict``.

    Both a fitted ``Ridge`` and a LightGBM ``Booster`` satisfy this, which is
    all :func:`predict_windows` needs of them. The return is ``object``
    because LightGBM's stubs admit sparse outputs; ``predict_windows``
    converts whatever comes back with ``np.asarray``.
    """

    def predict(self, x: np.ndarray, /) -> object: ...


def windowize(
    signals: np.ndarray, targets: np.ndarray, window: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a multi-channel shot into flattened fixed-length windows.

    Args:
        signals: [channels, length] input samples.
        targets: [length] target values.
        window: Number of samples per window.
        stride: Hop between successive window starts.

    Returns:
        ``(x, y)`` where ``x`` is [n_windows, channels * window] and ``y`` is
        [n_windows], the target at each window's centre.

    Raises:
        ValueError: If the window does not fit in the signal, or the signal
            and target lengths disagree.
    """
    if signals.ndim != 2:
        raise ValueError(f'signals must be [channels, length], got {signals.shape}')
    channels, length = signals.shape
    if targets.shape != (length,):
        raise ValueError(
            f'targets must be [{length}] to match signals, got {targets.shape}'
        )
    if window > length:
        raise ValueError(f'window {window} exceeds signal length {length}')
    if stride < 1:
        raise ValueError(f'stride must be >= 1, got {stride}')

    starts = np.arange(0, length - window + 1, stride)
    # Index once into a [n_windows, window] offset grid rather than looping:
    # a shot is 16384 samples and a sweep runs many of them.
    offsets = starts[:, None] + np.arange(window)[None, :]
    x = signals[:, offsets]  # [channels, n_windows, window]
    x = x.transpose(1, 0, 2).reshape(len(starts), channels * window)

    # Centre, not the last sample: the windowed torch models average a
    # prediction across the whole span, so the fair scalar label is the middle.
    y = targets[starts + window // 2]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def fit_lightgbm(
    x_train: np.ndarray, y_train: np.ndarray, params: dict[str, Any]
) -> lgb.Booster:
    """Fit a gradient-boosted regression tree ensemble.

    Args:
        x_train: [n_samples, n_features] training windows.
        y_train: [n_samples] targets.
        params: LightGBM parameters. Must include ``objective``: which loss
            the trees descend is a modelling choice, and defaulting it here
            would let a caller who forgot it silently get a different model
            rather than an error. ``num_boost_round``, if present, is consumed
            here rather than passed through, since LightGBM takes it as a
            keyword argument and not as a parameter.

    Returns:
        The trained ``Booster``.

    Raises:
        ImportError: If lightgbm is not installed.
        ValueError: If ``objective`` is absent from ``params``.
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError(
            'lightgbm is required for fit_lightgbm; it lives in the `simple` '
            'pixi feature'
        )
    if 'objective' not in params:
        raise ValueError(
            "params must name an 'objective'; it decides what is being fit, so "
            'it is for the caller to state (e.g. "regression")'
        )
    train_params = dict(params)
    num_boost_round = train_params.pop('num_boost_round', 100)
    # Unlike `objective`, verbosity cannot change the fitted model -- it only
    # decides whether LightGBM prints per-iteration chatter. Silence is the
    # useful default under a sweep that trains hundreds of boosters, and a
    # caller wanting the log can still pass its own `verbose`.
    train_params.setdefault('verbose', -1)
    dataset = lgb.Dataset(x_train, label=y_train)
    return lgb.train(train_params, dataset, num_boost_round=num_boost_round)


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Ridge:
    """Fit L2-regularized linear regression.

    The floor of the comparison: whatever a linear map of the raw window can
    do. A learned architecture that does not clear it is not earning anything.

    Args:
        x_train: [n_samples, n_features] training windows.
        y_train: [n_samples] targets.
        alpha: L2 penalty strength.

    Returns:
        The fitted ``Ridge``.
    """
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    return model


def predict_windows(model: SupportsPredict, x: np.ndarray) -> np.ndarray:
    """Predict one value per window.

    A single entry point for both baselines so callers do not branch on the
    model type.

    Args:
        model: A fitted ``Ridge`` or LightGBM ``Booster``.
        x: [n_samples, n_features] windows.

    Returns:
        [n_samples] predictions.
    """
    return np.asarray(model.predict(x)).ravel()


def predict_trace(
    model: SupportsPredict, signals: np.ndarray, window: int
) -> np.ndarray:
    """Predict a full-length trace for one shot from stride-1 window predictions.

    Each window's value is spread over every sample it covers and overlaps are
    averaged, which is what the windowed torch models do internally. Stride 1
    so the trace carries no piecewise-constant discretization floor of its own.

    Args:
        model: A fitted ``Ridge`` or LightGBM ``Booster``.
        signals: [channels, length] input samples.
        window: The window the model was fit on.

    Returns:
        [length] float64 prediction.
    """
    length = signals.shape[1]
    x, _ = windowize(signals, np.zeros(length), window, 1)
    kernel = np.ones(window)
    # Full convolution of the per-start values with a box of the window's
    # length is the per-sample sum over the windows covering it.
    accum = np.convolve(predict_windows(model, x), kernel)
    counts = np.convolve(np.ones(len(x)), kernel)
    return accum / counts
