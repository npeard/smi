"""Ridge and LightGBM on a grid's synthetic sims, scored like its network cells.

For every sim of a shipped grid (each channel subset, each sweep point) the
tabular models are fit on ``n_train_shots`` shots simulated from that sim with
``train_seed`` and scored by :func:`smi.experiments.evaluate.evaluate` on the
grid's own frozen test set (seed 1000), so their records carry the same
metrics, on the same shots, through the same code as the networks'. Families
are named ``ridge-w<window>`` and ``lightgbm-w<window>``; cell keys follow
:func:`smi.experiments.runner.cell_key`, so a recorded cell is skipped.

The target is the signed velocity (um/ms), as the grids train their networks
(``sign_invariant=False``); evaluate reports the sign-invariant variant beside
it as it does for every cell. On a sim whose network loss is sign-invariant
(Michelson, where the fringes do not fix the direction) neither fit has such a
loss, so they still fit the signed target and the record's config says
``sign_invariant_loss: False``; read ``velocity_rmse_sign_invariant`` there,
and expect a sign-blind fit to be handicapped against the networks.

Sizing, chosen so one sim's fits take about a minute on CPU rather than hours:
window 91 (the Barland receptive field, as the real-data tabular baselines in
``scripts/run_baseline_sweep.py`` use, so the two tables share a task),
training stride 64 and 512 training shots, a 130k x 273 design matrix. On real
data 200 -> 800 shots moved LightGBM 6% and ridge not at all, so 512 is past
the point where the fit is data-starved, and a noise-free sim is no harder to
cover than a noisy one. Traces are reconstructed at stride 1
(:func:`~smi.estimators.tabular.predict_trace`), so a prediction depends on
``2 * window - 1`` = 181 samples centred on it, which is the recorded
``receptive_field``; the ends are averaged over fewer windows, not padded.
"""

import dataclasses
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import Tensor

from smi.data.frozen import load_or_create
from smi.estimators.tabular import (
    CAUSAL,
    FIT_STRIDE,
    LIGHTGBM_AVAILABLE,
    LIGHTGBM_PARAMS,
    RIDGE_ALPHA,
    SupportsPredict,
    fit_lightgbm,
    fit_ridge,
    lgb,
    predict_trace,
    trace_span,
    windowize,
)
from smi.experiments.evaluate import evaluate
from smi.experiments.grids import grid
from smi.experiments.results_artifact import RunRecord, append_results, read_results
from smi.experiments.runner import build_sims, cell_key, runner_lock
from smi.theory.forward import SimConfig, simulate_batch

logger = logging.getLogger(__name__)

# Shots per simulate_batch call while drawing the training set; bounds memory.
SIM_CHUNK_SHOTS = 64
MODELS = ('ridge', 'lightgbm')


class TabularEstimator:
    """A fitted window model as an :class:`~smi.experiments.evaluate.Estimator`."""

    causal = CAUSAL
    device = 'cpu'

    def __init__(
        self,
        name: str,
        model_type: str,
        model: SupportsPredict,
        window: int,
        config: dict[str, Any],
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.model = model
        self.receptive_field: int | None = trace_span(window)
        self.window = window
        self.config = config
        self.total_params = _param_count(model)

    def predict_velocity(
        self, signals: Tensor, wavelengths_um: tuple[float, ...]
    ) -> Tensor:
        """Raw ``[B, C, L]`` -> ``[B, L]`` um/ms, one stride-1 trace per shot."""
        del wavelengths_um  # fit on the same subset by construction
        out = [predict_trace(self.model, s, self.window) for s in signals.numpy()]
        return torch.from_numpy(np.stack(out)).float()


def _param_count(model: SupportsPredict) -> int:
    """Ridge: weights + bias. Booster: leaf values, its fitted parameters."""
    if isinstance(model, Ridge):
        return int(np.size(model.coef_)) + 1
    if LIGHTGBM_AVAILABLE and isinstance(model, lgb.Booster):
        return int(sum(t['num_leaves'] for t in model.dump_model()['tree_info']))
    return 0


def training_windows(
    sim: SimConfig, n_shots: int, seed: int, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix of ``n_shots`` fresh shots: ``(x, y)`` at stride 64."""
    generator = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for start in range(0, n_shots, SIM_CHUNK_SHOTS):
        n = min(SIM_CHUNK_SHOTS, n_shots - start)
        batch = simulate_batch(sim, n, torch.device('cpu'), generator)
        for sig, vel in zip(batch.signals.numpy(), batch.velocity.numpy(), strict=True):
            x, y = windowize(sig, vel, window, FIT_STRIDE)
            xs.append(x)
            ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def fit(
    model_type: str, x: np.ndarray, y: np.ndarray, seed: int
) -> tuple[SupportsPredict, dict[str, Any]]:
    """Fit one model; returns it and the hyperparameters to record."""
    if model_type == 'ridge':
        return fit_ridge(x, y, RIDGE_ALPHA), {'alpha': RIDGE_ALPHA}
    if model_type == 'lightgbm':
        params = {**LIGHTGBM_PARAMS, 'seed': seed}
        return fit_lightgbm(x, y, params), params
    raise ValueError(f'unknown tabular model {model_type!r}; known: {MODELS}')


def run_tabular(
    grid_name: str,
    models: tuple[str, ...],
    window: int,
    out: Path,
    *,
    n_train_shots: int,
    train_seed: int,
    frozen_root: Path,
    lock_path: Path,
) -> list[RunRecord]:
    """Fit and score ``models`` on every sim of grid ``grid_name``.

    Holds the runner's lock, since both append to the same artifact.

    Returns:
        The records appended this call (recorded cells are skipped).

    Raises:
        ValueError: If ``train_seed`` is the test seed.
    """
    g = grid(grid_name)
    test = g.test_set
    if train_seed == test.seed:
        raise ValueError(f'train_seed {train_seed} would train on the test set')
    appended: list[RunRecord] = []
    with runner_lock(lock_path):
        done = {r.cell_key for r in read_results(out)} if Path(out).exists() else set()
        for entry in g.sims:
            for wavelengths in g.channel_subsets:
                for sim_name, sim, _sweep in build_sims(entry, wavelengths):
                    keys = {
                        m: cell_key(f'{m}-w{window}', wavelengths, sim_name, train_seed)
                        for m in models
                    }
                    todo = [m for m in models if keys[m] not in done]
                    if not todo:
                        continue
                    logger.info(
                        '%s: simulating %d training shots', sim_name, n_train_shots
                    )
                    x, y = training_windows(sim, n_train_shots, train_seed, window)
                    dataset = load_or_create(sim, test.seed, test.n_shots, frozen_root)
                    for m in todo:
                        start = time.perf_counter()
                        model, hparams = fit(m, x, y, train_seed)
                        config = {
                            'window': window,
                            'train_stride': FIT_STRIDE,
                            'n_train_shots': n_train_shots,
                            'sign_invariant_loss': False,
                            **hparams,
                        }
                        est = TabularEstimator(
                            f'{m}-w{window}', m, model, window, config
                        )
                        result = evaluate(
                            est,
                            dataset,
                            wavelengths,
                            test.n_shots,
                            converged=True,
                            train_seed=train_seed,
                        )
                        record = dataclasses.replace(
                            result.record,
                            cell_key=keys[m],
                            sim_name=sim_name,
                            wall_clock_s=time.perf_counter() - start,
                        )
                        append_results(out, [record])
                        appended.append(record)
                        logger.info(
                            '%s: velocity RMSE %.4g',
                            keys[m],
                            record.metrics['velocity_rmse']['mean'],
                        )
    return appended
