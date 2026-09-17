#!/usr/bin/env python3

"""Drive the learned-baseline sweep and record every run in one artifact.

Three comparison legs share this driver, so they share a training loop, a
test split and an evaluation path -- the only way their numbers belong in one
table:

1. TCN vs SCNN at two capacity points. ``SCNN`` subclasses ``TCN`` setting
   only ``_causal = False``, so the pair has identical parameter counts and
   identical field widths and differs only in where the context sits.
2. TCN/SCNN vs Barland at the matched parameter budget.
3. The simple floor: a windowed MLP, LightGBM and ridge regression.

Two invariants this module exists to hold, both of which would silently
invalidate the comparison if they slipped:

**The split seed is fixed while the model seed varies.** ``SPLIT_SEED`` is
never derived from the run seed. If it were, different architectures would be
scored on different test sets and the table would be meaningless.

**Every attempted configuration is recorded, including ones that trained
badly.** A run that diverges is a result -- it says where not to look next.
Failures are caught and stored with their diagnostic rather than dropped.
"""

import argparse
import contextlib
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from smi.analysis.baselines import (
    SupportsPredict,
    fit_lightgbm,
    fit_ridge,
    predict_windows,
    windowize,
)
from smi.analysis.datamodule import VelocityDataModule
from smi.analysis.metrics import displacement_from_velocity, summarize_per_shot
from smi.analysis.models.factory import create_model
from smi.analysis.results_artifact import RunRecord, write_results
from smi.analysis.training_interface import TrainingConfig, TrainingInterface
from smi.synthetic.coil_driver import MS_PER_S

logger = logging.getLogger(__name__)

#: Train/val/test split seed. Constant across every run in the sweep so all
#: architectures are scored on the same held-out shots.
SPLIT_SEED = 42

#: Model-initialization seeds. This is the axis that varies; the split does not.
DEFAULT_SEEDS = (0, 1, 2)

DATA_DIR = Path(__file__).parent / 'data'
CONFIG_DIR = Path(__file__).parent / 'models' / 'configs' / 'matched'
DEFAULT_OUTPUT = DATA_DIR / 'learned_results.json'

#: Dataset label -> HDF5 file, keyed as ``baseline_results.json`` keys them.
DATASETS: dict[str, str] = {
    'free-space': 'free-space-synchro_10k.h5',
    'mmfiber': 'mmfiber-synchro_10k.h5',
}

#: Torch configs to sweep, in the order the report tabulates them.
TORCH_CONFIGS: tuple[str, ...] = (
    'tcn-matched',
    'scnn-matched',
    'tcn-matched-small',
    'scnn-matched-small',
    'barland-matched',
    'mlp-matched',
    'tcn-varying',
    'tcn-repeated',
)

#: Models that see samples after the timestamp they predict.
#:
#: This is not "SCNN only". ``windowize`` labels each window at its centre and
#: the windowed torch models average one prediction across the whole span, so
#: Barland, MLP, LightGBM and Ridge all use lookahead that the TCN is denied.
#: Any table comparing them to the analytic baseline has to say so.
NONCAUSAL_MODELS = frozenset({'SCNN', 'Barland', 'MLP', 'LightGBM', 'Ridge'})

#: Window for the tabular baselines, matched to the Barland receptive field.
TABULAR_WINDOW = 91
#: Hop between tabular training windows. Dense strides would make a 16384-long
#: shot into 16k rows; 64 keeps the design matrix tractable without changing
#: what the model is asked to predict.
TABULAR_STRIDE = 64
#: Shots used to fit the tabular baselines. Both are fit on CPU from a dense
#: design matrix, so this is bounded by memory and fit time rather than by the
#: GPU budget the torch runs are bounded by. 800 shots is 204k rows and costs
#: about 12 s for the booster, so the floor is not being handicapped by a
#: budget that does not apply to it: going 200 -> 800 moved LightGBM from
#: 0.274 to 0.259 um/ms and ridge not at all.
TABULAR_FIT_SHOTS = 800
#: Shots the tabular baselines are scored on, from the same fixed test split.
TABULAR_EVAL_SHOTS = 100

LIGHTGBM_PARAMS: dict[str, Any] = {
    'objective': 'regression',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'num_boost_round': 200,
    'verbose': -1,
}
RIDGE_ALPHA = 1.0


@dataclass
class SweepSettings:
    """Knobs the CLI exposes, threaded through instead of read from globals.

    Attributes:
        epochs: Cap on training epochs. The sweep is budget limited, so this
            is a wall-clock decision and not a convergence criterion; it is
            recorded in every run's config so the report can say which.
        seeds: Model-initialization seeds to run per configuration.
        datasets: Dataset labels to evaluate.
        device: Explicit torch device, e.g. ``'cuda:0'``. This box has two
            unequal GPUs, so which one a run lands on changes its timing.
        eval_shots: Cap on test shots evaluated per torch run.
        check_val_every_n_epoch: Validation cadence.
        torch_configs: Config stems to sweep, so a partial rerun can name a
            subset without the module's default tuple being mutated.
        preload: Hold the whole dataset on the GPU instead of streaming it
            through a DataLoader.
    """

    epochs: int
    seeds: tuple[int, ...]
    datasets: tuple[str, ...]
    device: str
    eval_shots: int
    check_val_every_n_epoch: int
    torch_configs: tuple[str, ...] = TORCH_CONFIGS
    preload: bool = True


def device_name(device: str) -> str:
    """Human-readable name of the device runs are pinned to.

    Args:
        device: Torch device string.

    Returns:
        The CUDA device name, or the device string itself when not on CUDA.
    """
    torch_device = torch.device(device)
    if torch_device.type != 'cuda' or not torch.cuda.is_available():
        return str(device)
    return torch.cuda.get_device_name(torch_device)


def build_datamodule(
    dataset: str,
    batch_size: int,
    num_workers: int = 0,
    preload_device: str | None = None,
) -> VelocityDataModule:
    """Build the datamodule for one dataset at the fixed split seed.

    Args:
        dataset: Dataset label, a key of :data:`DATASETS`.
        batch_size: Loader batch size.
        num_workers: DataLoader worker processes.
        preload_device: Device to hold the whole file on, or None for the
            per-sample DataLoader path. The split is the same either way.

    Returns:
        A :class:`VelocityDataModule` whose split is seeded with
        :data:`SPLIT_SEED`, never with the run's model seed.
    """
    return VelocityDataModule(
        dataset_path=str(DATA_DIR / DATASETS[dataset]),
        split_ratios=(80, 10, 10),
        batch_size=batch_size,
        num_workers=num_workers,
        seed=SPLIT_SEED,
        preload_device=preload_device,
    )


def _sample_rate_hz(dataset: str) -> float:
    """Sample rate of a dataset's shots, in Hz.

    Args:
        dataset: Dataset label.

    Returns:
        Sample rate read from the dataset the datamodule would build.
    """
    module = build_datamodule(dataset, batch_size=1)
    module.setup()
    inner = module.train_dataset.dataset
    inner.open_hdf5()
    return float(inner.sample_rate)


def evaluate_per_shot(
    predictions: torch.Tensor, targets: torch.Tensor, dt_ms: float
) -> dict[str, dict[str, float]]:
    """Summarize velocity and displacement error one shot at a time.

    Per shot, not pooled: the analytic artifact reports a distribution over
    shots, and a pooled RMSE is not comparable to a median over shots.

    Displacement here is *derived* -- the cumulative integral of the predicted
    velocity -- while the analytic baseline fits displacement directly. The
    two displacement columns are therefore not measured the same way.

    Args:
        predictions: Predicted velocity, [shots, length], in um/ms.
        targets: True velocity, [shots, length], in um/ms.
        dt_ms: Sample spacing in milliseconds.

    Returns:
        Metric name -> the six summary keys of ``summarize_per_shot``.
    """
    velocity_err = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=1))

    pred_disp = displacement_from_velocity(predictions, dt_ms)
    true_disp = displacement_from_velocity(targets, dt_ms)
    # Both integrals start at zero by construction, so no offset removal is
    # needed before differencing them.
    disp_err = torch.sqrt(torch.mean((pred_disp - true_disp) ** 2, dim=1))

    return {
        'velocity_rmse_um_per_ms': summarize_per_shot(
            velocity_err.detach().cpu().numpy()
        ),
        'displacement_rmse_um': summarize_per_shot(disp_err.detach().cpu().numpy()),
    }


def _load_config(name: str, settings: SweepSettings, dataset: str) -> TrainingConfig:
    """Load a matched config and apply the sweep's overrides.

    Args:
        name: Config stem under ``configs/matched``.
        settings: Sweep settings supplying the epoch cap and device.
        dataset: Dataset label to point the config at.

    Returns:
        The loaded config with epochs, device and dataset overridden.
    """
    config = TrainingConfig.from_yaml(str(CONFIG_DIR / f'{name}.yaml'))
    config.training_config['max_epochs'] = settings.epochs
    config.training_config['T_0'] = settings.epochs
    config.training_config['use_logging'] = False
    config.data_config['dataset_file'] = DATASETS[dataset]

    torch_device = torch.device(settings.device)
    if torch_device.type == 'cuda':
        config.training_config['accelerator'] = 'gpu'
        config.training_config['devices'] = [torch_device.index or 0]
    else:
        config.training_config['accelerator'] = 'cpu'
        config.training_config['devices'] = 1
    return config


@torch.no_grad()
def _predict_test_set(
    interface: TrainingInterface, device: str, max_shots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the trained model over the held-out shots.

    Args:
        interface: A trained :class:`TrainingInterface`.
        device: Device to evaluate on.
        max_shots: Stop after this many shots.

    Returns:
        ``(predictions, targets)``, each [shots, length] in um/ms.
    """
    module = interface.lightning_module.to(device).eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    seen = 0
    for signals, velocity, _displacement in interface.datamodule.test_dataloader():
        out = module(signals.to(device))
        preds.append(out.squeeze(1).float().cpu())
        targets.append(velocity.float().cpu())
        seen += signals.shape[0]
        if seen >= max_shots:
            break
    return torch.cat(preds)[:max_shots], torch.cat(targets)[:max_shots]


def run_torch_config(
    name: str, dataset: str, seed: int, settings: SweepSettings, dt_ms: float
) -> RunRecord:
    """Train and evaluate one torch configuration.

    A run that raises is still a record: the exception text lands in the
    config under ``'error'`` and the metrics are NaN, because a suppressed
    failure would remove the evidence for where not to look next.

    Args:
        name: Config stem under ``configs/matched``.
        dataset: Dataset label.
        seed: Model-initialization seed. Does not touch the split.
        settings: Sweep settings.
        dt_ms: Sample spacing in milliseconds.

    Returns:
        The completed record.
    """
    config = _load_config(name, settings, dataset)
    model_hparams = dict(config.model_config)
    model_hparams['in_channels'] = config.data_config.get('num_pd_channels', 3)
    model_type = model_hparams['type']

    probe = create_model(model_hparams)
    total_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    receptive_field = int(probe.receptive_field)
    del probe

    record_config = dict(model_hparams)
    record_config.update(
        {
            'config_name': name,
            'max_epochs': settings.epochs,
            'batch_size': config.training_config['batch_size'],
            'optimizer': config.training_config['optimizer'],
            'learning_rate': config.training_config['learning_rate'],
            'split_seed': SPLIT_SEED,
            'budget_limited': True,
            'compiled': False,
        }
    )

    metrics: dict[str, dict[str, float]] = {}
    interface: TrainingInterface | None = None
    started = time.perf_counter()
    try:
        # Seeds the model initialization only. The split is seeded
        # separately and constantly with SPLIT_SEED.
        torch.manual_seed(seed)
        interface = TrainingInterface(
            config=config,
            experiment_name=f'{name}_{dataset}_s{seed}',
            check_val_every_n_epoch=settings.check_val_every_n_epoch,
        )
        if settings.preload:
            # The DataLoader path re-reads HDF5 and recomputes the velocity
            # FFT per item, which measured at ~33 s of the ~42 s epoch while
            # the model itself needs only ~9 s. Holding the file on the GPU
            # removes that. This contradicts the benchmark note on
            # VelocityDataModule, which measured a production-size TCN at
            # batch 32 and found the loader had headroom; these runs are a
            # 44k-parameter model at batch 16, cheap enough that the loader
            # is the bottleneck. The split is identical either way.
            interface.datamodule = build_datamodule(
                dataset,
                batch_size=config.training_config['batch_size'],
                preload_device=settings.device,
            )
        interface.train()
        preds, targets = _predict_test_set(
            interface, settings.device, settings.eval_shots
        )
        metrics = evaluate_per_shot(preds, targets, dt_ms)
    except Exception as exc:
        logger.exception('run failed: %s %s seed=%d', name, dataset, seed)
        record_config['error'] = f'{type(exc).__name__}: {exc}'
        metrics = _nan_metrics()
    finally:
        record_config['wall_clock_s'] = round(time.perf_counter() - started, 1)
        # Drop the interface before emptying the cache, not after. Preloading
        # puts a 3.28 GB resident copy of the dataset on the GPU per run, and
        # empty_cache only reclaims blocks nothing references -- with
        # `interface` still live it reclaims none of it, and the sweep walks
        # up to an OOM a few runs in.
        interface = None
        gc.collect()
        with contextlib.suppress(AttributeError, RuntimeError):
            torch.cuda.empty_cache()

    return RunRecord(
        model_type=model_type,
        config=record_config,
        total_params=total_params,
        receptive_field=receptive_field,
        sees_future=model_type in NONCAUSAL_MODELS,
        seed=seed,
        device=device_name(settings.device),
        dataset=dataset,
        metrics=metrics,
    )


def _nan_metrics() -> dict[str, dict[str, float]]:
    """Metrics for a run that did not produce predictions.

    Returns:
        Both metric names mapped to NaN under every summary key, so a failed
        run still has the shape a reader of the artifact expects.
    """
    nan = summarize_per_shot(np.array([float('nan')]))
    return {'velocity_rmse_um_per_ms': dict(nan), 'displacement_rmse_um': dict(nan)}


def _collect_shots(
    subset: Subset, count: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Pull raw signal/velocity arrays out of a split subset.

    Args:
        subset: A ``torch.utils.data.Subset`` from the datamodule's split.
        count: Number of shots to take, from the front of the subset.

    Returns:
        ``(signals, velocities)`` as numpy arrays, one entry per shot.
    """
    signals: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for idx in range(min(count, len(subset))):
        sig, vel, _disp = subset[idx]
        signals.append(np.asarray(sig, dtype=np.float32))
        velocities.append(np.asarray(vel, dtype=np.float32))
    return signals, velocities


def _tabular_predictions(
    model: SupportsPredict, signals: list[np.ndarray], length: int
) -> torch.Tensor:
    """Predict a full-length velocity trace per shot from window predictions.

    Each window yields one value at its centre; the trace is reconstructed by
    assigning that value to every position the window covers and averaging
    overlaps, which is what the windowed torch models do internally. Using a
    stride of 1 here would be exact but costs 16384 rows per shot, so the
    evaluation stride matches the fit stride and the gaps are filled by the
    same overlap average.

    Args:
        model: Fitted ridge or booster.
        signals: Per-shot [channels, length] arrays.
        length: Signal length.

    Returns:
        [shots, length] predicted velocity in um/ms.
    """
    out = np.zeros((len(signals), length), dtype=np.float32)
    starts = np.arange(0, length - TABULAR_WINDOW + 1, TABULAR_STRIDE)
    for row, sig in enumerate(signals):
        x, _ = windowize(
            sig, np.zeros(length, dtype=np.float32), TABULAR_WINDOW, TABULAR_STRIDE
        )
        values = predict_windows(model, x)
        accum = np.zeros(length, dtype=np.float64)
        counts = np.zeros(length, dtype=np.float64)
        for start, value in zip(starts, values, strict=True):
            accum[start : start + TABULAR_WINDOW] += value
            counts[start : start + TABULAR_WINDOW] += 1.0
        # The tail beyond the last complete window is never covered; hold the
        # last prediction there rather than leaving a zero that would flatter
        # or punish the model arbitrarily.
        covered = counts > 0
        out[row, covered] = (accum[covered] / counts[covered]).astype(np.float32)
        if not covered.all():
            out[row, ~covered] = out[row, covered][-1]
    return torch.from_numpy(out)


def run_tabular(
    kind: str, dataset: str, seed: int, settings: SweepSettings, dt_ms: float
) -> RunRecord:
    """Fit and evaluate one tabular baseline on the fixed split.

    Args:
        kind: ``'LightGBM'`` or ``'Ridge'``.
        dataset: Dataset label.
        seed: Recorded for symmetry with the torch runs. Ridge is
            deterministic and LightGBM is seeded through its params, so this
            moves the fit only for the latter.
        settings: Sweep settings.
        dt_ms: Sample spacing in milliseconds.

    Returns:
        The completed record.

    Raises:
        ValueError: If ``kind`` is not a known tabular baseline.
    """
    if kind not in ('LightGBM', 'Ridge'):
        raise ValueError(f'Unknown tabular baseline: {kind}')

    module = build_datamodule(dataset, batch_size=1)
    module.setup()
    train_signals, train_velocity = _collect_shots(
        module.train_dataset, TABULAR_FIT_SHOTS
    )
    test_signals, test_velocity = _collect_shots(
        module.test_dataset, min(TABULAR_EVAL_SHOTS, settings.eval_shots)
    )
    length = train_signals[0].shape[1]

    xs, ys = zip(
        *(
            windowize(sig, vel, TABULAR_WINDOW, TABULAR_STRIDE)
            for sig, vel in zip(train_signals, train_velocity, strict=True)
        ),
        strict=True,
    )
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    record_config: dict[str, Any] = {
        'config_name': kind.lower(),
        'window': TABULAR_WINDOW,
        'stride': TABULAR_STRIDE,
        'fit_shots': len(train_signals),
        'eval_shots': len(test_signals),
        'train_rows': int(x_train.shape[0]),
        'split_seed': SPLIT_SEED,
        'budget_limited': False,
    }

    started = time.perf_counter()
    metrics: dict[str, dict[str, float]] = {}
    total_params = 0
    try:
        if kind == 'LightGBM':
            params = dict(LIGHTGBM_PARAMS, seed=seed)
            record_config.update(params)
            model = fit_lightgbm(x_train, y_train, params)
            total_params = int(model.num_trees())
        else:
            record_config['alpha'] = RIDGE_ALPHA
            model = fit_ridge(x_train, y_train, RIDGE_ALPHA)
            total_params = int(model.coef_.size + 1)
        preds = _tabular_predictions(model, test_signals, length)
        targets = torch.from_numpy(np.stack(test_velocity))
        metrics = evaluate_per_shot(preds, targets, dt_ms)
    except Exception as exc:
        logger.exception('tabular run failed: %s %s seed=%d', kind, dataset, seed)
        record_config['error'] = f'{type(exc).__name__}: {exc}'
        metrics = _nan_metrics()
    finally:
        record_config['wall_clock_s'] = round(time.perf_counter() - started, 1)

    if kind == 'LightGBM':
        # Trees, not weights: a booster has no parameter count comparable to a
        # network's, and reporting one would invite a false equivalence. The
        # count recorded is the tree count and the key below says so.
        record_config['params_are'] = 'trees'
    else:
        record_config['params_are'] = 'coefficients'

    return RunRecord(
        model_type=kind,
        config=record_config,
        total_params=total_params,
        receptive_field=TABULAR_WINDOW,
        sees_future=kind in NONCAUSAL_MODELS,
        seed=seed,
        device='cpu',
        dataset=dataset,
        metrics=metrics,
    )


def run_sweep(settings: SweepSettings, output: Path) -> list[RunRecord]:
    """Run every leg and write the artifact.

    The artifact is rewritten after each record rather than once at the end,
    so a sweep interrupted partway leaves a valid file of the runs that did
    finish instead of nothing.

    Args:
        settings: Sweep settings.
        output: Artifact path.

    Returns:
        Every record produced, in run order.
    """
    records: list[RunRecord] = []
    for dataset in settings.datasets:
        dt_ms = MS_PER_S / _sample_rate_hz(dataset)
        for name in settings.torch_configs:
            for seed in settings.seeds:
                logger.info('run: %s %s seed=%d', name, dataset, seed)
                records.append(run_torch_config(name, dataset, seed, settings, dt_ms))
                write_results(output, records)
        for kind in ('LightGBM', 'Ridge'):
            for seed in settings.seeds:
                logger.info('run: %s %s seed=%d', kind, dataset, seed)
                records.append(run_tabular(kind, dataset, seed, settings, dt_ms))
                write_results(output, records)
    return records


def smoke(settings: SweepSettings) -> None:
    """Train one config briefly and print what the artifact would hold.

    Confirms the loss moves and the velocity RMSE is finite and near the
    analytic baseline's order of magnitude before an hours-long sweep is
    launched on the same code path.

    Args:
        settings: Sweep settings, with the epoch cap already lowered.
    """
    dataset = settings.datasets[0]
    dt_ms = MS_PER_S / _sample_rate_hz(dataset)
    record = run_torch_config('tcn-matched', dataset, 0, settings, dt_ms)
    logger.info('metrics: %s', json.dumps(record.metrics, indent=2))
    logger.info(
        'params=%d receptive_field=%d wall_clock_s=%s',
        record.total_params,
        record.receptive_field,
        record.config['wall_clock_s'],
    )


def main() -> None:
    """Parse arguments and run the sweep or the smoke check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Short single-run check.')
    parser.add_argument('--epochs', type=int, default=30, help='Epoch cap per run.')
    parser.add_argument(
        '--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help='Model seeds.'
    )
    parser.add_argument(
        '--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS)
    )
    parser.add_argument('--device', default='cuda:0', help='Explicit torch device.')
    parser.add_argument('--eval-shots', type=int, default=200)
    parser.add_argument('--check-val-every-n-epoch', type=int, default=2)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--configs', nargs='+', default=None, help='Subset to run.')
    parser.add_argument(
        '--no-preload',
        action='store_true',
        help='Stream through a DataLoader instead of holding the file on GPU.',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    settings = SweepSettings(
        epochs=2 if args.smoke else args.epochs,
        seeds=tuple(args.seeds),
        datasets=tuple(args.datasets),
        device=args.device,
        eval_shots=20 if args.smoke else args.eval_shots,
        check_val_every_n_epoch=1 if args.smoke else args.check_val_every_n_epoch,
        torch_configs=tuple(args.configs) if args.configs else TORCH_CONFIGS,
        preload=not args.no_preload,
    )

    if args.smoke:
        smoke(settings)
        return

    records = run_sweep(settings, args.output)
    logger.info('wrote %d records to %s', len(records), args.output)


if __name__ == '__main__':
    main()
