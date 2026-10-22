#!/usr/bin/env python
"""Ray Tune driver for the SMI hyperparameter / architecture search pipeline.

This module replaces the YAML Cartesian-product grid search that used to live in
:class:`smi.analysis.training_interface.TrainingConfig`. A search config is an
ordinary training YAML in which any field may be a list; list-valued fields in
the ``model`` / ``training`` / ``loss`` / ``data`` / ``synthetic`` sections are
interpreted as a ``ray.tune.choice([...])`` search dimension, and scalar fields
are fixed. Ray Tune then *samples* configurations (with an ASHA scheduler for
early stopping and fractional-GPU packing) instead of exhaustively enumerating
the grid.

The same :func:`train_func` is used on the dev Mac (CPU, tiny sizes, for tests)
and on the CUDA rig (GPU); the only difference is the ``accelerator`` / ``devices``
fields in the config and the ``gpu_fraction`` resource request.
"""

import logging
from pathlib import Path
from typing import Any

import lightning as lightning_module
import torch
import yaml
from ray import tune
from ray.tune import CheckpointConfig, RunConfig, TuneConfig, Tuner
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from torch.utils.data import DataLoader

from smi.analysis.datamodule import VelocityDataModule
from smi.analysis.ensemble import EnsembleModule, SyntheticEnsembleModule
from smi.analysis.lit_module import LitModule
from smi.analysis.models.base import Model
from smi.analysis.synthetic_lit_module import SyntheticIndexDataset, SyntheticLitModule
from smi.analysis.training_interface import TrainingConfig, TrainingInterface

logger = logging.getLogger(__name__)

#: The metric ASHA optimizes and the best config is selected by.
SEARCH_METRIC = 'val/total_unweighted_loss'
SEARCH_MODE = 'min'

#: Sections of the YAML config that contribute to the search space.
_CONFIG_SECTIONS = ('model', 'training', 'loss', 'data', 'synthetic')


def build_param_space(config_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map a loaded YAML config dict to a Ray Tune ``param_space``.

    For each of the ``model`` / ``training`` / ``loss`` / ``data`` / ``synthetic``
    sections, list-valued fields become ``ray.tune.choice([...])`` (one sampled
    element per trial) and scalar fields are passed through unchanged. This
    preserves the historical YAML grid semantics -- a field written as a list is
    a search dimension -- while letting Ray sample / early-stop rather than
    enumerate the full Cartesian product.

    A field whose intended value is itself a list (e.g. ``temporal_channels`` or
    ``wavelengths_nm``) is written in the YAML as a singleton list-of-lists,
    exactly as in the existing grid configs, so it round-trips as a single
    ``choice`` over inner lists.

    Args:
        config_dict: Parsed YAML config with top-level section keys.

    Returns:
        A ``param_space`` dict keyed by section name; each value is a dict of
        ``{field: value_or_tune.choice}``. Sections absent from the config
        (e.g. ``synthetic`` for real-data runs) are omitted.
    """
    param_space: dict[str, dict[str, Any]] = {}
    for section in _CONFIG_SECTIONS:
        raw = config_dict.get(section)
        if raw is None:
            continue
        section_space: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                section_space[key] = tune.choice(value)
            else:
                section_space[key] = value
        param_space[section] = section_space
    # The optional `ensemble` section passes through verbatim: its list-valued
    # fields (e.g. per_member_lr) are per-member specs for the vmap ensemble, NOT
    # tune.choice search dimensions. Ray passes plain values through unchanged.
    if config_dict.get('ensemble') is not None:
        param_space['ensemble'] = config_dict['ensemble']
    return param_space


def _config_from_sampled(sampled: dict[str, dict[str, Any]]) -> TrainingConfig:
    """Rebuild a :class:`TrainingConfig` from one sampled Tune config.

    Ray resolves each ``tune.choice`` to a concrete value before calling
    :func:`train_func`, so ``sampled`` is a plain nested dict mirroring the YAML
    sections. This reassembles the per-section hparams dicts that
    ``TrainingInterface`` / ``LitModule`` consume.

    Args:
        sampled: Nested dict with ``model`` / ``training`` / ``loss`` / ``data``
            and optionally ``synthetic`` sections of concrete values.

    Returns:
        A single-run :class:`TrainingConfig`.
    """
    return TrainingConfig(
        model_config=dict(sampled['model']),
        training_config=dict(sampled['training']),
        loss_config=dict(sampled['loss']),
        data_config=dict(sampled['data']),
        synthetic_config=(
            dict(sampled['synthetic']) if sampled.get('synthetic') is not None else None
        ),
    )


def train_func(config: dict[str, dict[str, Any]]) -> None:
    """Ray Tune trainable: train one sampled configuration.

    Reuses :class:`TrainingInterface` for data-path construction (real data via
    ``VelocityDataModule``; synthetic via ``SyntheticLitModule`` +
    ``SyntheticIndexDataset``), LightningModule creation (``LitModule`` real /
    ``SyntheticLitModule`` synthetic), and trainer setup. The only Tune-specific
    additions are:

    - a :class:`TuneReportCheckpointCallback` reporting :data:`SEARCH_METRIC`,
      injected via ``TrainingInterface(extra_callbacks=...)``;
    - ``check_val_every_n_epoch=1`` so the metric is reported every epoch (the
      single-run default of 5 would never report on a short ASHA budget).

    The same code path runs on CPU (tests) and GPU (rig) -- the accelerator and
    device count come straight from ``config['training']``.

    Args:
        config: One sampled point from the Tune ``param_space`` (nested by
            section). Ray has already resolved all ``tune.choice`` dimensions to
            concrete values.
    """
    # Inner level: if the config has an `ensemble` section, this trial trains a
    # vmap-ensemble over the within-architecture axis (seeds / loss weights /
    # per-member lr) instead of a single model. It reports the same SEARCH_METRIC
    # (its best member's unweighted loss) so ASHA can compare both trial kinds.
    if config.get('ensemble') is not None:
        _train_ensemble(config)
        return

    # Single-model trial: save a checkpoint each validation so the best trial's
    # weights can be exported afterward (see :func:`export_best_model`).
    report_callback = TuneReportCheckpointCallback(
        metrics={SEARCH_METRIC: SEARCH_METRIC},
        on='validation_end',
        save_checkpoints=True,
    )

    training_config = _config_from_sampled(config)
    experiment_name = training_config.training_config.get(
        'experiment_name', training_config.model_config['type']
    )

    interface = TrainingInterface(
        config=training_config,
        experiment_name=experiment_name,
        extra_callbacks=[report_callback],
        check_val_every_n_epoch=1,
    )
    interface.train()


def _resolve_dataset_path(data_config: dict[str, Any]) -> str:
    """Resolve the HDF5 dataset path for the real-data ensemble path.

    Accepts an explicit ``dataset_path``, or resolves ``data_dir`` + ``dataset_file``
    relative to the package root (mirroring ``TrainingInterface.setup_data``).
    """
    if data_config.get('dataset_path'):
        return str(data_config['dataset_path'])
    base_dir = Path(__file__).parent.parent  # the `smi` package dir
    data_dir = str(data_config['data_dir']).lstrip('./')
    return str(base_dir / data_dir / data_config['dataset_file'])


def _train_ensemble(config: dict[str, dict[str, Any]]) -> None:
    """Train a vmap-ensemble for one Tune trial (inner within-architecture fan).

    Builds a :class:`SyntheticEnsembleModule` (on-GPU shared synthetic batches) or
    a :class:`EnsembleModule` over real data (``VelocityDataModule``), per the
    ``synthetic`` section, and fits it while reporting :data:`SEARCH_METRIC`. The
    architecture is fixed by ``config['model']``; the ``ensemble`` section
    supplies the per-member axis (``size``/``seeds``, ``per_member_lr``,
    ``velocity_loss_weights``, ``displacement_loss_weights``, ``dropouts``).

    Checkpointing is disabled here: an ensemble checkpoint is not a single
    deployable model, so :func:`export_best_model` targets single-model trials;
    use :meth:`EnsembleModule.best_member_model` to extract a member in-process.
    """
    report_callback = TuneReportCheckpointCallback(
        metrics={SEARCH_METRIC: SEARCH_METRIC},
        on='validation_end',
        save_checkpoints=False,
    )
    model_hparams = dict(config['model'])
    training = config['training']
    loss = config['loss']
    ensemble = config['ensemble']
    synthetic = config.get('synthetic')

    seeds = ensemble.get('seeds') or list(range(int(ensemble['size'])))
    common = dict(
        target=training.get('target', 'velocity'),
        lr=float(ensemble.get('lr', training.get('learning_rate', 1e-3))),
        velocity_loss_weights=ensemble.get(
            'velocity_loss_weights', loss.get('velocity_loss_weight', 1.0)
        ),
        displacement_loss_weights=ensemble.get(
            'displacement_loss_weights', loss.get('displacement_loss_weight', 1.0)
        ),
        dropouts=ensemble.get('dropouts'),
        per_member_lr=ensemble.get('per_member_lr'),
    )
    batch_size = int(training['batch_size'])

    if synthetic is not None and synthetic.get('use_synthetic_training', False):
        wavelengths_nm = synthetic['wavelengths_nm']
        model_hparams['in_channels'] = len(wavelengths_nm)
        module: EnsembleModule = SyntheticEnsembleModule(
            model_hparams,
            seeds,
            wavelengths_nm=wavelengths_nm,
            start_freq=synthetic.get('start_freq', 1.0),
            end_freq=synthetic.get('end_freq', 1000.0),
            max_displacement_um=synthetic.get('max_displacement_um', 5.0),
            **common,
        )
        steps = int(synthetic.get('steps_per_epoch', 100))
        val_steps = int(synthetic.get('val_steps', max(1, steps // 5)))
        train_loader = DataLoader(
            SyntheticIndexDataset(steps * batch_size),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            SyntheticIndexDataset(val_steps * batch_size), batch_size=batch_size
        )
    else:
        data = config['data']
        model_hparams['in_channels'] = int(data.get('num_pd_channels', 3))
        module = EnsembleModule(model_hparams, seeds, **common)
        datamodule = VelocityDataModule(
            dataset_path=_resolve_dataset_path(data),
            split_ratios=tuple(data.get('split_ratios', [80, 10, 10])),
            batch_size=batch_size,
            num_workers=int(data.get('num_workers', 0)),
            num_pd_channels=int(data.get('num_pd_channels', 3)),
        )
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

    trainer = lightning_module.Trainer(
        max_epochs=int(training['max_epochs']),
        accelerator=training.get('accelerator', 'cpu'),
        devices=training.get('devices', 1),
        callbacks=[report_callback],
        check_val_every_n_epoch=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_loader, val_loader)


def run_search(
    config_path: str,
    *,
    num_samples: int = 10,
    gpu_fraction: float = 0.0,
    cpus_per_trial: int = 1,
    max_concurrent_trials: int | None = None,
    max_t: int | None = None,
    grace_period: int = 1,
    search_alg: str = 'random',
    storage_path: str | None = None,
) -> tune.ResultGrid | Any:
    """Run a Ray Tune search defined by a YAML config and return the best result.

    Builds the search space from the YAML (see :func:`build_param_space`), runs
    ``num_samples`` trials with an :class:`ASHAScheduler` for early stopping, and
    packs trials onto the GPU via fractional-GPU resources. Exposes the
    single-GPU packing knobs from the design (``gpu_fraction``,
    ``max_concurrent_trials``, ``num_samples``).

    Args:
        config_path: Path to the search YAML (list-valued fields = search dims).
        num_samples: Number of configurations Ray samples and trains.
        gpu_fraction: Fraction of a GPU each trial requests. ``0`` for CPU-only
            (dev Mac / tests); on the 24 GB rig, ``~1/gpu_fraction`` trials pack
            per GPU (memory is not isolated, so cap with ``max_concurrent_trials``).
        cpus_per_trial: CPUs reserved per trial.
        max_concurrent_trials: Hard cap on simultaneous trials (CUDA-context
            overhead, not weights, is the real limit for tiny models).
        max_t: ASHA ``max_t`` (max training iterations / epochs before a trial
            may run to completion). Defaults to the config's ``max_epochs``.
        grace_period: ASHA ``grace_period`` (min iterations before a trial may be
            early-stopped).
        search_alg: ``'random'`` (default; ASHA over randomly sampled configs) or
            ``'optuna'`` (Optuna TPE sampler over the same space, which tends to
            find good configs in fewer samples for continuous/ordinal spaces).
        storage_path: Where Ray persists results. Defaults to Ray's default.

    Returns:
        The best :class:`ray.tune.Result` by :data:`SEARCH_METRIC` (``min``).
        The best single-model trial carries a checkpoint usable by
        :func:`export_best_model`.
    """
    with Path(config_path).open() as f:
        config_dict = yaml.safe_load(f)

    param_space = build_param_space(config_dict)

    if max_t is None:
        # max_epochs may be a search dimension; take the largest candidate.
        max_epochs = config_dict['training']['max_epochs']
        max_t = max(max_epochs) if isinstance(max_epochs, list) else int(max_epochs)

    scheduler = ASHAScheduler(
        metric=SEARCH_METRIC, mode=SEARCH_MODE, max_t=max_t, grace_period=grace_period
    )

    trainable = tune.with_resources(
        train_func, {'CPU': cpus_per_trial, 'GPU': gpu_fraction}
    )

    if search_alg == 'optuna':
        searcher: OptunaSearch | None = OptunaSearch(
            metric=SEARCH_METRIC, mode=SEARCH_MODE
        )
    elif search_alg == 'random':
        searcher = None  # Tune's default random sampling over the param_space.
    else:
        raise ValueError(f"search_alg must be 'random' or 'optuna', got {search_alg!r}")

    tuner = Tuner(
        trainable,
        param_space=param_space,
        tune_config=TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=searcher,
            max_concurrent_trials=max_concurrent_trials,
        ),
        # Keep only the best-scoring checkpoint per trial so the best trial's
        # checkpoint is the one with the lowest SEARCH_METRIC (for export).
        run_config=RunConfig(
            storage_path=storage_path,
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute=SEARCH_METRIC,
                checkpoint_score_order=SEARCH_MODE,
            ),
        ),
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric=SEARCH_METRIC, mode=SEARCH_MODE)
    logger.info(
        'Best config (%s=%s): %s',
        SEARCH_METRIC,
        best_result.metrics.get(SEARCH_METRIC),
        best_result.config,
    )
    return best_result


def export_best_model(best_result: Any, output_path: str) -> str:
    """Export the best single-model trial as a TorchScript artifact.

    Loads the best trial's checkpoint, rebuilds its LightningModule
    (``SyntheticLitModule`` if the trial was synthetic, else ``LitModule``),
    scripts the baked-normalization :class:`Model` wrapper via
    :meth:`Model.to_torchscript`, and writes it to ``output_path``. The scripted
    model takes raw device signals and returns raw velocity/displacement.

    Args:
        best_result: A ``ray.tune.Result`` (e.g. from :func:`run_search`) for a
            single-model trial. Ensemble trials are not checkpointed; use
            :meth:`smi.analysis.ensemble.EnsembleModule.best_member_model` for
            those.
        output_path: Destination ``.pt`` path for the TorchScript model.

    Returns:
        ``output_path``.

    Raises:
        ValueError: If the result has no checkpoint (e.g. an ensemble trial) or
            no Lightning checkpoint file is found.
    """
    if best_result.checkpoint is None:
        raise ValueError(
            'Best result has no checkpoint to export (ensemble trials are not '
            'checkpointed -- use EnsembleModule.best_member_model instead).'
        )

    config = best_result.config or {}
    synthetic = config.get('synthetic')
    is_synthetic = synthetic is not None and synthetic.get('use_synthetic_training')
    module_cls: type[LitModule] = SyntheticLitModule if is_synthetic else LitModule

    with best_result.checkpoint.as_directory() as ckpt_dir:
        # Lightning checkpoints are usually ``*.ckpt``; Ray's
        # TuneReportCheckpointCallback saves the file literally named
        # ``checkpoint`` (no extension), so fall back to that.
        ckpt_files = sorted(Path(ckpt_dir).rglob('*.ckpt')) or [
            p for p in Path(ckpt_dir).rglob('checkpoint') if p.is_file()
        ]
        if not ckpt_files:
            raise ValueError(f'No Lightning checkpoint found under {ckpt_dir}')
        lit = module_cls.load_from_checkpoint(str(ckpt_files[0]), map_location='cpu')

    # torch.compile (GPU runs) wraps the model; unwrap to the raw Model.
    model = getattr(lit.model, '_orig_mod', lit.model)
    model.eval()

    # Bake out any training-time weight parametrizations (e.g. weight_norm):
    # TorchScript cannot script live parametrizations, and at inference the
    # effective weight is fixed, so folding it in is equivalent.
    parametrize = torch.nn.utils.parametrize
    for submodule in model.modules():
        if parametrize.is_parametrized(submodule):
            for tensor_name in list(submodule.parametrizations):
                parametrize.remove_parametrizations(
                    submodule, tensor_name, leave_parametrized=True
                )

    if isinstance(model, Model):
        scripted = model.to_torchscript()
    else:
        scripted = torch.jit.script(model)
    torch.jit.save(scripted, output_path)
    logger.info('Exported best model to %s', output_path)
    return output_path
