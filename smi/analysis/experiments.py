"""Resumable experiment runner: a grid of train-then-evaluate cells.

A grid YAML expands to cells ``family x channel subset x sim x seed`` (plus one
optional classical oracle per ``sim x subset``). Each cell trains one network on
its own simulator config, then scores it on the frozen test set of that config
through :func:`smi.analysis.evaluate.evaluate`. A cell's record carries its
``cell_key``, ``sim_config_hash``, ``simulator_version`` and training budget
(``BUDGET_FIELDS``), plus a ``run_fingerprint`` hashing the whole resolved
training config and test set; a cell recorded with sim, simulator and
fingerprint matching is skipped, so an interrupted grid resumes where it
stopped. A recorded key whose setup has since changed raises, naming the
differing config blocks, rather than mixing the two in one table.

Grid YAML keys (all required except where noted):

* ``families``: ``[{name, config}]``; ``config`` is a model YAML, relative to
  the grid file.
* ``channel_subsets``: wavelength lists (um). Never part of a ``sim`` block.
* ``sims``: ``[{name, preset, feedback_c, alpha, override?, sweep?,
  sign_invariant?}]``. ``preset`` is ``nominal`` or ``noise_free``
  (:class:`SimConfig` constructors); ``override`` replaces fields after that;
  ``sweep: {field: [values]}`` expands into one sim per value, named
  ``<name>-<field>=<value>``. ``sign_invariant`` overrides the grid's for this
  sim (Michelson sims need it; see ``loss.sign_invariant``).
* ``seeds``: training seeds.
* ``oracle``: also score ``InverseFitEstimator`` on every (sim, subset).
* ``loss: {sign_invariant}``: the default for every sim.
* ``synthetic: {steps_per_epoch, val_steps, train_crop_length}``. A model
  whose ``requires_full_length`` is set trains on full shots
  (``train_crop_length`` ``None``) whatever the grid says; the record's
  ``train_crop_length`` reports what it trained on.
* ``training``: shallow overrides of the family's ``training`` block (device,
  epoch cap). ``convergence.monitor`` is derived from the sim's sign
  invariance, so one family YAML serves Michelson and SMI sims.
* ``test_set: {seed, n_shots}``.

``--rescore-metrics`` instead re-scores every recorded cell that lacks a
metric added to :func:`~smi.analysis.evaluate.evaluate` after it ran
(``RESCORED_METRICS``), from its checkpoint on its frozen test set, without
retraining or touching its fingerprint.
"""

import argparse
import copy
import dataclasses
import fcntl
import fnmatch
import functools
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning
import yaml

from smi.analysis.evaluate import (
    CheckpointEstimator,
    Estimator,
    InverseFitEstimator,
    evaluate,
)
from smi.analysis.models.factory import create_model
from smi.analysis.results_artifact import (
    RunRecord,
    append_results,
    read_results,
    write_results,
)
from smi.analysis.training_interface import (
    TrainingConfig,
    TrainingInterface,
    convergence_monitor,
)
from smi.synthetic.forward import NOMINAL_SWEEPS, SIMULATOR_VERSION, SimConfig
from smi.synthetic.frozen import DEFAULT_ROOT, load_or_create

logger = logging.getLogger(__name__)

ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = ANALYSIS_DIR / 'data' / 'synthetic_results.json'
DEFAULT_LOCK = ANALYSIS_DIR / 'data' / '.runner.lock'
DEFAULT_RUNS_DIR = ANALYSIS_DIR.parents[1] / 'runs'

# Sim fields whose YAML lists must become tuples (SimConfig coerces the float
# ones, but the sweep and override values are compared and named before that).
_TUPLE_FIELDS = frozenset(
    name for name, values in NOMINAL_SWEEPS.items() if isinstance(values[0], tuple)
)
ORACLE_FAMILY = 'inverse_fit'
# Record fields that, with the sim config, define what a cell trained on; the
# cell key names neither, so resume compares them too.
BUDGET_FIELDS = (
    'steps_per_epoch',
    'batch_size',
    'train_crop_length',
    'max_epochs',
    'T_0',
    'lr',
)
# Metrics evaluate gained after records were written; rescore_metrics adds them.
RESCORED_METRICS = ('displacement_rmse', 'displacement_rmse_sign_invariant')
# A rescored velocity RMSE must reproduce the record to this (relative), or the
# checkpoint or test set is not the one the record describes. CPU re-scoring of
# MPS-trained checkpoints reproduced it to <= 1.2e-6.
RESCORE_REL_TOL = 1e-4


class RunnerLockedError(RuntimeError):
    """Another live runner holds the lockfile."""


@dataclass(frozen=True)
class Cell:
    """One unit of work: a trained network (or the oracle) on one sim/subset.

    ``seed`` is ``None`` for the oracle, which is not trained. ``sweep`` is
    the ``(field, value)`` a ``sweep`` entry set for this sim, ``None`` for a
    sim that is not a sweep point.
    """

    key: str
    family: str
    family_config: Path
    wavelengths_um: tuple[float, ...]
    sim_name: str
    sim: SimConfig
    sweep: tuple[str, object] | None
    sign_invariant: bool
    seed: int | None


def _fmt(value: object) -> str:
    """Compact, filename-safe spelling of a swept value."""
    if isinstance(value, (tuple, list)):
        return '-'.join(_fmt(v) for v in value)
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def _as_field_value(name: str, value: object) -> object:
    return tuple(value) if name in _TUPLE_FIELDS else value


def subset_label(wavelengths_um: tuple[float, ...]) -> str:
    """``ch635`` or ``ch635-675-515``: wavelengths in nm, in channel order."""
    return 'ch' + '-'.join(str(round(w * 1000)) for w in wavelengths_um)


def cell_key(
    family: str, wavelengths_um: tuple[float, ...], sim_name: str, seed: int | None
) -> str:
    """Stable, human-readable, filename-safe cell identifier."""
    tail = 'oracle' if seed is None else f'seed{seed}'
    return f'{family}__{subset_label(wavelengths_um)}__{sim_name}__{tail}'


def _build_sims(
    entry: dict[str, Any], wavelengths_um: tuple[float, ...]
) -> list[tuple[str, SimConfig, tuple[str, object] | None]]:
    """Expand one ``sims`` entry (and its sweep) for a channel subset.

    Each item is ``(sim name, sim, (field, value) swept or None)``.
    """
    preset = entry['preset']
    args = {
        'wavelengths_um': wavelengths_um,
        'feedback_c': entry['feedback_c'],
        'alpha': entry['alpha'],
    }
    nominal = SimConfig.nominal(**args)
    if preset == 'nominal':
        base = nominal
    elif preset == 'noise_free':
        base = SimConfig.noise_free(
            **args,
            sample_rate=nominal.sample_rate,
            start_freq_hz=nominal.start_freq_hz,
            end_freq_hz=nominal.end_freq_hz,
            drive_gain=nominal.drive_gain,
        )
    else:
        raise ValueError(f'sim {entry["name"]!r}: unknown preset {preset!r}')
    override = {k: _as_field_value(k, v) for k, v in entry['override'].items()}
    base = dataclasses.replace(base, **override)
    sweep = entry['sweep']
    if not sweep:
        return [(entry['name'], base, None)]
    # One field per sweep: a Cartesian product would hide cell counts.
    ((field, values),) = sweep.items()
    return [
        (
            f'{entry["name"]}-{field}={_fmt(value)}',
            dataclasses.replace(base, **{field: _as_field_value(field, value)}),
            (field, _as_field_value(field, value)),
        )
        for value in values
    ]


def expand_grid(path: Path) -> tuple[dict[str, Any], list[Cell]]:
    """Parse a grid YAML and expand it into its cells, in a deterministic order.

    Order is sim, then subset, then (oracle first) family and seed, so a
    ``--only`` filter on a sim name picks a contiguous block.

    Returns:
        ``(grid, cells)``: the parsed YAML and its cells.

    Raises:
        ValueError: On duplicate cell keys (two entries with the same name).
    """
    path = Path(path)
    grid = yaml.safe_load(path.read_text())
    cells: list[Cell] = []
    for sim_entry in grid['sims']:
        entry = {'override': {}, 'sweep': None, **sim_entry}
        sign_invariant = bool(
            entry.get('sign_invariant', grid['loss']['sign_invariant'])
        )
        for subset in grid['channel_subsets']:
            wavelengths = tuple(float(w) for w in subset)
            for sim_name, sim, sweep in _build_sims(entry, wavelengths):
                if grid['oracle']:
                    cells.append(
                        Cell(
                            cell_key(ORACLE_FAMILY, wavelengths, sim_name, None),
                            ORACLE_FAMILY,
                            path,
                            wavelengths,
                            sim_name,
                            sim,
                            sweep,
                            sign_invariant,
                            None,
                        )
                    )
                cells.extend(
                    Cell(
                        cell_key(family['name'], wavelengths, sim_name, seed),
                        family['name'],
                        (path.parent / family['config']).resolve(),
                        wavelengths,
                        sim_name,
                        sim,
                        sweep,
                        sign_invariant,
                        int(seed),
                    )
                    for family in grid['families']
                    for seed in grid['seeds']
                )
    keys = [c.key for c in cells]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f'duplicate cell keys: {dupes}')
    return grid, cells


def _kill_children() -> None:
    # pkill -P: direct children only, which is all this process spawns.
    subprocess.run(  # noqa: S603 -- fixed argv, own PID
        ['pkill', '-TERM', '-P', str(os.getpid())],  # noqa: S607 -- pkill is on PATH on POSIX
        check=False,
    )


@contextmanager
def runner_lock(path: Path) -> Iterator[None]:
    """Hold the single-instance lock for the runner's lifetime.

    The lock is an exclusive ``flock`` on the file, which the kernel drops when
    the process dies, so there is no stale-lock reclaim and no window where a
    half-written file looks abandoned. The PID written into it is for humans.
    The file is never unlinked: removing it would let a second runner lock a
    new inode while the first still holds the old one. INT and TERM kill this
    process's children and exit ``128 + signal``, since killing a wrapper alone
    would leave detached children running.

    Raises:
        RunnerLockedError: If another live process holds the lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = os.read(fd, 32).decode().strip() or 'unknown'
        os.close(fd)
        raise RunnerLockedError(f'another runner (pid {holder}) holds {path}') from None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())

    def _terminate(signum: int, _frame: object) -> None:
        _kill_children()
        raise SystemExit(128 + signum)

    previous = {
        s: signal.signal(s, _terminate) for s in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        os.close(fd)


@functools.cache
def _family(text: str) -> tuple[dict[str, Any], bool]:
    """A family YAML's parse and its model's ``requires_full_length``.

    Keyed by content, not path, so an edited file is never served stale; a
    grid's cells share a handful of families, each built once.
    """
    family = yaml.safe_load(text)
    return family, create_model(family['model']).requires_full_length


def training_config(cell: Cell, grid: dict[str, Any]) -> TrainingConfig:
    """The resolved training config of a trained ``cell`` of ``grid``."""
    cached, requires_full_length = _family(cell.family_config.read_text())
    # A copy: TrainingInterface writes in_channels into its model config.
    family = copy.deepcopy(cached)
    training = {**family['training'], **grid['training']}
    loss = {**family['loss'], 'sign_invariant': cell.sign_invariant}
    # Derived, not chosen: the monitor must follow the loss (Task 11 rule), and
    # a grid mixes Michelson (sign-invariant) and SMI sims over one family. So
    # a family YAML's own ``convergence.monitor`` line is overridden here.
    training['convergence'] = {
        **training['convergence'],
        'monitor': convergence_monitor(loss),
    }
    synthetic = {
        'use_synthetic_training': True,
        **grid['synthetic'],
        'sim': dataclasses.asdict(cell.sim),
    }
    if requires_full_length:
        synthetic['train_crop_length'] = None
    return TrainingConfig(
        model_config=family['model'],
        training_config=training,
        data_config=family['data'],
        loss_config=loss,
        synthetic_config=synthetic,
    )


def _budget(config: TrainingConfig) -> dict[str, Any]:
    """The ``BUDGET_FIELDS`` of a cell's training config."""
    synthetic, training = config.synthetic_config, config.training_config
    return {
        'steps_per_epoch': synthetic['steps_per_epoch'],
        'batch_size': training['batch_size'],
        'train_crop_length': synthetic['train_crop_length'],
        'max_epochs': training['max_epochs'],
        'T_0': training['T_0'],
        'lr': float(training['learning_rate']),
    }


def _fingerprint(
    cell: Cell, grid: dict[str, Any], config: TrainingConfig | None
) -> dict[str, Any]:
    """What a cell trained and was scored on, as its record carries it.

    ``run_fingerprint`` hashes every top-level block of the resolved training
    config plus the test set, because the six ``BUDGET_FIELDS`` miss e.g. the
    architecture, weight decay and loss weights; ``fingerprint_blocks`` keeps
    the per-block hashes so a mismatch can name what changed. The oracle has no
    training config, so its blocks are the sim and test set only.
    """
    test = grid['test_set']
    blocks: dict[str, Any] = {'test_set': test}
    if config is None:
        budget = dict.fromkeys(BUDGET_FIELDS)
        blocks['sim'] = dataclasses.asdict(cell.sim)
    else:
        budget = _budget(config)
        blocks.update(
            model=config.model_config,
            training=config.training_config,
            data=config.data_config,
            loss=config.loss_config,
            synthetic=config.synthetic_config,
        )
    block_hashes = {k: _stable_hash(v) for k, v in blocks.items()}
    return {
        'sim_config_hash': cell.sim.hash(),
        'simulator_version': SIMULATOR_VERSION,
        'run_fingerprint': _stable_hash([block_hashes, SIMULATOR_VERSION]),
        'fingerprint_blocks': block_hashes,
        **budget,
    }


def _stable_hash(value: object) -> str:
    """Short sha1 of the JSON form of ``value`` (YAML-loaded, so JSON-safe)."""
    blob = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()[:16]


def _is_done(cell: Cell, want: dict[str, Any], records: list[RunRecord]) -> bool:
    """Whether ``cell`` is recorded under its current sim, simulator and setup.

    ``want`` is the cell's :func:`_fingerprint`.

    Raises:
        ValueError: If its key is recorded only under another setup.
    """
    recorded = [r for r in records if r.cell_key == cell.key]
    if not recorded:
        return False
    comparable = ('sim_config_hash', 'simulator_version', 'run_fingerprint')
    if any(all(getattr(r, k) == want[k] for k in comparable) for r in recorded):
        return True
    have = recorded[0]
    old_blocks = have.fingerprint_blocks or {}
    differing = sorted(
        b
        for b in old_blocks.keys() | want['fingerprint_blocks'].keys()
        if old_blocks.get(b) != want['fingerprint_blocks'].get(b)
    )
    diff = {
        k: (getattr(have, k), want[k])
        for k in ('sim_config_hash', 'simulator_version', *BUDGET_FIELDS)
        if getattr(have, k) != want[k]
    }
    raise ValueError(
        f'{cell.key} is recorded under a different setup: differing blocks '
        f'{differing or "(record predates fingerprints)"}; readable fields '
        f'(recorded, grid) {diff}; write to another --out or remove those '
        'records rather than mixing them'
    )


def _train(
    cell: Cell, config: TrainingConfig, checkpoint: Path
) -> tuple[bool, str, dict[str, Any]]:
    """Train ``cell`` and copy its best checkpoint to ``checkpoint``.

    Returns:
        ``(converged, device, record fields)``; the fields say how far past its
        best the final epoch was, and the budget behind the cell.
    """
    assert cell.seed is not None
    lightning.seed_everything(cell.seed, workers=True)
    interface = TrainingInterface(config=config, experiment_name=cell.key)
    converged = interface.train()
    if not interface.best_model_path:
        raise RuntimeError(
            f'{cell.key}: no checkpoint was saved (no epoch produced a finite '
            'monitored validation value)'
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(interface.best_model_path, checkpoint)
    fields = {
        'epochs_run': interface.epochs_run,
        'best_monitor': interface.best_monitor,
        'final_monitor': interface.final_monitor,
        'zero_predictor_monitor': interface.zero_predictor_monitor,
    }
    return converged, str(interface.trainer.strategy.root_device), fields


def run_cell(
    cell: Cell,
    grid: dict[str, Any],
    config: TrainingConfig | None,
    fingerprint: dict[str, Any],
    out: Path,
    runs_dir: Path,
    frozen_root: Path,
) -> None:
    """Train (unless oracle), evaluate and append the cell's record.

    ``config`` is the cell's :func:`_training_config` (None for the oracle),
    and ``fingerprint`` its :func:`_fingerprint`.
    """
    test = grid['test_set']
    dataset = load_or_create(cell.sim, test['seed'], test['n_shots'], frozen_root)
    start = time.perf_counter()
    est: Estimator
    extra: dict[str, Any] = {
        k: fingerprint[k]
        for k in ('run_fingerprint', 'fingerprint_blocks', *BUDGET_FIELDS)
    }
    if config is None:
        est = InverseFitEstimator.for_dataset(dataset, test['n_shots'])
        converged = True
    else:
        ckpt = runs_dir / f'{cell.key}.ckpt'
        converged, device, trained = _train(cell, config, ckpt)
        extra.update(trained)
        est = CheckpointEstimator(ckpt, device)
    result = evaluate(
        est, dataset, cell.wavelengths_um, test['n_shots'], converged, cell.seed
    )
    record = dataclasses.replace(
        result.record,
        cell_key=cell.key,
        sim_name=cell.sim_name,
        wall_clock_s=time.perf_counter() - start,
        **extra,
    )
    append_results(out, [record])


def run_grid(
    path: Path,
    out: Path,
    *,
    runs_dir: Path,
    lock_path: Path,
    frozen_root: Path,
    only: list[str] | None,
) -> int:
    """Run every cell of a grid not already recorded in ``out``.

    Args:
        path: Grid YAML.
        out: Results artifact; its records' ``cell_key`` values mark done cells.
        runs_dir: Where each cell's checkpoint is written, named by its key.
        lock_path: Single-instance lockfile.
        frozen_root: Cache directory of frozen test sets.
        only: Glob patterns over cell keys; a cell runs if any matches.
            ``None`` runs the whole grid (for a TCN-only pilot, ``tcn*``).

    Returns:
        The number of cells run this call.

    Raises:
        RunnerLockedError: If another runner is alive.
    """
    grid, cells = expand_grid(path)
    if only is not None:
        cells = [c for c in cells if any(fnmatch.fnmatch(c.key, p) for p in only)]
    # Resolved once, before any training: TrainingInterface edits its config.
    configs = {
        c.key: None if c.seed is None else training_config(c, grid) for c in cells
    }
    fingerprints = {c.key: _fingerprint(c, grid, configs[c.key]) for c in cells}
    with runner_lock(lock_path):
        # Read under the lock: the other runner may have just finished.
        records = read_results(out) if Path(out).exists() else []
        todo = [c for c in cells if not _is_done(c, fingerprints[c.key], records)]
        logger.info(
            '%d cells selected, %d done, %d to run',
            len(cells),
            len(cells) - len(todo),
            len(todo),
        )
        for i, cell in enumerate(todo, 1):
            logger.info('[%d/%d] %s', i, len(todo), cell.key)
            run_cell(
                cell,
                grid,
                configs[cell.key],
                fingerprints[cell.key],
                Path(out),
                runs_dir,
                frozen_root,
            )
    return len(todo)


def rescore_metrics(
    out: Path,
    *,
    runs_dir: Path,
    lock_path: Path,
    frozen_root: Path,
    device: str,
    include_oracle: bool,
) -> int:
    """Add ``RESCORED_METRICS`` to every recorded cell that lacks them.

    Each cell is re-scored from ``runs_dir/<cell_key>.ckpt`` (or, with
    ``include_oracle``, by the oracle) on its frozen set, found from the
    record itself: ``record.dataset`` is the set's file stem. Only the new
    metric keys are copied in; velocity metrics, bins, fingerprints, budget
    and wall clock stay as recorded, so resume still skips every cell. The
    artifact is written once, at the end, so a refused cell leaves it as it
    was.

    Args:
        out: Results artifact, rewritten in place.
        runs_dir: Directory of the cells' checkpoints.
        lock_path: Single-instance lockfile shared with ``run_grid``, so a
            rescore cannot race a training grid writing the same artifact.
        frozen_root: Cache directory of frozen test sets.
        device: Device the checkpoints are evaluated on.
        include_oracle: Also re-score ``inverse_fit`` records (minutes per
            cell on CPU, against seconds for a network).

    Returns:
        The number of records rescored.

    Raises:
        RunnerLockedError: If another runner is alive.
        ValueError: If a re-scored cell's sim hash differs from its record's,
            or its velocity RMSE mean differs by more than
            ``RESCORE_REL_TOL``.
    """
    with runner_lock(lock_path):
        records = read_results(out)
        n = 0
        for record in records:
            if record.cell_key is None or RESCORED_METRICS[0] in record.metrics:
                continue
            dataset = Path(frozen_root) / f'{record.dataset}.h5'
            est: Estimator
            if record.model_type == ORACLE_FAMILY:
                if not include_oracle:
                    continue
                est = InverseFitEstimator.for_dataset(dataset, None)
            else:
                ckpt = Path(runs_dir) / f'{record.cell_key}.ckpt'
                if not ckpt.exists():
                    logger.warning('%s: no checkpoint at %s', record.cell_key, ckpt)
                    continue
                est = CheckpointEstimator(ckpt, device)
            assert record.wavelengths_um is not None
            assert record.converged is not None
            logger.info('rescoring %s', record.cell_key)
            # n_shots None: a frozen set holds exactly the shots its cell was
            # scored on (the stem's n<n_shots>).
            new = evaluate(
                est,
                dataset,
                tuple(record.wavelengths_um),
                None,
                record.converged,
                record.train_seed,
            ).record
            if new.sim_config_hash != record.sim_config_hash:
                raise ValueError(
                    f'{record.cell_key}: {dataset} has sim hash '
                    f'{new.sim_config_hash}, record says {record.sim_config_hash}'
                )
            have = record.metrics['velocity_rmse']['mean']
            got = new.metrics['velocity_rmse']['mean']
            if abs(got - have) > RESCORE_REL_TOL * abs(have):
                raise ValueError(
                    f'{record.cell_key}: rescored velocity_rmse mean {got} does '
                    f'not reproduce the recorded {have} to {RESCORE_REL_TOL}; '
                    'wrong checkpoint or test set'
                )
            record.metrics.update({k: new.metrics[k] for k in RESCORED_METRICS})
            n += 1
        if n:
            write_results(out, records)
    return n


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exits non-zero when another runner holds the lock."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        'grid', type=Path, nargs='?', help='grid YAML (omit with --rescore-metrics)'
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        '--only', action='append', help='glob over cell keys; repeatable (OR)'
    )
    parser.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        '--rescore-metrics',
        action='store_true',
        help='add RESCORED_METRICS to recorded cells instead of running a grid',
    )
    parser.add_argument(
        '--include-oracle',
        action='store_true',
        help='with --rescore-metrics, also re-score the inverse_fit oracle',
    )
    parser.add_argument(
        '--device', default='cpu', help='with --rescore-metrics: evaluation device'
    )
    args = parser.parse_args(argv)
    if args.rescore_metrics == (args.grid is not None):
        parser.error('give either a grid or --rescore-metrics, not both or neither')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        if args.rescore_metrics:
            n = rescore_metrics(
                args.out,
                runs_dir=args.runs_dir,
                lock_path=args.lock,
                frozen_root=args.frozen_root,
                device=args.device,
                include_oracle=args.include_oracle,
            )
            sys.stdout.write(f'rescored {n} records\n')
            return 0
        n = run_grid(
            args.grid,
            args.out,
            runs_dir=args.runs_dir,
            lock_path=args.lock,
            frozen_root=args.frozen_root,
            only=args.only,
        )
    except RunnerLockedError as e:
        sys.stderr.write(f'refusing to start: {e}\n')
        return 1
    sys.stdout.write(f'ran {n} cells\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
