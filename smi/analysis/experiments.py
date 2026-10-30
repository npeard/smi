"""Resumable experiment runner: a grid of train-then-evaluate cells.

A grid YAML expands to cells ``family x channel subset x sim x seed`` (plus one
optional classical oracle per ``sim x subset``). Each cell trains one network on
its own simulator config, then scores it on the frozen test set of that config
through :func:`smi.analysis.evaluate.evaluate`. A cell's record carries its
``cell_key``; keys already in the results artifact are skipped, so an
interrupted grid resumes where it stopped.

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
* ``synthetic: {steps_per_epoch, val_steps, train_crop_length}``.
* ``training``: shallow overrides of the family's ``training`` block (device,
  epoch cap). ``convergence.monitor`` is derived from the sim's sign
  invariance, so one family YAML serves Michelson and SMI sims.
* ``test_set: {seed, n_shots}``.
"""

import argparse
import dataclasses
import fnmatch
import logging
import os
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
from smi.analysis.results_artifact import append_results, read_results
from smi.analysis.training_interface import (
    TrainingConfig,
    TrainingInterface,
    convergence_monitor,
)
from smi.synthetic.forward import NOMINAL_SWEEPS, SimConfig
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


class RunnerLockedError(RuntimeError):
    """Another live runner holds the lockfile."""


@dataclass(frozen=True)
class Cell:
    """One unit of work: a trained network (or the oracle) on one sim/subset.

    ``seed`` is ``None`` for the oracle, which is not trained.
    """

    key: str
    family: str
    family_config: Path
    wavelengths_um: tuple[float, ...]
    sim_name: str
    sim: SimConfig
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
) -> list[tuple[str, SimConfig]]:
    """Expand one ``sims`` entry (and its sweep) for a channel subset."""
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
        return [(entry['name'], base)]
    # One field per sweep: a Cartesian product would hide cell counts.
    ((field, values),) = sweep.items()
    return [
        (
            f'{entry["name"]}-{field}={_fmt(value)}',
            dataclasses.replace(base, **{field: _as_field_value(field, value)}),
        )
        for value in values
    ]


def expand_grid(path: Path) -> list[Cell]:
    """Expand a grid YAML into its cells, in a deterministic order.

    Order is sim, then subset, then (oracle first) family and seed, so a
    ``--only`` filter on a sim name picks a contiguous block.

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
            for sim_name, sim in _build_sims(entry, wavelengths):
                if grid['oracle']:
                    cells.append(
                        Cell(
                            cell_key(ORACLE_FAMILY, wavelengths, sim_name, None),
                            ORACLE_FAMILY,
                            path,
                            wavelengths,
                            sim_name,
                            sim,
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
    return cells


def _pid_alive(pid: int) -> bool:
    """Signal-0 probe; POSIX only (the runner targets the MPS Mac)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_children() -> None:
    # pkill -P: direct children only, which is all this process spawns.
    subprocess.run(  # noqa: S603 -- fixed argv, own PID
        ['pkill', '-TERM', '-P', str(os.getpid())],  # noqa: S607 -- pkill is on PATH on POSIX
        check=False,
    )


@contextmanager
def runner_lock(path: Path) -> Iterator[None]:
    """Hold the single-instance lockfile, containing this process's PID.

    A lock whose PID is dead is reclaimed with a warning. INT and TERM kill
    this process's children, release the lock and exit ``128 + signal``, since
    killing a wrapper alone would leave detached children running.

    Raises:
        RunnerLockedError: If a live process holds the lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                holder = int(path.read_text().strip())
            except (ValueError, FileNotFoundError):
                holder = None
            if holder is not None and _pid_alive(holder):
                raise RunnerLockedError(
                    f'another runner (pid {holder}) holds {path}'
                ) from None
            logger.warning('reclaiming stale runner lock %s (pid %s)', path, holder)
            path.unlink(missing_ok=True)
    with os.fdopen(fd, 'w') as f:
        f.write(str(os.getpid()))

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
        path.unlink(missing_ok=True)


def _training_config(cell: Cell, grid: dict[str, Any]) -> TrainingConfig:
    family = yaml.safe_load(cell.family_config.read_text())
    training = {**family['training'], **grid['training']}
    loss = {**family['loss'], 'sign_invariant': cell.sign_invariant}
    # Derived, not chosen: the monitor must follow the loss (Task 11 rule), and
    # a grid mixes Michelson (sign-invariant) and SMI sims over one family.
    training['convergence'] = {
        **training['convergence'],
        'monitor': convergence_monitor(loss),
    }
    synthetic = {
        'use_synthetic_training': True,
        **grid['synthetic'],
        'sim': dataclasses.asdict(cell.sim),
    }
    return TrainingConfig(
        model_config=family['model'],
        training_config=training,
        data_config=family['data'],
        loss_config=loss,
        synthetic_config=synthetic,
    )


def _train(
    cell: Cell, grid: dict[str, Any], checkpoint: Path
) -> tuple[Path, bool, str]:
    """Train ``cell`` and save its final weights; returns (path, converged, device)."""
    assert cell.seed is not None
    lightning.seed_everything(cell.seed, workers=True)
    interface = TrainingInterface(
        config=_training_config(cell, grid), experiment_name=cell.key
    )
    converged = interface.train()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    interface.trainer.save_checkpoint(checkpoint)
    return checkpoint, converged, str(interface.trainer.strategy.root_device)


def run_cell(
    cell: Cell, grid: dict[str, Any], out: Path, runs_dir: Path, frozen_root: Path
) -> None:
    """Train (unless oracle), evaluate and append the cell's record."""
    test = grid['test_set']
    dataset = load_or_create(cell.sim, test['seed'], test['n_shots'], frozen_root)
    start = time.perf_counter()
    est: Estimator
    if cell.seed is None:
        est = InverseFitEstimator.for_dataset(dataset, test['n_shots'])
        converged = True
    else:
        ckpt, converged, device = _train(cell, grid, runs_dir / f'{cell.key}.ckpt')
        est = CheckpointEstimator(ckpt, device)
    result = evaluate(
        est, dataset, cell.wavelengths_um, test['n_shots'], converged, cell.seed
    )
    record = dataclasses.replace(
        result.record,
        cell_key=cell.key,
        sim_name=cell.sim_name,
        wall_clock_s=time.perf_counter() - start,
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
    grid = yaml.safe_load(Path(path).read_text())
    cells = expand_grid(path)
    if only is not None:
        cells = [c for c in cells if any(fnmatch.fnmatch(c.key, p) for p in only)]
    with runner_lock(lock_path):
        # Read under the lock: the other runner may have just finished.
        done = {r.cell_key for r in read_results(out)} if Path(out).exists() else set()
        todo = [c for c in cells if c.key not in done]
        logger.info(
            '%d cells selected, %d done, %d to run', len(cells), len(done), len(todo)
        )
        for i, cell in enumerate(todo, 1):
            logger.info('[%d/%d] %s', i, len(todo), cell.key)
            run_cell(cell, grid, Path(out), runs_dir, frozen_root)
    return len(todo)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exits non-zero when another runner holds the lock."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('grid', type=Path, help='grid YAML')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        '--only', action='append', help='glob over cell keys; repeatable (OR)'
    )
    parser.add_argument('--runs-dir', type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
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
