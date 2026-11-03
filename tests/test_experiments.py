"""Experiment runner: grid expansion, resumability, locking, record contents."""

import copy
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from smi.core.acquisition import STEMLAB_125_14
from smi.core.kinematics import derivative_displacement
from smi.core.paths import REPO_ROOT
from smi.data.datasets import select_pd_keys
from smi.experiments import grid as grid_module
from smi.experiments import runner as experiments
from smi.experiments.evaluate import load_lit_module
from smi.experiments.grid import EvalSet, Family, Grid, Sim, Synthetic
from smi.experiments.grids import GRIDS
from smi.experiments.results_artifact import read_results
from smi.experiments.runner import (
    RunnerLockedError,
    cell_fingerprint,
    expand_grid,
    rescore_metrics,
    run_grid,
    runner_lock,
    training_config,
)
from smi.learning import configs, synthetic_lit_module
from smi.learning.configs import MODEL_CONFIGS, model_config
from smi.learning.lit_module import LitModule
from smi.learning.models.factory import create_model
from smi.learning.training_interface import SIGNED_RMSE_KEY
from smi.theory.forward import SimConfig, simulate_batch
from tests._helpers import script_log

TINY = 'tiny-tcn'


@pytest.fixture(autouse=True)
def small_simulations(small_chunks, monkeypatch):
    monkeypatch.setattr(synthetic_lit_module, 'NORM_STATS_SHOTS', 2)


@pytest.fixture(autouse=True)
def tiny_family(monkeypatch):
    """Serve a 2-layer TCN as ``tiny-tcn``; tests edit it in place.

    The registry is read-only, so the lookup the grid and runner make is
    patched instead; each call hands out a fresh copy, as the real one does.
    """
    family = model_config('matched/tcn-matched')
    family.model_config['temporal_channels'] = [4, 4]

    def lookup(name):
        return copy.deepcopy(family) if name == TINY else configs.model_config(name)

    for module in (grid_module, experiments):
        monkeypatch.setattr(module, 'model_config', lookup)
    return family


SMI = Sim('smi', 'noise_free', feedback_c=0.2, alpha=3.0)
TRAINING = {
    'accelerator': 'cpu',
    'devices': 1,
    'max_epochs': 1,
    'batch_size': 2,
    'use_logging': False,
}


def _grid(**overrides):
    """A tiny CPU grid whose family is the served 2-layer TCN."""
    grid = Grid(
        families=(Family('tcn', TINY),),
        channel_subsets=((0.635, 0.6748, 0.515),),
        sims=(SMI,),
        seeds=(0, 1),
        oracle=False,
        sign_invariant=False,
        synthetic=Synthetic(steps_per_epoch=1, val_steps=1, train_crop_length=256),
        training=TRAINING,
        test_set=EvalSet(seed=5, n_shots=2),
    )
    return replace(grid, **overrides)


def _run(grid, tmp_path, only=None):
    return run_grid(
        grid,
        tmp_path / 'results.json',
        runs_dir=tmp_path / 'runs' / 'grid',
        lock_path=tmp_path / '.runner.lock',
        frozen_root=tmp_path / 'frozen',
        only=only,
    )


def test_two_cell_grid_writes_complete_records_and_resumes(tmp_path):
    grid = _grid()
    assert _run(grid, tmp_path) == 2
    records = read_results(tmp_path / 'results.json')
    assert [r.cell_key for r in records] == [
        'tcn__ch635-675-515__smi__seed0',
        'tcn__ch635-675-515__smi__seed1',
    ]
    for r, seed in zip(records, (0, 1), strict=True):
        assert r.train_seed == seed
        assert r.causal
        assert r.receptive_field > 1
        assert r.converged is False  # one epoch hits the cap
        assert r.precision == '32-true'
        assert r.device == 'cpu'
        assert r.sim_config_hash
        assert r.sim_name == 'smi'
        assert r.wall_clock_s > 0
        assert r.epochs_run == 1
        assert r.best_monitor <= r.final_monitor
        assert r.zero_predictor_monitor > 0
        assert (r.train_crop_length, r.steps_per_epoch, r.batch_size) == (256, 1, 2)
        family = model_config('matched/tcn-matched').training_config
        assert (r.max_epochs, r.T_0) == (1, family['T_0'])
        assert r.lr == float(family['learning_rate'])
        assert {'velocity_rmse', 'velocity_rmse_sign_invariant'} <= set(r.metrics)
        assert (tmp_path / 'runs' / 'grid' / f'{r.cell_key}.ckpt').exists()
    # Rerun: every key is present, nothing is trained or appended.
    assert _run(grid, tmp_path) == 0
    assert len(read_results(tmp_path / 'results.json')) == 2


def _one_cell(tmp_path, **overrides):
    grid = _grid(seeds=(0,), **overrides)
    assert _run(grid, tmp_path) == 1
    return grid


def test_displacement_target_cell_trains_and_scores_its_differentiated_output(
    tmp_path, tiny_family
):
    grid = _grid(seeds=(0,))
    tiny_family.training_config['target'] = 'displacement'
    tiny_family.loss_config.update(
        velocity_loss_weight=0.0, displacement_loss_weight=1.0
    )
    assert _run(grid, tmp_path) == 1
    (record,) = read_results(tmp_path / 'results.json')
    ckpt = tmp_path / 'runs' / 'grid' / f'{record.cell_key}.ckpt'
    module = load_lit_module(ckpt, device='cpu')
    assert module.target == 'displacement'
    # Scored by hand: the raw output is displacement, so it is differentiated
    # with the training code's sample rate before the velocity RMSE.
    with h5py.File(tmp_path / 'frozen' / f'{record.dataset}.h5', 'r') as f:
        keys = select_pd_keys(f, tuple(record.wavelengths_um))
        signals = torch.from_numpy(np.stack([f[k][:] for k in keys], axis=1))
        velocity = torch.from_numpy(f['velocity'][:])
        sample_rate = float(f.attrs['sample_rate'])
    with torch.no_grad():
        displacement = module.model(signals)[:, 0]
    pred = derivative_displacement(displacement, sample_rate)
    rmse = torch.sqrt(torch.mean((pred - velocity) ** 2, dim=-1)).mean()
    assert record.metrics['velocity_rmse']['mean'] == pytest.approx(
        float(rmse), rel=1e-5
    )
    assert 'displacement_rmse' in record.metrics


def test_resume_refuses_a_key_recorded_under_another_budget(tmp_path):
    _one_cell(tmp_path)
    grid = _grid(seeds=(0,), training={**TRAINING, 'max_epochs': 2})
    with pytest.raises(ValueError, match=r'differing blocks.*training'):
        _run(grid, tmp_path)
    assert len(read_results(tmp_path / 'results.json')) == 1


def test_resume_refuses_a_key_recorded_under_another_unbudgeted_setting(tmp_path):
    """Settings outside the six budget fields still change what a cell trained."""
    grid = _one_cell(tmp_path)
    decayed = replace(grid, training={**TRAINING, 'weight_decay': 0.123})
    with pytest.raises(ValueError, match=r'differing blocks.*training'):
        _run(decayed, tmp_path)
    with pytest.raises(ValueError, match=r'differing blocks.*test_set'):
        _run(replace(grid, test_set=EvalSet(seed=5, n_shots=3)), tmp_path)
    assert len(read_results(tmp_path / 'results.json')) == 1


def test_resume_refuses_a_key_recorded_under_another_architecture(
    tmp_path, tiny_family
):
    grid = _one_cell(tmp_path)
    tiny_family.model_config['temporal_channels'] = [4, 8]
    with pytest.raises(ValueError, match=r'differing blocks.*model'):
        _run(grid, tmp_path)


def test_resume_skips_an_unchanged_cell_by_fingerprint(tmp_path):
    grid = _one_cell(tmp_path)
    (record,) = read_results(tmp_path / 'results.json')
    assert record.run_fingerprint
    assert _run(grid, tmp_path) == 0


def test_resume_refuses_a_key_recorded_under_another_sim(tmp_path):
    _one_cell(tmp_path)
    sims = (replace(SMI, override={'snr_db': 30.0}),)
    grid = _grid(seeds=(0,), sims=sims)
    with pytest.raises(ValueError, match='sim_config_hash'):
        _run(grid, tmp_path)


def test_full_length_model_ignores_the_grid_crop(tmp_path):
    """The offline neural operator trains on full shots when the grid crops."""
    family = Family('no', 'matched/neuraloperator-offline-19k')
    grid = _grid(families=(family,))
    cell, *_ = expand_grid(grid)
    config = experiments.training_config(cell, grid)
    assert config.synthetic_config['train_crop_length'] is None
    assert config.synthetic_config['steps_per_epoch'] == 1  # grid value kept


def test_family_crop_does_not_override_the_grid_crop(tmp_path, tiny_family):
    tiny_family.synthetic_config = {'train_crop_length': None}
    grid = _grid()
    cell, *_ = expand_grid(grid)
    config = experiments.training_config(cell, grid)
    assert config.synthetic_config['train_crop_length'] == 256


DISPLACEMENT_KEYS = ('displacement_rmse', 'displacement_rmse_sign_invariant')


def _strip_displacement_metrics(out, edit_velocity=1.0):
    """Rewrite ``out`` as records scored before evaluate had displacement.

    ``edit_velocity`` scales each recorded velocity RMSE mean, to fake a
    record that does not belong to its checkpoint or test set.
    """
    payload = json.loads(out.read_text())
    for records in payload['datasets'].values():
        for record in records:
            for key in DISPLACEMENT_KEYS:
                del record['metrics'][key]
            record['metrics']['velocity_rmse']['mean'] *= edit_velocity
    # write_results' own formatting, so an untouched record stays byte-equal.
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


class _ConstantOracle:
    name = model_type = 'inverse_fit'
    causal = False
    total_params = 0
    receptive_field = None
    device = 'cpu'
    config: dict = {}  # noqa: RUF012 -- stub attribute, never mutated

    def predict_velocity(self, signals, _wavelengths_um):
        return signals[:, 0] * 0 + 1.0


def _rescore(tmp_path, include_oracle=False):
    return rescore_metrics(
        tmp_path / 'results.json',
        runs_dir=tmp_path / 'runs',
        lock_path=tmp_path / '.runner.lock',
        frozen_root=tmp_path / 'frozen',
        device='cpu',
        include_oracle=include_oracle,
    )


def test_rescore_restores_exactly_what_a_fresh_run_records(tmp_path):
    """Only the new metric keys are added; fingerprints and the rest are kept."""
    _run(_grid(), tmp_path)
    out = tmp_path / 'results.json'
    fresh = out.read_text()
    _strip_displacement_metrics(out)
    assert out.read_text() != fresh
    assert _rescore(tmp_path) == 2
    assert out.read_text() == fresh
    # Idempotent: every record now carries the metric, so nothing is rescored.
    assert _rescore(tmp_path) == 0
    assert out.read_text() == fresh
    assert _run(_grid(), tmp_path) == 0  # resume still skips both


# NaN: a comparison written as "differs by more than tol" is False for NaN, so
# a corrupt recorded mean would have slipped through as a match.
@pytest.mark.parametrize('edit', [1.0 + 2e-4, math.nan])
def test_rescore_refuses_a_velocity_mismatch_and_writes_nothing(tmp_path, edit):
    _one_cell(tmp_path)
    out = tmp_path / 'results.json'
    _strip_displacement_metrics(out, edit_velocity=edit)
    before = out.read_text()
    with pytest.raises(ValueError, match='velocity_rmse'):
        _rescore(tmp_path)
    assert out.read_text() == before


def test_rescore_scores_oracle_records_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiments.InverseFitEstimator, 'for_dataset', lambda *_: _ConstantOracle()
    )
    _run(_grid(oracle=True, seeds=()), tmp_path)
    out = tmp_path / 'results.json'
    fresh = out.read_text()
    _strip_displacement_metrics(out)
    assert _rescore(tmp_path) == 0
    assert _rescore(tmp_path, include_oracle=True) == 1
    assert out.read_text() == fresh


def test_rescore_names_an_empty_runs_dir(tmp_path):
    _one_cell(tmp_path)
    _strip_displacement_metrics(tmp_path / 'results.json')
    with pytest.raises(FileNotFoundError, match='grid directories'):
        rescore_metrics(
            tmp_path / 'results.json',
            runs_dir=tmp_path / 'nothing',
            lock_path=tmp_path / '.runner.lock',
            frozen_root=tmp_path / 'frozen',
            device='cpu',
            include_oracle=False,
        )


def test_only_filters_cells_by_glob(tmp_path):
    grid = _grid()
    assert _run(grid, tmp_path, only=['*seed1']) == 1
    keys = [r.cell_key for r in read_results(tmp_path / 'results.json')]
    assert keys == ['tcn__ch635-675-515__smi__seed1']


def test_runner_evaluates_the_best_checkpoint(tmp_path, monkeypatch):
    """With epoch 1 scripted best and epoch 2 worse, epoch 1's weights are scored."""
    scripted = [1.0, 0.1, 5.0]  # monitored value by epoch: the middle one is best
    snapshots: dict[int, dict] = {}
    real_log = LitModule.log

    def log(self, name, value, *args, **kwargs):
        if name == SIGNED_RMSE_KEY:
            epoch = self.trainer.current_epoch
            snapshots[epoch] = {
                k: v.detach().clone() for k, v in self.state_dict().items()
            }
            value = torch.tensor(scripted[epoch])
        return real_log(self, name, value, *args, **kwargs)

    monkeypatch.setattr(LitModule, 'log', log)
    _run(_grid(training={**TRAINING, 'max_epochs': 3}, seeds=(0,)), tmp_path)
    (record,) = read_results(tmp_path / 'results.json')
    assert record.epochs_run == 3
    assert record.best_monitor == pytest.approx(0.1)
    assert record.final_monitor == pytest.approx(5.0)
    ckpt = torch.load(
        tmp_path / 'runs' / 'grid' / f'{record.cell_key}.ckpt', weights_only=False
    )
    saved = ckpt['state_dict']
    best, last = snapshots[1], snapshots[2]
    assert any(not torch.equal(best[k], last[k]) for k in best), 'epochs identical'
    for k, v in best.items():
        torch.testing.assert_close(saved[k], v, atol=0, rtol=0)


@pytest.mark.parametrize(('zero_scale', 'converged'), [(1e6, True), (1e-6, False)])
def test_patience_exhausted_cell_is_converged_only_below_the_zero_predictor(
    tmp_path, monkeypatch, zero_scale, converged
):
    # Scale the logged zero-predictor value far above (a real plateau) or far
    # below (a stall at the zero-output plateau) whatever the tiny TCN reaches.
    script_log(monkeypatch, None, zero=lambda v: v * zero_scale)
    training = {
        **TRAINING,
        'max_epochs': 40,
        'convergence': {
            'monitor': 'unused',
            'patience_epochs': 5,
            'min_rel_delta': 0.9,
            'min_epochs': 0,
        },
    }
    _run(_grid(seeds=(0,), training=training), tmp_path)
    (record,) = read_results(tmp_path / 'results.json')
    assert record.converged is converged
    assert record.epochs_run < 40
    assert (record.best_monitor < record.zero_predictor_monitor) is converged


def test_oracle_cell_is_untrained_and_converged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        experiments.InverseFitEstimator, 'for_dataset', lambda *_: _ConstantOracle()
    )
    grid = _grid(oracle=True, seeds=())
    assert _run(grid, tmp_path) == 1
    (record,) = read_results(tmp_path / 'results.json')
    assert record.cell_key == 'inverse_fit__ch635-675-515__smi__oracle'
    assert record.converged is True
    assert record.train_seed is None
    assert not (tmp_path / 'runs').exists()  # nothing trained, no checkpoint
    assert _run(grid, tmp_path) == 0  # the oracle's record resumes too


def test_sweep_and_subsets_expand_to_stable_readable_keys(tmp_path):
    grid = _grid(
        channel_subsets=((0.635,), (0.635, 0.6748)),
        sims=(
            Sim(
                'nominal',
                'nominal',
                feedback_c=0.2,
                alpha=3.0,
                sweep=('visibility_range', ((0.5, 1.0), (0.9, 1.0))),
            ),
        ),
        seeds=(0,),
        oracle=True,
    )
    cells = expand_grid(grid)
    # 2 subsets x 2 sweep points x (oracle + 1 family x 1 seed)
    assert len(cells) == 8
    assert cells[0].key == 'inverse_fit__ch635__nominal-visibility_range=0.5-1__oracle'
    assert cells[1].key == 'tcn__ch635__nominal-visibility_range=0.5-1__seed0'
    assert cells[-1].sim.visibility_range == (0.9, 1.0)
    assert cells[-1].wavelengths_um == (0.635, 0.6748)
    assert expand_grid(grid) == cells
    # Resume matches records by key, so the sweep field is carried beside the
    # key rather than recovered from it, and the keys stay as they were.
    assert [(c.key, c.sweep) for c in cells if c.seed is None] == [
        (
            'inverse_fit__ch635__nominal-visibility_range=0.5-1__oracle',
            ('visibility_range', (0.5, 1.0)),
        ),
        (
            'inverse_fit__ch635__nominal-visibility_range=0.9-1__oracle',
            ('visibility_range', (0.9, 1.0)),
        ),
        (
            'inverse_fit__ch635-675__nominal-visibility_range=0.5-1__oracle',
            ('visibility_range', (0.5, 1.0)),
        ),
        (
            'inverse_fit__ch635-675__nominal-visibility_range=0.9-1__oracle',
            ('visibility_range', (0.9, 1.0)),
        ),
    ]


def test_a_sim_without_a_sweep_carries_none(tmp_path):
    cells = expand_grid(_grid())
    assert {c.sweep for c in cells} == {None}


def test_shipped_grids_expand_and_name_existing_configs():
    # 14 families (LSTM, PatchTransformer, UNetTransformer and NeuralOperator
    # counted once per direction) x 3 seeds, plus the oracle.
    for name, n_cells in (
        ('q3_noise_free', 7 * (3 * 14 + 1)),
        ('q1_channels', 14 * (3 * 14 + 1)),
    ):
        cells = expand_grid(GRIDS[name])
        assert len(cells) == n_cells
        assert all(
            c.family_config in MODEL_CONFIGS for c in cells if c.seed is not None
        )
        # Michelson cannot show direction; no SMI sim may claim sign invariance.
        for c in cells:
            assert c.sign_invariant == (c.sim.feedback_c == 0.0)


def test_displacement_grid_retargets_the_s1_cells_and_nothing_else():
    """Each family is its matched config with only the target, its monitor and
    the loss weights changed, on S1's sims and test sets, so each cell pairs
    with its S1 cell.
    """
    grid, s1_grid = GRIDS['q3_s1d_displacement'], GRIDS['q3_s1_families']
    cells, s1_cells = expand_grid(grid), expand_grid(s1_grid)
    assert len(cells) == 4 * 2
    assert grid.test_set == s1_grid.test_set
    s1_sims = {c.sim_name: c.sim for c in s1_cells}
    for cell in cells:
        assert cell.sim == s1_sims[cell.sim_name]
        name = cell.family_config
        assert name is not None
        assert name.startswith('displacement/')
        retargeted = model_config(name)
        matched = model_config(name.replace('displacement/', 'matched/', 1))
        matched.training_config['target'] = 'displacement'
        matched.training_config['convergence']['monitor'] = 'val/displacement_rmse_um'
        matched.loss_config.update(
            velocity_loss_weight=0.0, displacement_loss_weight=1.0
        )
        assert retargeted == matched, name
        monitor = training_config(cell, grid).training_config['convergence']
        assert monitor['monitor'] == (
            'val/displacement_rmse_sign_invariant_um'
            if cell.sign_invariant
            else 'val/displacement_rmse_um'
        )


def test_shipped_velocity_cells_keep_their_fingerprints():
    """Resume matches a recorded cell by its fingerprint, so a code change that
    moved one would make every recorded cell of that grid retrain or be refused.
    The golden values were computed at adb199e, before the convergence monitor
    followed the target, and migrated once when the model configs dropped
    their stale ``data_dir`` and string-typed numerics; they cover every
    velocity-target cell of every grid.
    """
    golden = json.loads(
        (Path(__file__).parent / 'golden' / 'grid_fingerprints.json').read_text()
    )
    for name, want in golden.items():
        grid = GRIDS[name]
        cells = expand_grid(grid)
        got = {
            c.key: cell_fingerprint(
                c, grid, None if c.seed is None else training_config(c, grid)
            )['run_fingerprint']
            for c in cells
            if c.key in want
        }
        assert got == want, name


def test_lock_refuses_second_holder_and_releases(tmp_path):
    lock = tmp_path / '.runner.lock'
    with runner_lock(lock):
        assert lock.read_text() == str(os.getpid())
        with pytest.raises(RunnerLockedError), runner_lock(lock):
            pass
        assert lock.exists()  # the refused attempt did not steal it


def test_leftover_lockfile_of_a_dead_process_is_acquirable(tmp_path):
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    lock = tmp_path / '.runner.lock'
    lock.write_text(str(dead.pid))
    with runner_lock(lock):
        assert lock.read_text() == str(os.getpid())


def test_lock_held_by_live_process_survives_an_emptied_lockfile(tmp_path):
    lock = tmp_path / '.runner.lock'
    code = (
        'import time\n'
        'from pathlib import Path\n'
        'from smi.experiments.runner import runner_lock\n'
        f'with runner_lock(Path({str(lock)!r})):\n'
        "    print('held', flush=True)\n"
        '    time.sleep(60)\n'
    )
    holder = subprocess.Popen(
        [sys.executable, '-c', code], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == 'held'
        # Invariant: a live holder keeps the lock even when its file holds no PID.
        lock.write_text('')
        with pytest.raises(RunnerLockedError), runner_lock(lock):
            pass
    finally:
        holder.kill()
        holder.wait()


def test_concurrent_runner_process_exits_nonzero(tmp_path):
    # A shipped grid: the child process takes a grid name. It is refused at
    # the lock, before anything trains.
    lock = tmp_path / '.runner.lock'
    with runner_lock(lock):
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / 'scripts' / 'run_experiments.py'),
                'q3_s0c_long',
                '--out',
                str(tmp_path / 'results.json'),
                '--lock',
                str(lock),
                '--runs-dir',
                str(tmp_path / 'runs' / 'grid'),
                '--frozen-root',
                str(tmp_path / 'frozen'),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode != 0
    assert 'holds' in proc.stderr
    assert not (tmp_path / 'results.json').exists()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_sigterm_kills_children_and_releases_lock(tmp_path):
    lock = tmp_path / '.runner.lock'
    code = (
        'import subprocess, sys, time\n'
        'from pathlib import Path\n'
        'from smi.experiments.runner import runner_lock\n'
        f'with runner_lock(Path({str(lock)!r})):\n'
        '    child = subprocess.Popen(\n'
        "        [sys.executable, '-c', 'import time; time.sleep(60)']\n"
        '    )\n'
        '    print(child.pid, flush=True)\n'
        '    time.sleep(60)\n'
    )
    proc = subprocess.Popen(
        [sys.executable, '-c', code], stdout=subprocess.PIPE, text=True
    )
    child_pid = int(proc.stdout.readline())
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    deadline = time.time() + 10
    while _pid_alive(child_pid) and time.time() < deadline:
        # A killed child stays a zombie until reaped; init reaps it shortly.
        time.sleep(0.1)
    assert not _pid_alive(child_pid)


def test_resume_refuses_a_key_recorded_by_another_simulator(tmp_path, monkeypatch):
    grid = _one_cell(tmp_path)
    monkeypatch.setattr(experiments, 'SIMULATOR_VERSION', -1)
    with pytest.raises(ValueError, match='simulator_version'):
        _run(grid, tmp_path)


def test_grid_naming_an_unknown_family_fails_when_built(tmp_path):
    with pytest.raises(
        KeyError, match=r"unknown model config 'matched/tnc-44k'.*tcn-44k"
    ):
        _grid(families=(Family('x', 'matched/tnc-44k'),))


def test_sim_naming_an_unknown_field_fails_when_built():
    with pytest.raises(ValueError, match=r"'snr' is not a SimConfig field.*snr_db"):
        replace(SMI, override={'snr': 30.0})
    with pytest.raises(ValueError, match='not a SimConfig field'):
        replace(SMI, sweep=('feedbak_c', (0.1,)))


def test_r1_r2_r4_grids_expand_to_their_planned_cells():
    # R1: N = 1..6 x 2 families x 2 sims at seed 0; seeds 1-2 at N = 1, 3 only.
    count = expand_grid(GRIDS['q1_r1_count'])
    assert len(count) == 6 * 2 * 2
    assert sorted({len(c.wavelengths_um) for c in count}) == [1, 2, 3, 4, 5, 6]
    seeds = expand_grid(GRIDS['q1_r1_seeds'])
    assert len(seeds) == 2 * 2 * 2 * 2
    assert {len(c.wavelengths_um) for c in seeds} == {1, 3}
    # R1 pairs: the two rig pairs q1_r1_count's N = 2 cell leaves out.
    pairs = expand_grid(GRIDS['q1_r1_pairs'])
    assert len(pairs) == 2 * 2 * 2
    assert {c.wavelengths_um for c in pairs} == {(0.635, 0.515), (0.6748, 0.515)}
    # R2: one color repeated N times; quadrature at N = 2, three-phase at N = 3.
    for name, n in (('q1_r2_colors', 2), ('q1_r2_colors_n3', 3)):
        cells = expand_grid(GRIDS[name])
        assert len(cells) == 2 * 2
        for c in cells:
            assert c.wavelengths_um == (0.635,) * n
            assert c.sim.channel_phase_offsets_rad == pytest.approx(
                (0.0, math.pi / 2)
                if n == 2
                else (0.0, 2 * math.pi / 3, 4 * math.pi / 3)
            )
    # R4: 3 families x (7 knobs at nominal + 2 stressed rungs + nominal point).
    noise = expand_grid(GRIDS['q2_r4_noise'])
    assert len(noise) == 3 * 10
    assert all(c.family_config in MODEL_CONFIGS for c in noise)


def test_q2_r4_noise_runs_each_knob_at_its_nominal_value():
    grid = GRIDS['q2_r4_noise']
    nominal = SimConfig.nominal(
        STEMLAB_125_14, grid.channel_subsets[0], feedback_c=0.2, alpha=3.0
    )
    swept = {s.sweep[0]: s.sweep[1] for s in grid.sims if s.sweep is not None}
    # The ladder's ADC rung fixes its own full scale; 14 bits is the board's.
    assert swept.pop('adc_bits') == (nominal.adc_bits,) == (14,)
    stressed_values = {'phase_drift_rad_per_sqrt_s': 116.0723093699996, 'snr_db': 20.0}
    for field, values in swept.items():
        stressed = (stressed_values[field],) if field in stressed_values else ()
        assert values == (getattr(nominal, field), *stressed), field


def _q3_fiber_grid():
    return GRIDS['q3_fiber']


def test_q3_fiber_grid_expands_to_two_families_by_four_arms():
    cells = expand_grid(_q3_fiber_grid())
    assert len(cells) == 2 * 4
    channels = {c.sim_name: c.sim.n_signal_channels for c in cells}
    assert channels == {'nominal': 3, 'fiber': 3, 'fiber-monitor': 4, 'fiber-pbs': 6}
    monitor = next(c for c in cells if c.sim_name == 'fiber-monitor')
    assert monitor.wavelengths_um == (0.635, 0.6748, 0.515, 0.785)
    assert monitor.key.split('__')[1] == 'ch635-675-515-785'
    # One frozen test set per arm: every arm is a distinct sim.
    assert len({c.sim.hash() for c in cells}) == 4


def test_q3_fiber_families_take_each_arms_input_channels():
    grid = _q3_fiber_grid()
    for cell in expand_grid(grid):
        hp = {**training_config(cell, grid).model_config}
        hp['in_channels'] = cell.sim.n_signal_channels
        model = create_model(hp).eval()
        batch = simulate_batch(cell.sim, 1, torch.device('cpu'), torch.Generator())
        x = batch.signals[..., :4096]
        assert x.shape[1] == cell.sim.n_signal_channels
        with torch.no_grad():
            assert model(x).shape[-1] == 4096, cell.key


@pytest.mark.parametrize('arm', ['fiber-monitor', 'fiber-pbs'])
def test_fiber_arm_trains_and_scores_end_to_end(tmp_path, arm):
    (sim,) = [s for s in _q3_fiber_grid().sims if s.name == arm]
    grid = _grid(seeds=(0,), sims=(sim,))
    assert _run(grid, tmp_path) == 1
    (record,) = read_results(tmp_path / 'results.json')
    assert math.isfinite(record.metrics['velocity_rmse']['mean'])
