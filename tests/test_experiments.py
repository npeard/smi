"""Experiment runner: grid expansion, resumability, locking, record contents."""

import os
import signal
import subprocess
import sys
import time

import pytest
import torch
import yaml

from smi.analysis import experiments, synthetic_lit_module
from smi.analysis.experiments import (
    RunnerLockedError,
    expand_grid,
    run_grid,
    runner_lock,
)
from smi.analysis.lit_module import LitModule
from smi.analysis.results_artifact import read_results
from smi.analysis.training_interface import SIGNED_RMSE_KEY

MATCHED = experiments.ANALYSIS_DIR / 'models' / 'configs' / 'matched'


@pytest.fixture(autouse=True)
def small_simulations(small_chunks, monkeypatch):
    monkeypatch.setattr(synthetic_lit_module, 'NORM_STATS_SHOTS', 2)


def _grid(tmp_path, family_synthetic=None, **overrides):
    """A tiny CPU grid whose family is a 2-layer TCN."""
    family = yaml.safe_load((MATCHED / 'tcn-matched.yaml').read_text())
    family['model']['temporal_channels'] = [4, 4]
    if family_synthetic is not None:
        family['synthetic'] = family_synthetic
    (tmp_path / 'tiny-tcn.yaml').write_text(yaml.safe_dump(family))
    grid = {
        'families': [{'name': 'tcn', 'config': 'tiny-tcn.yaml'}],
        'channel_subsets': [[0.635, 0.6748, 0.515]],
        'sims': [
            {'name': 'smi', 'preset': 'noise_free', 'feedback_c': 0.2, 'alpha': 3.0}
        ],
        'seeds': [0, 1],
        'oracle': False,
        'loss': {'sign_invariant': False},
        'synthetic': {'steps_per_epoch': 1, 'val_steps': 1, 'train_crop_length': 256},
        'training': {
            'accelerator': 'cpu',
            'devices': 1,
            'max_epochs': 1,
            'batch_size': 2,
            'use_logging': False,
        },
        'test_set': {'seed': 5, 'n_shots': 2},
    }
    grid.update(overrides)
    path = tmp_path / 'grid.yaml'
    path.write_text(yaml.safe_dump(grid))
    return path


def _run(path, tmp_path, only=None):
    return run_grid(
        path,
        tmp_path / 'results.json',
        runs_dir=tmp_path / 'runs',
        lock_path=tmp_path / '.runner.lock',
        frozen_root=tmp_path / 'frozen',
        only=only,
    )


def test_two_cell_grid_writes_complete_records_and_resumes(tmp_path):
    path = _grid(tmp_path)
    assert _run(path, tmp_path) == 2
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
        assert (r.train_crop_length, r.steps_per_epoch, r.batch_size) == (256, 1, 2)
        family = yaml.safe_load((tmp_path / 'tiny-tcn.yaml').read_text())['training']
        assert (r.max_epochs, r.T_0) == (1, family['T_0'])
        assert r.lr == float(family['learning_rate'])
        assert {'velocity_rmse', 'velocity_rmse_sign_invariant'} <= set(r.metrics)
        assert (tmp_path / 'runs' / f'{r.cell_key}.ckpt').exists()
    # Rerun: every key is present, nothing is trained or appended.
    assert _run(path, tmp_path) == 0
    assert len(read_results(tmp_path / 'results.json')) == 2


def _one_cell(tmp_path, **overrides):
    path = _grid(tmp_path, seeds=[0], **overrides)
    assert _run(path, tmp_path) == 1
    return path


def test_resume_refuses_a_key_recorded_under_another_budget(tmp_path):
    _one_cell(tmp_path)
    budget = {
        'accelerator': 'cpu',
        'devices': 1,
        'max_epochs': 2,
        'batch_size': 2,
        'use_logging': False,
    }
    path = _grid(tmp_path, seeds=[0], training=budget)
    with pytest.raises(ValueError, match=r'differing blocks.*training'):
        _run(path, tmp_path)
    assert len(read_results(tmp_path / 'results.json')) == 1


def test_resume_refuses_a_key_recorded_under_another_unbudgeted_setting(tmp_path):
    """Settings outside the six budget fields still change what a cell trained."""
    path = _one_cell(tmp_path)
    grid = yaml.safe_load(path.read_text())
    grid['training']['weight_decay'] = 0.123
    path.write_text(yaml.safe_dump(grid))
    with pytest.raises(ValueError, match=r'differing blocks.*training'):
        _run(path, tmp_path)
    grid['training'].pop('weight_decay')
    grid['test_set']['n_shots'] = 3
    path.write_text(yaml.safe_dump(grid))
    with pytest.raises(ValueError, match=r'differing blocks.*test_set'):
        _run(path, tmp_path)
    assert len(read_results(tmp_path / 'results.json')) == 1


def test_resume_refuses_a_key_recorded_under_another_architecture(tmp_path):
    path = _one_cell(tmp_path)
    family = yaml.safe_load((tmp_path / 'tiny-tcn.yaml').read_text())
    family['model']['temporal_channels'] = [4, 8]
    (tmp_path / 'tiny-tcn.yaml').write_text(yaml.safe_dump(family))
    with pytest.raises(ValueError, match=r'differing blocks.*model'):
        _run(path, tmp_path)


def test_resume_skips_an_unchanged_cell_by_fingerprint(tmp_path):
    path = _one_cell(tmp_path)
    (record,) = read_results(tmp_path / 'results.json')
    assert record.run_fingerprint
    assert _run(path, tmp_path) == 0


def test_resume_refuses_a_key_recorded_under_another_sim(tmp_path):
    _one_cell(tmp_path)
    sims = [
        {
            'name': 'smi',
            'preset': 'noise_free',
            'feedback_c': 0.2,
            'alpha': 3.0,
            'override': {'snr_db': 30.0},
        }
    ]
    path = _grid(tmp_path, seeds=[0], sims=sims)
    with pytest.raises(ValueError, match='sim_config_hash'):
        _run(path, tmp_path)


def test_full_length_model_ignores_the_grid_crop(tmp_path):
    """The offline neural operator trains on full shots when the grid crops."""
    family = 'neuraloperator-offline-19k.yaml'
    (tmp_path / family).write_text((MATCHED / family).read_text())
    path = _grid(tmp_path, families=[{'name': 'no', 'config': family}])
    cell = expand_grid(path)[0]
    config = experiments._training_config(cell, yaml.safe_load(path.read_text()))
    assert config.synthetic_config['train_crop_length'] is None
    assert config.synthetic_config['steps_per_epoch'] == 1  # grid value kept


def test_family_crop_does_not_override_the_grid_crop(tmp_path):
    path = _grid(tmp_path, family_synthetic={'train_crop_length': None})
    cell = expand_grid(path)[0]
    config = experiments._training_config(cell, yaml.safe_load(path.read_text()))
    assert config.synthetic_config['train_crop_length'] == 256


def test_only_filters_cells_by_glob(tmp_path):
    path = _grid(tmp_path)
    assert _run(path, tmp_path, only=['*seed1']) == 1
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
    path = _grid(tmp_path)
    grid = yaml.safe_load(path.read_text())
    grid['training']['max_epochs'] = 3
    grid['seeds'] = [0]
    path.write_text(yaml.safe_dump(grid))
    _run(path, tmp_path)
    (record,) = read_results(tmp_path / 'results.json')
    assert record.epochs_run == 3
    assert record.best_monitor == pytest.approx(0.1)
    assert record.final_monitor == pytest.approx(5.0)
    ckpt = torch.load(tmp_path / 'runs' / f'{record.cell_key}.ckpt', weights_only=False)
    saved = ckpt['state_dict']
    best, last = snapshots[1], snapshots[2]
    assert any(not torch.equal(best[k], last[k]) for k in best), 'epochs identical'
    for k, v in best.items():
        torch.testing.assert_close(saved[k], v, atol=0, rtol=0)


def test_patience_exhausted_cell_records_converged(tmp_path):
    path = _grid(tmp_path)
    grid = yaml.safe_load(path.read_text())
    grid['seeds'] = [0]
    grid['training'].update(
        max_epochs=40,
        convergence={'monitor': 'unused', 'patience_epochs': 5, 'min_rel_delta': 0.9},
    )
    path.write_text(yaml.safe_dump(grid))
    _run(path, tmp_path)
    (record,) = read_results(tmp_path / 'results.json')
    assert record.converged is True
    assert record.epochs_run < 40


def test_oracle_cell_is_untrained_and_converged(tmp_path, monkeypatch):
    class Stub:
        name = model_type = 'inverse_fit'
        causal = False
        total_params = 0
        receptive_field = 16384
        device = 'cpu'
        config: dict = {}  # noqa: RUF012 -- stub attribute, never mutated

        def predict_velocity(self, signals, _wavelengths_um):
            return signals[:, 0] * 0 + 1.0

    monkeypatch.setattr(
        experiments.InverseFitEstimator, 'for_dataset', lambda *_: Stub()
    )
    path = _grid(tmp_path, oracle=True, seeds=[])
    assert _run(path, tmp_path) == 1
    (record,) = read_results(tmp_path / 'results.json')
    assert record.cell_key == 'inverse_fit__ch635-675-515__smi__oracle'
    assert record.converged is True
    assert record.train_seed is None
    assert not (tmp_path / 'runs').exists()  # nothing trained, no checkpoint
    assert _run(path, tmp_path) == 0  # the oracle's record resumes too


def test_sweep_and_subsets_expand_to_stable_readable_keys(tmp_path):
    path = _grid(
        tmp_path,
        channel_subsets=[[0.635], [0.635, 0.6748]],
        sims=[
            {
                'name': 'nominal',
                'preset': 'nominal',
                'feedback_c': 0.2,
                'alpha': 3.0,
                'sweep': {'visibility_range': [[0.5, 1.0], [0.9, 1.0]]},
            }
        ],
        seeds=[0],
        oracle=True,
    )
    cells = expand_grid(path)
    # 2 subsets x 2 sweep points x (oracle + 1 family x 1 seed)
    assert len(cells) == 8
    assert cells[0].key == 'inverse_fit__ch635__nominal-visibility_range=0.5-1__oracle'
    assert cells[1].key == 'tcn__ch635__nominal-visibility_range=0.5-1__seed0'
    assert cells[-1].sim.visibility_range == (0.9, 1.0)
    assert cells[-1].wavelengths_um == (0.635, 0.6748)
    assert expand_grid(path) == cells


def test_shipped_grids_expand_and_name_existing_configs():
    # 14 families (LSTM, PatchTransformer, UNetTransformer and NeuralOperator
    # counted once per direction) x 3 seeds, plus the oracle.
    for name, n_cells in (
        ('q3_noise_free', 7 * (3 * 14 + 1)),
        ('q1_channels', 14 * (3 * 14 + 1)),
    ):
        path = experiments.ANALYSIS_DIR / 'experiments' / f'{name}.yaml'
        cells = expand_grid(path)
        assert len(cells) == n_cells
        assert all(c.family_config.exists() for c in cells)
        # Michelson cannot show direction; no SMI sim may claim sign invariance.
        for c in cells:
            assert c.sign_invariant == (c.sim.feedback_c == 0.0)


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
        'from smi.analysis.experiments import runner_lock\n'
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
    path = _grid(tmp_path)
    lock = tmp_path / '.runner.lock'
    with runner_lock(lock):
        proc = subprocess.run(
            [
                sys.executable,
                '-m',
                'smi.analysis.experiments',
                str(path),
                '--out',
                str(tmp_path / 'results.json'),
                '--lock',
                str(lock),
                '--runs-dir',
                str(tmp_path / 'runs'),
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
        'from smi.analysis.experiments import runner_lock\n'
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
    path = _one_cell(tmp_path)
    monkeypatch.setattr(experiments, 'SIMULATOR_VERSION', -1)
    with pytest.raises(ValueError, match='simulator_version'):
        _run(path, tmp_path)
