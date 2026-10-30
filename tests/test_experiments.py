"""Experiment runner: grid expansion, resumability, locking, record contents."""

import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

from smi.analysis import experiments, synthetic_lit_module
from smi.analysis.experiments import (
    RunnerLockedError,
    expand_grid,
    run_grid,
    runner_lock,
)
from smi.analysis.results_artifact import read_results
from smi.synthetic import frozen

MATCHED = experiments.ANALYSIS_DIR / 'models' / 'configs' / 'matched'


@pytest.fixture(autouse=True)
def small_simulations(monkeypatch):
    monkeypatch.setattr(frozen, 'CHUNK_SHOTS', 2)
    monkeypatch.setattr(synthetic_lit_module, 'NORM_STATS_SHOTS', 2)


def _grid(tmp_path, **overrides):
    """A tiny CPU grid whose family is a 2-layer TCN."""
    family = yaml.safe_load((MATCHED / 'tcn-matched.yaml').read_text())
    family['model']['temporal_channels'] = [4, 4]
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
        assert {'velocity_rmse', 'velocity_rmse_sign_invariant'} <= set(r.metrics)
        assert (tmp_path / 'runs' / f'{r.cell_key}.ckpt').exists()
    assert not (tmp_path / '.runner.lock').exists()
    # Rerun: every key is present, nothing is trained or appended.
    assert _run(path, tmp_path) == 0
    assert len(read_results(tmp_path / 'results.json')) == 2


def test_only_filters_cells_by_glob(tmp_path):
    path = _grid(tmp_path)
    assert _run(path, tmp_path, only=['*seed1']) == 1
    keys = [r.cell_key for r in read_results(tmp_path / 'results.json')]
    assert keys == ['tcn__ch635-675-515__smi__seed1']


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
    for name, n_cells in (('q3_noise_free', 7 * 3 * 2 + 7), ('q1_channels', 14 * 7)):
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
    assert not lock.exists()


def test_stale_lock_is_reclaimed_with_warning(tmp_path, caplog):
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    lock = tmp_path / '.runner.lock'
    lock.write_text(str(dead.pid))
    with runner_lock(lock):
        assert lock.read_text() == str(os.getpid())
    assert 'stale' in caplog.text


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
    assert not lock.exists()
    deadline = time.time() + 10
    while experiments._pid_alive(child_pid) and time.time() < deadline:
        # A killed child stays a zombie until reaped; init reaps it shortly.
        time.sleep(0.1)
    assert not experiments._pid_alive(child_pid)
