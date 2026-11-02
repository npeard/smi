"""Pytest-wide configuration.

Pin native-math thread counts to 1 for the test session. On this Intel-mac /
conda setup the pypi torch wheel bundles its own OpenMP runtime while conda
numpy/scipy/h5py use the conda ``llvm-openmp``; with multiple threads the two
runtimes collide and segfault the suite once several modules are exercised in a
single process (each test file passes in isolation, but the full run crashes).
Forcing a single thread sidesteps the collision. The tests are small so the
speed cost is negligible, and training is unaffected (it does not import this
file). OpenMP/BLAS read these variables when their thread pool first
initializes, so setting them here -- before any test runs a heavy numeric op --
is early enough even though numpy is already imported.
"""

import os
import shutil
import subprocess
import sys

import pytest

from smi.core.paths import REPO_ROOT

for _var in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_var, '1')


# A plugin, not an import: pytest loads it after this module has run, so torch
# loads only once the thread pins above are in the environment.
pytest_plugins = ['tests._sims']


# Every library default that names a repo path, as (dotted module attribute,
# subpath under the test's tmp_path). Patched by name so torch loads after the
# thread pins above, not at import. The shared TrainingInterface directory also
# gave every xdist worker the same ``epoch=<n>.ckpt`` names, and one run's
# save_top_k cleanup deleted another's.
REDIRECTED_DEFAULTS = (
    ('smi.learning.training_interface.TrainingInterface.CHECKPOINT_DIR', 'train'),
    ('smi.experiments.q1_binned.GRID_RUNS_DIR', 'grids'),
    ('smi.experiments.q1_binned.DEFAULT_ROOT', 'frozen'),
    ('smi.data.frozen.DEFAULT_ROOT', 'frozen'),
)


@pytest.fixture(autouse=True)
def private_write_defaults(tmp_path, monkeypatch):
    """Send every default output location under the test's tmp_path."""
    for name, sub in REDIRECTED_DEFAULTS:
        monkeypatch.setattr(name, tmp_path / sub)
    monkeypatch.setenv('WANDB_DIR', str(tmp_path))
    monkeypatch.setenv('WANDB_MODE', 'offline')


def _git_status() -> str:
    # Untracked-but-ignored paths (runs/, data/) are excluded by default, so a
    # real run writing there concurrently cannot trip the check.
    git = shutil.which('git')
    assert git is not None
    return subprocess.run(
        [git, 'status', '--porcelain'],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _is_worker(config) -> bool:
    return hasattr(config, 'workerinput')


def pytest_sessionstart(session):
    if not _is_worker(session.config):
        session.config.stash_status = _git_status()


def pytest_sessionfinish(session, exitstatus):
    """Fail the session if it changed the tracked or unignored files in the repo."""
    config = session.config
    if _is_worker(config) or not hasattr(config, 'stash_status'):
        return
    before = set(config.stash_status.splitlines())
    changed = sorted(before ^ set(_git_status().splitlines()))
    if changed:
        sys.stderr.write(f'\nthe test session changed the repo: {changed}\n')
        session.exitstatus = 1
