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

import pytest

from tests._repo_outputs import repo_output_files

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


@pytest.fixture(autouse=True)
def private_train_dir(tmp_path, monkeypatch):
    """Send every TrainingInterface's checkpoints under the test's tmp_path.

    The class default is the repo's ``runs/train``, which a test must not
    write: one shared directory also gave every xdist worker the same
    ``epoch=<n>.ckpt`` names, and one run's save_top_k cleanup deleted another's.
    """
    # By dotted name so torch loads after the thread pins above, not at import.
    monkeypatch.setattr(
        'smi.learning.training_interface.TrainingInterface.CHECKPOINT_DIR', tmp_path
    )


@pytest.fixture(scope='session', autouse=True)
def no_repo_writes():
    """Fail the session if it added a file to the run tree or the legacy dirs."""
    before = repo_output_files()
    yield
    added = sorted(repo_output_files() - before)
    assert not added, f'the test session wrote into the repo: {added}'
