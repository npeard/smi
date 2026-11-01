"""The package layering is enforced by import-linter, run from the suite."""

import importlib

import pytest
from importlinter.cli import lint_imports

from smi.core.paths import CONFIGS_DIR, DATA_DIR, FROZEN_DIR, REPO_ROOT, RESULTS_DIR


def test_contracts_hold():
    assert lint_imports(config_filename=str(REPO_ROOT / 'pyproject.toml')) == 0


@pytest.mark.parametrize(
    'name', ['smi.analysis', 'smi.synthetic', 'smi.redpitaya', 'smi.main']
)
def test_old_modules_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_paths_exist():
    # Checkpoints are gitignored and created on first training run.
    for path in (CONFIGS_DIR, RESULTS_DIR, DATA_DIR, FROZEN_DIR):
        assert path.is_dir(), path


def test_stemlab_contract_catches_a_new_smi_module():
    """A module added anywhere outside smi.acquisition must not reach stemlab."""
    violator = REPO_ROOT / 'smi' / 'core' / '_stemlab_violation.py'
    violator.write_text('import stemlab\n')
    try:
        assert lint_imports(config_filename=str(REPO_ROOT / 'pyproject.toml')) != 0
    finally:
        violator.unlink()
