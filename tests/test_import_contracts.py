"""The package layering is enforced by import-linter, run from the suite."""

import importlib

import pytest
from importlinter.cli import lint_imports

from smi.core.paths import CONFIGS_DIR, DATA_DIR, FROZEN_DIR, REPO_ROOT, RESULTS_DIR


def test_contracts_hold():
    assert lint_imports(config_filename=str(REPO_ROOT / 'pyproject.toml')) == 0


def test_old_analysis_package_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('smi.analysis')


def test_paths_exist():
    # Checkpoints are gitignored and created on first training run.
    for path in (CONFIGS_DIR, RESULTS_DIR, DATA_DIR, FROZEN_DIR):
        assert path.is_dir(), path
