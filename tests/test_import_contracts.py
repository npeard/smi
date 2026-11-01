"""The package layering is enforced by import-linter, run from the suite."""

from pathlib import Path

from importlinter.cli import lint_imports

REPO = Path(__file__).resolve().parent.parent


def test_contracts_hold():
    assert lint_imports(config_filename=str(REPO / 'pyproject.toml')) == 0
