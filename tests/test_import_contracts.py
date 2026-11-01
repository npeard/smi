"""The package layering is enforced by import-linter, run from the suite."""

import shutil
import subprocess
import sys
from pathlib import Path

from importlinter.cli import lint_imports

from smi.core.paths import CONFIGS_DIR, DATA_DIR, FROZEN_DIR, REPO_ROOT, RESULTS_DIR


def test_contracts_hold():
    assert lint_imports(config_filename=str(REPO_ROOT / 'pyproject.toml')) == 0


def test_paths_exist():
    # Checkpoints are gitignored and created on first training run.
    for path in (CONFIGS_DIR, RESULTS_DIR, DATA_DIR, FROZEN_DIR):
        assert path.is_dir(), path


def _only_python_sources(directory: str, names: list[str]) -> list[str]:
    # import-linter reads only .py files; skip caches and any bulky artifacts.
    return [
        n
        for n in names
        if n == '__pycache__'
        or not (n.endswith('.py') or (Path(directory) / n).is_dir())
    ]


def test_stemlab_contract_catches_a_new_smi_module(tmp_path):
    """A module added anywhere outside smi.acquisition must not reach stemlab.

    Linted on a copy so the live tree is never written to. ``python -c`` puts
    its cwd, the copy, first on sys.path, ahead of the installed smi.
    """
    shutil.copytree(REPO_ROOT / 'smi', tmp_path / 'smi', ignore=_only_python_sources)
    (tmp_path / 'smi' / 'core' / '_stemlab_violation.py').write_text('import stemlab\n')
    shutil.copy(REPO_ROOT / 'pyproject.toml', tmp_path)
    code = (
        'import sys; from importlinter.cli import lint_imports; '
        "sys.exit(lint_imports(config_filename='pyproject.toml', no_cache=True))"
    )
    run = subprocess.run(
        [sys.executable, '-c', code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode != 0, run.stdout
    assert '_stemlab_violation' in run.stdout
