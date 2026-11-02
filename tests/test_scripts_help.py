"""Every CLI in ``scripts/`` parses ``--help`` and exits 0."""

import subprocess
import sys

import pytest

from smi.core.paths import REPO_ROOT

SCRIPTS = sorted((REPO_ROOT / 'scripts').glob('*.py'))


@pytest.mark.parametrize('script', SCRIPTS, ids=lambda p: p.name)
def test_help_exits_zero(script):
    proc = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'usage' in proc.stdout.lower()
