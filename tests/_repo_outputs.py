"""The directories a test session must leave untouched."""

from smi.core.paths import REPO_ROOT, RUNS_DIR

# `checkpoints/` and `lightning_logs/` are where training wrote before
# `runs/train`; watching them catches a regression to a default directory.
WATCHED_DIRS = (RUNS_DIR, REPO_ROOT / 'checkpoints', REPO_ROOT / 'lightning_logs')


def repo_output_files() -> set[str]:
    """Every file now under the watched directories."""
    return {str(p) for d in WATCHED_DIRS for p in d.rglob('*') if p.is_file()}
