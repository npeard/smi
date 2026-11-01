"""A stale config ``data_dir`` must fail loudly, not be created."""

import pytest

from smi.core.paths import REPO_ROOT
from smi.learning.training_interface import resolve_data_dir


def test_existing_data_dir_resolves_against_repo_root():
    assert resolve_data_dir('./data') == REPO_ROOT / 'data'


def test_missing_data_dir_raises_and_is_not_created():
    with pytest.raises(FileNotFoundError, match='does not exist'):
        resolve_data_dir('./analysis/data')
    assert not (REPO_ROOT / 'analysis').exists()
