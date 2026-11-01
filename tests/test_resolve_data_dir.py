"""A stale config ``data_dir`` must fail loudly, not be created."""

import pytest

from smi.core.paths import REPO_ROOT
from smi.learning.training_interface import resolve_data_dir


def test_existing_data_dir_resolves_against_repo_root():
    assert resolve_data_dir('./data') == (REPO_ROOT / 'data').resolve()


def test_missing_data_dir_raises_and_is_not_created():
    with pytest.raises(FileNotFoundError, match='does not exist'):
        resolve_data_dir('./analysis/data')
    assert not (REPO_ROOT / 'analysis').exists()


def test_absolute_data_dir_stays_absolute(tmp_path):
    assert resolve_data_dir(str(tmp_path)) == tmp_path.resolve()


def test_parent_relative_data_dir_keeps_its_parent_part():
    assert resolve_data_dir(f'../{REPO_ROOT.name}') == REPO_ROOT.resolve()
    with pytest.raises(FileNotFoundError) as err:
        resolve_data_dir('../no-such-dir')
    assert str(REPO_ROOT.resolve().parent / 'no-such-dir') in str(err.value)
