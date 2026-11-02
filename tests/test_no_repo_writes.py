"""The session guard in conftest sees writes and tests do not make them."""

from smi.core.paths import RUNS_DIR
from smi.learning.training_interface import TrainingInterface
from tests import _repo_outputs


def test_guard_sees_a_new_file(tmp_path, monkeypatch):
    monkeypatch.setattr(_repo_outputs, 'WATCHED_DIRS', (tmp_path,))
    before = _repo_outputs.repo_output_files()
    (tmp_path / 'a.ckpt').write_text('x')
    assert _repo_outputs.repo_output_files() - before == {str(tmp_path / 'a.ckpt')}


def test_training_checkpoints_are_redirected_out_of_the_run_tree(tmp_path):
    assert tmp_path == TrainingInterface.CHECKPOINT_DIR
    assert RUNS_DIR not in tmp_path.parents
