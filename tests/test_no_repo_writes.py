"""Default write locations resolve under tmp_path, not the repo."""

from smi.core.paths import REPO_ROOT
from smi.experiments import runner
from smi.learning.training_interface import TrainingInterface


def test_default_write_locations_are_private_to_the_test(tmp_path):
    defaults = [
        TrainingInterface.CHECKPOINT_DIR,
        runner.GRID_RUNS_DIR,
        runner.DEFAULT_OUT,
        runner.DEFAULT_LOCK,
        runner.DEFAULT_ROOT,
    ]
    for path in defaults:
        assert tmp_path in path.parents, path
        assert REPO_ROOT not in path.parents or tmp_path.is_relative_to(REPO_ROOT)
