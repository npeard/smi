"""Logic checks of the report table scripts (sagnac_table, channel_analytics)."""

import importlib.util
import math

import numpy as np

from smi.core.paths import REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / 'scripts' / f'{name}.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allowed_swing_keeps_effective_feedback_below_one():
    sagnac = _load('sagnac_table')
    assert math.isinf(sagnac.allowed_swing(0.99, math.pi / 2))
    swing = sagnac.allowed_swing(1.3, math.pi / 2)
    assert math.isclose(1.3 * math.cos((math.pi / 2 - swing) / 2), 1.0)


def test_profiled_information_matches_schur_complement():
    channels = _load('channel_analytics')
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4))
    fisher = a @ a.T + 4 * np.eye(4)
    expected = 1 / np.linalg.inv(fisher)[0, 0]
    assert math.isclose(channels.profiled_velocity_information(fisher), expected)
