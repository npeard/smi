"""Logic checks of the report table scripts (sagnac_table)."""

import importlib.util
import math

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

