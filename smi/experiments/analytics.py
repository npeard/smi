"""Helpers shared by the analytic report tables and the grids they mirror.

scripts/channel_analytics.py and scripts/fiber_analytics.py assemble the
tables; the rules here are the ones a grid or a test must agree with.
"""

import numpy as np

from smi.theory.bounds import IdealShots


def channels(shots: IdealShots, idx: list[int]) -> IdealShots:
    """``shots`` restricted to channels ``idx``: the same draws, fewer channels."""
    return IdealShots(
        shots.displacement,
        shots.velocity,
        shots.theta[:, idx],
        shots.visibility[:, idx],
    )


def correlation_condition(fisher: np.ndarray) -> float:
    """Condition number of the unit-diagonal (correlation) form: unit-free."""
    d = 1 / np.sqrt(np.diag(fisher))
    return float(np.linalg.cond(fisher * d[:, None] * d[None, :]))
