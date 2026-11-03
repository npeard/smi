"""Static channel phase offsets for ``SimConfig.channel_phase_offsets_rad``."""

import math

import numpy as np


def offsets_quadrature(n: int) -> tuple[float, ...]:
    """Known static offsets of an ``n``-copy one-color array.

    2 pi k / N, except N = 2, where that gives (0, pi) -- the first channel
    negated under a cosine, no new information -- so N = 2 takes (0, pi/2).
    """
    step = math.pi / 2 if n == 2 else 2 * math.pi / n
    return tuple(step * k for k in range(n))


def random_offsets(index: int, n_ch: int) -> tuple[float, ...]:
    """Offsets ``(0, u_1, ...)`` of ``phase_random`` configuration ``index``."""
    rest = np.random.default_rng(index).uniform(0, 2 * math.pi, n_ch - 1)
    return (0.0, *map(float, rest))
