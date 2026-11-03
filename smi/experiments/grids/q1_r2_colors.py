"""R2 (Q1.2): a one-color array at N = 2 against N distinct colors.

0.635 um repeated N times with known static offsets on one shared mirror
phase: (0, pi/2) at N = 2 and 2 pi k / 3 at N = 3. Under a cosine the naive
2 pi k / N at N = 2 gives (0, pi), the first channel negated -- no quadrature
and no new information -- so N = 2 takes the quadrature pair; N = 3 keeps the
symmetric three-phase set, which already spans both quadratures. Families,
sims and budget are q1_r1_count's, whose N = 2 and 3 cells are the
distinct-color arm. One grid per N (this one N = 2, q1_r2_colors_n3 N = 3):
the offsets' length must match the subset's.
Run: pixi run experiments q1_r2_colors
"""

import dataclasses
import math

from smi.experiments.grid import Grid
from smi.experiments.grids.q1_r1_count import GRID as COUNT


def one_color(n: int) -> Grid:
    """q1_r1_count at 0.635 um x ``n``; offsets as in the module docstring."""
    step = math.pi / 2 if n == 2 else 2 * math.pi / n
    offsets = tuple(step * k for k in range(n))
    return dataclasses.replace(
        COUNT,
        channel_subsets=((0.635,) * n,),
        sims=tuple(
            dataclasses.replace(
                s,
                name=f'{s.name}-1color',
                override={**s.override, 'channel_phase_offsets_rad': offsets},
            )
            for s in COUNT.sims
        ),
    )


GRID = one_color(2)
