"""R1 seed spread: q1_r1_count's cells at N = 1 and 3, seeds 1-2.

Its own grid because a grid's seeds apply to every subset.
Run: pixi run experiments q1_r1_seeds
"""

import dataclasses

from smi.experiments.grids.q1_r1_count import COUNT_SUBSETS
from smi.experiments.grids.q1_r1_count import GRID as COUNT

GRID = dataclasses.replace(
    COUNT, channel_subsets=(COUNT_SUBSETS[0], COUNT_SUBSETS[2]), seeds=(1, 2)
)
