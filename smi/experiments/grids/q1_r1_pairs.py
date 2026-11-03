"""R1 (Q1.1) at N = 2: the other two rig pairs.

q1_r1_count's N = 2 cell is the first two colors, (0.635, 0.6748) um, the
closest pair and so the smallest synthetic-wavelength spread; one pair cannot
say whether the N = 2 gain is a property of N or of that pair. This grid runs
the remaining rig pairs with q1_r1_count's families, sims, budget and test set.
Run: pixi run experiments q1_r1_pairs
"""

import dataclasses

from smi.experiments.grid import RIG_WAVELENGTHS_UM
from smi.experiments.grids.q1_r1_count import GRID as COUNT

_A, _B, _C = RIG_WAVELENGTHS_UM

GRID = dataclasses.replace(COUNT, channel_subsets=((_A, _C), (_B, _C)))
