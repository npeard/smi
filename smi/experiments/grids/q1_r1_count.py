"""R1 (Q1.1): marginal gain per added channel, N = 1..6.

The rig's three colors, then extra colors inside the rig's range (0.785,
0.850, 0.405 um), added in that order. Families are the best causal (TCAN-44k)
and best offline (UNetTransformer-offline-44k) of the noise-free C = 0.2 table
(q3_s1_families). Noise-free SMI C = 0.2 and the nominal point; seed 0 (seeds
1-2 at N = 1 and 3 are q1_r1_seeds). Sim and family names match q1_s2_subsets.
Run: pixi run experiments q1_r1_count
"""

from smi.experiments.grid import (
    FAMILY_BUDGET,
    MPS,
    RIG_WAVELENGTHS_UM,
    TEST_SET,
    Family,
    Grid,
    Sim,
)

_COLORS_UM = (*RIG_WAVELENGTHS_UM, 0.785, 0.850, 0.405)
#: The first N colors, N = 1..6.
COUNT_SUBSETS = tuple(_COLORS_UM[:n] for n in range(1, len(_COLORS_UM) + 1))

GRID = Grid(
    families=(
        Family('tcan-44k', 'matched/tcan-44k'),
        Family('unettransformer-offline-44k', 'matched/unettransformer-offline-44k'),
    ),
    channel_subsets=COUNT_SUBSETS,
    sims=(
        Sim('s2-smi-a3', 'noise_free', feedback_c=0.2, alpha=3.0),
        Sim('nominal', 'nominal', feedback_c=0.2, alpha=3.0),
    ),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # The family schedule (T_0 = 30) under a 100-epoch cap, as q1_s2_subsets.
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
