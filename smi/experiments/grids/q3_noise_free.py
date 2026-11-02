"""Q3: noise-free accuracy ceiling vs feedback strength C, on all three channels.

Michelson (C = 0) cannot show direction, so it trains and is scored with the
sign-invariant loss; every C > 0 sim with alpha != 0 can.
Run: python -m smi.experiments.runner q3_noise_free
Every existing family at the 44k budget (Task 19); later families extend this.
"""

from smi.experiments.grid import (
    FAMILY_BUDGET,
    MICHELSON,
    MPS,
    RIG_WAVELENGTHS_UM,
    STEMS_44K,
    TEST_SET,
    Family,
    Grid,
    Sim,
)

GRID = Grid(
    families=tuple(Family(stem, f'matched/{stem}') for stem in STEMS_44K),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=(
        MICHELSON,
        Sim(
            'smi',
            'noise_free',
            feedback_c=0.2,
            alpha=3.0,
            sweep=('feedback_c', (0.05, 0.1, 0.2, 0.4, 0.7, 0.9)),
        ),
    ),
    seeds=(0, 1, 2),
    oracle=True,
    sign_invariant=False,  # default for sims that do not set their own
    # Each family's own schedule (30 epochs, T_0 = 30), not S1's T_0 = 90.
    synthetic=FAMILY_BUDGET,
    training=MPS,
    test_set=TEST_SET,
)
