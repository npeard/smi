"""Q1: which photodiode channels carry the information.

Every non-empty subset of the three rig lasers, at the nominal weak-feedback
SMI point, noise-free (ceiling) and with the nominal noise model.
Run: python -m smi.experiments.runner q1_channels
Every existing family at the 44k budget (Task 19); later families extend this.
"""

from smi.experiments.grid import (
    ALL_SUBSETS,
    FAMILY_BUDGET,
    MPS,
    STEMS_44K,
    TEST_SET,
    Family,
    Grid,
    Sim,
)

GRID = Grid(
    families=tuple(Family(stem, f'matched/{stem}') for stem in STEMS_44K),
    channel_subsets=ALL_SUBSETS,
    sims=(
        Sim('noise-free', 'noise_free', feedback_c=0.2, alpha=3.0),
        Sim('nominal', 'nominal', feedback_c=0.2, alpha=3.0),
    ),
    seeds=(0, 1, 2),
    oracle=True,
    sign_invariant=False,
    # Each family's own schedule (30 epochs, T_0 = 30), not S1's T_0 = 90.
    synthetic=FAMILY_BUDGET,
    training=MPS,
    test_set=TEST_SET,
)
