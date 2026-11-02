"""Q1 stage S2: whether multi-channel beats one channel when nothing masks it.

The nominal-point sweep (q1_s4_snr) is phase-drift limited: every subset sits
near the zero predictor there. Here all 7 subsets run noise-free, Michelson
(sign-invariant) and SMI C = 0.2, alpha = 3, for the TCN and TCAN (the best
causal family at C = 0.2 in S1). The oracle row is run for Michelson only (it
decodes under the Michelson model).
Run: pixi run experiments q1_s2_subsets
"""

from smi.experiments.grid import (
    ALL_SUBSETS,
    FAMILY_BUDGET,
    MPS,
    TEST_SET,
    Family,
    Grid,
    Sim,
)

GRID = Grid(
    families=(
        Family('tcn-44k', 'matched/tcn-44k'),
        Family('tcan-44k', 'matched/tcan-44k'),
    ),
    channel_subsets=ALL_SUBSETS,
    sims=(
        Sim(
            's2-michelson', 'noise_free', feedback_c=0.0, alpha=0.0, sign_invariant=True
        ),
        Sim('s2-smi-a3', 'noise_free', feedback_c=0.2, alpha=3.0),
    ),
    seeds=(0,),
    oracle=True,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # The family schedule (T_0 = 30) under a 100-epoch cap, not S1's T_0 = 90.
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
