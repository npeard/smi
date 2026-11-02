"""Q3 stage S0c-long: one 270-epoch run (T_0 = 270, 400-epoch cap).

The C = 0.2 cell of q3_s0c_ceiling (TCN-44k, all three channels, noise-free,
alpha = 3), as evidence of how far the fixed-budget "ceiling" still moves past
the adopted 90-epoch schedule.
Run: python -m smi.experiments.runner q3_s0c_long
"""

from smi.experiments.grid import (
    FAMILY_BUDGET,
    MPS,
    RIG_WAVELENGTHS_UM,
    TEST_SET,
    Grid,
    Sim,
    tcn_44k,
)

GRID = Grid(
    families=tcn_44k('tcn-44k-T270'),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=(Sim('smi-a3-feedback_c=0.2', 'noise_free', feedback_c=0.2, alpha=3.0),),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    training={**MPS, 'max_epochs': 400, 'T_0': 270},
    test_set=TEST_SET,
)
