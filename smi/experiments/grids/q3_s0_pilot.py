"""Q3 stage S0 pilot: TCN-44k only, all three channels, one seed.

It sizes the full Q3/Q1 grids (wall_clock_s and epochs_run per cell) and gives
the first accuracy-vs-C curve. Michelson (C = 0) is sign-invariant by
necessity; the alpha = 0 control at C = 0.2 keeps the d -> -d symmetry and so
should look like Michelson under the signed loss, while alpha = 3 and 5 can
show direction.
Run: python -m smi.experiments.runner q3_s0_pilot
"""

from smi.experiments.grid import (
    FAMILY_BUDGET,
    MICHELSON,
    MPS,
    RIG_WAVELENGTHS_UM,
    TEST_SET,
    Grid,
    Sim,
    tcn_44k,
)

GRID = Grid(
    families=tcn_44k('tcn-44k'),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=(
        MICHELSON,
        Sim(
            'smi-a3',
            'noise_free',
            feedback_c=0.2,
            alpha=3.0,
            sweep=('feedback_c', (0.05, 0.1, 0.2, 0.4, 0.7, 0.9)),
        ),
        Sim('smi-a0', 'noise_free', feedback_c=0.2, alpha=0.0),
        Sim('smi-a5', 'noise_free', feedback_c=0.2, alpha=5.0),
    ),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # pilot: let plateau stopping, not the cap, end runs
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
