"""Q3 stage S0c: the headline accuracy-vs-C curve at the 90-epoch schedule.

S0b showed the 30-epoch plateau is schedule-limited, so this adopts the
90-epoch cosine schedule; one 270-epoch run of the C = 0.2 cell (q3_s0c_long)
is evidence of how far the fixed-budget "ceiling" still moves. TCN-44k, all
three channels, noise-free, alpha = 3, with the alpha = 0 and alpha = 5
controls at C = 0.2. Michelson and C = 0.2 at T_0 = 90 already exist from S0b
(same cell keys, skipped).
Run: pixi run experiments q3_s0c_ceiling
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
    families=tcn_44k('tcn-44k-T90'),
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
    # S1 runs the oracle at Michelson and C = 0.2; inverse_fit assumes the
    # cosine fringe.
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    training={**MPS, 'max_epochs': 150, 'T_0': 90},
    test_set=TEST_SET,
)
