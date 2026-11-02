"""Q3 stage S0b: whether the noise-free "ceiling" is schedule-limited.

The S0 pilot's runs all stopped near epoch 38 -- the bottom of the 30-epoch
cosine cycle plus patience 10 -- so the plateau may be the schedule's, not the
model's. Same TCN-44k cells with a 3x longer cosine period; if RMSE moves by
more than ~10%, every later stage uses the longer schedule.
Run: pixi run experiments q3_s0b_schedule
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
        Sim('smi-a3-feedback_c=0.2', 'noise_free', feedback_c=0.2, alpha=3.0),
    ),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    training={**MPS, 'max_epochs': 150, 'T_0': 90},
    test_set=TEST_SET,
)
