"""Q3 stage S3s: run-to-run spread of the noise ladder's budget.

T_0 = 30, one seed per cell. Single-seed ladder rows differ by less than the
spread the S0 C-curve shows at this budget, so the noise-free base (the
ladder's origin) runs at three seeds and the nominal point at two more; the
ladder is read against that spread.
Run: pixi run experiments q3_s3_spread
"""

from smi.experiments.grid import (
    FAMILY_BUDGET,
    MPS,
    RIG_OPERATING_POINT,
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
        Sim('ladder-base', 'noise_free', **RIG_OPERATING_POINT),
        Sim('nominal', 'nominal', **RIG_OPERATING_POINT),
    ),
    seeds=(0, 1, 2),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # The family schedule (T_0 = 30) under a 100-epoch cap, not S1's T_0 = 90.
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
