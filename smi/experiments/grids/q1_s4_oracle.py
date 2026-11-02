"""Q1 stage S4-oracle: the classical inverse_fit decoder on the S4 frozen sets.

For the 3-channel subset and the 515 nm channel (chosen in advance: section 1
shows the shortest wavelength carries the most Fisher information), SNR swept.
inverse_fit costs ~8.5 min per 64-shot set, so only these two subsets.
Run: pixi run experiments q1_s4_oracle
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
    families=tcn_44k('tcn-44k'),
    channel_subsets=((0.515,), RIG_WAVELENGTHS_UM),
    sims=(
        Sim(
            'q1-nominal',
            'nominal',
            feedback_c=0.2,
            alpha=3.0,
            sweep=('snr_db', (0.0, 10.0, 20.0, 30.0, 40.0)),
        ),
    ),
    seeds=(0,),
    oracle=True,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # The family schedule (T_0 = 30) under a 100-epoch cap, not S1's T_0 = 90.
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
