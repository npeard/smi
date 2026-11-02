"""Q1 stage S4: whether, and where, multi-channel beats any single channel.

TCN-44k on all 7 subsets of the rig wavelengths, at the nominal operating point
(SMI C = 0.2, alpha = 3, every knob nominal) with the SNR swept. The classical
oracle runs separately (q1_s4_oracle) on the 3-channel and best single-channel
subsets only, because inverse_fit costs ~8.5 min per set.
Run: python -m smi.experiments.runner q1_s4_snr
"""

from smi.experiments.grid import (
    ALL_SUBSETS,
    FAMILY_BUDGET,
    MPS,
    TEST_SET,
    Grid,
    Sim,
    tcn_44k,
)

GRID = Grid(
    families=tcn_44k('tcn-44k'),
    channel_subsets=ALL_SUBSETS,
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
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # The family schedule (T_0 = 30) under a 100-epoch cap, not S1's T_0 = 90.
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
