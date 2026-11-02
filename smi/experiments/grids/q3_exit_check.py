"""Report section 1.1 demonstration of the Michelson global-sign symmetry.

The same causal TCN-44k on noise-free three-channel data, trained (a) on
Michelson with the plain signed loss (predicted to collapse toward zero), (b)
on Michelson with the sign-invariant loss, and (c) on weak-feedback SMI
(C = 0.2, alpha = 3) with the signed loss. A short fixed budget (20 epochs);
the point is the ordering, not a ceiling.
Run: pixi run experiments q3_exit_check
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
    families=tcn_44k('tcn-44k-E20'),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=(
        Sim(
            'michelson-signed',
            'noise_free',
            feedback_c=0.0,
            alpha=0.0,
            sign_invariant=False,
        ),
        Sim(
            'michelson-signinv',
            'noise_free',
            feedback_c=0.0,
            alpha=0.0,
            sign_invariant=True,
        ),
        Sim('smi-c0.2-signed', 'noise_free', feedback_c=0.2, alpha=3.0),
    ),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    training={**MPS, 'max_epochs': 20, 'T_0': 20},
    test_set=TEST_SET,
)
