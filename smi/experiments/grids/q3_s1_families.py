"""Q3/Q2 stage S1: every family at the 44k budget, at two feedback settings.

All three channels, noise-free: Michelson (sign-invariant; the "is the
estimator the limit" test) and weak-feedback SMI at C = 0.2 (alpha = 3); the C
curve itself is the TCN's (q3_s0c_ceiling). Causal families form the primary
column, offline families the secondary.
Run: pixi run experiments q3_s1_families
"""

from smi.experiments.grid import (
    FAMILY_BUDGET,
    MICHELSON,
    MPS,
    RIG_OPERATING_POINT,
    RIG_WAVELENGTHS_UM,
    STEMS_44K,
    TEST_SET,
    Family,
    Grid,
    Sim,
)

GRID = Grid(
    families=tuple(
        Family(f'{stem}-T90', f'matched/{stem}')
        for stem in STEMS_44K
        # mamba-44k runs as its own stage, q3_s1m_mamba (deferred for its scan).
        if stem != 'mamba-44k'
    ),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=(MICHELSON, Sim('smi-a3', 'noise_free', **RIG_OPERATING_POINT)),
    seeds=(0,),
    oracle=True,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # T_0 = 90, S0b: the 30-epoch plateau is schedule-limited.
    training={**MPS, 'max_epochs': 150, 'T_0': 90},
    test_set=TEST_SET,
)
