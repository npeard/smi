"""Q3 stage S3m: matched training length for the leave-one-out's headline row.

Removing phase drift cut the nominal error ~44% (q3_s3_leave_one_out), but that
cell trained ~64 epochs against ~35 for the nominal point, and the noise-free
base itself gains 36-49% from longer training. Here the nominal point, the
nominal point without phase drift, and the noise-free base all train exactly
60 epochs, so the comparison is at equal length; so does the ladder's
phase-drift sweep.
Run: python -m smi.experiments.runner q3_s3_matched_length
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
from smi.experiments.grids.q3_s3_noise_ladder import PHASE_DRIFT_SWEEP

GRID = Grid(
    families=tcn_44k('tcn-44k-E60'),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=(
        Sim('nominal', 'nominal', feedback_c=0.2, alpha=3.0),
        Sim(
            'nominal-no-phasedrift',
            'nominal',
            feedback_c=0.2,
            alpha=3.0,
            override={'phase_drift_rad_per_sqrt_s': 0.0},
        ),
        Sim('ladder-base', 'noise_free', feedback_c=0.2, alpha=3.0),
        # The ladder's phase-drift rows trained 36-67 epochs; at equal length
        # the curve says what a partial drift cut buys.
        Sim(
            'ladder-phasedrift',
            'noise_free',
            feedback_c=0.2,
            alpha=3.0,
            sweep=PHASE_DRIFT_SWEEP,
        ),
    ),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    training={
        **MPS,
        # Exactly 60 epochs, the bottom of the second 30-epoch cosine cycle:
        # patience longer than the cap disables plateau stopping, so every
        # cell trains the same length and length cannot explain a difference.
        'max_epochs': 60,
        'convergence': {
            # The runner derives the monitor from the loss.
            'monitor': 'val/velocity_rmse_um_per_ms',
            'patience_epochs': 1000,
            'min_rel_delta': 0.01,
            'min_epochs': 30,
        },
    },
    test_set=TEST_SET,
)
