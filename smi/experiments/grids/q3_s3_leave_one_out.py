"""Q3 stage S3l: leave-one-out at the nominal operating point.

Each sim is the nominal point (TCN-44k, 3 channels, SMI C = 0.2, alpha = 3)
with ONE knob returned to its noise-free value (SimConfig.noise_free). The
ladder sweeps knobs alone from the noise-free base, so it cannot say what the
nominal error would be with one term removed -- the question a hardware fix
asks. Read against the nominal point's three-seed spread (q3_s3_spread).
Run: python -m smi.experiments.runner q3_s3_leave_one_out
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

_NOISE_FREE = {
    'snr': {'snr_db': None},
    'visibility': {'visibility_range': (1.0, 1.0)},
    'dc': {'dc_offset_range': (0.0, 0.0)},
    'ampdrift': {'amplitude_drift_rel': 0.0},
    'phasedrift': {'phase_drift_rad_per_sqrt_s': 0.0},
    'contrastdrift': {'contrast_drift_rel': 0.0},
    'pdbw': {'pd_bandwidth_hz': None},
    'adc': {'adc_bits': None},
}

GRID = Grid(
    families=tcn_44k('tcn-44k'),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=tuple(
        Sim(f'nominal-no-{knob}', 'nominal', feedback_c=0.2, alpha=3.0, override=o)
        for knob, o in _NOISE_FREE.items()
    ),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # The family schedule (T_0 = 30) under a 100-epoch cap, not S1's T_0 = 90.
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
