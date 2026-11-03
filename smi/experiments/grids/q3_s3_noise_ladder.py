"""Q3 stage S3: the noise ladder.

TCN-44k, all three channels, SMI C = 0.2, alpha = 3. Each detector/drift knob
is swept ALONE from the noise-free base over its NOMINAL_SWEEPS range
(smi/theory/forward.py), then every knob at its nominal value together (the
nominal operating point). Which single knob moves RMSE most, at its nominal
value, is the case for or against new electronics. shot_noise is omitted
alone: it needs an SNR and so is only meaningful inside the nominal point.
drive_gain is motion scale, not noise.
Run: pixi run experiments q3_s3_noise_ladder
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

PHASE_DRIFT_SWEEP = (
    'phase_drift_rad_per_sqrt_s',
    (
        10.74743605277774,
        21.49487210555548,
        35.31967896870391,
        58.0361546849998,
        116.0723093699996,
    ),
)
_SWEEPS = {
    'snr': ('snr_db', (10.0, 20.0, 30.0, 40.0, 50.0)),
    'visibility': (
        'visibility_range',
        ((0.1, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (0.9, 1.0)),
    ),
    'dc': ('dc_offset_range', ((-0.1, 0.1), (-0.3, 0.3), (-1.0, 1.0))),
    'ampdrift': ('amplitude_drift_rel', (0.02, 0.05, 0.1, 0.2, 0.5)),
    'phasedrift': PHASE_DRIFT_SWEEP,
    'contrastdrift': ('contrast_drift_rel', (0.02, 0.05, 0.1, 0.2, 0.5)),
    'pdbw': ('pd_bandwidth_hz', (200000.0, 100000.0, 30000.0, 10000.0, 3000.0)),
}

GRID = Grid(
    families=tcn_44k('tcn-44k'),
    channel_subsets=(RIG_WAVELENGTHS_UM,),
    sims=(
        *(
            Sim(f'ladder-{knob}', 'noise_free', **RIG_OPERATING_POINT, sweep=s)
            for knob, s in _SWEEPS.items()
        ),
        Sim(
            'ladder-adc',
            'noise_free',
            **RIG_OPERATING_POINT,
            override={'adc_full_scale': 2.0},
            sweep=('adc_bits', (14, 12, 10, 8)),
        ),
        Sim('nominal', 'nominal', **RIG_OPERATING_POINT),
        # best C of the noise-free curve (monotone in C up to 0.9, q3_s0c_ceiling)
        Sim('nominal-c0.9', 'nominal', feedback_c=0.9, alpha=3.0),
    ),
    seeds=(0,),
    oracle=False,
    sign_invariant=False,
    synthetic=FAMILY_BUDGET,
    # The family schedule (T_0 = 30) under a 100-epoch cap, not S1's T_0 = 90.
    training={**MPS, 'max_epochs': 100},
    test_set=TEST_SET,
)
