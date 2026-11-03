"""R4 (Q2.2): the noise ladder over the top three noise-free families.

TCAN-44k, UNetTransformer-offline-44k and PatchTransformer-offline-44k, the
best three at C = 0.2 in q3_s1_families; LightGBM, the better tabular model,
runs on this grid through scripts/run_tabular_synthetic.py. Each knob of
q3_s3_noise_ladder at its NOMINAL value alone, plus the nominal point: the
full ladder (44 sims) at these families' wall times is ~40 h, while the
question R4 asks (which knob costs most, at nominal) needs only these.
Nominal alone cannot show robustness as a knob worsens, so the two knobs that
dominate the TCN ladder get one stressed rung each: phase drift at 116 rad/s^0.5
(5.4 fringes, 3.3x nominal; TCN RMSE 0.080 -> 0.27 across the ladder) and SNR
20 dB (10 dB below nominal; 0.15, vs 0.12 at 30 dB). Amplitude drift is next
(0.16 at 0.5) but a third rung would push the grid past ~8 h. The pdbw knob is
dropped: at its nominal 100 kHz the TCN matches the noise-free base (0.078 vs
0.080), so its three cells pay for half the rungs. 30 cells at TCAN-44k's
~19 min median: ~9 h.
Sim names match the ladder's, so the TCN row compares cell for cell.
Run: pixi run experiments q2_r4_noise
"""

import dataclasses

from smi.core.acquisition import STEMLAB_125_14
from smi.experiments.grid import RIG_OPERATING_POINT, Family, Sim
from smi.experiments.grids.q3_s3_noise_ladder import GRID as LADDER
from smi.theory.forward import SimConfig

_NOMINAL = SimConfig.nominal(
    STEMLAB_125_14, LADDER.channel_subsets[0], **RIG_OPERATING_POINT
)


#: One stressed ladder value per dominant knob (see the module docstring).
STRESSED = {'phase_drift_rad_per_sqrt_s': 116.0723093699996, 'snr_db': 20.0}
_DROPPED = ('pd_bandwidth_hz',)


def nominal_value(field: str, values: tuple[object, ...]) -> object:
    """The ladder value of ``field`` that equals its nominal-rig value."""
    # The ladder's ADC rung sets its own full scale, so 14 bits is its nominal.
    nominal = 14 if field == 'adc_bits' else getattr(_NOMINAL, field)
    (point,) = [v for v in values if v == nominal]
    return point


def _rungs(sim: Sim) -> Sim:
    assert sim.sweep is not None
    field, values = sim.sweep
    stressed = (STRESSED[field],) if field in STRESSED else ()
    points = (nominal_value(field, values), *stressed)
    return dataclasses.replace(sim, sweep=(field, points))


GRID = dataclasses.replace(
    LADDER,
    families=tuple(
        Family(stem, f'matched/{stem}')
        for stem in (
            'tcan-44k',
            'unettransformer-offline-44k',
            'patchtransformer-offline-44k',
        )
    ),
    sims=(
        *(
            _rungs(s)
            for s in LADDER.sims
            if s.sweep is not None and s.sweep[0] not in _DROPPED
        ),
        *(s for s in LADDER.sims if s.name == 'nominal'),
    ),
)
