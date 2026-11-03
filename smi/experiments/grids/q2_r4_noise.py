"""R4 (Q2.2): the noise ladder over the top three noise-free families.

TCAN-44k, UNetTransformer-offline-44k and PatchTransformer-offline-44k, the
best three at C = 0.2 in q3_s1_families; LightGBM, the better tabular model,
runs on this grid through scripts/run_tabular_synthetic.py. Each knob of
q3_s3_noise_ladder at its NOMINAL value alone, plus the nominal point: the
full ladder (44 sims) at these families' wall times is ~40 h, while the
question R4 asks (which knob costs most, at nominal) needs only these.
Sim names match the ladder's, so the TCN row compares cell for cell.
Run: pixi run experiments q2_r4_noise
"""

import dataclasses

from smi.core.acquisition import STEMLAB_125_14
from smi.experiments.grid import Family, Sim
from smi.experiments.grids.q3_s3_noise_ladder import GRID as LADDER
from smi.theory.forward import SimConfig

_NOMINAL = SimConfig.nominal(
    STEMLAB_125_14, LADDER.channel_subsets[0], feedback_c=0.2, alpha=3.0
)


def _at_nominal(sim: Sim) -> Sim:
    assert sim.sweep is not None
    field, values = sim.sweep
    value = getattr(_NOMINAL, field)
    # The ladder's ADC rung sets its own full scale, so 14 bits is its nominal.
    (point,) = [v for v in values if v == value or (field == 'adc_bits' and v == 14)]
    return dataclasses.replace(sim, sweep=(field, (point,)))


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
        *(_at_nominal(s) for s in LADDER.sims if s.sweep is not None),
        *(s for s in LADDER.sims if s.name == 'nominal'),
    ),
)
