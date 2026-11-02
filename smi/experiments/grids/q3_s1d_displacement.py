"""Q3/Q2 stage S1d: the stalled families retrained on displacement.

The families that stalled at the zero-output plateau under the velocity target
(LSTM, LSTM-bidir on both sims; offline neural operator at C = 0.2), retrained
on displacement, plus a TCN control that tells whether retargeting costs a
family that already trains. Same sims, channels, budget and frozen test sets
as q3_s1_families, so each cell pairs with its S1 velocity-target cell on
identical shots. Decision metrics are fixed in the round-2 rerun plan, section
(b).
Run: pixi run experiments q3_s1d_displacement
"""

from dataclasses import replace

from smi.experiments.grid import Family
from smi.experiments.grids.q3_s1_families import GRID as S1

GRID = replace(
    S1,
    families=tuple(
        Family(f'{stem}-disp-T90', f'displacement/{stem}')
        for stem in (
            'lstm-44k',
            'lstm-bidir-44k',
            'neuraloperator-offline-44k',
            'tcn-44k',
        )
    ),
    oracle=False,
)
