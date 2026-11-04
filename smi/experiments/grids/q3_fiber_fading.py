"""Q3 H4: whether static fading costs TCAN-44k by decorrelating colours.

In q3_fiber_sweep TCAN-44k keeps 0.12 of its signal at N = 8 static modes,
far less than the C_eff, SNR or sign losses predict (fiber_sweep_mechanism.json).
At spectral_corr_width_um = 0.01 the three colours fade independently, which
would break the cross-colour agreement fast-motion estimation leans on. Arms,
all static (no drift), TCAN-44k seed 0, the sweep's budget and test sets:

- ``modes-8-corr``: N = 8, three colours sharing one speckle realization
  (width 100 um: per-shot |kappa| correlation across colours >= 0.9999).
- ``modes-{1,4,8}-1color``: 0.635 um alone, so a one-colour fading cost is
  measured against its own N = 1 baseline.

H4 predicts ``modes-8-corr`` recovers most of the loss against the sweep's
``modes-8`` and the one-colour retained fraction (8 vs 1) is well above 0.12.
4 cells at ~20-27 min: ~1.5-2 h.
Run: pixi run experiments q3_fiber_fading
"""

import dataclasses

from smi.experiments.grid import Sim
from smi.experiments.grids.q3_fiber_sweep import GRID as SWEEP
from smi.experiments.grids.q3_fiber_sweep import STATIC_SINGLE_MODE
from smi.theory.fiber import FiberConfig

#: Wide enough that every rig colour shares one speckle draw.
CORRELATED_WIDTH_UM = 100.0
ONE_COLOR_UM = (0.635,)
ONE_COLOR_MODES = (1, 4, 8)

_TEMPLATE = next(s for s in SWEEP.sims if s.name == 'modes-8')


def _sim(name: str, fiber: FiberConfig, **extra: object) -> Sim:
    return dataclasses.replace(
        _TEMPLATE, name=name, override={**_TEMPLATE.override, 'fiber': fiber, **extra}
    )


GRID = dataclasses.replace(
    SWEEP,
    sims=(
        _sim(
            'modes-8-corr',
            dataclasses.replace(
                STATIC_SINGLE_MODE,
                n_modes=8,
                spectral_corr_width_um=CORRELATED_WIDTH_UM,
            ),
        ),
        *(
            _sim(
                f'modes-{n}-1color',
                dataclasses.replace(STATIC_SINGLE_MODE, n_modes=n),
                wavelengths_um=ONE_COLOR_UM,
            )
            for n in ONE_COLOR_MODES
        ),
    ),
)
