"""Q3 fiber degradation: TCAN-44k as each fiber impairment turns on alone.

q3_fiber's one point (8 modes, speckle drift 3 rad/s^0.5, path drift
2.8 um/s^0.5, all at once) left every arm at the zero predictor, while the
real MMF data shows learning; that point cannot say which impairment costs
what. Here each knob is swept from zero with the others off:

- ``modes-N``: a static fiber of N = 1, 2, 4, 8 spatial modes (coupling
  fading, E|kappa|^2 ~ 2 / (2N + 1)); no drift.
- ``speckle-r``: speckle drift r rad/s^0.5 on the N = 1 fiber (the least
  fading one, so the drift is not confounded with coupling loss).
- ``path-r``: common path drift r um/s^0.5 on the N = 1 static fiber.

Plus ``nominal`` (free space, the zero point; q2_r4_noise's cell, same
fingerprint) and ``fiber`` (q3_fiber's combined point, same fingerprint), both
already recorded, so resume skips them. The drift ladders step by ~3x up to
q3_fiber's values; the 0.1 rungs bracket the onset that
fiber_sweep_analytics.json predicts. Every sim is scored on its own frozen set
(seed 1000, 64 shots) whose motion is drawn identically, so the points compare
shot for shot (tests/test_experiments.py pins this). 12 new cells at q3_fiber's
~20-27 min TCAN-44k wall times: ~5 h, ~6-7 h under MPS contention.
Run: pixi run experiments q3_fiber_sweep
"""

import dataclasses

from smi.experiments.grids.q3_fiber import FIBER
from smi.experiments.grids.q3_fiber import GRID as Q3_FIBER
from smi.theory.fiber import FiberConfig

MODES = (1, 2, 4, 8)
SPECKLE_RAD_PER_SQRT_S = (0.1, 0.3, 1.0, 3.0)
PATH_UM_PER_SQRT_S = (0.1, 0.3, 1.0, 2.8)

#: The zero point of every fiber knob: one mode, nothing drifting.
STATIC_SINGLE_MODE = dataclasses.replace(
    FIBER, n_modes=1, drift_rad_per_sqrt_s=0.0, path_drift_um_per_sqrt_s=0.0
)


def sweep_fibers() -> dict[str, FiberConfig]:
    """Arm name -> fiber of every one-knob point."""
    base = STATIC_SINGLE_MODE
    return {
        **{f'modes-{n}': dataclasses.replace(base, n_modes=n) for n in MODES},
        **{
            f'speckle-{r:g}': dataclasses.replace(base, drift_rad_per_sqrt_s=r)
            for r in SPECKLE_RAD_PER_SQRT_S
        },
        **{
            f'path-{r:g}': dataclasses.replace(base, path_drift_um_per_sqrt_s=r)
            for r in PATH_UM_PER_SQRT_S
        },
    }


_NOMINAL, _COMBINED = (s for s in Q3_FIBER.sims if s.name in ('nominal', 'fiber'))

GRID = dataclasses.replace(
    Q3_FIBER,
    families=Q3_FIBER.families[:1],
    sims=(
        _NOMINAL,
        _COMBINED,
        *(
            dataclasses.replace(
                _COMBINED, name=name, override={**_COMBINED.override, 'fiber': fib}
            )
            for name, fib in sweep_fibers().items()
        ),
    ),
)
