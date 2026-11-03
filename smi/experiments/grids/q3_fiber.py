"""R7 (Q3.1/Q3.2): what a multimode fiber costs, and what recovers it.

TCAN-44k and UNetTransformer-offline-44k (best causal and offline at the
nominal C = 0.2, alpha = 3 rig, q2_r4_noise) under four arms:

- ``nominal``: free space, the control (q2_r4_noise's nominal cell).
- ``fiber``: the fiber alone; fiber_results.json finds the Fisher matrix
  singular here, so the network can only learn the fiber-phase-blind part.
- ``fiber-monitor``: plus a 0.785 um dichroic monitor channel, which sees
  the fiber phase only (0.19 of free-space information analytically).
- ``fiber-4color``: the control for the monitor arm: the same 0.785 um color
  as a 4th channel through the target, not the monitor path, so the monitor's
  benefit is separated from that of one more channel.
- ``fiber-pbs``: plus a P_perp channel per color (2C inputs); the analytic
  corr(P_perp, |kappa|^2) ~ -1.3/sqrt(N) bounds what it can tell the network.

The fiber point is where fiber_results.json has the effect largest but not
saturated: N = 8 modes, 3 rad/s^0.5 speckle drift, 2.8 um/s^0.5 path drift,
no dispersion, 0.01 um spectral width. Each arm is its own sim, so each is
scored on its own frozen test set (seed 1000, 64 shots): the arms differ in
input channels, so no one set could serve them all. The sets are paired all the
same: the motion is drawn identically in every arm, so all five share the
velocity and displacement shot for shot (tests/test_experiments.py pins this).
10 cells (2 families x 5 arms, one seed) at TCAN-44k's ~20 min and
UNetTransformer-offline-44k's ~11 min medians, plus the fiber sim's overhead:
~3-4 h.
Run: pixi run experiments q3_fiber
"""

import dataclasses

from smi.experiments.grid import RIG_OPERATING_POINT, RIG_WAVELENGTHS_UM, Sim
from smi.experiments.grids.q2_r4_noise import GRID as R4
from smi.learning.configs import R8_FIBER
from smi.theory.fiber import FiberConfig

MONITOR_UM = 0.785
# Defined once with the R8 flows, which train on this grid's fiber arm.
FIBER = R8_FIBER

# Both 4-channel arms add the same color, so they differ only in its path.
_FOUR_COLORS = (*RIG_WAVELENGTHS_UM, MONITOR_UM)


def _arm(name: str, fiber: FiberConfig | None, **override: object) -> Sim:
    return Sim(
        name, 'nominal', **RIG_OPERATING_POINT, override={'fiber': fiber, **override}
    )


GRID = dataclasses.replace(
    R4,
    families=R4.families[:2],
    sims=(
        Sim('nominal', 'nominal', **RIG_OPERATING_POINT),
        _arm('fiber', FIBER),
        _arm('fiber-4color', FIBER, wavelengths_um=_FOUR_COLORS),
        _arm(
            'fiber-monitor',
            dataclasses.replace(FIBER, monitor_wavelengths_um=(MONITOR_UM,)),
            wavelengths_um=_FOUR_COLORS,
        ),
        _arm('fiber-pbs', dataclasses.replace(FIBER, pbs_channel=True)),
    ),
)
