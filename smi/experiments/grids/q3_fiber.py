"""R7 (Q3.1/Q3.2): what a multimode fiber costs, and what recovers it.

TCAN-44k and UNetTransformer-offline-44k (best causal and offline at the
nominal C = 0.2, alpha = 3 rig, q2_r4_noise) under four arms:

- ``nominal``: free space, the control (q2_r4_noise's nominal cell).
- ``fiber``: the fiber alone; fiber_results.json finds the Fisher matrix
  singular here, so the network can only learn the fiber-phase-blind part.
- ``fiber-monitor``: plus a 0.785 um dichroic monitor channel, which sees
  the fiber phase only (0.19 of free-space information analytically).
- ``fiber-pbs``: plus a P_perp channel per color (2C inputs); the analytic
  corr(P_perp, |kappa|^2) ~ -1.3/sqrt(N) bounds what it can tell the network.

The fiber point is where fiber_results.json has the effect largest but not
saturated: N = 8 modes, 3 rad/s^0.5 speckle drift, 2.8 um/s^0.5 path drift,
no dispersion, 0.01 um spectral width. Each arm is its own sim, so each is
scored on its own frozen test set (seed 1000, 64 shots): the arms differ in
input channels, so no one set could serve them all.
Run: pixi run experiments q3_fiber
"""

import dataclasses

from smi.experiments.grid import RIG_WAVELENGTHS_UM, Sim
from smi.experiments.grids.q2_r4_noise import GRID as R4
from smi.theory.fiber import FiberConfig

MONITOR_UM = 0.785
FIBER = FiberConfig(
    n_modes=8,
    drift_rad_per_sqrt_s=3.0,
    path_drift_um_per_sqrt_s=2.8,
    dispersion_per_um=0.0,
    spectral_corr_width_um=0.01,
    monitor_wavelengths_um=(),
    pbs_channel=False,
)


def _arm(name: str, fiber: FiberConfig | None, **override: object) -> Sim:
    return Sim(
        name,
        'nominal',
        feedback_c=0.2,
        alpha=3.0,
        override={'fiber': fiber, **override},
    )


GRID = dataclasses.replace(
    R4,
    families=R4.families[:2],
    sims=(
        Sim('nominal', 'nominal', feedback_c=0.2, alpha=3.0),
        _arm('fiber', FIBER),
        _arm(
            'fiber-monitor',
            dataclasses.replace(FIBER, monitor_wavelengths_um=(MONITOR_UM,)),
            wavelengths_um=(*RIG_WAVELENGTHS_UM, MONITOR_UM),
        ),
        _arm('fiber-pbs', dataclasses.replace(FIBER, pbs_channel=True)),
    ),
)
