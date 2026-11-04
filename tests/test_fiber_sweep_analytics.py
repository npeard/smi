"""coupling_stats: the fiber coupling's fringe scale and drift, by knob."""

import dataclasses
import math

import pytest

from smi.experiments.diagnostics import coupling_stats
from smi.experiments.grids.q3_fiber_sweep import GRID, STATIC_SINGLE_MODE
from smi.experiments.runner import expand_grid

_SIM = next(c.sim for c in expand_grid(GRID) if c.sim_name == 'modes-1')


def _stats(**knobs):
    fib = dataclasses.replace(STATIC_SINGLE_MODE, **knobs)
    return coupling_stats(dataclasses.replace(_SIM, fiber=fib), 64, 0)


def test_static_fiber_has_no_in_shot_fade_or_phase():
    s = _stats()
    assert s['in_shot_fade_rel'] == pytest.approx(0, abs=1e-5)
    assert s['speckle_phase_rms_um'] == pytest.approx(0, abs=1e-6)
    assert s['snr_loss_db'] == pytest.approx(-10 * math.log10(s['mean_kappa2']))


def test_more_modes_lose_more_coupling_near_coe():
    # COE 2 / (2N + 1) at N = 8; fiber.py documents a ~3% departure.
    assert _stats(n_modes=8)['mean_kappa2'] == pytest.approx(2 / 17, rel=0.25)


def test_speckle_drift_moves_the_coupling_phase_monotonically():
    slow, fast = (
        _stats(drift_rad_per_sqrt_s=r)['speckle_phase_rms_um'] for r in (0.1, 3)
    )
    assert 0 < slow < fast
