"""Q1's binned errors: fringe-extremum distance and per-bin pooled RMSE."""

import math

import numpy as np
import pytest
import torch

from smi.experiments.q1_binned import (
    binned_rmse,
    extremum_distance,
    fringe_phase,
    nearest_extremum_distances,
)
from smi.theory.forward import fringe


@pytest.mark.parametrize(
    ('phase', 'fringes'),
    [
        (0.0, 0.0),
        (math.pi, 0.0),
        (-3 * math.pi, 0.0),
        (math.pi / 2, 0.25),
        (-math.pi / 2, 0.25),
        (2 * math.pi + 0.1 * 2 * math.pi, 0.1),
        (math.pi - 0.05 * 2 * math.pi, 0.05),
    ],
)
def test_extremum_distance_is_fraction_of_a_fringe(phase, fringes):
    assert extremum_distance(np.array([phase]))[0] == pytest.approx(fringes, abs=1e-12)


@pytest.mark.parametrize(('c', 'alpha'), [(0.0, 0.0), (0.2, 3.0), (0.7, 5.0)])
def test_fringe_phase_reproduces_the_simulated_fringe(c, alpha):
    # cos(phi_F) is the fringe the simulator draws, so its extrema are the
    # fringe's extrema, not the pre-feedback phase's.
    theta = torch.linspace(-40.0, 40.0, 4001, dtype=torch.float64)
    phi_f = fringe_phase(theta.numpy(), c, alpha)
    np.testing.assert_allclose(
        np.cos(phi_f), fringe(theta, c, alpha).numpy(), atol=1e-9
    )


def test_any_and_all_take_the_nearest_and_farthest_channel():
    quarter = math.pi / 2
    phase = np.array([[[0.0, quarter, 0.2 * quarter], [quarter, quarter, 0.0]]])
    distances = nearest_extremum_distances(phase, (0.635, 0.515))
    np.testing.assert_allclose(distances['635'], [[0.0, 0.25, 0.05]])
    np.testing.assert_allclose(distances['515'], [[0.25, 0.25, 0.0]])
    np.testing.assert_allclose(distances['any'], [[0.0, 0.25, 0.0]])
    np.testing.assert_allclose(distances['all'], [[0.25, 0.25, 0.05]])


def test_single_channel_has_no_any_or_all():
    distances = nearest_extremum_distances(np.zeros((2, 1, 5)), (0.515,))
    assert set(distances) == {'515'}


def test_binned_rmse_of_a_uniform_ramp():
    # Error equal to the binned value on a uniform ramp: the pooled RMSE of
    # bin [a, b] is sqrt((a^2 + a b + b^2) / 3).
    x = np.linspace(0.0, 1.0, 400_001)
    edges = np.linspace(0.0, 1.0, 5)
    rmse, n = binned_rmse(x**2, x, edges)
    a, b = edges[:-1], edges[1:]
    np.testing.assert_allclose(rmse, np.sqrt((a**2 + a * b + b**2) / 3), rtol=1e-4)
    assert n.sum() == x.size


def test_binned_rmse_rises_toward_the_extremum_only_where_it_should():
    # A single-channel-like error that grows toward the extremum, and a flat
    # one: the binning must recover both shapes.
    rng = np.random.default_rng(0)
    d = rng.uniform(0.0, 0.25, 200_000)
    edges = np.linspace(0.0, 0.25, 6)
    rising, _ = binned_rmse((0.25 - d) ** 2, d, edges)
    flat, _ = binned_rmse(np.full_like(d, 0.01), d, edges)
    assert np.all(np.diff(rising) < 0)
    np.testing.assert_allclose(flat, 0.1)


def test_binned_rmse_puts_the_top_edge_in_the_last_bin():
    rmse, n = binned_rmse(
        np.array([4.0, 9.0]), np.array([0.0, 1.0]), np.array([0, 0.5, 1])
    )
    np.testing.assert_array_equal(n, [1, 1])
    np.testing.assert_allclose(rmse, [2.0, 3.0])
