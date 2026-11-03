"""The per-frequency Wiener bound behind the q3_fiber diagnosis."""

import numpy as np

from smi.experiments.diagnostics import wiener_rmse


def test_single_channel_matches_scalar_wiener():
    s_v, s_e = np.array([4.0, 1.0]), np.array([[[1.0, 3.0]]])
    got = wiener_rmse(s_v, np.zeros(2), s_e, np.array([1]))
    want = np.sqrt(np.mean(s_v * s_e[0, 0] / (s_v + s_e[0, 0])))
    np.testing.assert_allclose(got, [want])


def test_monitor_cancels_common_term():
    # Noise-free monitor of the common term: v is recovered exactly.
    s_v, s_w = np.ones(3), 100 * np.ones(3)
    s_e = np.zeros((1, 2, 3))
    s_e[0, 0] = 1e-12
    s_e[0, 1] = 1e-12
    got = wiener_rmse(s_v, s_w, s_e, np.array([1, 0]))
    np.testing.assert_allclose(got, [0.0], atol=1e-4)


def test_unmonitored_common_term_degrades():
    s_v, s_w, s_e = np.ones(3), 100 * np.ones(3), 1e-6 * np.ones((1, 1, 3))
    got = wiener_rmse(s_v, s_w, s_e, np.array([1]))
    np.testing.assert_allclose(got, [np.sqrt(100 / 101)], rtol=1e-4)
