"""The per-frequency Wiener bound behind the q3_fiber diagnosis."""

import numpy as np
import pytest
import torch

from smi.experiments.diagnostics import (
    colour_fading_verdict,
    explained_fraction,
    matched_retention,
    per_shot_excess,
    rescale_to,
    wiener_rmse,
)


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


def test_rescale_to_maps_each_shot_onto_target_stats():
    gen = torch.Generator().manual_seed(0)
    sig = torch.randn(4, 2, 500, generator=gen) * torch.tensor([[[0.1], [5.0]]]) + 3
    mean, std = torch.tensor([1.0, -2.0]), torch.tensor([0.5, 2.0])
    out = rescale_to(sig, mean, std)
    torch.testing.assert_close(out.mean(-1), mean.expand(4, 2), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.std(-1), std.expand(4, 2))


def test_explained_fraction_spans_perfect_to_zero_predictor():
    assert explained_fraction(0.0, 2.0) == 1.0
    assert explained_fraction(2.0, 2.0) == 0.0


def test_matched_retention_interpolates_between_rungs():
    curve = {0.0: 0.0, 0.2: 0.8, 0.4: 1.0}
    assert matched_retention(curve, 0.2, 0.1) == 0.5
    assert matched_retention(curve, 0.2, 0.3) == 0.9 / 0.8


def test_per_shot_excess_separates_speed_from_amplitude():
    speed = np.array([0.1, 0.2, 0.4, 0.8])
    amplitude = np.array([0.5, 0.9, 0.1, 0.5])
    base = np.full(4, 0.1)
    faded = np.sqrt(base**2 + speed**2)  # excess MSE grows with speed only
    out = per_shot_excess(base, faded, speed, amplitude)
    assert out['corr_excess_mse_vs_speed'] > 0.9
    assert abs(out['corr_excess_mse_vs_amplitude']) < 0.5


def test_colour_fading_verdict_reads_recovery_and_one_colour_retention():
    supported = colour_fading_verdict(
        three_m1=0.6, three_m8=0.1, three_m8_corr=0.5, one_m1=0.4, one_m8=0.3
    )
    assert supported['corr_recovery'] == pytest.approx(0.8)
    assert supported['one_colour_retention'] == pytest.approx(0.75)
    assert supported['three_colour_retention'] == pytest.approx(0.1 / 0.6)
    assert supported['verdict'] == 'supported'
    rejected = colour_fading_verdict(
        three_m1=0.6, three_m8=0.1, three_m8_corr=0.12, one_m1=0.4, one_m8=0.05
    )
    assert rejected['verdict'] == 'rejected'
    mixed = colour_fading_verdict(
        three_m1=0.6, three_m8=0.1, three_m8_corr=0.5, one_m1=0.4, one_m8=0.05
    )
    assert mixed['verdict'] == 'mixed'
