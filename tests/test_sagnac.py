"""Checks of the Sagnac self-mixing signal model and its acceleration Fisher."""

import math

import pytest
import torch

from smi.theory import sagnac

_F64 = torch.float64


def test_two_beam_field_matches_loop_reflection():
    """Summing the CW and CCW fields of a moving off-center element gives
    cos(kappa tau v(t_mid) / 2) times the common phase, exactly for a quadratic x(t).
    """
    kappa, tau, t1 = 4 * math.pi / 1.55, 0.3, 0.1
    x0, v0, acc = 0.2, 1.7, -2.5
    t = torch.linspace(0.0, 3.0, 101, dtype=_F64)

    def x(s):
        return x0 + v0 * s + 0.5 * acc * s**2

    phi_cw, phi_ccw = kappa * x(t - t1), kappa * x(t - t1 - tau)
    field = 0.5 * (torch.exp(1j * phi_cw) + torch.exp(1j * phi_ccw))
    t_mid = t - t1 - 0.5 * tau
    dphi = sagnac.nonreciprocal_phase(kappa, tau, v0 + acc * t_mid)
    expected = sagnac.loop_reflection(dphi) * torch.exp(0.5j * (phi_cw + phi_ccw))
    torch.testing.assert_close(field, expected.to(field.dtype))


@pytest.mark.parametrize(('feedback_c', 'alpha'), [(0.3, 3.0), (0.8, 5.0)])
def test_signal_matches_feedback_equation(feedback_c, alpha):
    """Signal solves phi_F = phi_0 - C c sin(phi_F + atan alpha), y = c cos phi_F."""
    phi0 = torch.linspace(0.0, 2 * math.pi, 257, dtype=_F64)
    dphi = 0.9
    c = math.cos(dphi / 2)
    y = sagnac.sagnac_signal(phi0, dphi, feedback_c, alpha)
    phi_f = torch.arccos(torch.clamp(y / c, -1, 1))
    # cos is not injective; check the residual on both branches, take the best.
    beta = math.atan(alpha)
    res = torch.stack(
        [p - phi0 + feedback_c * c * torch.sin(p + beta) for p in (phi_f, -phi_f)]
    )
    res = torch.remainder(res + math.pi, 2 * math.pi) - math.pi
    assert res.abs().min(dim=0).values.max() < 1e-9


@pytest.mark.parametrize(('feedback_c', 'alpha'), [(0.0, 3.0), (0.3, 3.0), (0.8, 5.0)])
def test_fisher_matches_finite_difference(feedback_c, alpha):
    """Analytic acceleration Fisher equals (dy/da / sigma)^2 by central difference."""
    phi0 = torch.linspace(0.1, 2 * math.pi, 97, dtype=_F64)
    bias, gain, amp, sigma, h = math.pi / 2, 0.7, 0.9, 0.05, 1e-6
    fisher = sagnac.fisher_acceleration(phi0, bias, gain, amp, sigma, feedback_c, alpha)

    def y(acc):
        return amp * sagnac.sagnac_signal(phi0, bias + gain * acc, feedback_c, alpha)

    fd = ((y(h) - y(-h)) / (2 * h) / sigma) ** 2
    torch.testing.assert_close(fisher, fd, rtol=1e-2, atol=1e-6 * fd.max().item())


def test_zero_feedback_is_linear_interferometer():
    """C -> 0: Fisher -> (amp gain sin(dphi/2) cos(phi0) / (2 sigma))^2, gain 1."""
    phi0 = torch.linspace(0.0, 2 * math.pi, 65, dtype=_F64)
    bias, gain, amp, sigma = 1.1, 0.4, 0.8, 0.1
    linear = (amp * gain * math.sin(bias / 2) * torch.cos(phi0) / (2 * sigma)) ** 2
    for c in (0.0, 1e-7):
        got = sagnac.fisher_acceleration(phi0, bias, gain, amp, sigma, c, 3.0)
        torch.testing.assert_close(got, linear, rtol=1e-5, atol=1e-9)
    assert sagnac.feedback_gain(1e-7, 3.0, bias) == pytest.approx(1.0, rel=1e-5)
