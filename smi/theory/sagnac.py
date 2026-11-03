"""Signal model and acceleration Fisher information of a Sagnac self-mixing loop.

Analytic study for Q4; the derivation, its warrants and references live in the
report (report.typ, section sec-q4), and this module only states the model.
A 2x2 coupler closed on a loop returns to the laser the field
2 i sqrt(rho (1 - rho)) cos(dphi / 2) exp(i Sigma / 2): the nonreciprocal phase
dphi sets only the return amplitude c = cos(dphi / 2), while the return phase
(hence the self-mixing phase phi_0) keeps the reciprocal fiber phase. An
element crossed by the two beams tau apart gives dphi = kappa tau v. With
feedback C_0 at full reflection the laser sees
phi_F = phi_0 - C_0 c sin(phi_F + atan alpha), y = amp c cos(phi_F), and
C_0 c < 1 must hold over the whole dphi swing (weak-feedback regime).
"""

import math

import torch

from smi.theory.bounds import fringe_slope
from smi.theory.forward import fringe, smi_excess_phase

# Points of the uniform phi_0 grid in ``feedback_gains``; the integrand is smooth
# and periodic, so the trapezoid rule converges geometrically.
_PHASE_GRID = 4096


def nonreciprocal_phase(
    kappa: float, tau: float, velocity: torch.Tensor
) -> torch.Tensor:
    """Dphi = kappa tau v: the delayed difference of a moving element's phase."""
    return kappa * tau * velocity


def loop_reflection(dphi: torch.Tensor | float) -> torch.Tensor:
    """Loop-mirror field amplitude cos(dphi / 2), normalized to 1 at dphi = 0."""
    return torch.cos(torch.as_tensor(dphi) / 2)


def _check_bias(dphi: float) -> float:
    if not abs(dphi) < math.pi:
        raise ValueError(f'|dphi| must be < pi so the return is nonzero, got {dphi}')
    return math.cos(dphi / 2)


def sagnac_signal(
    phi0: torch.Tensor, dphi: float, feedback_c: float, alpha: float
) -> torch.Tensor:
    """Y / amp = c cos(phi_F) with feedback C_0 c, c = cos(dphi / 2)."""
    c = _check_bias(dphi)
    return c * fringe(phi0, feedback_c * c, alpha)


def _signal_slope(
    phi0: torch.Tensor, dphi: float, feedback_c: float, alpha: float
) -> torch.Tensor:
    """d(y / amp)/dc = cos phi_F + c C_0 dcos(phi_F)/dC at C = C_0 c.

    dphi_F/dC = -sin(phi_F + beta) / (1 + C cos(phi_F + beta)) by implicit
    differentiation, so dcos(phi_F)/dC = -fringe_slope * sin(phi_F + beta).
    """
    c = _check_bias(dphi)
    cc = feedback_c * c
    if cc == 0.0:
        return torch.cos(phi0)
    delta, _ = smi_excess_phase(phi0, cc, alpha)
    phi_f = phi0 + delta
    dcos_dc = -fringe_slope(phi0, cc, alpha) * torch.sin(phi_f + math.atan(alpha))
    return torch.cos(phi_f) + cc * dcos_dc


def fisher_acceleration(
    phi0: torch.Tensor,
    dphi: float,
    gain: float,
    amp: float,
    sigma: float,
    feedback_c: float,
    alpha: float,
) -> torch.Tensor:
    """Acceleration Fisher information of one sample, with phi_0 known.

    Chain rule through dphi = bias + gain a and dc/d(dphi) = -sin(dphi / 2) / 2:
    I_a = (amp gain sin(dphi / 2) / 2 * d(y/amp)/dc / sigma)^2. Known phi_0 is
    a genie bound: per item 2 of the module docstring, fiber drift moves phi_0.
    """
    slope = _signal_slope(phi0, dphi, feedback_c, alpha)
    return (amp * gain * math.sin(dphi / 2) / 2 * slope / sigma) ** 2


def feedback_gains(feedback_c: float, alpha: float, dphi: float) -> tuple[float, float]:
    """(phase-averaged, best-phase) Fisher at C_0 over C_0 = 0, at fixed amp.

    Averaged: 2 <S'^2> over uniform phi_0 (C_0 = 0 gives <cos^2 phi_0> = 1/2).
    Best: max over phi_0 of S'^2 (1 at C_0 = 0), the genie operating point when
    phi_0 can be held where the slope peaks.
    """
    phi0 = torch.linspace(0, 2 * math.pi, _PHASE_GRID + 1, dtype=torch.float64)[:-1]
    sq = _signal_slope(phi0, dphi, feedback_c, alpha).pow(2)
    return 2 * sq.mean().item(), sq.max().item()
