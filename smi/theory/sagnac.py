"""Signal model and acceleration Fisher information of a Sagnac self-mixing loop.

Analytic study for Q4 (no rig exists). The laser feeds a 2x2 coupler whose two
output ports are joined by a loop of one-way delay T; the loop mirror returns
light to the laser. Results, derived step by step in the functions below:

1. Loop reflection (Mortimore 1988). For coupler power split rho, the CW beam
   takes bar-then-cross and the CCW beam cross-then-bar, so both pick up one
   factor i sqrt(rho (1 - rho)) and the returned field is
   ``2 i sqrt(rho (1 - rho)) cos(dphi / 2) exp(i Sigma / 2)``, with
   Sigma = phi_CW + phi_CCW and dphi = phi_CW - phi_CCW. The nonreciprocal phase
   dphi sets only the return AMPLITUDE; the return PHASE is the reciprocal mean
   Sigma / 2, which carries every reciprocal fiber-phase fluctuation of the loop.
2. Self-mixing reads the round-trip phase phi_0 = omega tau_ext of the return
   (Lang-Kobayashi 1980), which includes Sigma / 2 and the lead fiber. So the
   loop does NOT make the self-mixing phase insensitive to reciprocal fiber
   noise: it cancels reciprocal noise in the loop's reflectivity only. The
   premise of Q4 holds for a conventional (photodetector at the coupler's
   second port) Sagnac readout, not for self-mixing read through the laser.
3. Kinematics. A phase element kappa x(t) a time offset tau off the loop midpoint
   is crossed by the two beams tau apart, so dphi = kappa (x(t - t1) - x(t - t1
   - tau)) = kappa tau v(t_mid), exact for quadratic x (``nonreciprocal_phase``).
   It encodes the element's VELOCITY, scaled by the delay offset. Rotation adds
   8 pi A Omega / (lambda c) (Post 1967). Rigid linear acceleration of the
   whole loop adds no nonreciprocal phase: a uniformly accelerated frame is
   static (Rindler metric, g_0i = 0), so propagation is time-reversal symmetric
   and the CW and CCW phases are equal; only the rotation (g_0i != 0) term
   survives. Its time variation enters through the delayed difference of the
   reciprocal term, of order k L (a L / c^2) (T da/dt / a), below 1e-15 rad for
   a 1 m loop. Acceleration therefore enters only through a proof mass moving
   the element: x = -a / (s^2 + omega_0 s / Q + omega_0^2), so dphi is jerk below
   resonance, acceleration (gain Q / omega_0) at resonance, frame velocity above.
   ``gain`` below is d(dphi)/da, i.e. kappa tau |H_v| for that transfer.
4. Signal. With feedback strength C_0 at full reflection and c = cos(dphi / 2)
   (|dphi| < pi), the return scales C and the modulation depth by c:
   phi_F = phi_0 - C_0 c sin(phi_F + atan alpha), y = amp c cos(phi_F).
"""

import math

import torch

from smi.theory.bounds import fringe_slope
from smi.theory.forward import fringe, smi_excess_phase

# Points of the uniform phi_0 grid in ``feedback_gain``; the integrand is smooth
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


def feedback_gain(feedback_c: float, alpha: float, dphi: float) -> float:
    """Phase-averaged Fisher at C_0 over C_0 = 0, at fixed modulation depth amp.

    Equals 2 <(d(y/amp)/dc)^2> over uniform phi_0, since C_0 = 0 gives
    <cos^2 phi_0> = 1/2.
    """
    phi0 = torch.linspace(0, 2 * math.pi, _PHASE_GRID + 1, dtype=torch.float64)[:-1]
    return 2 * _signal_slope(phi0, dphi, feedback_c, alpha).pow(2).mean().item()
