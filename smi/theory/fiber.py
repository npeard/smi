"""Multimode-fiber transmission model feeding the self-mixing simulator (R7).

Physics, and the assumptions the Q3.1/Q3.2 analyses rest on:

- The fiber carries N spatial modes x 2 polarizations (M = 2N). Basis index
  ``pol * N + mode``: indices < N are the laser's (x) polarization, mode 0 of
  them is the mode the laser launches into and the only one its cavity
  couples back to.
- One-way transmission ``T(t, lambda) = exp(i 2 pi p(t) n(lambda) / lambda)
  U(t, lambda)``. ``p(t)`` is a common optical-path drift (thermal / strain,
  um), the same for every mode and color; ``U = exp(i H)`` is the speckle
  (mode-mixing) part, H Hermitian.
- Reciprocity: a reciprocal fiber traversed backwards in the same fixed basis
  is ``T^T`` (not ``T^H``), so after an ideal distal mirror the returned field
  is ``r = T^T T e_0``, and the round-trip matrix ``T^T T`` is symmetric. The
  coherent feedback coupling is the projection onto the laser mode,
  ``kappa = (T^T T)_00``: |kappa| <= 1 scales the return-field AMPLITUDE, so
  it multiplies C (and the fringe modulation depth, which is linear in the
  return amplitude for weak feedback); arg(kappa) adds to the round-trip
  phase.
- The PBS after the collimated fiber output sends the whole orthogonal (y)
  polarization to one large-area PD: ``P_perp = sum_{m} |r_{N+m}|^2`` over
  all N modes. Lossless, so ``|kappa|^2 <= 1 - P_perp`` with equality only
  at N = 1, where corr(P_perp, |kappa|^2) is exactly -1. Exact at the speckle
  knots; between them the linear interpolation is renormalized per sample
  onto the same bound.
- Fiber phase vs wavelength: ``4 pi p(t) n(lambda) / lambda`` round trip
  with ``n(lambda) / n_ref = 1 + dispersion_per_um (lambda - 1 um)``; with
  zero dispersion it scales as 1/lambda exactly like the target's
  ``4 pi d / lambda``, which is why colors through fiber and target cannot
  separate the two (Q3.1). The speckle phase arg(U^T U)_00 does not scale
  as 1/lambda but is itself an unknown per-color nuisance.
- Spectral decorrelation: per-color generators are mixtures of independent
  GUE draws with Gaussian correlation ``exp(-(dlambda / w)^2 / 2)``; colors
  many w apart see independent speckle, colors well inside w the same.
- Drift: H random-walks with per-element std ``drift_rad_per_sqrt_s`` per
  sqrt(s), starting from a fully mixed draw (element std ``_MIXED_STD``, so
  the eigenphases wrap several times: a long fiber, not a short one).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

# Element std of the initial generator: eigenvalues spread over ~2 pi sqrt(M),
# a fully mode-mixed fiber of meters, not centimeters. exp(i H) has Haar
# eigenvectors but its wrapped eigenphases lack CUE level repulsion, so small N
# departs from Haar: E|kappa|^2 is 0.73 at N = 1 against the COE 2/3, and
# within 3% of the COE 2 / (2N + 1) from N = 4 (fiber_analytics.py; raising
# this std to 4 pi does not move it).
_MIXED_STD = math.pi
# Speckle is evaluated on this many knots per shot and linearly interpolated:
# exponentiating an M x M generator per sample is too costly, and the mixing
# drift is slow next to the sample rate. The common path phase, which can be
# fast, is applied per sample.
FIBER_KNOTS = 64
# Reference wavelength of the dispersion slope (um).
_DISPERSION_REF_UM = 1.0


@dataclass(frozen=True)
class FiberConfig:
    """A multimode fiber between the lasers and the target, no defaults.

    Attributes:
        n_modes: Spatial modes N (2N field components with polarization).
        drift_rad_per_sqrt_s: Random-walk rate of the speckle generator,
            per element (rad / sqrt(s)).
        path_drift_um_per_sqrt_s: Random-walk rate of the common one-way
            optical path p(t) (um / sqrt(s)); 0 is a static path.
        dispersion_per_um: Relative slope of n(lambda) seen by the path drift
            (1/um), 0 is pure 1/lambda scaling.
        spectral_corr_width_um: Gaussian scale w of the speckle generator
            correlation across wavelength (um).
        monitor_wavelengths_um: The channel wavelengths (possibly none) that
            a dichroic at the distal face returns before the target: those
            channels carry fiber phase only.
        pbs_channel: Append one P_perp input channel per color.
    """

    n_modes: int
    drift_rad_per_sqrt_s: float
    path_drift_um_per_sqrt_s: float
    dispersion_per_um: float
    spectral_corr_width_um: float
    monitor_wavelengths_um: tuple[float, ...]
    pbs_channel: bool

    def __post_init__(self) -> None:
        # JSON/YAML give a list; equality and the hash must not depend on it.
        object.__setattr__(
            self, 'monitor_wavelengths_um', tuple(self.monitor_wavelengths_um)
        )
        if self.n_modes < 1:
            raise ValueError(f'n_modes must be >= 1, got {self.n_modes}')
        if self.spectral_corr_width_um <= 0:
            raise ValueError('spectral_corr_width_um must be > 0')
        if min(self.drift_rad_per_sqrt_s, self.path_drift_um_per_sqrt_s) < 0:
            raise ValueError('drift rates must be >= 0')

    def fiber_phase_scale(self, wavelength_um: float) -> float:
        """d(round-trip fiber phase)/dp in rad/um: 4 pi n(lambda) / lambda."""
        rel = 1 + self.dispersion_per_um * (wavelength_um - _DISPERSION_REF_UM)
        return 4 * math.pi * rel / wavelength_um


@dataclass
class FiberDraw:
    """Per-sample fiber outputs for one batch."""

    kappa: torch.Tensor  # [B, C, L] complex64: (T^T T)_00, path phase included
    p_perp: torch.Tensor  # [B, C, L] float32: returned orthogonal-pol power


def _gue(
    shape: tuple[int, ...],
    m: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Hermitian draws with E|H_ij|^2 = 1, complex64 [..., m, m]."""
    re = torch.randn(*shape, m, m, device=device, generator=generator)
    im = torch.randn(*shape, m, m, device=device, generator=generator)
    a = torch.complex(re, im)
    return (a + a.mH) / 2


def spectral_mixing(wavelengths_um: tuple[float, ...], width_um: float) -> torch.Tensor:
    """Cholesky factor [C, C] of the Gaussian spectral correlation matrix.

    Computed in NumPy float64 (setup only) and returned float32: the 1e-9
    ridge that keeps near-identical colors positive definite is below float32
    resolution next to the unit diagonal, and torch float64 would break the
    no-float64 rule of the torch path (MPS has none).
    """
    lam = np.asarray(wavelengths_um, dtype=np.float64)
    corr = np.exp(-0.5 * ((lam[:, None] - lam[None, :]) / width_um) ** 2)
    corr = corr + 1e-9 * np.eye(len(lam))
    return torch.from_numpy(np.linalg.cholesky(corr).astype(np.float32))


def round_trip(t: torch.Tensor) -> torch.Tensor:
    """Round-trip matrix T^T T of a reciprocal fiber (plain transpose)."""
    return t.transpose(-1, -2) @ t


def speckle_unitaries(
    cfg: FiberConfig,
    wavelengths_um: tuple[float, ...],
    batch: int,
    n_knots: int,
    knot_dt_s: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """U(t_knot, lambda_c) = exp(i H), complex64 [B, C, K, M, M]."""
    m, n_ch = 2 * cfg.n_modes, len(wavelengths_um)
    mix = spectral_mixing(wavelengths_um, cfg.spectral_corr_width_um).to(device)
    gen = _MIXED_STD * _gue((batch, n_ch, 1), m, device, generator)
    if cfg.drift_rad_per_sqrt_s > 0 and n_knots > 1:
        step = cfg.drift_rad_per_sqrt_s * math.sqrt(knot_dt_s)
        walk = step * _gue((batch, n_ch, n_knots - 1), m, device, generator)
        gen = torch.cat([gen, gen + walk.cumsum(2)], 2)
    # Independent sources j mixed into colors c: correlated across color.
    gen = torch.einsum('cj,bjkxy->bckxy', mix.to(gen.dtype), gen)
    u = torch.linalg.matrix_exp((1j * gen).contiguous())
    # A static fiber is exponentiated once and shared by every knot.
    return u.expand(batch, n_ch, n_knots, m, m)


def _interp(knots: torch.Tensor, length: int) -> torch.Tensor:
    """Linear interpolation of [..., K] knots spanning the shot onto L samples."""
    n_knots = knots.shape[-1]
    if n_knots == 1:
        return knots.expand(*knots.shape[:-1], length)
    pos = torch.linspace(0, n_knots - 1, length, device=knots.device)
    lo = pos.floor().long().clamp(max=n_knots - 2)
    frac = (pos - lo).to(knots.dtype)
    return knots[..., lo] * (1 - frac) + knots[..., lo + 1] * frac


def simulate_fiber(
    cfg: FiberConfig,
    wavelengths_um: tuple[float, ...],
    batch: int,
    length: int,
    sample_rate: float,
    device: torch.device,
    generator: torch.Generator | None,
) -> FiberDraw:
    """Draw kappa and P_perp per shot, color and sample.

    Draw order: initial speckle generator, its walk (if drifting), the path
    walk (if drifting); an off knob draws nothing.
    """
    n_knots = min(FIBER_KNOTS, length)
    knot_dt = length / sample_rate / max(n_knots - 1, 1)
    u = speckle_unitaries(
        cfg, wavelengths_um, batch, n_knots, knot_dt, device, generator
    )
    r = round_trip(u)[..., :, 0]  # returned field for launch e_0, [B, C, K, M]
    n = cfg.n_modes
    kappa_knots = r[..., 0]
    perp_knots = (r[..., n:].abs() ** 2).sum(-1)
    kappa = _interp(kappa_knots, length)
    p_perp = _interp(perp_knots, length).clamp(min=0.0)
    # Interpolating between unitary knots can break |kappa|^2 + P_perp <= 1
    # by O(step^2); scale both powers back onto it (a no-op at the knots).
    excess = (kappa.abs() ** 2 + p_perp).clamp(min=1.0)
    kappa, p_perp = kappa / excess.sqrt(), p_perp / excess
    if cfg.path_drift_um_per_sqrt_s > 0:
        sigma = cfg.path_drift_um_per_sqrt_s / math.sqrt(sample_rate)
        path = sigma * torch.randn(
            batch, 1, length, device=device, generator=generator
        ).cumsum(-1)
        scale = torch.tensor(
            [cfg.fiber_phase_scale(lam) for lam in wavelengths_um], device=device
        ).view(1, -1, 1)
        kappa = kappa * torch.polar(torch.ones_like(path), scale * path)
    return FiberDraw(kappa=kappa.to(torch.complex64), p_perp=p_perp.float())
