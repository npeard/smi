"""Identifiability, Fisher information and Bayesian CRBs for SMI velocity.

Analysis tools (report section 1), float64 on CPU; nothing here belongs on a
training path. The measurement model is the ideal one of ``simulate_batch``:
channel k reads ``V_k F(theta_k) + noise`` with ``theta_k = 4 pi d / lambda_k
+ phi_k``, ``F = forward.fringe`` and white Gaussian noise of std
``forward.noise_std(snr_db)``. The velocity bounds are bounds for the fitted
Gauss-Markov surrogate prior (``motion_prior``), with genie per-channel phases
and visibilities; drifts, DC offset, PD low-pass and ADC are omitted. They are
not a floor for the full simulator: ``sample_trajectory`` is a finite line
spectrum (about 33 in-band lines, per-shot U(0, 1) amplitude), which an estimator
that knows the line structure can exploit beyond the surrogate, and amplitude
or contrast drift raise E[a^2 V^2] above the value used here.

The key derivation steps are kept in the docstrings below.
"""

import functools
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.optimize
import torch

from smi.core.acquisition import AcquisitionConfig
from smi.theory.forward import (
    SimConfig,
    channel_phases,
    drive,
    fringe,
    noise_std,
    sample_trajectory,
    smi_excess_phase,
)

_F64 = torch.float64
_CPU = torch.device('cpu')

# Shots and seed of the trajectory sample the motion prior is fit to (the same
# 256-shot, seed-0 sample NOMINAL_DRIVE_GAIN was calibrated on).
_PRIOR_SHOTS = 256
_PRIOR_SEED = 0
# Phase grid of the mirrored-hypothesis search in sign_information, refined by
# Brent within one grid step. The mismatch has at most a few minima per 2 pi.
_SIGN_PHASE_GRID = 64


def fringe_slope(theta: torch.Tensor, feedback_c: float, alpha: float) -> torch.Tensor:
    """dF/dtheta of ``fringe``: -sin(phi_F) / (1 + C cos(phi_F + atan(alpha))).

    Implicit differentiation of phi_F = theta - C sin(phi_F + atan(alpha)); the
    denominator is >= 1 - C > 0 in the weak-feedback regime.
    """
    if feedback_c == 0.0:
        return -torch.sin(theta)
    delta, _ = smi_excess_phase(theta, feedback_c, alpha)
    phi_f = theta + delta
    return -torch.sin(phi_f) / (1 + feedback_c * torch.cos(phi_f + math.atan(alpha)))


def fisher_single(
    theta: torch.Tensor,
    amp: float,
    vis: float,
    sigma: float,
    wavelength_um: float,
    feedback_c: float,
    alpha: float,
) -> torch.Tensor:
    """Displacement Fisher information of one sample of one channel (1/um^2).

    The sample is ``amp (1 + vis F(theta)) - 1 + N(0, sigma^2)``, so the
    information is ``(amp vis (4 pi / lambda) F'(theta) / sigma)^2``; C = 0 is
    the Michelson ``sin^2(theta)`` form, zero at fringe extrema.
    """
    k = 4 * math.pi / wavelength_um
    return (amp * vis * k * fringe_slope(theta, feedback_c, alpha) / sigma) ** 2


def phase_factor(feedback_c: float, alpha: float) -> float:
    """E over uniform theta of F'(theta)^2; 1/2 for Michelson.

    Closed form [1 - cos(2 beta) (1 - s) / (1 + s)] / (2 s), with
    s = sqrt(1 - C^2), beta = atan(alpha). Steps: substitute theta -> phi_F,
    dtheta = (1 + C cos(phi_F + beta)) dphi_F (monotone for C < 1), giving
    <sin^2(phi_F) / (1 + C cos(phi_F + beta))> over uniform phi_F; with
    u = phi_F + beta, sin^2(u - beta) = (1 - cos 2u cos 2beta - sin 2u
    sin 2beta) / 2, the sin 2u term averages to zero (odd), <1 / (1 + C cos u)>
    = 1/s (Gradshteyn-Ryzhik 3.613) and <cos 2u / (1 + C cos u)> = (1 - s)^2 /
    (s C^2) = (1 - s) / (s (1 + s)). Above 1/2 and rising with C for
    |alpha| > 1, so feedback adds magnitude information at the rig's alpha.
    """
    s = math.sqrt(1 - feedback_c**2)
    cos2b = (1 - alpha**2) / (1 + alpha**2)
    return (1 - cos2b * (1 - s) / (1 + s)) / (2 * s)


def _check_noise(cfg: SimConfig) -> float:
    if cfg.snr_db is None:
        raise ValueError('bounds need a finite noise level; snr_db is None')
    if cfg.shot_noise:
        raise ValueError('bounds assume additive noise; shot_noise must be False')
    return noise_std(cfg.snr_db)


def expected_information(cfg: SimConfig) -> float:
    """Per-sample displacement information summed over channels (1/um^2).

    Expectation over the uniform fringe phase (closed form, ``phase_factor``)
    and V ~ U(visibility_range), at unit amplitude. Channels add because their
    noises are independent, so their log-likelihoods and Fisher information add.
    """
    sigma = _check_noise(cfg)
    lo, hi = cfg.visibility_range
    mean_v2 = (lo * lo + lo * hi + hi * hi) / 3
    factor = phase_factor(cfg.feedback_c, cfg.alpha)
    return sum(
        mean_v2 * factor * (4 * math.pi / lam / sigma) ** 2
        for lam in cfg.wavelengths_um
    )


@dataclass
class IdealShots:
    """Shots of the ideal measurement model, float64 on CPU."""

    displacement: torch.Tensor  # [B, L] um
    velocity: torch.Tensor  # [B, L] um/ms
    theta: torch.Tensor  # [B, K, L] rad: 4 pi d / lambda_k + phi_k
    visibility: torch.Tensor  # [B, K]


def ideal_shots(cfg: SimConfig, n_shots: int, seed: int) -> IdealShots:
    """Trajectories from ``sample_trajectory`` with uniform phases, U visibility."""
    gen = torch.Generator().manual_seed(seed)
    waveform, coil = drive(cfg.acquisition, cfg.start_freq_hz, cfg.end_freq_hz)
    _, velocity, displacement = sample_trajectory(
        n_shots, waveform, coil, cfg.drive_gain, cfg.sample_rate, _CPU, gen
    )
    k = len(cfg.wavelengths_um)
    phase = channel_phases(cfg.channel_phase_offsets_rad, (n_shots, k), _CPU, _F64, gen)
    lo, hi = cfg.visibility_range
    vis = lo + (hi - lo) * torch.rand(n_shots, k, generator=gen, dtype=_F64)
    lam = torch.tensor(cfg.wavelengths_um, dtype=_F64).view(1, -1, 1)
    d = displacement.to(_F64)
    target = (4 * math.pi * d[:, None] / lam) * _sees_target(cfg)
    return IdealShots(d, velocity.to(_F64), target + phase, vis)


def _sees_target(cfg: SimConfig) -> torch.Tensor:
    """[1, K, 1] mask: 0 for each dichroic monitor channel (fiber only), else 1."""
    mask = torch.ones(1, len(cfg.wavelengths_um), 1, dtype=_F64)
    monitors = () if cfg.fiber is None else cfg.fiber.monitor_wavelengths_um
    for lam in monitors:
        mask[0, cfg.wavelengths_um.index(lam)] = 0.0
    return mask


def fisher_matrix(cfg: SimConfig, n_shots: int, seed: int) -> np.ndarray:
    """Mean per-sample Fisher matrix of (velocity, phases), from ``ideal_shots``."""
    return fisher_from_shots(cfg, ideal_shots(cfg, n_shots, seed))


def fisher_from_shots(cfg: SimConfig, shots: IdealShots) -> np.ndarray:
    """Mean per-sample Fisher matrix of (v, phase nuisances) on given shots.

    Model: channel k, sample n reads ``V_k F(theta_kn) + N(0, sigma^2)`` with
    theta_kn = 4 pi (d_n + v t_n) / lambda_k + phase_k, where v is a velocity
    offset (um/ms) entering as a displacement ramp over the shot, t_n in ms
    centred on the shot middle. Gaussian-likelihood Fisher: sum over (k, n) of
    g g^T / sigma^2, averaged over the L samples and the shots (unit amplitude),
    with g_v = V_k F'_kn 4 pi t_n / lambda_k (F' = ``fringe_slope``).

    The nuisances follow ``cfg.channel_phase_offsets_rad``:
    - None (independent paths): one unknown phi_k per channel, matrix
      [K+1, K+1], g_phi_k = V_k F'_kn on channel k only. J is then
      block-arrowhead, so the profiled velocity information
      J_vv - J_vp J_pp^-1 J_pv is exactly the sum of per-channel profiled
      information (R1).
    - Set (shared target): one unknown phi_0, offsets known, matrix [2, 2],
      g_phi0 = V_k F'_kn on every channel. The offsets now matter: they
      change where each channel's fringe slope vanishes and how the
      channels' phi_0 information correlates with v (R2).
    With ``cfg.fiber`` set, a last nuisance column is the fiber path drift
    rate u (um/ms of one-way optical path, a ramp like v): g_u = V_k F'_kn
    t_n ``fiber_phase_scale(lambda_k)``, on every channel, since every color
    crosses the fiber (Q3.1). A dichroic monitor channel has g_v = 0. With
    zero dispersion g_u equals the summed g_v of the target channels, so
    without a monitor J is exactly singular. The fiber coupling |kappa| is
    taken as known and unit (genie), so these are fading-free bounds.
    The velocity entry scales as T^2 with shot length T, so compare at fixed
    length. Local (CRB) only: global sign and fringe-order ambiguities are
    ``sign_information``'s.
    """
    sigma = _check_noise(cfg)
    n_shots, n_ch, length = shots.theta.shape
    t_ms = (torch.arange(length, dtype=_F64) - (length - 1) / 2) * (
        1e3 / cfg.sample_rate
    )
    lam = torch.tensor(cfg.wavelengths_um, dtype=_F64).view(1, -1, 1)
    slope = shots.visibility[..., None] * fringe_slope(
        shots.theta, cfg.feedback_c, cfg.alpha
    )
    if cfg.channel_phase_offsets_rad is None:
        phase_jac = torch.eye(n_ch, dtype=_F64)[None, :, None]
    else:
        phase_jac = torch.ones(1, n_ch, 1, 1, dtype=_F64)
    g_v = slope * (4 * math.pi / lam) * t_ms * _sees_target(cfg)
    columns = [g_v[..., None], slope[..., None] * phase_jac]
    if cfg.fiber is not None:
        scale = [cfg.fiber.fiber_phase_scale(x) for x in cfg.wavelengths_um]
        g_u = slope * torch.tensor(scale, dtype=_F64).view(1, -1, 1) * t_ms
        columns.append(g_u[..., None])
    grads = torch.cat(columns, -1)
    flat = grads.reshape(-1, grads.shape[-1])
    return (flat.T @ flat).numpy() / (n_shots * length * sigma**2)


def _mirror_mismatch(
    target: torch.Tensor,
    mirror: torch.Tensor,
    phi: torch.Tensor | float,
    feedback_c: float,
    alpha: float,
) -> torch.Tensor:
    """Least-squares residual energy of ``target`` [L] by a + b F(mirror + phi).

    One value per phi [G]; offset a is free, gain b >= 0 (``sign_information``).
    """
    shape = fringe(
        mirror + torch.as_tensor(phi, dtype=_F64).reshape(-1, 1), feedback_c, alpha
    )
    g = shape - shape.mean(-1, keepdim=True)
    t = target - target.mean()
    proj = (g @ t).clamp(min=0) ** 2 / (g * g).sum(-1).clamp(min=1e-300)
    return (t * t).sum() - proj


def sign_information(cfg: SimConfig, n_shots: int, seed: int) -> float:
    """Upper bound on the sign evidence (nats) of one full shot, mean over shots.

    Per full shot of the cfg's length L, so it scales as L / sigma^2; the
    per-sample value is this / L. For each shot and channel: the KL
    divergence between the noise law under the true mean ``V F(theta)`` and
    the closest mean ``a + b F(-4 pi d / lambda + phi')`` of the sign-flipped
    hypothesis on the same trajectory, with offset a free, gain b >= 0 (least
    squares) and phase phi' free (grid, then Brent). Channels add. Gain b is
    kept >= 0 because photodiode polarity is a fixed, known property of the
    rig and the learned models see a fixed (not per-shot) normalization, so an
    inverted fringe is not a nuisance an estimator faces. A free b would fit
    b ~ -1, phi' + pi and leave C cos(beta) instead of C sin(beta).

    Zero exactly for C = 0 (phi' = -phi) and alpha = 0 (F even in theta).
    Leading order in C: F = cos(theta) + C sin(theta) sin(theta + beta) +
    O(C^2), so the b = 1, phi' = -phi mirror leaves the residual V C sin(beta)
    sin(2 theta); (a, b, phi') move only along {1, cos, sin}, which is
    orthogonal to it under uniform phase coverage, so to O(C^4)
    I = sum_k L C^2 E[V^2] alpha^2 / (4 sigma^2 (1 + alpha^2)).

    It is the expected log-likelihood ratio against the composite sign-flipped
    hypothesis with the trajectory held fixed, so an UPPER bound on sign
    evidence: freeing the trajectory removes it for K = 1, where the warped
    trajectory theta'' = phi_F + C sin(phi_F - beta) (a fringe-rate warp
    theta'' - theta = -2 C sin(beta) cos(phi_F), i.e. a displacement warp of
    depth C sin(beta) lambda / (2 pi)) gives F(-theta'') = F(theta) exactly.
    Single-channel sign evidence therefore comes only from the motion
    prior's penalty on that warp, not from the fringe shape. For K >= 2 the
    per-channel warps differ (they depend on each channel's phi_F and
    lambda), so no common trajectory mirrors all channels: the evidence is
    real but smaller than this fixed-trajectory value.

    With ``channel_phase_offsets_rad`` set the channels share one phase, so
    the mirror fits ONE phi' with the offsets known. Then even C = 0 carries
    sign evidence when the offsets differ: cos(-x + phi' + o_k) matches
    cos(x + phi_0 + o_k) only if phi' = -phi_0 - 2 o_k for every k at once
    (quadrature detection); equal offsets reduce to the Michelson zero.
    """
    sigma = _check_noise(cfg)
    shots = ideal_shots(cfg, n_shots, seed)
    c, alpha = cfg.feedback_c, cfg.alpha
    n_ch = len(cfg.wavelengths_um)
    lam = torch.tensor(cfg.wavelengths_um, dtype=_F64).view(1, -1, 1)
    target = shots.visibility[..., None] * fringe(shots.theta, c, alpha)
    mirror = -4 * math.pi * shots.displacement[:, None] / lam
    offsets = cfg.channel_phase_offsets_rad
    if offsets is None:
        # Independent paths: each channel's mirror phase is fitted on its own.
        groups = [[k] for k in range(n_ch)]
        known = torch.zeros(n_ch, dtype=_F64)
    else:
        # Shared target: one mirror phase for all channels, offsets known.
        groups = [list(range(n_ch))]
        known = torch.tensor(offsets, dtype=_F64)
    total = 0.0
    for b in range(n_shots):
        for group in groups:
            residual = _best_mirror(
                [target[b, k] for k in group],
                [mirror[b, k] + known[k] for k in group],
                c,
                alpha,
            )
            total += residual / (2 * sigma**2)
    return total / n_shots


def _best_mirror(
    targets: list[torch.Tensor],
    mirrors: list[torch.Tensor],
    feedback_c: float,
    alpha: float,
) -> float:
    """Least residual energy over one phase phi' shared by the channels given."""

    def mismatch(phi: torch.Tensor | float) -> torch.Tensor:
        parts = [
            _mirror_mismatch(t, m, phi, feedback_c, alpha)
            for t, m in zip(targets, mirrors, strict=True)
        ]
        return torch.stack(parts).sum(0)

    grid = torch.arange(_SIGN_PHASE_GRID, dtype=_F64) * (2 * math.pi / _SIGN_PHASE_GRID)
    coarse = mismatch(grid)
    best = float(grid[int(coarse.argmin())])
    step = 2 * math.pi / _SIGN_PHASE_GRID
    fine = scipy.optimize.minimize_scalar(
        lambda p: float(mismatch(p)[0]),
        bounds=(best - step, best + step),
        method='bounded',
        options={'xatol': 1e-10},
    )
    return max(min(float(fine.fun), float(coarse.min())), 0.0)


@dataclass
class StateSpace:
    """Discrete linear-Gaussian motion prior x_{n+1} = F x_n + w, w ~ N(0, Q).

    State x = [d (um), v (um/ms), a (um/ms^2)] at the acquisition sample
    interval. ``psd_rms_db`` and ``in_band_fraction`` report the fit (see
    ``motion_prior``); the ``empirical_*`` fields are the fitted sample's.
    """

    transition: np.ndarray  # [n, n]
    process_cov: np.ndarray  # [n, n]
    initial_cov: np.ndarray  # [n, n]; stationary for motion_prior's output
    displacement_row: np.ndarray  # [n]: d = row @ x
    velocity_row: np.ndarray  # [n]: v = row @ x
    dt_ms: float
    psd_rms_db: float
    in_band_fraction: float
    empirical_velocity_var: float
    empirical_displacement_var: float

    def velocity_var(self) -> float:
        """Velocity variance under ``initial_cov``."""
        return float(self.velocity_row @ self.initial_cov @ self.velocity_row)

    def displacement_var(self) -> float:
        """Displacement variance under ``initial_cov``."""
        return float(self.displacement_row @ self.initial_cov @ self.displacement_row)


def _continuous_model(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(A, B) of d = K / ((s + p) (s^2 + 2 zeta w s + w^2)) white, state [d, v, a].

    One real pole (the slow end of the band; it keeps d stationary) and one
    resonance (the coil's, which the pre-compensation does not flatten on
    the acquisition time base); ``params`` are the logs of K, p, w, zeta.
    """
    gain, p, w, zeta = np.exp(params)
    # Monic denominator (s + p)(s^2 + 2 zeta w s + w^2) in companion form.
    a2, a1, a0 = p + 2 * zeta * w, 2 * zeta * w * p + w * w, p * w * w
    a_mat = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-a0, -a1, -a2]])
    return a_mat, np.array([0.0, 0.0, gain])


def _velocity_psd(params: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Two-sided velocity PSD (um/ms)^2 ms at angular frequencies omega (rad/ms)."""
    gain, p, w, zeta = np.exp(params)
    s = 1j * omega
    den = (s + p) * (s * s + 2 * zeta * w * s + w * w)
    return (omega * gain / np.abs(den)) ** 2


def _stationary_cov(params: np.ndarray) -> np.ndarray:
    a_mat, b = _continuous_model(params)
    return scipy.linalg.solve_continuous_lyapunov(a_mat, -np.outer(b, b))


@functools.cache
def _fit_prior(
    acquisition: AcquisitionConfig,
    drive_gain: float,
    start_hz: float,
    end_hz: float,
    sample_rate: float,
) -> StateSpace:
    cfg_drive = drive(acquisition, start_hz, end_hz)
    gen = torch.Generator().manual_seed(_PRIOR_SEED)
    _, v, d = sample_trajectory(
        _PRIOR_SHOTS, *cfg_drive, drive_gain, sample_rate, _CPU, gen
    )
    v, d = v.to(_F64).numpy(), d.to(_F64).numpy()
    n = v.shape[-1]
    fs_khz = sample_rate / 1000
    # One-sided variance per DFT bin; the Waveform's lines sit exactly on these
    # bins (its grid is 1 / shot length), so each bin is one drive line.
    line_var = (np.abs(np.fft.rfft(v, axis=-1)) ** 2).mean(0) * 2 / n**2
    freq = np.fft.rfftfreq(n, 1 / fs_khz)
    band = (freq >= start_hz / 1000) & (freq <= end_hz / 1000) & (line_var > 0)
    omega = 2 * math.pi * freq[band]
    # A stationary PSD S puts 2 S(f) df of one-sided variance in a bin of width df.
    target = np.log10(line_var[band] / (2 * fs_khz / n))
    var_d = float(d.var(axis=-1).mean())
    var_v = float(v.var(axis=-1).mean())

    def residual(params: np.ndarray) -> np.ndarray:
        model = np.log10(_velocity_psd(params, omega))
        # The lines say nothing below the lowest one, so the slow pole is set by
        # the displacement variance; and the resonance is narrower than the line
        # spacing, so line-centre samples alone misplace its area. Both
        # variances are matched in the same log10 units, weighted as heavily as
        # all lines together.
        cov = _stationary_cov(params)
        var_terms = np.log10([cov[0, 0] / var_d, cov[1, 1] / var_v])
        return np.append(model - target, len(omega) * var_terms)

    peak = omega[np.argmax(target)]
    start = np.log([10 ** (target.max() / 2) * peak**2, omega[0] / 4, peak, 0.05])
    fit = scipy.optimize.least_squares(residual, start)
    params = fit.x
    rms_db = float(10 * np.sqrt(np.mean(residual(params)[:-2] ** 2)))

    a_mat, b = _continuous_model(params)
    p_inf = _stationary_cov(params)
    edge = np.linspace(start_hz / 1000, end_hz / 1000, 20001) * 2 * math.pi
    in_band = 2 * np.trapezoid(_velocity_psd(params, edge), edge) / (2 * math.pi)

    # Exact zero-order discretization (Van Loan): F = e^{A dt}, Q = int e^{As} B
    # B^T e^{A^T s} ds over one sample.
    dt = 1 / fs_khz
    k = len(b)
    van_loan = np.zeros((2 * k, 2 * k))
    van_loan[:k, :k] = -a_mat
    van_loan[:k, k:] = np.outer(b, b)
    van_loan[k:, k:] = a_mat.T
    e = scipy.linalg.expm(van_loan * dt)
    f = e[k:, k:].T
    q = f @ e[:k, k:]
    return StateSpace(
        transition=f,
        process_cov=0.5 * (q + q.T),
        initial_cov=p_inf,
        displacement_row=np.array([1.0, 0.0, 0.0]),
        velocity_row=np.array([0.0, 1.0, 0.0]),
        dt_ms=dt,
        psd_rms_db=rms_db,
        in_band_fraction=float(in_band / p_inf[1, 1]),
        empirical_velocity_var=var_v,
        empirical_displacement_var=var_d,
    )


def motion_prior(cfg: SimConfig) -> StateSpace:
    """Linear-Gaussian prior fit to the PSD of ``sample_trajectory`` at ``cfg``.

    The in-band velocity line powers of 256 seeded shots are matched in log10
    by a third-order model (one real pole, one resonance), whose slow pole
    also matches the displacement variance. The per-shot U(0, 1) amplitude draw
    makes the true motion a Gaussian scale mixture; the prior keeps only its
    mean PSD, so bounds under it are bounds for that second-order surrogate.
    """
    return _fit_prior(
        cfg.acquisition,
        cfg.drive_gain,
        cfg.start_freq_hz,
        cfg.end_freq_hz,
        cfg.sample_rate,
    )


def _riccati(prior: StateSpace, info: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filtered and predicted covariances for a scalar d-measurement of info[n].

    The PCRB recursion's linear-Gaussian-dynamics form: J_n^{-1} = P_n with
    P_pred = F P F^T + Q and P^{-1} = P_pred^{-1} + info c_d c_d^T, done in
    covariance form so neither Q nor P needs inverting (Q is nearly singular
    at this sample rate) and info = 0 (a blind sample) is exact.
    """
    f, q, cd = prior.transition, prior.process_cov, prior.displacement_row
    n_steps, k = len(info), len(cd)
    filt, pred = np.empty((n_steps, k, k)), np.empty((n_steps, k, k))
    p = prior.initial_cov
    for t in range(n_steps):
        if t:
            p = f @ p @ f.T + q
        pred[t] = p
        pc = p @ cd
        p = p - np.outer(pc, pc) * (info[t] / (1 + info[t] * (cd @ pc)))
        p = 0.5 * (p + p.T)
        filt[t] = p
    return filt, pred


def information_bounds(
    prior: StateSpace, info: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal (filtering) and offline (smoothing) velocity MSE bounds, (um/ms)^2.

    ``info[n]`` is the displacement information of sample n (1/um^2).

    Causal: the Tichavsky-Muravchik-Nehorai (1998) posterior-CRB recursion
    J_{n+1} = D22 - D21 (J_n + D11)^{-1} D12 with D11 = F^T Q^{-1} F,
    D12 = -F^T Q^{-1}, D22 = Q^{-1} + info c_d c_d^T reduces, by the matrix
    inversion lemma, to J_{n+1} = (Q + F J_n^{-1} F^T)^{-1} + info c_d c_d^T:
    the Kalman covariance recursion of a linear d-measurement (``_riccati``).
    Offline: the Bayesian (Van Trees) information matrix is block-tridiagonal
    (Gauss-Markov prior information plus blockdiag(info c_d c_d^T)); it equals
    the information matrix of that linear-Gaussian surrogate, so its inverse
    diagonal blocks are the surrogate's RTS smoother covariances.

    Both bounds are local and sign-resolved: they see neither the global sign
    nor the fringe order. At C = 0 (or alpha = 0), where the sign is
    unobservable, they stay finite while the signed-MSE floor is the prior
    variance; compare them there with a sign-invariant RMSE.
    """
    filt, pred = _riccati(prior, info.to(_F64).numpy())
    smooth = filt.copy()
    f = prior.transition
    for t in range(len(filt) - 2, -1, -1):
        gain = np.linalg.solve(pred[t + 1], f @ filt[t]).T
        smooth[t] = filt[t] + gain @ (smooth[t + 1] - pred[t + 1]) @ gain.T
    cv = prior.velocity_row
    return (
        torch.from_numpy(np.einsum('i,tij,j->t', cv, filt, cv)),
        torch.from_numpy(np.einsum('i,tij,j->t', cv, smooth, cv)),
    )


def pcrb_filter(prior: StateSpace, cfg: SimConfig, n_steps: int) -> torch.Tensor:
    """Causal Bayesian CRB on velocity, (um/ms)^2, per sample [n_steps].

    Tichavsky posterior-CRB recursion (``information_bounds``; local and
    sign-resolved, see there), with the measurement term averaged over the
    trajectory and uniform phase: theta mod 2 pi is uniform whatever d is, so
    that average is the same for every sample (``expected_information``) and
    exact without Monte Carlo. Averaging J over the genie phase gives
    (E J)^{-1} <= E[J^{-1}] by Jensen, so this is still a valid, slightly
    looser, bound than the phase-conditioned one.
    """
    info = torch.full((n_steps,), expected_information(cfg), dtype=_F64)
    return information_bounds(prior, info)[0]


def bcrb_smoother(prior: StateSpace, cfg: SimConfig, n_steps: int) -> torch.Tensor:
    """Offline Bayesian CRB on velocity, (um/ms)^2, per sample [n_steps].

    Smoothing counterpart of ``pcrb_filter``, same phase-averaged information;
    local and sign-resolved (``information_bounds``).
    """
    info = torch.full((n_steps,), expected_information(cfg), dtype=_F64)
    return information_bounds(prior, info)[1]


def realization_bounds(
    prior: StateSpace, cfg: SimConfig, theta: torch.Tensor, visibility: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal and smoothing velocity bounds along ONE realized phase history.

    ``theta`` [K, L] and ``visibility`` [K] are the channels of
    ``cfg.wavelengths_um`` (pass a subset config for a channel subset). The
    measurement information is evaluated at the realized phases instead of
    averaged over them, so the bound shows where a channel goes blind (fringe
    extrema). It conditions on the phase history, so it is a diagnostic of
    where information is lost, not a Bayesian bound for that one shot.
    """
    sigma = _check_noise(cfg)
    if theta.shape[0] != len(cfg.wavelengths_um):
        raise ValueError('theta needs one row per cfg.wavelengths_um channel')
    info = sum(
        fisher_single(
            theta[k].to(_F64),
            1.0,
            float(visibility[k]),
            sigma,
            lam,
            cfg.feedback_c,
            cfg.alpha,
        )
        for k, lam in enumerate(cfg.wavelengths_um)
    )
    return information_bounds(prior, info)
