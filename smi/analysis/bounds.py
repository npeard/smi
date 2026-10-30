"""Identifiability, Fisher information and Bayesian CRBs for SMI velocity.

Analysis tools (report section 1), float64 on CPU; nothing here belongs on a
training path. The measurement model is the ideal one of ``simulate_batch``:
channel k reads ``V_k F(theta_k) + noise`` with ``theta_k = 4 pi d / lambda_k
+ phi_k``, ``F = forward.fringe`` and white Gaussian noise of std
``forward.noise_std(snr_db)``. The detector nuisances (DC offset, amplitude,
contrast and phase drift, PD low-pass, ADC) are left out, and phi_k and V_k are
known to the bound (a genie), so every bound here is a lower bound for the full
simulator too: each omitted term only adds unknowns or noise.

Derivations, with each step's warrant, are in the Task 12 report; the
docstrings below state results.
"""

import functools
import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.optimize
import torch

from smi.synthetic.forward import (
    SimConfig,
    _drive,
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

    Closed form [1 - cos(2 atan(alpha)) (1 - s) / (1 + s)] / (2 s), with
    s = sqrt(1 - C^2) (report, step 2): above 1/2 and rising with C for
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
    noises are independent (report, step 3).
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
    waveform, coil = _drive(cfg.start_freq_hz, cfg.end_freq_hz)
    _, velocity, displacement = sample_trajectory(
        n_shots, waveform, coil, cfg.drive_gain, cfg.sample_rate, _CPU, gen
    )
    k = len(cfg.wavelengths_um)
    phase = 2 * math.pi * torch.rand(n_shots, k, 1, generator=gen, dtype=_F64)
    lo, hi = cfg.visibility_range
    vis = lo + (hi - lo) * torch.rand(n_shots, k, generator=gen, dtype=_F64)
    lam = torch.tensor(cfg.wavelengths_um, dtype=_F64).view(1, -1, 1)
    d = displacement.to(_F64)
    return IdealShots(d, velocity.to(_F64), 4 * math.pi * d[:, None] / lam + phase, vis)


def _mirror_mismatch(
    target: torch.Tensor,
    mirror: torch.Tensor,
    phi: torch.Tensor | float,
    feedback_c: float,
    alpha: float,
) -> torch.Tensor:
    """Energy of ``target`` [L] off span{1, F(mirror + phi)}, for each phi [G]."""
    shape = fringe(
        mirror + torch.as_tensor(phi, dtype=_F64).reshape(-1, 1), feedback_c, alpha
    )
    g = shape - shape.mean(-1, keepdim=True)
    t = target - target.mean()
    proj = (g @ t) ** 2 / (g * g).sum(-1).clamp(min=1e-300)
    return (t * t).sum() - proj


def sign_information(cfg: SimConfig, n_shots: int, seed: int) -> float:
    """Mean evidence (nats) a shot carries against its own mirror image.

    For each shot and channel, the Kullback-Leibler divergence between the
    noise distribution under the true mean ``V F(theta)`` and the closest
    mean of the mirrored hypothesis d -> -d: ``a + b F(-4 pi d / lambda +
    phi')`` with offset a, gain b and phase phi' free (least squares in a, b;
    grid then Brent in phi'). Channels add (they share the trajectory, each
    has its own nuisances). Zero means p(signals | v) = p(signals | -v) up to
    those nuisances, so no estimator can tell the sign: exactly so for C = 0
    (phi' = -phi) and for alpha = 0 (F is then even in theta). It is the
    expected log-likelihood ratio of a sign test against the exact mirror
    only, so an upper bound on the evidence against every sign-flipped
    alternative.
    """
    sigma = _check_noise(cfg)
    shots = ideal_shots(cfg, n_shots, seed)
    c, alpha = cfg.feedback_c, cfg.alpha
    grid = torch.arange(_SIGN_PHASE_GRID, dtype=_F64) * (2 * math.pi / _SIGN_PHASE_GRID)
    total = 0.0
    for b in range(n_shots):
        for k, lam in enumerate(cfg.wavelengths_um):
            target = shots.visibility[b, k] * fringe(shots.theta[b, k], c, alpha)
            mirror = -4 * math.pi * shots.displacement[b] / lam

            coarse = _mirror_mismatch(target, mirror, grid, c, alpha)
            best = float(grid[int(coarse.argmin())])
            step = 2 * math.pi / _SIGN_PHASE_GRID
            fine = scipy.optimize.minimize_scalar(
                lambda p, t=target, m=mirror: float(
                    _mirror_mismatch(t, m, p, c, alpha)[0]
                ),
                bounds=(best - step, best + step),
                method='bounded',
                options={'xatol': 1e-10},
            )
            total += max(min(float(fine.fun), float(coarse.min())), 0.0) / (
                2 * sigma**2
            )
    return total / n_shots


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
    drive_gain: float, start_hz: float, end_hz: float, sample_rate: float
) -> StateSpace:
    cfg_drive = _drive(start_hz, end_hz)
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
    also matches the displacement variance. The per-shot U(0, 1) power draw
    makes the true motion a Gaussian scale mixture; the prior keeps only its
    mean PSD, so bounds under it are bounds for that second-order surrogate.
    """
    return _fit_prior(
        cfg.drive_gain, cfg.start_freq_hz, cfg.end_freq_hz, cfg.sample_rate
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

    ``info[n]`` is the displacement information of sample n (1/um^2). The
    smoothing bound is the velocity block of the inverse of the block-
    tridiagonal Bayesian information matrix, evaluated by the RTS backward
    pass, which yields exactly those diagonal blocks (report, step 5).
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

    Tichavsky posterior-CRB recursion, with the measurement term averaged over
    the trajectory and uniform phase: that average is the same for every
    sample (``expected_information``), so it is exact without Monte Carlo.
    """
    info = torch.full((n_steps,), expected_information(cfg), dtype=_F64)
    return information_bounds(prior, info)[0]


def bcrb_smoother(prior: StateSpace, cfg: SimConfig, n_steps: int) -> torch.Tensor:
    """Offline Bayesian CRB on velocity, (um/ms)^2, per sample [n_steps]."""
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
