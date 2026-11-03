"""Checks of the identifiability, Fisher information and Bayesian CRB tools."""

import dataclasses
import math

import numpy as np
import pytest
import sympy as sp
import torch

from smi.core.acquisition import STEMLAB_125_14
from smi.data.frozen import RIG_WAVELENGTHS_UM
from smi.theory import bounds
from smi.theory.forward import SimConfig, fringe, noise_std, smi_excess_phase


def _cfg(wavelengths_um, feedback_c, alpha):
    return SimConfig.nominal(STEMLAB_125_14, wavelengths_um, feedback_c, alpha)


@pytest.fixture(scope='module')
def prior():
    return bounds.motion_prior(_cfg(RIG_WAVELENGTHS_UM, 0.2, 3.0))


@pytest.mark.parametrize(('feedback_c', 'alpha'), [(0.0, 3.0), (0.2, 3.0), (0.7, 5.0)])
def test_fisher_matches_sympy(feedback_c, alpha):
    """fisher_single equals E[score^2] of the Gaussian likelihood, by sympy."""
    d, y, phi_f, amp, vis, sigma, lam, c, beta, eps = sp.symbols(
        'd y phi_F a V sigma lambda C beta epsilon', real=True
    )
    # phi_F(d) is implicit: phi_F = phi_0(d) - C sin(phi_F + beta), phi_0 = 4 pi d
    # / lambda + phi. Implicit differentiation of that residual gives dphi_F/dd.
    residual = phi_f - 4 * sp.pi * d / lam + c * sp.sin(phi_f + beta)
    dphi_dd = -sp.diff(residual, d) / sp.diff(residual, phi_f)
    mean = amp * (1 + vis * sp.cos(phi_f)) - 1
    loglik = -((y - mean) ** 2) / (2 * sigma**2)
    score = sp.diff(loglik, phi_f) * dphi_dd
    # E over y ~ N(mean, sigma^2): substitute y = mean + sigma eps, E[eps^2] = 1.
    score_sq = sp.expand((score**2).subs(y, mean + sigma * eps))
    info = score_sq.coeff(eps, 2)
    assert sp.simplify(score_sq.coeff(eps, 1)) == 0
    info_fn = sp.lambdify((phi_f, amp, vis, sigma, lam, c, beta), info, 'numpy')

    theta = torch.linspace(-7.0, 7.0, 101, dtype=torch.float64)
    args = (0.9, 0.7, 0.03, 0.635, feedback_c, alpha)
    got = bounds.fisher_single(theta, *args).numpy()
    delta, _ = smi_excess_phase(theta, feedback_c, alpha)
    want = info_fn((theta + delta).numpy(), *args[:-1], math.atan(alpha))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9 * want.max())


@pytest.mark.parametrize(
    ('feedback_c', 'alpha'), [(0.0, 3.0), (0.2, 0.0), (0.2, 3.0), (0.9, 5.0)]
)
def test_expected_information_matches_quadrature(feedback_c, alpha):
    """Closed-form phase average equals the uniform-phase average of fisher_single."""
    cfg = _cfg(RIG_WAVELENGTHS_UM, feedback_c, alpha)
    # Periodic trapezoid rule over one fringe: exact to rounding for this
    # analytic integrand at this many nodes.
    theta = torch.arange(4096, dtype=torch.float64) * (2 * math.pi / 4096)
    lo, hi = cfg.visibility_range
    mean_v2 = (lo * lo + lo * hi + hi * hi) / 3
    sigma = noise_std(cfg.snr_db)
    want = sum(
        bounds.fisher_single(theta, 1.0, 1.0, sigma, lam, feedback_c, alpha).mean()
        * mean_v2
        for lam in RIG_WAVELENGTHS_UM
    )
    assert bounds.expected_information(cfg) == pytest.approx(float(want), rel=1e-10)


def test_k_channel_fi_adds():
    """Autograd FI of the joint K-channel likelihood is the sum of the singles."""
    feedback_c, alpha, sigma = 0.4, 3.0, 0.02
    vis = torch.tensor([0.9, 0.6, 0.75], dtype=torch.float64)
    phase0 = torch.tensor([0.3, 2.0, -1.1], dtype=torch.float64)
    lam = torch.tensor(RIG_WAVELENGTHS_UM, dtype=torch.float64)
    for d_value in (0.0, 0.11, 0.37, 1.9):
        d = torch.tensor(d_value, dtype=torch.float64, requires_grad=True)
        theta = 4 * math.pi * d / lam + phase0
        mean = 1 + vis * fringe(theta, feedback_c, alpha) - 1
        jac = torch.stack(
            [torch.autograd.grad(m, d, retain_graph=True)[0] for m in mean]
        )
        joint = float((jac**2).sum() / sigma**2)
        singles = sum(
            float(
                bounds.fisher_single(
                    theta[k].detach(),
                    1.0,
                    float(vis[k]),
                    sigma,
                    RIG_WAVELENGTHS_UM[k],
                    feedback_c,
                    alpha,
                )
            )
            for k in range(3)
        )
        assert joint == pytest.approx(singles, rel=1e-8)


@pytest.mark.parametrize('n_channels', [2, 3, 5])
def test_spaced_offsets_remove_michelson_blind_spots(n_channels):
    """Equal-wavelength offsets j pi / K sum sin^2 to K / 2 at every phase."""
    amp, vis, sigma, lam = 1.0, 0.8, 0.02, 0.635
    peak = (amp * vis * 4 * math.pi / lam / sigma) ** 2
    theta = torch.tensor([0.0, 0.4, 2.3], dtype=torch.float64)
    total = sum(
        bounds.fisher_single(
            theta + j * math.pi / n_channels, amp, vis, sigma, lam, 0.0, 0.0
        )
        for j in range(n_channels)
    )
    assert torch.allclose(total, torch.full_like(theta, n_channels / 2 * peak))


def test_sign_information_zero_for_michelson():
    for alpha in (0.0, 3.0):
        cfg = _cfg(RIG_WAVELENGTHS_UM[:2], 0.0, alpha)
        assert bounds.sign_information(cfg, 2, 0) < 1e-6


def test_sign_information_zero_for_alpha_zero():
    for feedback_c in (0.2, 0.7):
        cfg = _cfg(RIG_WAVELENGTHS_UM[:1], feedback_c, 0.0)
        assert bounds.sign_information(cfg, 2, 0) < 1e-6


def test_sign_information_increases_with_c():
    values = [
        bounds.sign_information(_cfg(RIG_WAVELENGTHS_UM[:1], c, 3.0), 4, 0)
        for c in (0.05, 0.2, 0.7)
    ]
    assert values[0] > 1.0
    assert values[0] < values[1] < values[2]


@pytest.mark.parametrize('wavelengths_um', [RIG_WAVELENGTHS_UM[:1], RIG_WAVELENGTHS_UM])
@pytest.mark.parametrize('alpha', [3.0, 5.0])
def test_sign_information_matches_leading_order(wavelengths_um, alpha):
    """At small C, sign_information equals its O(C^2) form shot by shot.

    The residual V C sin(beta) sin(2 theta) off the tangent span {1, cos theta,
    sin theta} of (a, b, phi'), per realization: with uniform phase coverage
    that is L C^2 V^2 sin^2(beta) / (4 sigma^2) per channel, but the low-power
    shots of the U(0, 1) mixture span few fringes, so the closed form's
    expectation is only reached on average. Neglected terms are O(C^2)
    relative: measured 2e-4 at C = 0.02, 2e-3 at C = 0.05.
    """
    feedback_c, n_shots = 0.02, 4
    cfg = _cfg(wavelengths_um, feedback_c, alpha)
    shots = bounds.ideal_shots(cfg, n_shots, 0)
    sigma = noise_std(cfg.snr_db)
    sin_beta = math.sin(math.atan(alpha))
    total = 0.0
    for theta, vis in zip(
        shots.theta.flatten(0, 1), shots.visibility.flatten(), strict=True
    ):
        residual = vis * feedback_c * sin_beta * torch.sin(2 * theta)
        tangent = torch.stack([torch.ones_like(theta), theta.cos(), theta.sin()], 1)
        fit = torch.linalg.lstsq(tangent, residual[:, None]).solution[:, 0]
        off = residual - tangent @ fit
        total += float(off @ off) / (2 * sigma**2)
    got = bounds.sign_information(cfg, n_shots, 0)
    assert got == pytest.approx(total / n_shots, rel=1e-3)


@pytest.mark.parametrize(('feedback_c', 'alpha'), [(0.2, 3.0), (0.7, 5.0)])
def test_k1_warped_mirror_reproduces_signal(feedback_c, alpha):
    """Theta'' = phi_F + C sin(phi_F - beta) gives F(-theta'') = F(theta).

    So with the trajectory free, one channel's fringe shape carries no sign
    evidence; sign_information's fixed-trajectory value is an upper bound.
    """
    shots = bounds.ideal_shots(_cfg(RIG_WAVELENGTHS_UM[:1], feedback_c, alpha), 1, 0)
    theta = shots.theta[0, 0]
    delta, _ = smi_excess_phase(theta, feedback_c, alpha)
    phi_f = theta + delta
    warped = phi_f + feedback_c * torch.sin(phi_f - math.atan(alpha))
    torch.testing.assert_close(
        fringe(-warped, feedback_c, alpha),
        fringe(theta, feedback_c, alpha),
        rtol=0,
        atol=1e-12,
    )
    # The warp is a fringe-rate one, of depth 2 C sin(beta) in phase.
    depth = float((warped - theta).abs().max())
    assert depth == pytest.approx(2 * feedback_c * math.sin(math.atan(alpha)), rel=1e-3)


def test_motion_prior_matches_trajectory_psd(prior):
    """The fitted prior reproduces the in-band line powers of sample_trajectory."""
    assert prior.psd_rms_db < 1.0
    assert prior.in_band_fraction > 0.95
    assert prior.velocity_var() == pytest.approx(prior.empirical_velocity_var, rel=0.1)
    assert prior.displacement_var() == pytest.approx(
        prior.empirical_displacement_var, rel=0.1
    )


def _dense_posterior(prior, info, n_steps):
    """Velocity posterior variances by joint Gaussian conditioning, no Riccati.

    Measurement z_n = d_n + e_n, var(e_n) = 1 / info[n]; a sample with
    info[n] = 0 is not measured at all. The joint prior
    covariance of (x_0 .. x_{L-1}) is cov(x_i, x_j) = F^(i-j) P_j for i >= j.
    """
    f, n = prior.transition, prior.transition.shape[0]
    p = [prior.initial_cov]
    for _ in range(n_steps - 1):
        p.append(f @ p[-1] @ f.T + prior.process_cov)
    sig = np.zeros((n_steps * n, n_steps * n))
    for j in range(n_steps):
        block = p[j]
        for i in range(j, n_steps):
            sig[i * n : (i + 1) * n, j * n : (j + 1) * n] = block
            sig[j * n : (j + 1) * n, i * n : (i + 1) * n] = block.T
            block = f @ block
    h = np.kron(np.eye(n_steps), prior.displacement_row[None, :])
    cv = prior.velocity_row

    def velocity_var(rows):
        rows = [r for r in rows if info[r] > 0]
        hs = h[rows]
        gain = np.linalg.solve(hs @ sig @ hs.T + np.diag(1 / info[rows]), hs @ sig)
        post = sig - sig @ hs.T @ gain
        return np.array(
            [
                cv @ post[t * n : (t + 1) * n, t * n : (t + 1) * n] @ cv
                for t in range(n_steps)
            ]
        )

    smooth = velocity_var(list(range(n_steps)))
    causal = np.array([velocity_var(list(range(t + 1)))[t] for t in range(n_steps)])
    return causal, smooth


# Per um^2: a linear d-measurement with 0.022 um noise; and a time-varying one
# with blind samples (info 0), the shape realization_bounds feeds in.
_CONSTANT_INFO = np.full(40, 2.0e3)
_VARYING_INFO = 2.0e3 * np.sin(np.arange(40) * 0.4) ** 2
_VARYING_INFO[[0, 7, 8, 20]] = 0.0


@pytest.mark.parametrize('info', [_CONSTANT_INFO, _VARYING_INFO])
def test_pcrb_equals_kalman_for_linear_measurement(prior, info):
    causal, _ = _dense_posterior(prior, info, 40)
    got, _ = bounds.information_bounds(prior, torch.from_numpy(info))
    np.testing.assert_allclose(got.numpy(), causal, rtol=1e-7)


@pytest.mark.parametrize('info', [_CONSTANT_INFO, _VARYING_INFO])
def test_smoother_matches_dense_information_inverse(prior, info):
    _, smooth = _dense_posterior(prior, info, 40)
    _, got = bounds.information_bounds(prior, torch.from_numpy(info))
    np.testing.assert_allclose(got.numpy(), smooth, rtol=1e-7)


def test_smoother_bound_le_filter_bound(prior):
    cfg = _cfg(RIG_WAVELENGTHS_UM, 0.2, 3.0)
    causal = bounds.pcrb_filter(prior, cfg, 500)
    smooth = bounds.bcrb_smoother(prior, cfg, 500)
    assert torch.all(smooth <= causal * (1 + 1e-9))
    assert smooth[0] < 0.5 * causal[0]
    assert torch.isclose(smooth[-1], causal[-1], rtol=1e-9)


def test_realization_bounds_more_channels_lower(prior):
    cfg = _cfg(RIG_WAVELENGTHS_UM, 0.2, 3.0)
    shots = bounds.ideal_shots(cfg, 1, 0)
    theta, vis = shots.theta[0, :, :2000], shots.visibility[0]
    three = bounds.realization_bounds(prior, cfg, theta, vis)
    one = bounds.realization_bounds(
        prior,
        dataclasses.replace(cfg, wavelengths_um=RIG_WAVELENGTHS_UM[:1]),
        theta[:1],
        vis[:1],
    )
    for many, few in zip(three, one, strict=True):
        assert torch.all(many <= few * (1 + 1e-9))


def _sample_prior(prior, n_shots, n_steps, rng):
    root = _psd_sqrt(prior.process_cov)
    x = rng.standard_normal((n_shots, len(root))) @ _psd_sqrt(prior.initial_cov).T
    out = [x]
    for _ in range(n_steps - 1):
        x = x @ prior.transition.T + rng.standard_normal(x.shape) @ root.T
        out.append(x)
    return np.stack(out, axis=1)  # [B, L, n]


def _psd_sqrt(m):
    w, v = np.linalg.eigh(m)
    return v * np.sqrt(np.clip(w, 0, None))


def _ekf(prior, y, lam, phase0, vis, sigma, feedback_c, alpha):
    """EKF on the SMI fringe with known phases and visibilities (one shot)."""
    f, q, cd = prior.transition, prior.process_cov, prior.displacement_row
    x, p = np.zeros(len(cd)), prior.initial_cov.copy()
    out = []
    for n in range(y.shape[-1]):
        if n:
            x, p = f @ x, f @ p @ f.T + q
        for k, wl in enumerate(lam):
            theta = torch.tensor(
                4 * math.pi * cd @ x / wl + phase0[k], dtype=torch.float64
            )
            pred = vis[k] * float(fringe(theta, feedback_c, alpha))
            slope = vis[k] * float(bounds.fringe_slope(theta, feedback_c, alpha))
            h = slope * 4 * math.pi / wl * cd
            s = h @ p @ h + sigma**2
            gain = p @ h / s
            x = x + gain * (y[k, n] - pred)
            p = p - np.outer(gain, h @ p)
        out.append(x @ prior.velocity_row)
    return np.array(out)


def test_ekf_mse_above_bound(prior):
    """An EKF's empirical velocity MSE is not below the causal bound."""
    cfg = _cfg(RIG_WAVELENGTHS_UM, 0.2, 3.0)
    # Scored after a 50-sample transient: before it the error is dominated by
    # each shot's initial velocity draw, which a few shots sample poorly.
    n_shots, n_steps, skip = 24, 300, 50
    rng = np.random.default_rng(0)
    # Pin the initial displacement to 0.01 um (a linear-Gaussian update of the
    # stationary prior) so the EKF starts on the right fringe; truth and bound
    # share this prior, so the comparison stays a Bayesian one.
    p0 = prior.initial_cov
    cd = prior.displacement_row
    gain = p0 @ cd / (cd @ p0 @ cd + 1e-4)
    local = dataclasses.replace(prior, initial_cov=p0 - np.outer(gain, cd @ p0))
    bound = bounds.pcrb_filter(local, cfg, n_steps).numpy()

    x = _sample_prior(local, n_shots, n_steps, rng)
    sigma = noise_std(cfg.snr_db)
    lam = np.array(RIG_WAVELENGTHS_UM)
    errs = []
    for b in range(n_shots):
        phase0 = rng.uniform(0, 2 * math.pi, 3)
        vis = rng.uniform(cfg.visibility_range[0], cfg.visibility_range[1], 3)
        theta = 4 * math.pi * (x[b] @ cd)[None] / lam[:, None] + phase0[:, None]
        clean = vis[:, None] * fringe(torch.from_numpy(theta), 0.2, 3.0).numpy()
        y = clean + sigma * rng.standard_normal(clean.shape)
        v_hat = _ekf(local, y, RIG_WAVELENGTHS_UM, phase0, vis, sigma, 0.2, 3.0)
        errs.append(np.mean((v_hat - x[b] @ local.velocity_row)[skip:] ** 2))
    errs = np.array(errs)
    mse, se = errs.mean(), errs.std(ddof=1) / math.sqrt(n_shots)
    assert mse >= bound[skip:].mean() - 3 * se
    # Measured 1.16x: a near-efficient EKF; a bound far below every estimator
    # would pass the line above while being useless.
    assert mse < 2 * bound[skip:].mean()


def _velocity_marginal(j):
    """Velocity information with the nuisance phases profiled out (Schur)."""
    return j[0, 0] - j[0, 1:] @ np.linalg.solve(j[1:, 1:], j[1:, 0])


def test_fisher_matrix_additive_independent_channels():
    """Independent phases: 3-channel profiled velocity information equals the
    sum of single-channel matrices computed on the same (sliced) draws.
    """
    cfg = _cfg((0.635, 0.6748, 0.515), 0.2, 3.0)
    shots = bounds.ideal_shots(cfg, 4, 0)
    j = bounds.fisher_from_shots(cfg, shots)
    assert j.shape == (4, 4)
    singles = 0.0
    for k, lam in enumerate(cfg.wavelengths_um):
        one = dataclasses.replace(cfg, wavelengths_um=(lam,))
        sub = dataclasses.replace(
            shots,
            theta=shots.theta[:, k : k + 1],
            visibility=shots.visibility[:, k : k + 1],
        )
        singles += _velocity_marginal(bounds.fisher_from_shots(one, sub))
    assert _velocity_marginal(j) == pytest.approx(singles, rel=1e-6)


def test_fisher_matrix_shared_phase_depends_on_offsets():
    """Shared phase: 2 pi k / 3 offsets make sum_k F'^2 constant at C = 0, so
    v decouples from phi_0 exactly; zero offsets leave them coupled.
    """

    def corr(offsets):
        cfg = dataclasses.replace(
            _cfg((0.635,) * 3, 0.0, 3.0),
            channel_phase_offsets_rad=offsets,
            visibility_range=(1.0, 1.0),
        )
        j = bounds.fisher_matrix(cfg, n_shots=4, seed=0)
        assert j.shape == (2, 2)
        return abs(j[0, 1]) / math.sqrt(j[0, 0] * j[1, 1])

    spread = corr(tuple(2 * math.pi * k / 3 for k in range(3)))
    assert spread < 1e-9
    assert corr((0.0, 0.0, 0.0)) > 1e3 * spread + 1e-4


def test_fisher_matrix_matches_finite_difference():
    """One channel: J equals the Gaussian-likelihood Fisher of the mean signal,
    whose derivatives are taken by central differences of ``fringe``.
    """
    cfg = _cfg((0.635,), 0.2, 3.0)
    sigma = noise_std(cfg.snr_db)
    j = bounds.fisher_matrix(cfg, n_shots=2, seed=3)

    shots = bounds.ideal_shots(cfg, 2, 3)
    length = shots.theta.shape[-1]
    t_ms = (torch.arange(length, dtype=torch.float64) - (length - 1) / 2) * (
        1e3 / cfg.sample_rate
    )
    k = 4 * math.pi / 0.635
    eps = 1e-6

    def mean(dv, dphi):
        theta = shots.theta[:, 0] + k * dv * t_ms + dphi
        return shots.visibility[:, :1] * fringe(theta, cfg.feedback_c, cfg.alpha)

    g_v = (mean(eps, 0) - mean(-eps, 0)) / (2 * eps)
    g_p = (mean(0, eps) - mean(0, -eps)) / (2 * eps)
    grads = torch.stack([g_v, g_p], -1).reshape(-1, 2).numpy()
    fd = grads.T @ grads / (grads.shape[0] * sigma**2)
    np.testing.assert_allclose(j, fd, rtol=1e-2)
