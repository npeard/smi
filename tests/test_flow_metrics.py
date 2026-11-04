"""Probabilistic scores of the R8 flow posteriors, on cases with known answers."""

import math

import numpy as np
import pytest
import torch

from smi.core.metrics import band_coverage, crps_gaussian, crps_samples
from smi.experiments.flow_eval import (
    align_signs,
    conditioning_sensitivity,
    magnitude_scores,
    multimodal_fraction,
    sign_structure,
    signed_scores,
)


def test_band_coverage_of_calibrated_draws():
    """Truth and draws from the same N(0, 1): the 90% band covers 0.90."""
    g = torch.Generator().manual_seed(0)
    draws = torch.randn(4000, 8, 500, generator=g)
    truth = torch.randn(8, 500, generator=g)
    assert abs(band_coverage(draws, truth, 0.9) - 0.9) < 0.01


def test_band_coverage_is_nominal_at_64_draws():
    """64 calibrated draws read 0.9, as the ensemble's analytic band does.

    Interpolated sample quantiles read ~0.873 here. 20000 truths put the MC
    error near 0.002 for both.
    """
    g = torch.Generator().manual_seed(5)
    n = 20000
    draws = torch.randn(64, n, generator=g)
    truth = torch.randn(n, generator=g)
    mc = 3 * math.sqrt(0.9 * 0.1 / n)
    assert abs(band_coverage(draws, truth, 0.9) - 0.9) < mc
    # flow_eval.ensemble_scores' band with the true mean and std.
    z = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95))
    analytic = float((truth.abs() <= z).float().mean())
    assert abs(analytic - 0.9) < mc
    assert abs(band_coverage(draws, truth, 0.9) - analytic) < 2 * mc


def test_band_coverage_slot_weights():
    """S = 3, level 0.5: m = 1 slot per tail, so ranks 1 and 2 count 1."""
    draws = torch.tensor([1.0, 2.0, 3.0])[:, None].expand(3, 4)
    truth = torch.tensor([0.5, 1.5, 2.5, 3.5])
    assert band_coverage(draws, truth, 0.5) == 0.5
    # level 0.75: m = 0.5, so the boundary slots 0 and 3 count 1/2.
    assert band_coverage(draws, truth, 0.75) == 0.75


def test_band_coverage_of_overconfident_draws():
    """Draws at std 0.5 for N(0, 1) truth: P(|Z| < 1.645 * 0.5) = 0.589."""
    g = torch.Generator().manual_seed(1)
    draws = 0.5 * torch.randn(4000, 8, 500, generator=g)
    truth = torch.randn(8, 500, generator=g)
    want = math.erf(1.6449 * 0.5 / math.sqrt(2))
    assert abs(band_coverage(draws, truth, 0.9) - want) < 0.01


def test_crps_gaussian_closed_form():
    """CRPS(N(0, 1), 0) = 2 phi(0) - 1 / sqrt(pi); it scales with sigma."""
    want = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
    got = crps_gaussian(torch.zeros(3), torch.tensor([1.0, 2.0, 0.5]), torch.zeros(3))
    assert torch.allclose(got, torch.tensor([want, 2 * want, 0.5 * want]), rtol=1e-5)


def test_crps_samples_matches_gaussian():
    """The sample CRPS of N(mu, s) draws converges to the closed form."""
    g = torch.Generator().manual_seed(2)
    mu, s = 0.3, 1.5
    draws = mu + s * torch.randn(20000, 2, generator=g)
    truth = torch.tensor([0.0, 2.0])
    want = crps_gaussian(torch.full((2,), mu), torch.full((2,), s), truth)
    assert torch.allclose(crps_samples(draws, truth), want, rtol=0.02)


def test_crps_samples_is_unbiased_at_64_draws():
    """Averaged over replicates of 64 draws, the fair CRPS hits the closed form.

    The S^2 (V-statistic) form sits ~E|X - X'| / (2 S) = 2 / sqrt(pi) / 128
    ~ 0.0088 high for N(0, 1); 4000 replicates put the MC error near 0.002.
    """
    g = torch.Generator().manual_seed(4)
    draws = torch.randn(64, 4000, dtype=torch.float64, generator=g)
    truth = torch.randn(4000, dtype=torch.float64, generator=g)
    got = crps_samples(draws, truth)
    want = crps_gaussian(torch.zeros(4000), torch.ones(4000), truth.float()).double()
    diff = got - want
    assert abs(diff.mean()) < 3 * diff.std() / math.sqrt(len(diff))


def test_crps_samples_of_a_point_mass():
    """One repeated draw x: CRPS = |x - y|."""
    draws = torch.full((5, 1), 2.0)
    assert torch.allclose(crps_samples(draws, torch.tensor([0.5])), torch.tensor([1.5]))


def _columns(rng: np.random.Generator, kind: str) -> np.ndarray:
    n, cols = 64, 200
    if kind == 'gauss':
        return rng.standard_normal((n, cols))
    if kind == 'exponential':
        return rng.exponential(size=(n, cols))
    if kind == 'lognormal':
        return rng.lognormal(size=(n, cols))
    sign = rng.choice([-1.0, 1.0], size=(n, cols))
    return 3 * sign + 0.3 * rng.standard_normal((n, cols))


@pytest.mark.parametrize(
    ('kind', 'low', 'high'),
    [
        ('gauss', 0.0, 0.02),
        # Skewed but unimodal: BIC alone flagged ~99% of these at 64 draws.
        ('exponential', 0.0, 0.05),
        ('lognormal', 0.0, 0.05),
        ('mixture', 0.95, 1.0),
    ],
)
def test_multimodal_fraction(kind: str, low: float, high: float):
    """Only a well-separated two-mode mixture counts as multimodal."""
    frac = multimodal_fraction(_columns(np.random.default_rng(3), kind))
    assert low <= frac <= high


def _sign_flipped_draws(seed: int, flip_within: bool) -> tuple[torch.Tensor, ...]:
    """Draws equal to truth up to a per-draw sign (or per-half sign)."""
    g = torch.Generator().manual_seed(seed)
    truth = torch.randn(4, 64, generator=g) + 3.0
    sign = torch.where(torch.rand(32, 4, 1, generator=g) < 0.5, -1.0, 1.0)
    draws = sign * truth
    if flip_within:
        draws[..., 32:] *= -1
    return draws, truth


def test_magnitude_scores_ignore_sign():
    """Sign-flipped exact draws: zero speed RMSE however the signs fall."""
    draws, truth = _sign_flipped_draws(0, flip_within=False)
    out = magnitude_scores(draws, truth)
    assert out['speed_rmse']['max'] < 1e-6
    assert out['crps_um_per_ms']['max'] < 1e-6


def test_sign_structure_global_vs_within_shot():
    """Whole-draw flips read zero segment flips; half-shot flips read half."""
    draws, truth = _sign_flipped_draws(1, flip_within=False)
    glob = sign_structure(draws, truth, segment=16)
    assert glob['segment_flip_fraction'] == 0.0
    assert 0.3 < glob['draw_sign_agreement'] < 0.7
    assert glob['sample_mass_bimodal'] == 1.0
    within = sign_structure(*_sign_flipped_draws(1, flip_within=True), segment=16)
    assert within['segment_flip_fraction'] == 0.5


def test_align_signs_recovers_truth_given_a_right_reference():
    """A reference with the right sign undoes global and per-segment flips."""
    draws, truth = _sign_flipped_draws(2, flip_within=False)
    assert torch.equal(align_signs(draws, truth, None), truth.expand_as(draws))
    draws, truth = _sign_flipped_draws(2, flip_within=True)
    assert torch.equal(align_signs(draws, truth, 16), truth.expand_as(draws))
    # A partial trailing block takes the last full block's sign.
    assert torch.equal(
        align_signs(draws[..., :60], truth[..., :60], 16),
        truth[..., :60].expand_as(draws[..., :60]),
    )


def test_signed_scores_of_exact_draws():
    """Draws equal to truth: zero RMSE and CRPS."""
    _, truth = _sign_flipped_draws(3, flip_within=False)
    out = signed_scores(truth.expand(8, -1, -1), truth)
    assert out['posterior_mean_velocity_rmse']['max'] < 1e-6
    assert out['crps_um_per_ms']['max'] < 1e-6


class _Prior:
    """A sampler that ignores its conditioning entirely."""

    device = torch.device('cpu')

    def sample(self, cond, n, _n_steps, generator):
        return torch.randn(n, *cond.shape, generator=generator)


def test_conditioning_sensitivity_of_a_prior_sampler_is_zero():
    assert conditioning_sensitivity(_Prior(), torch.ones(2, 3, 32), 4, 0) == 0.0
