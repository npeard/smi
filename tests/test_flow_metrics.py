"""Probabilistic scores of the R8 flow posteriors, on cases with known answers."""

import math

import numpy as np
import torch

from smi.core.metrics import band_coverage, crps_gaussian, crps_samples
from smi.experiments.flow_eval import multimodal_fraction


def test_band_coverage_of_calibrated_draws():
    """Truth and draws from the same N(0, 1): the 90% band covers 0.90."""
    g = torch.Generator().manual_seed(0)
    draws = torch.randn(4000, 8, 500, generator=g)
    truth = torch.randn(8, 500, generator=g)
    assert abs(band_coverage(draws, truth, 0.9) - 0.9) < 0.01


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


def test_crps_samples_of_a_point_mass():
    """One repeated draw x: CRPS = |x - y|."""
    draws = torch.full((5, 1), 2.0)
    assert torch.allclose(crps_samples(draws, torch.tensor([0.5])), torch.tensor([1.5]))


def test_multimodal_fraction():
    """A +/-3 two-mode mixture is multimodal; a Gaussian is not."""
    rng = np.random.default_rng(3)
    n = 64
    gauss = rng.standard_normal((n, 40))
    sign = rng.choice([-1.0, 1.0], size=(n, 40))
    mixture = 3 * sign + 0.3 * rng.standard_normal((n, 40))
    assert multimodal_fraction(gauss) < 0.1
    assert multimodal_fraction(mixture) > 0.9
