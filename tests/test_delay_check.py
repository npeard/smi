"""Tests for the velocity/photodiode delay estimator."""

import numpy as np

from smi.analysis.delay_check import estimate_lag


def test_recovers_known_injected_lag():
    """A signal shifted by a known amount reports that lag, within one sample."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(4096)
    shift = 17
    shifted = np.roll(base, shift)

    result = estimate_lag(target=base, signal=shifted, max_lag=100)

    assert result.is_interior_peak
    assert abs(result.lag_samples - shift) <= 1


def test_reports_no_interior_peak_for_uncorrelated_input():
    """Uncorrelated inputs must not be reported as a confident lag."""
    rng = np.random.default_rng(1)
    result = estimate_lag(
        target=rng.standard_normal(4096), signal=rng.standard_normal(4096), max_lag=100
    )

    assert not result.is_interior_peak


def test_zero_lag_is_interior_and_recovered():
    """An unshifted pair reports lag 0, not an edge artifact."""
    rng = np.random.default_rng(2)
    base = rng.standard_normal(4096)

    result = estimate_lag(target=base, signal=base.copy(), max_lag=100)

    assert result.is_interior_peak
    assert result.lag_samples == 0
