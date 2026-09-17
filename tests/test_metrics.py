"""Tests for evaluation metrics in stored units."""

import numpy as np
import torch

from smi.analysis.metrics import (
    count_parameters,
    displacement_from_velocity,
    summarize_per_shot,
    velocity_rmse,
)
from smi.analysis.models.factory import create_model

#: Dataset sample rate (Hz) after the decimate-by-256 acquisition setting.
SAMPLE_RATE_HZ = 488281.25
DT_MS = 1.0e3 / SAMPLE_RATE_HZ


def test_velocity_rmse_is_zero_for_exact_prediction():
    x = torch.randn(4, 1, 512)
    assert float(velocity_rmse(x, x)) == 0.0


def test_velocity_rmse_matches_hand_computation():
    pred = torch.tensor([[[1.0, 2.0, 3.0]]])
    target = torch.tensor([[[1.0, 4.0, 3.0]]])
    # errors: 0, 2, 0 -> mean square 4/3 -> rmse 2/sqrt(3)
    expected = 2.0 / np.sqrt(3.0)
    assert (
        float(velocity_rmse(pred, target)) == np.float32(expected).item()
        or abs(float(velocity_rmse(pred, target)) - expected) < 1e-6
    )


def test_summarize_per_shot_keys_match_baseline_artifact():
    """The six keys must match baseline_results.json exactly."""
    summary = summarize_per_shot(np.arange(101, dtype=float))
    assert set(summary) == {'mean', 'median', 'p25', 'p75', 'min', 'max'}


def test_summarize_per_shot_values_match_numpy():
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    summary = summarize_per_shot(values)
    assert summary['mean'] == float(np.mean(values))
    assert summary['median'] == float(np.median(values))
    assert summary['p25'] == float(np.percentile(values, 25))
    assert summary['p75'] == float(np.percentile(values, 75))
    assert summary['min'] == 1.0
    assert summary['max'] == 100.0


def test_displacement_from_velocity_accepts_the_model_output_shape():
    """Models emit [batch, 1, length]; the integrator underneath takes 2-D."""
    velocity = torch.ones(2, 1, 8)
    displacement = displacement_from_velocity(velocity, DT_MS)
    assert displacement.shape == velocity.shape
    # Constant 1 um/ms integrates to a ramp of dt_ms per sample, starting at 0.
    expected = torch.arange(8, dtype=torch.float32) * DT_MS
    assert torch.allclose(displacement[0, 0], expected, atol=1e-9)


def test_displacement_from_velocity_preserves_two_dimensional_input():
    velocity = torch.ones(2, 8)
    assert displacement_from_velocity(velocity, DT_MS).shape == (2, 8)


# The matched-parameter budget: Barland at three channels. Its single-channel
# count (57,153) matches the 2021 paper's stated total exactly, which is what
# confirms the port is faithful.
BARLAND_HPARAMS = {
    'type': 'Barland',
    'window_size': 256,
    'in_channels': 3,
    'activation': 'LeakyReLU',
    'dropout': 0.1,
    'use_weight_norm': False,
    'window_stride': 64,
}


def test_count_parameters_agrees_with_instantiated_model():
    hparams = dict(BARLAND_HPARAMS)
    model = create_model(hparams)
    assert count_parameters(hparams) == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def test_barland_three_channel_budget_is_stable():
    """The matched-parameter budget every other model is held to."""
    assert count_parameters(dict(BARLAND_HPARAMS)) == 57377
