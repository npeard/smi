"""Tests for the windowed MLP baseline."""

import torch

from smi.analysis.models.factory import create_model


def _hparams(**overrides) -> dict:
    base = {
        'type': 'MLP',
        'window_size': 256,
        'in_channels': 3,
        'hidden_sizes': [64, 32],
        'activation': 'GELU',
        'dropout': 0.0,
        'window_stride': 64,
    }
    base.update(overrides)
    return base


def test_mlp_output_shape_matches_input_length():
    model = create_model(_hparams())
    out = model(torch.randn(2, 3, 2048))
    assert out.shape == (2, 1, 2048)


def test_mlp_handles_length_not_divisible_by_stride():
    """Right-padding must cover a ragged tail, as BarlandCNN's does."""
    model = create_model(_hparams())
    out = model(torch.randn(1, 3, 2000))
    assert out.shape == (1, 1, 2000)


def test_mlp_reports_its_parameter_count():
    model = create_model(_hparams())
    assert model.total_params == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def test_mlp_receptive_field_is_the_window():
    """A dense layer over the window sees all of it and nothing beyond it."""
    model = create_model(_hparams())
    assert model.receptive_field == 256
