"""Empirical receptive-field and causality tests.

The property under test is a formula, so asserting it against another
formula proves nothing. These tests measure the field by backpropagating
from a single output timestep and counting which input timesteps receive
gradient -- ground truth the formula cannot talk its way around.
"""

import pytest
import torch

from smi.analysis.models.scnn import SCNN, SCNNConfig
from smi.analysis.models.tcan import TCAN, TCANConfig
from smi.analysis.models.tcn import TCN, TCNConfig


def measure_receptive_field(
    model: torch.nn.Module, in_channels: int, length: int
) -> tuple[int, bool]:
    """Measure the true receptive field by gradient flow.

    Returns:
        (field_width, sees_future) where field_width is the number of input
        timesteps with nonzero gradient, and sees_future reports whether any
        timestep strictly after the probed output index received gradient.
    """
    model.eval()
    x = torch.zeros(1, in_channels, length, requires_grad=True)
    out = model(x)
    probe = length // 2
    model.zero_grad()
    out[0, 0, probe].backward()

    grad = x.grad
    assert grad is not None
    touched = (grad.abs().sum(dim=1)[0] > 0).nonzero().flatten()
    field_width = int(touched.numel())
    sees_future = bool((touched > probe).any().item())
    return field_width, sees_future


def _config(kernel_size: int, dilation_base: int, channels: list[int]) -> TCNConfig:
    return TCNConfig(
        sequence_length=2048,
        in_channels=3,
        activation='GELU',
        use_layer_norm=False,
        use_weight_norm=False,
        kernel_size=kernel_size,
        temporal_channels=channels,
        dilation_base=dilation_base,
        dropout=0.0,
    )


@pytest.mark.parametrize(
    ('kernel_size', 'dilation_base', 'channels'),
    [(3, 2, [8, 8, 16]), (5, 2, [8, 16]), (3, 3, [8, 8, 16]), (7, 2, [8, 16])],
)
def test_receptive_field_matches_measurement(kernel_size, dilation_base, channels):
    """The reported receptive field equals the measured one."""
    model = TCN(_config(kernel_size, dilation_base, channels))
    measured, _ = measure_receptive_field(model, in_channels=3, length=2048)
    assert model.receptive_field == measured


def test_causal_tcn_never_sees_the_future():
    """A causal TCN must receive no gradient from samples after the probe."""
    model = TCN(_config(3, 2, [8, 8, 16]))
    _, sees_future = measure_receptive_field(model, in_channels=3, length=2048)
    assert not sees_future


def test_scnn_does_see_the_future():
    """SCNN is non-causal by construction; that is the whole difference."""
    config = SCNNConfig(
        sequence_length=2048,
        in_channels=3,
        activation='GELU',
        use_layer_norm=False,
        use_weight_norm=False,
        kernel_size=3,
        temporal_channels=[8, 8, 16],
        dilation_base=2,
        dropout=0.0,
    )
    _, sees_future = measure_receptive_field(SCNN(config), in_channels=3, length=2048)
    assert sees_future


def test_tcan_receptive_field_does_not_double_count_the_centre():
    """TCAN sums two TCN fields, each of which already includes its centre.

    Without the correction the reported field is one larger than the measured
    one -- an off-by-one that is invisible unless the field is measured.
    """
    config = TCANConfig(
        sequence_length=2048,
        in_channels=3,
        activation='GELU',
        use_layer_norm=False,
        use_weight_norm=False,
        siamese_kernel_size=3,
        siamese_channels=[8, 16],
        siamese_dilation_base=2,
        atten_heads=2,
        decoder_kernel_size=3,
        decoder_channels=[16, 16],
        decoder_dilation_base=2,
        dropout=0.0,
    )
    model = TCAN(config)
    measured, _ = measure_receptive_field(model, in_channels=3, length=2048)
    assert model.receptive_field == measured
