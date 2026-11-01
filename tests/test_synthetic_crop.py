"""Training-length budget: random crops keep signals and targets aligned."""

import torch

from smi.learning.models.factory import create_model
from smi.learning.synthetic_lit_module import random_crop


def test_crop_keeps_signals_and_targets_aligned():
    batch, length, crop = 5, 40, 12
    ramp = torch.arange(length, dtype=torch.float32)
    signals = ramp.expand(batch, 3, length).clone()
    velocity = ramp.expand(batch, length).clone()
    displacement = 2 * ramp.expand(batch, length)

    s, v, d = random_crop(crop, signals, velocity, displacement, generator=None)

    assert s.shape == (batch, 3, crop)
    assert v.shape == d.shape == (batch, crop)
    # The ramp's value is its absolute position: equal values mean same window.
    for c in range(3):
        torch.testing.assert_close(s[:, c], v)
    torch.testing.assert_close(d, 2 * v)
    # Windows are contiguous and in range.
    assert torch.equal(v.diff(dim=-1), torch.ones(batch, crop - 1))
    assert v.min() >= 0
    assert v.max() <= length - 1


def test_crop_start_varies_per_shot_and_none_is_identity():
    x = torch.arange(1000.0).expand(64, 1000)
    (cropped,) = random_crop(10, x, generator=None)
    assert cropped[:, 0].unique().numel() > 1
    (same,) = random_crop(None, x, generator=None)
    assert same is x


def test_causal_model_on_crop_matches_full_shot_beyond_receptive_field():
    torch.manual_seed(0)
    model = create_model(
        dict(
            type='TCN',
            sequence_length=512,
            in_channels=3,
            activation='GELU',
            use_layer_norm=True,
            use_weight_norm=False,
            kernel_size=3,
            temporal_channels=[8, 8, 8],
            dilation_base=2,
            dropout=0.0,
        )
    ).eval()
    rf = model.receptive_field
    x = torch.randn(4, 3, 512)
    start, crop = 200, 256
    with torch.no_grad():
        full = model(x)
        part = model(x[..., start : start + crop])
    # Past the first rf - 1 samples a causal output sees only the window.
    torch.testing.assert_close(
        part[..., rf - 1 :],
        full[..., start + rf - 1 : start + crop],
        atol=1e-5,
        rtol=1e-5,
    )
