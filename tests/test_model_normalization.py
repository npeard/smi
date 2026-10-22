"""Tests for the Model normalization wrapper and FeatureMap."""

import numpy as np
import torch

from smi.analysis.models.base import FeatureMap, Model
from smi.analysis.models.tcn import TCN, TCNConfig

SEQ_LEN = 256
IN_CHANNELS = 3


def _tiny_tcn() -> TCN:
    """A small TCN suitable for fast CPU tests."""
    config = TCNConfig(
        sequence_length=SEQ_LEN,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=False,
        use_weight_norm=False,
        kernel_size=3,
        temporal_channels=[4, 4],
        dilation_base=2,
        dropout=0.0,
    )
    return TCN(config)


def test_feature_map_is_identity():
    """FeatureMap returns its input unchanged."""
    x = torch.randn(2, IN_CHANNELS, SEQ_LEN)
    torch.testing.assert_close(FeatureMap()(x), x)


def test_forward_shape_and_raw_scale():
    """The wrapper preserves shape and de-normalizes the output to raw scale."""
    inner = _tiny_tcn()
    # Distinct, non-trivial stats so scaling is observable.
    model = Model(
        inner,
        input_mean=np.array([1.0, 2.0, 3.0]),
        input_std=np.array([0.5, 0.5, 0.5]),
        output_mean=10.0,
        output_std=100.0,
    )
    model.eval()
    x = torch.randn(4, IN_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 1, SEQ_LEN)

    # The wrapped output equals inner(normalized) * output_std + output_mean.
    with torch.no_grad():
        x_norm = (x - model.input_mean) / model.input_std
        expected = inner(x_norm) * 100.0 + 10.0
    torch.testing.assert_close(out, expected)


def test_internal_input_is_standardized():
    """Inputs drawn from the stats distribution are ~0 mean / unit std internally."""
    inner = _tiny_tcn()
    means = np.array([5.0, -2.0, 0.3])
    stds = np.array([2.0, 0.7, 1.5])
    model = Model(
        inner, input_mean=means, input_std=stds, output_mean=0.0, output_std=1.0
    )

    rng = np.random.default_rng(0)
    # Raw inputs distributed per-channel as N(mean, std^2).
    raw = np.stack(
        [rng.normal(means[c], stds[c], size=(2000,)) for c in range(IN_CHANNELS)]
    )[None]  # [1, C, N]
    x = torch.tensor(raw, dtype=torch.float32)
    normed = (x - model.input_mean) / model.input_std
    per_channel_mean = normed.mean(dim=(0, 2))
    per_channel_std = normed.std(dim=(0, 2))
    torch.testing.assert_close(
        per_channel_mean, torch.zeros(IN_CHANNELS), atol=0.1, rtol=0
    )
    torch.testing.assert_close(
        per_channel_std, torch.ones(IN_CHANNELS), atol=0.1, rtol=0
    )


def test_identity_wrapper_is_passthrough():
    """Model.identity leaves the signal untouched (matches the bare inner model)."""
    inner = _tiny_tcn()
    model = Model.identity(inner, IN_CHANNELS)
    model.eval()
    x = torch.randn(2, IN_CHANNELS, SEQ_LEN)
    with torch.no_grad():
        torch.testing.assert_close(model(x), inner(x))


def test_from_registry_stats_selects_named_features():
    """from_registry_stats wires the named feature stats into the buffers."""
    inner = _tiny_tcn()
    stats = {
        'a': {'mean': 1.0, 'std': 2.0},
        'b': {'mean': 3.0, 'std': 4.0},
        'c': {'mean': 5.0, 'std': 6.0},
        'velocity': {'mean': 7.0, 'std': 8.0},
    }
    model = Model.from_registry_stats(inner, ['a', 'b', 'c'], 'velocity', stats)
    torch.testing.assert_close(
        model.input_mean.flatten(), torch.tensor([1.0, 3.0, 5.0])
    )
    torch.testing.assert_close(model.input_std.flatten(), torch.tensor([2.0, 4.0, 6.0]))
    assert float(model.output_mean) == 7.0
    assert float(model.output_std) == 8.0
