#!/usr/bin/env python3
"""Sanity tests for forward pass of CNN, TCN, SCNN, and other models."""

import pytest
import torch
from torch.nn import functional

from smi.learning.models.barland_cnn import BarlandCNN, BarlandCNNConfig
from smi.learning.models.lstm import LSTM, LSTMConfig
from smi.learning.models.mamba import Mamba, MambaConfig
from smi.learning.models.neural_operator import (
    CausalSpectralConv1d,
    NeuralOperator,
    NeuralOperatorConfig,
)
from smi.learning.models.patch_transformer import (
    PatchTransformer,
    PatchTransformerConfig,
)
from smi.learning.models.scnn import SCNN, SCNNConfig
from smi.learning.models.tcan import TCAN, TCANConfig
from smi.learning.models.tcn import TCN, TCNConfig
from smi.learning.models.unet_transformer import UNetTransformer, UNetTransformerConfig
from tests._devices import devices

SEQUENCE_LENGTH = 256
IN_CHANNELS = 3
BATCH_SIZE = 2


@pytest.fixture
def barland_model():
    config = BarlandCNNConfig(
        window_size=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='LeakyReLU',
        dropout=0.1,
        use_weight_norm=False,
        window_stride=128,
        channels=[16, 32, 64, 64],
    )
    return BarlandCNN(config).eval()


@pytest.fixture
def tcn_model():
    config = TCNConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=7,
        temporal_channels=[16, 32, 64, 32, 16],
        dilation_base=2,
        dropout=0.0,
    )
    return TCN(config).eval()


SIGNAL_LENGTH = 1024  # longer than the 256-sample window to exercise the striding


@pytest.fixture
def stemtcan_model():
    config = TCANConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        siamese_kernel_size=7,
        siamese_channels=[8, 8, 16],
        siamese_dilation_base=2,
        atten_heads=4,
        decoder_kernel_size=7,
        decoder_channels=[32, 16],
        decoder_dilation_base=2,
        dropout=0.0,
    )
    return TCAN(config).eval()


class TestTCANForward:
    def test_output_shape(self, stemtcan_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = stemtcan_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, stemtcan_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = stemtcan_model(x)
        assert torch.isfinite(out).all(), 'TCAN output contains non-finite values'

    def test_zero_input(self, stemtcan_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = stemtcan_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH)
        assert torch.isfinite(out).all()

    def test_encode_shape(self, stemtcan_model):
        """encode() must return per-channel features of the right shape."""
        embed_dim = stemtcan_model.config.siamese_channels[-1]
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            features, channel_features = stemtcan_model.encode(x)
        assert features.shape == (BATCH_SIZE, IN_CHANNELS * embed_dim, SEQUENCE_LENGTH)
        assert len(channel_features) == IN_CHANNELS
        for cf in channel_features:
            assert cf.shape == (BATCH_SIZE, embed_dim, SEQUENCE_LENGTH)


class TestBarlandCNNForward:
    def test_output_shape(self, barland_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SIGNAL_LENGTH)
        with torch.no_grad():
            out = barland_model(x)
        assert out.shape == (BATCH_SIZE, 1, SIGNAL_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SIGNAL_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, barland_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SIGNAL_LENGTH)
        with torch.no_grad():
            out = barland_model(x)
        assert torch.isfinite(out).all(), 'BarlandCNN output contains non-finite values'

    def test_zero_input(self, barland_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SIGNAL_LENGTH)
        with torch.no_grad():
            out = barland_model(x)
        assert out.shape == (BATCH_SIZE, 1, SIGNAL_LENGTH)
        assert torch.isfinite(out).all()


@pytest.fixture
def lstm_model():
    config = LSTMConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        hidden_size=32,
        num_layers=2,
        dropout=0.0,
        bidirectional=False,
    )
    return LSTM(config).eval()


@pytest.fixture
def lstm_bidirectional_model():
    config = LSTMConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        hidden_size=32,
        num_layers=2,
        dropout=0.0,
        bidirectional=True,
    )
    return LSTM(config).eval()


class TestLSTMForward:
    def test_output_shape(self, lstm_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = lstm_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, lstm_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = lstm_model(x)
        assert torch.isfinite(out).all(), 'LSTM output contains non-finite values'

    def test_zero_input(self, lstm_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = lstm_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH)
        assert torch.isfinite(out).all()

    def test_bidirectional_output_shape(self, lstm_bidirectional_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = lstm_bidirectional_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )


@pytest.fixture
def mamba_model():
    config = MambaConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        d_model=16,
        d_state=8,
        d_conv=4,
        expand=2,
        num_layers=2,
        use_layer_norm=True,
    )
    return Mamba(config).eval()


class TestMambaForward:
    def test_output_shape(self, mamba_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = mamba_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, mamba_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = mamba_model(x)
        assert torch.isfinite(out).all(), 'Mamba output contains non-finite values'

    def test_zero_input(self, mamba_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = mamba_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH)
        assert torch.isfinite(out).all()


class TestTCNForward:
    def test_output_shape(self, tcn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = tcn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, tcn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = tcn_model(x)
        assert torch.isfinite(out).all(), 'TCN output contains non-finite values'

    def test_zero_input(self, tcn_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = tcn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH)
        assert torch.isfinite(out).all()


@pytest.fixture
def scnn_model():
    config = SCNNConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=7,
        temporal_channels=[16, 32, 64, 32, 16],
        dilation_base=2,
        dropout=0.0,
    )
    return SCNN(config).eval()


class TestSCNNForward:
    def test_output_shape(self, scnn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = scnn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH), (
            f'Expected shape ({BATCH_SIZE}, 1, {SEQUENCE_LENGTH}), got {out.shape}'
        )

    def test_output_is_finite(self, scnn_model):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = scnn_model(x)
        assert torch.isfinite(out).all(), 'SCNN output contains non-finite values'

    def test_zero_input(self, scnn_model):
        x = torch.zeros(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = scnn_model(x)
        assert out.shape == (BATCH_SIZE, 1, SEQUENCE_LENGTH)
        assert torch.isfinite(out).all()


def _patch_transformer(causal: bool) -> PatchTransformer:
    config = PatchTransformerConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        patch_len=8,
        stride=5,
        d_model=16,
        n_heads=2,
        n_layers=2,
        ffn_dim=32,
        attention_span=2,
        dropout=0.0,
        causal=causal,
    )
    return PatchTransformer(config).eval()


@pytest.mark.parametrize('causal', [True, False], ids=['causal', 'offline'])
class TestPatchTransformerForward:
    # 253 is not a multiple of the stride: the last token is partial, and the
    # upsampled output must still be cropped back to the input length.
    @pytest.mark.parametrize('length', [SEQUENCE_LENGTH, 253])
    def test_output_shape(self, causal, length):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, length)
        with torch.no_grad():
            out = _patch_transformer(causal)(x)
        assert out.shape == (BATCH_SIZE, 1, length)

    def test_output_is_finite(self, causal):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = _patch_transformer(causal)(x)
        assert torch.isfinite(out).all()

    def test_input_shorter_than_one_patch(self, causal):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, 3)
        with torch.no_grad():
            out = _patch_transformer(causal)(x)
        assert out.shape == (BATCH_SIZE, 1, 3)
        assert torch.isfinite(out).all()


def _unet_transformer(causal: bool) -> UNetTransformer:
    config = UNetTransformerConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        channels=[8, 8, 16],
        kernel_size=4,
        pool=2,
        n_heads=2,
        n_layers=1,
        ffn_dim=32,
        attention_span=2,
        dropout=0.0,
        causal=causal,
    )
    return UNetTransformer(config).eval()


@pytest.mark.parametrize('causal', [True, False], ids=['causal', 'offline'])
class TestUNetTransformerForward:
    # The total pooling factor is 4: 253 and 254 leave a partial coarse token
    # at each level, and 1 and 3 are shorter than one bottleneck token.
    @pytest.mark.parametrize('length', [SEQUENCE_LENGTH, 253, 254, 3, 1])
    def test_output_shape(self, causal, length):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, length)
        with torch.no_grad():
            out = _unet_transformer(causal)(x)
        assert out.shape == (BATCH_SIZE, 1, length)
        assert torch.isfinite(out).all()

    def test_rejects_heads_that_do_not_divide_the_bottleneck(self, causal):
        with pytest.raises(ValueError, match='not divisible by n_heads'):
            UNetTransformerConfig(
                sequence_length=SEQUENCE_LENGTH,
                in_channels=IN_CHANNELS,
                channels=[8, 8, 15],
                kernel_size=4,
                pool=2,
                n_heads=2,
                n_layers=1,
                ffn_dim=32,
                attention_span=2,
                dropout=0.0,
                causal=causal,
            )


def _neural_operator_config(causal: bool) -> NeuralOperatorConfig:
    return NeuralOperatorConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        width=8,
        n_layers=2,
        modes=4,
        kernel_len=9 if causal else None,
    )


@pytest.mark.parametrize('causal', [True, False], ids=['causal', 'offline'])
class TestNeuralOperatorForward:
    # Odd, even, shorter than the kernel, and shorter than the retained modes.
    @pytest.mark.parametrize('length', [SEQUENCE_LENGTH, 253, 5, 1])
    def test_output_shape(self, causal, length):
        x = torch.randn(BATCH_SIZE, IN_CHANNELS, length)
        with torch.no_grad():
            out = NeuralOperator(_neural_operator_config(causal)).eval()(x)
        assert out.shape == (BATCH_SIZE, 1, length)
        assert torch.isfinite(out).all()


def test_neural_operator_rejects_modes_the_kernel_cannot_hold():
    """Modes past the kernel grid's (K + 1) // 2 would be dead parameters."""
    with pytest.raises(ValueError, match='modes 6 exceeds'):
        NeuralOperatorConfig(
            sequence_length=SEQUENCE_LENGTH,
            in_channels=IN_CHANNELS,
            width=8,
            n_layers=2,
            modes=6,
            kernel_len=9,
        )


@pytest.mark.parametrize('device', devices())
@pytest.mark.parametrize(('length', 'kernel_len'), [(100, 22), (7, 22), (64, 1)])
def test_causal_operator_matches_direct_conv(device, length, kernel_len):
    """The zero-padded FFT product is the linear, left-padded convolution.

    Too little padding would wrap the kernel's tail onto the first samples;
    conv1d with K - 1 zeros on the left is the causal convolution by
    definition. conv1d correlates, so the lag axis is flipped.
    """
    torch.manual_seed(0)
    width, modes = 5, (kernel_len + 1) // 2
    layer = CausalSpectralConv1d(width, modes, kernel_len).to(device)
    x = torch.randn(2, width, length, device=device)
    kernel = layer.kernel()  # [in, out, lag]
    direct = functional.conv1d(
        functional.pad(x, (kernel_len - 1, 0)), kernel.transpose(0, 1).flip(-1)
    )
    with torch.no_grad():
        torch.testing.assert_close(layer(x), direct, rtol=1e-4, atol=1e-5)
