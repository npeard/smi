#!/usr/bin/env python3
"""Sanity tests for forward pass of CNN, TCN, SCNN, and other models."""

import pytest
import torch

from smi.analysis.models.barland_cnn import BarlandCNN, BarlandCNNConfig
from smi.analysis.models.lstm import LSTM, LSTMConfig
from smi.analysis.models.mamba import Mamba, MambaConfig
from smi.analysis.models.scnn import SCNN, SCNNConfig
from smi.analysis.models.tcan import TCAN, TCANConfig
from smi.analysis.models.tcn import TCN, TCNConfig

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
