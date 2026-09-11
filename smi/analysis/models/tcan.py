#!/usr/bin/env python3

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .cross_attention_block import CrossAttentionBlock
from .tcn import TCN, TCNConfig

logger = logging.getLogger(__name__)


@dataclass
class TCANConfig:
    """Configuration for Temporal Convolutional Attention Network.

    Args:
        sequence_length: Length of input/output sequence
        in_channels: Number of input channels (e.g. 3 for interferometer data)
        activation: Activation function name ('GELU', 'ReLU', etc.)
        use_layer_norm: Whether to use layer normalization in temporal blocks
        use_weight_norm: Whether to apply weight normalization to conv layers

        siamese_kernel_size: Kernel size for the siamese encoder TCN
        siamese_channels: Output channel widths per layer of the siamese encoder TCN
        siamese_dilation_base: Dilation base for the siamese encoder TCN

        atten_heads: Number of attention heads (must divide siamese_channels[-1])

        decoder_kernel_size: Kernel size for the decoder TCN
        decoder_channels: Output channel widths per layer of the decoder TCN
        decoder_dilation_base: Dilation base for the decoder TCN
    """

    sequence_length: int
    in_channels: int
    activation: str
    use_layer_norm: bool
    use_weight_norm: bool

    siamese_kernel_size: int
    siamese_channels: list[int]
    siamese_dilation_base: int

    atten_heads: int

    decoder_kernel_size: int
    decoder_channels: list[int]
    decoder_dilation_base: int
    dropout: float

    def __post_init__(self):
        embed_dim = self.siamese_channels[-1]
        if embed_dim % self.atten_heads != 0:
            raise ValueError(
                f'siamese_channels[-1] ({embed_dim}) must be divisible by '
                f'atten_heads ({self.atten_heads})'
            )


def _make_siamese_tcn(config: TCANConfig) -> TCN:
    """Build a single-input-channel TCN for use as the siamese encoder.

    The output projection (to 1 channel) is replaced with Identity so the
    network outputs ``siamese_channels[-1]`` feature maps instead.
    """
    tcn_config = TCNConfig(
        sequence_length=config.sequence_length,
        in_channels=1,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        use_weight_norm=config.use_weight_norm,
        dropout=config.dropout,
        kernel_size=config.siamese_kernel_size,
        temporal_channels=config.siamese_channels,
        dilation_base=config.siamese_dilation_base,
    )
    tcn = TCN(tcn_config)
    # Encoder TCN drops its final 1x1 projection (kept as a feature extractor);
    # replacing the typed Conv1d attribute with Identity is intentional.
    tcn.projection = nn.Identity()  # ty: ignore[invalid-assignment]
    return tcn


def _make_decoder_tcn(config: TCANConfig) -> TCN:
    """Build the decoder TCN that maps all attended channel features to 1 output."""
    embed_dim = config.siamese_channels[-1]
    tcn_config = TCNConfig(
        sequence_length=config.sequence_length,
        in_channels=config.in_channels * embed_dim,
        activation=config.activation,
        use_layer_norm=config.use_layer_norm,
        use_weight_norm=config.use_weight_norm,
        dropout=config.dropout,
        kernel_size=config.decoder_kernel_size,
        temporal_channels=config.decoder_channels,
        dilation_base=config.decoder_dilation_base,
    )
    return TCN(tcn_config)


class TCAN(nn.Module):
    """Temporal Convolutional Attention Network.

    Architecture:
    1. Siamese encoder: a shared-weight TCN applied independently to each input
       channel, producing per-channel feature maps.
    2. Cross-attention: cross-channel attention fuses information across channels
       at each time step independently (no temporal lookahead).
    3. Decoder: a single TCN that takes all attended channel features concatenated
       along the channel dimension and produces a 1-channel sequence output.

    The per-channel encoder outputs are exposed via ``encode()`` so that VICReg
    regularisation can be applied from LitModule when ``vicreg_weight > 0``.
    """

    def __init__(self, config: TCANConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        embed_dim = config.siamese_channels[-1]
        # Cache as a plain int attribute so forward() does not touch the
        # dataclass config (which TorchScript cannot type).
        self.embed_dim = embed_dim

        # Shared-weight encoder TCN (one instance, reused for every channel)
        self.siamese_encoder = _make_siamese_tcn(config)

        # Cross-channel attention (operates over n_channels tokens at each time step)
        self.cross_attention = CrossAttentionBlock(
            n_channels=config.in_channels,
            embed_dim=embed_dim,
            num_heads=config.atten_heads,
            dropout=config.dropout,
        )

        # Decoder TCN: in_channels * embed_dim -> 1
        self.decoder = _make_decoder_tcn(config)

        logger.info(f'Number of parameters in TCAN: {self.total_params:,}')

    @property
    @torch.jit.unused
    def receptive_field(self) -> int:
        """Calculate the receptive field of the TCAN.

        The encoder and decoder TCNs are in series, so their receptive fields add.
        The cross-channel attention has no temporal span (same time step only).
        """
        return self.siamese_encoder.receptive_field + self.decoder.receptive_field

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run the siamese encoder on each input channel independently.

        Args:
            x: [batch, in_channels, seq_len]

        Returns:
            features: [batch, in_channels * embed_dim, seq_len] concatenated
                per-channel encoder outputs, ready for cross-attention.
            channel_features: list of in_channels tensors each
                [batch, embed_dim, seq_len] -- exposed for VICReg.
        """
        batch_size, _, seq_len = x.shape
        # Fold channels into batch for a single batched forward pass through
        # the shared encoder.
        # [batch * in_channels, 1, seq_len] -> [batch * in_channels, embed_dim, seq_len]
        x_flat = x.reshape(batch_size * self.in_channels, 1, seq_len)
        out = self.siamese_encoder(x_flat)
        embed_dim = out.shape[1]
        # Unfold: [batch, in_channels, embed_dim, seq_len]
        out = out.reshape(batch_size, self.in_channels, embed_dim, seq_len)
        channel_features = [out[:, c] for c in range(self.in_channels)]
        features = out.reshape(batch_size, self.in_channels * embed_dim, seq_len)
        return features, channel_features

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through TCAN.

        Args:
            x: [batch, in_channels, seq_len]

        Returns:
            [batch, 1, seq_len]
        """
        batch_size, _, seq_len = x.shape
        embed_dim = self.embed_dim

        # Siamese encoder: per-channel feature extraction
        features, _ = self.encode(x)
        # features: [batch, in_channels * embed_dim, seq_len]

        # Cross-channel attention at each time step.
        # Reshape to [batch * seq_len, in_channels, embed_dim] so that
        # attention tokens are the per-channel feature vectors at each time step.
        features = features.view(batch_size, self.in_channels, embed_dim, seq_len)
        features = features.permute(0, 3, 1, 2).reshape(
            batch_size * seq_len, self.in_channels, embed_dim
        )
        features = self.cross_attention(features)
        # Reshape back to [batch, in_channels * embed_dim, seq_len]
        features = features.reshape(batch_size, seq_len, self.in_channels * embed_dim)
        features = features.permute(0, 2, 1).contiguous()

        # Decoder: all attended channel features -> 1-channel output
        return self.decoder(features)  # [batch, 1, seq_len]
