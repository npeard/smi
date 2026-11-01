#!/usr/bin/env python3
"""1-D UNet with a Transformer bottleneck for per-sample velocity regression.

The audio and speech-enhancement shape: a convolutional encoder halves (or
divides by ``pool``) the resolution level by level, a stack of banded
self-attention layers mixes tokens at the coarsest level, and a decoder
upsamples back, concatenating each level's encoder features through a skip
connection. Attention reuses :class:`~.patch_transformer.TransformerLayer`, so
the bottleneck has the same finite, relative-position band as the patch
Transformer and the field can be matched to the convolutional families'.

Causal variant: every convolution is left-padded, coarse token ``j`` pools the
``pool`` finer samples that end at ``j * pool`` (complete at the first sample
of its cell), and a fine sample is upsampled from the coarse token whose cell
holds it -- the current coarse token, never a later one. Skips carry encoder
features at the same time index, which are causal themselves.
"""

import logging
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor, nn
from torch.nn import functional

from .base import SequenceModel
from .patch_transformer import TransformerLayer

logger = logging.getLogger(__name__)


@dataclass
class UNetTransformerConfig:
    """Configuration for the UNet-Transformer.

    Args:
        sequence_length: Nominal input length (unused by the network, which
            accepts any length; kept for parity with the other configs).
        in_channels: Number of input channels.
        channels: Encoder width per level, finest first. Its length minus one
            is the number of pooling steps; the last entry is the bottleneck
            Transformer's width.
        kernel_size: Kernel of every encoder and decoder convolution.
        pool: Average-pooling factor between levels (and upsampling factor
            back).
        n_heads: Bottleneck attention heads; must divide ``channels[-1]``.
        n_layers: Bottleneck Transformer layers.
        ffn_dim: Hidden width of each layer's feed-forward network.
        attention_span: Bottleneck tokens attended on each side of a token
            (past side only when causal), excluding the token itself.
        dropout: Dropout on attention weights and residual branches.
        causal: True for left-padded convolutions, causal pooling and
            past-only attention.
    """

    sequence_length: int
    in_channels: int
    channels: list[int]
    kernel_size: int
    pool: int
    n_heads: int
    n_layers: int
    ffn_dim: int
    attention_span: int
    dropout: float
    causal: bool

    def __post_init__(self):
        """Validate the geometry the receptive-field formula relies on.

        Raises:
            ValueError: If there is no pooling level, the pool or kernel is
                degenerate, the span or depth is below one, or the heads do
                not divide the bottleneck width.
        """
        if len(self.channels) < 2:
            raise ValueError('channels needs at least two levels (one pooling step)')
        if self.pool < 2 or self.kernel_size < 1:
            raise ValueError('pool must be >= 2 and kernel_size >= 1')
        if self.attention_span < 1 or self.n_layers < 1:
            raise ValueError('attention_span and n_layers must both be >= 1')
        if self.channels[-1] % self.n_heads != 0:
            raise ValueError(
                f'bottleneck width {self.channels[-1]} is not divisible by '
                f'n_heads {self.n_heads}'
            )


class UNetTransformer(SequenceModel):
    """UNet encoder/decoder around a banded-attention Transformer bottleneck.

    Levels 0..D with D = len(channels) - 1; level i runs at 1 / pool**i of the
    input rate. Encoder: conv at level 0, then (pool, conv) per level. Decoder,
    from level D - 1 down to 0: nearest upsampling of the coarser output,
    concatenation with the level's encoder features, conv. A 1x1 projection
    gives one output per sample.
    """

    def __init__(self, config: UNetTransformerConfig):
        super().__init__()
        self.config = config
        self.pool = config.pool
        self.n_levels = len(config.channels) - 1
        k = config.kernel_size
        p = config.pool
        span = config.attention_span
        if config.causal:
            self.conv_pad = (k - 1, 0)
            # Coarse token j averages fine samples jp - p + 1 .. jp.
            self.pool_left = p - 1
            left, right = span, 0
        else:
            self.conv_pad = ((k - 1) // 2, k // 2)
            # Coarse token j averages its own cell, jp .. jp + p - 1.
            self.pool_left = 0
            left, right = span, span

        widths = config.channels
        self.encoder = nn.ModuleList(
            [
                nn.Conv1d(c_in, c_out, k)
                for c_in, c_out in pairwise([config.in_channels, *widths])
            ]
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    widths[-1],
                    config.n_heads,
                    config.ffn_dim,
                    left,
                    right,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(widths[-1])
        # In the order forward uses them, coarsest first: decoder[j] produces
        # level i = D - 1 - j from level i + 1 and the level-i skip.
        self.decoder = nn.ModuleList(
            [
                nn.Conv1d(widths[i + 1] + widths[i], widths[i], k)
                for i in reversed(range(self.n_levels))
            ]
        )
        self.projection = nn.Conv1d(widths[0], 1, 1)

        logger.info(
            f'Number of parameters in {self.__class__.__name__}: {self.total_params:,}'
        )

    def _downsample(self, x: Tensor) -> Tensor:
        # ceil(L / pool) tokens either way. The causal left padding already
        # completes the last token; offline, the right padding does.
        length = x.shape[-1]
        n_tokens = (length + self.pool - 1) // self.pool
        right = n_tokens * self.pool - length - self.pool_left
        x = functional.pad(x, (self.pool_left, max(right, 0)))
        return functional.avg_pool1d(x, self.pool, self.pool)

    def forward(self, x: Tensor) -> Tensor:
        """Map [batch, in_channels, length] to [batch, 1, length]."""
        skips: list[Tensor] = []
        h = x
        for i, conv in enumerate(self.encoder):
            if i > 0:
                h = self._downsample(h)
            h = functional.gelu(conv(functional.pad(h, self.conv_pad)))
            skips.append(h)

        h = h.transpose(1, 2)  # [batch, tokens, width]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h).transpose(1, 2)

        for j, conv in enumerate(self.decoder):
            skip = skips[self.n_levels - 1 - j]
            up = h.repeat_interleave(self.pool, dim=-1)[..., : skip.shape[-1]]
            h = torch.cat([up, skip], dim=1)
            h = functional.gelu(conv(functional.pad(h, self.conv_pad)))
        return self.projection(h)

    @property
    @torch.jit.unused
    def receptive_field(self) -> int:
        """Receptive field in input samples: worst-phase lookback or width.

        Pooling makes the field periodic in the total stride P = pool**D, not
        constant, so the reported value is its maximum over the phase of t --
        the most context any output can draw on, which is what the matched
        comparison holds fixed. With K = kernel_size, p = pool, D pooling
        steps, N = n_layers * attention_span, G = (p**D - 1) / (p - 1) and
        S = (p**(D + 1) - 1) / (p - 1):

        * Encoder level D's lookback past its token's time is
          (K - 1) S + P - 1: K - 1 coarse steps of each of D + 1 convolutions
          plus p - 1 fine steps of each of D poolings.
        * The decoder convolutions reach back K - 1 tokens per level before
          their upsampling; nesting the floors, fine output t reads bottleneck
          token m = floor((t - (K - 1) G) / P), so t - mP ranges over
          (K - 1) G .. (K - 1) G + P - 1 with the phase of t.
        * Attention adds N bottleneck tokens.

        Causal lookback, inclusive of t: maximum (K - 1)(G + S) + (N + 2)P - 1
        and minimum P - 1 less, (K - 1)(G + S) + (N + 1)P. For the matched
        causal configs (K = 4, p = 2, D = 2, N = 12) that is 85 at worst and
        82 at best.

        Offline, with the centred convolutions padding a = (K - 1) // 2 left
        and b = K // 2 right: the decoder reads bottleneck tokens
        floor((t - aG) / P) .. floor((t + bG) / P), at most
        ceil((K - 1) G / P) + 1 of them, each attending N either side and
        holding an encoder footprint of P + (K - 1) S. Width, maximum:
        (ceil((K - 1) G / P) + 1 + 2N) P + (K - 1) S; the minimum is P less
        when P does not divide (K - 1) G, and equal otherwise.
        Not symmetric about t when K is even: for the matched offline configs
        (K = 4) the total width is 81..85 over phase, right-heavy by 8..12
        samples.

        ``tests/test_receptive_field.py`` checks both against a gradient probe
        swept over the phase.
        """
        c = self.config
        k, p, d = c.kernel_size, c.pool, self.n_levels
        period = p**d
        g = (period - 1) // (p - 1)
        s = (p ** (d + 1) - 1) // (p - 1)
        n = c.n_layers * c.attention_span
        if c.causal:
            return (k - 1) * (g + s) + (n + 2) * period - 1
        bottleneck_tokens = -(-((k - 1) * g) // period) + 1
        return (bottleneck_tokens + 2 * n) * period + (k - 1) * s

    @property
    @torch.jit.unused
    def causal(self) -> bool:
        """Chosen by the config."""
        return self.config.causal

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        """Calculate the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
