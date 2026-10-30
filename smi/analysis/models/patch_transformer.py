#!/usr/bin/env python3
"""Patch Transformer for per-sample velocity regression.

The signal is cut into patches of ``patch_len`` samples every ``stride``
samples (the PatchTST tokenisation), each patch becomes one token, and a stack
of pre-norm Transformer layers mixes tokens. Attention is local: each token
sees ``attention_span`` tokens on each attended side, so the receptive field is
finite and can be matched to the convolutional families'. Global attention
over a 4096-sample crop would give a field set by the crop, not the model.

The per-sample output at ``t`` combines the token whose stride cell holds
``t`` (repeated across the cell), a learned embedding of ``t``'s phase inside
the cell, and a short convolution over the raw samples around ``t``. In the
causal variant every one of those is built from samples at or before ``t``.
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

logger = logging.getLogger(__name__)


@dataclass
class PatchTransformerConfig:
    """Configuration for the patch Transformer.

    Args:
        sequence_length: Nominal input length (unused by the network, which
            accepts any length; kept for parity with the other configs).
        in_channels: Number of input channels.
        patch_len: Samples per patch (token).
        stride: Samples between consecutive patch starts; one token per
            stride cell. Must not exceed ``patch_len``.
        d_model: Token width.
        n_heads: Attention heads; must divide ``d_model``.
        n_layers: Transformer layers.
        ffn_dim: Hidden width of each layer's feed-forward network.
        attention_span: Tokens attended on each side of a token (past side
            only when causal), excluding the token itself.
        dropout: Dropout on attention weights and residual branches.
        causal: True for past-only patches, attention and output path.
    """

    sequence_length: int
    in_channels: int
    patch_len: int
    stride: int
    d_model: int
    n_heads: int
    n_layers: int
    ffn_dim: int
    attention_span: int
    dropout: float
    causal: bool

    def __post_init__(self):
        """Validate the geometry the receptive-field formula relies on.

        Raises:
            ValueError: If the stride outruns the patch (samples between
                patches would reach no token, so the field would have gaps
                and the property would over-report), if the span or depth is
                below one, or if the heads do not divide the width.
        """
        if self.stride > self.patch_len:
            raise ValueError(
                f'stride {self.stride} exceeds patch_len {self.patch_len}: '
                'samples between patches would reach no token.'
            )
        if self.attention_span < 1 or self.n_layers < 1:
            raise ValueError('attention_span and n_layers must both be >= 1')
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f'd_model {self.d_model} is not divisible by n_heads {self.n_heads}'
            )


class LocalSelfAttention(nn.Module):
    """Multi-head self-attention restricted to a band of nearby tokens.

    Keys and values are gathered into a ``[left + 1 + right]``-wide window per
    query with ``unfold``, so cost and memory are linear in the token count and
    no token outside the band can contribute, by construction rather than by a
    mask over a full score matrix. Position enters only as a learned per-head
    bias per window offset: relative, so the model is shift-equivariant in
    whole strides, and confined to the band, so it cannot leak.
    """

    def __init__(
        self, d_model: int, n_heads: int, left: int, right: int, dropout: float
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.left = left
        self.right = right
        self.window = left + 1 + right
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, self.window))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Attend within the band.

        Args:
            x: [batch, tokens, d_model]

        Returns:
            Tensor of the same shape.
        """
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, tokens, head_dim]
        q = qkv[0]
        # Window slot w of token i holds token i + w - left.
        k = functional.pad(qkv[1], (0, 0, self.left, self.right)).unfold(
            2, self.window, 1
        )  # [batch, heads, tokens, head_dim, window]
        v = functional.pad(qkv[2], (0, 0, self.left, self.right)).unfold(
            2, self.window, 1
        )
        scores = torch.einsum('bhtd,bhtdw->bhtw', q, k) / math.sqrt(self.head_dim)
        scores = scores + self.rel_bias[None, :, None, :]
        # Slots that fall in the zero padding are not tokens; slot `left` is
        # the token itself, so no row is ever fully masked.
        key_pos = (
            torch.arange(t, device=x.device)[:, None]
            + torch.arange(self.window, device=x.device)[None, :]
            - self.left
        )
        outside = (key_pos < 0) | (key_pos >= t)
        scores = scores.masked_fill(outside, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        y = torch.einsum('bhtw,bhtdw->bhtd', attn, v)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class TransformerLayer(nn.Module):
    """Pre-norm Transformer layer around :class:`LocalSelfAttention`."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        left: int,
        right: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = LocalSelfAttention(d_model, n_heads, left, right, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention then feed-forward, each as a residual branch."""
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class PatchTransformer(nn.Module):
    """Patch-token Transformer with banded attention and per-sample output.

    Geometry, with P = patch_len, S = stride, N = n_layers * attention_span.
    Token k owns the stride cell [kS, kS + S).

    * Causal: token k's patch is [kS - P + 1, kS], so it is complete at the
      first sample of its cell; attention reaches tokens k - N .. k; the local
      convolution covers [t - S + 1, t].
    * Offline: token k's patch is centred on its cell, starting at
      kS - (P - S) // 2; attention reaches k - N .. k + N; the local
      convolution is centred on t.
    """

    def __init__(self, config: PatchTransformerConfig):
        super().__init__()
        self.config = config
        self.patch_len = config.patch_len
        self.stride = config.stride
        p, s = config.patch_len, config.stride
        span = config.attention_span
        if config.causal:
            self.patch_left = p - 1
            self.local_pad = (s - 1, 0)
            left, right = span, 0
        else:
            self.patch_left = (p - s) // 2
            self.local_pad = ((s - 1) // 2, s // 2)
            left, right = span, span

        self.embed = nn.Conv1d(config.in_channels, config.d_model, p, stride=s)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    config.d_model,
                    config.n_heads,
                    config.ffn_dim,
                    left,
                    right,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        # A token stands for its whole cell, so the head needs to know where
        # in the cell t sits, and -- in the causal variant, where the token
        # stops at the cell's first sample -- what arrived since.
        self.phase = nn.Parameter(torch.zeros(s, config.d_model))
        self.local = nn.Conv1d(config.in_channels, config.d_model, s)
        self.head_norm = nn.LayerNorm(config.d_model)
        self.projection = nn.Linear(config.d_model, 1)

        logger.info(
            f'Number of parameters in {self.__class__.__name__}: {self.total_params:,}'
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map [batch, in_channels, length] to [batch, 1, length]."""
        length = x.shape[-1]
        n_tokens = (length + self.stride - 1) // self.stride
        # Right padding only completes the last token; in the causal variant
        # the left padding already provides it, hence the clamp at zero.
        right = max(
            (n_tokens - 1) * self.stride + self.patch_len - length - self.patch_left, 0
        )
        tokens = self.embed(functional.pad(x, (self.patch_left, right)))
        h = tokens.transpose(1, 2)  # [batch, tokens, d_model]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        per_sample = h.repeat_interleave(self.stride, dim=1)[:, :length]
        phase = self.phase.repeat(n_tokens, 1)[:length]
        local = self.local(functional.pad(x, self.local_pad)).transpose(1, 2)
        z = functional.gelu(self.head_norm(per_sample + phase + local))
        return self.projection(z).transpose(1, 2)

    @property
    @torch.jit.unused
    def receptive_field(self) -> int:
        """Receptive field in input samples: worst-case lookback or total width.

        The field is periodic in the stride, not constant, so the reported
        value is its maximum over the phase of t in its cell -- the context a
        sample can actually draw on, which is what the matched comparison
        holds fixed.

        Causal, t = kS + r with 0 <= r < S: the oldest sample reached is the
        start of token k - N's patch, (k - N)S - P + 1, so the lookback
        (inclusive of t) is NS + r + P: smallest at r = 0, NS + P, and
        largest at r = S - 1, (N + 1)S + P - 1 -- 76 and 85 for the matched
        causal configs (N = 6, S = 10, P = 16). Offline: tokens
        k - N .. k + N cover 2NS + P contiguous samples whatever r is, and
        the local convolution stays inside them because N >= 1. Both rely
        on stride <= patch_len, which the config enforces;
        ``tests/test_receptive_field.py`` checks them against a gradient
        probe swept over the phase.
        """
        c = self.config
        n = c.n_layers * c.attention_span
        if c.causal:
            return (n + 1) * c.stride + c.patch_len - 1
        return 2 * n * c.stride + c.patch_len

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        """Calculate the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
