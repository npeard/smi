#!/usr/bin/env python3

from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention


class CrossAttentionBlock(nn.Module):
    """Cross-channel attention block for multi-channel time series.

    At each time step, the ``n_channels`` feature vectors (each of size
    ``embed_dim``) are treated as a small sequence of length ``n_channels``
    and standard (unmasked) multi-head self-attention is applied across them.
    This allows channels to exchange information at the same point in time
    without any temporal lookahead, preserving causality.

    Input / output shape: [batch * seq_len, n_channels, embed_dim]
    The calling code in TCAN reshapes to/from this layout.
    """

    def __init__(self, n_channels: int, embed_dim: int, num_heads: int, dropout: float):
        """Initialize cross-channel attention block.

        Args:
            n_channels: Number of input channels (sequence length for attention)
            embed_dim: Embedding dimension per channel
            num_heads: Number of attention heads (must divide embed_dim)
            dropout: Dropout probability
        """
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, (
            f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})'
        )

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Apply cross-channel attention at each time step.

        Args:
            x: [batch * seq_len, n_channels, embed_dim]

        Returns:
            Output tensor of same shape as input
        """
        bs, n_ch, ed = x.shape  # bs = batch * seq_len
        head_dim = ed // self.num_heads

        Q = self.query_proj(x)  # [bs, n_ch, ed]
        K = self.key_proj(x)
        V = self.value_proj(x)

        # Reshape for multi-head: [bs, num_heads, n_ch, head_dim]
        Q = Q.reshape(bs, n_ch, self.num_heads, head_dim).transpose(1, 2)
        K = K.reshape(bs, n_ch, self.num_heads, head_dim).transpose(1, 2)
        V = V.reshape(bs, n_ch, self.num_heads, head_dim).transpose(1, 2)

        attn_out = scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout.p if self.training else 0.0
        )  # [bs, num_heads, n_ch, head_dim]

        attn_out = attn_out.transpose(1, 2).reshape(bs, n_ch, ed)

        output = self.out_proj(attn_out)
        output = self.dropout(output)

        return x + output
