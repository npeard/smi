#!/usr/bin/env python3

from torch import Tensor, nn

from .chomp1d import Chomp1d
from .transpose import Transpose

act_fn_by_name = {
    'LeakyReLU': nn.LeakyReLU(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
}


class TemporalBlock(nn.Module):
    """Single block of temporal convolutions with dilation.

    When causal=True (default), left-only padding + Chomp1d ensures the model
    cannot look ahead in time.  When causal=False, symmetric padding is used
    instead, giving the block a symmetric receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        padding: int,
        activation: str,
        use_layer_norm: bool,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        if causal:
            # Left-only padding: pad the full amount on the left, then chomp right
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.chomp1 = Chomp1d(padding)
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
            self.chomp2 = Chomp1d(padding)
        else:
            # Symmetric padding: use half the causal padding on each side
            sym_padding = padding // 2
            self.conv1 = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=sym_padding,
                dilation=dilation,
            )
            self.chomp1 = nn.Identity()
            self.conv2 = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=sym_padding,
                dilation=dilation,
            )
            self.chomp2 = nn.Identity()

        # Select normalization layer
        if use_layer_norm:
            # LayerNorm across channel dimension requires transpose
            self.norm = nn.Sequential(
                Transpose(1, 2), nn.LayerNorm(in_channels), Transpose(1, 2)
            )
        else:
            self.norm = nn.Identity()

        # Select activation function
        if activation in act_fn_by_name:
            self.activation_fn = act_fn_by_name[activation]
        else:
            raise ValueError(f'Unknown activation: {activation}')

        self.dropout = nn.Dropout1d(dropout)

        self.net = nn.Sequential(
            self.norm,
            self.conv1,
            self.chomp1,
            self.activation_fn,
            self.dropout,
            self.conv2,
            self.chomp2,
            self.activation_fn,
            self.dropout,
        )

        # 1x1 convolution for residual connection if input and output channels differ
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.activation = self.activation_fn

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)
