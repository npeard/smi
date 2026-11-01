#!/usr/bin/env python3

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .sliding_window import SlidingWindowModel

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'Tanh': nn.Tanh()}

# The published geometry: every conv layer has this kernel and is followed by
# a max-pool of this size, and the dense head reads this many pooled positions.
KERNEL_SIZE = 7
POOL_SIZE = 2
HEAD_POSITIONS = 10


def conv_position_field(n_layers: int) -> int:
    """Samples one conv-stack position spans, counting kernels only.

    Layer ``i`` (from 0) sees its kernel at stride ``POOL_SIZE**i``; each pool
    is counted as pure subsampling, so its own window adds nothing (91 for the
    four-layer published network, where the pools' windows would make it 98).
    """
    return 1 + (KERNEL_SIZE - 1) * sum(POOL_SIZE**i for i in range(n_layers))


@dataclass
class BarlandCNNConfig:
    # Common parameters (consistent across all model configs)
    window_size: int  # length of each input window (256)
    in_channels: int
    activation: str
    dropout: float

    use_weight_norm: bool

    # BarlandCNN specific parameters
    window_stride: int  # controls the sliding window stride in the forward pass
    # Output widths of the four conv layers; [16, 32, 64, 64] is the published
    # network (57,153 parameters at one channel). Exposed so the model can be
    # held to the matched parameter budgets without altering its depth, kernel
    # sizes or receptive field.
    channels: list[int]


class BarlandCNN(SlidingWindowModel):
    def __init__(self, config: BarlandCNNConfig):
        super().__init__(config.window_size, config.window_stride)
        self.config = config
        self.in_channels = config.in_channels
        c1, c2, c3, c4 = config.channels
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.in_channels, c1, kernel_size=KERNEL_SIZE),
            # Lout = 250, given L = 256
            act_fn_by_name[config.activation],
            nn.MaxPool1d(POOL_SIZE),
            # Lout = 125, given L = 250
            nn.Conv1d(c1, c2, kernel_size=KERNEL_SIZE),
            # Lout = 119, given L = 125
            act_fn_by_name[config.activation],
            nn.MaxPool1d(POOL_SIZE),
            # Lout = 59, given L = 119
            nn.Conv1d(c2, c3, kernel_size=KERNEL_SIZE),
            # Lout = 53, given L = 59
            act_fn_by_name[config.activation],
            nn.MaxPool1d(POOL_SIZE),
            # Lout = 26, given L = 53
            nn.Dropout(config.dropout),
            nn.Conv1d(c3, c4, kernel_size=KERNEL_SIZE),
            # Lout = 20, given L = 26
            act_fn_by_name[config.activation],
            nn.MaxPool1d(POOL_SIZE),
            # Lout = 10, given L = 20
        )
        self.fc_layers = nn.Sequential(
            # length of input = c4 filters * HEAD_POSITIONS remaining time steps
            nn.Linear(c4 * HEAD_POSITIONS, 16),
            act_fn_by_name[config.activation],
            nn.Linear(16, 1),
        )

        self._initialize_weights(config)

    def _initialize_weights(self, config: BarlandCNNConfig) -> None:
        """Initialize model weights based on configuration."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                if config.activation in ['ReLU', 'LeakyReLU']:
                    nn.init.kaiming_normal_(
                        module.weight, mode='fan_out', nonlinearity='relu'
                    )
                elif config.activation == 'Tanh':
                    nn.init.xavier_normal_(module.weight)
                else:  # GELU and others
                    nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

                if config.use_weight_norm:
                    nn.utils.parametrizations.weight_norm(module)

    @property
    @torch.jit.unused
    def receptive_field(self) -> int:
        """Receptive field in input samples.

        The head reads HEAD_POSITIONS pooled positions, and walking them back
        through each conv + pool layer (a pool doubles the samples, a conv
        adds kernel_size - 1) gives the samples one window's prediction
        depends on: 250 of the published 256, the pooling floors dropping the
        tail. The dense head combines the conv positions (``conv_position_field``
        spans each alone), and the overlap average widens the window's field to
        every covering window.
        """
        window_field = HEAD_POSITIONS
        for _ in self.config.channels:
            window_field = POOL_SIZE * window_field + KERNEL_SIZE - 1
        return self.averaged_span(window_field)

    def _forward_windows(self, x: Tensor) -> Tensor:
        """Run a batch of windows through the CNN.

        Args:
            x: [n_windows, channels, sequence_length]

        Returns:
            [n_windows, 1] predictions
        """
        out = self.conv_layers(x)  # [n_windows, 64, 10]
        out = out.view(out.size(0), 1, -1)  # [n_windows, 1, 640]
        return self.fc_layers(out).squeeze(-1)  # [n_windows, 1]
