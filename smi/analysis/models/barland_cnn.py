#!/usr/bin/env python3

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .sliding_window import SlidingWindowModel

act_fn_by_name = {'LeakyReLU': nn.LeakyReLU(), 'Tanh': nn.Tanh()}


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


class BarlandCNN(SlidingWindowModel):
    def __init__(self, config: BarlandCNNConfig):
        super().__init__(config.window_size, config.window_stride)
        self.config = config
        self.in_channels = config.in_channels
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.in_channels, 16, kernel_size=7),
            # Lout = 250, given L = 256
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 125, given L = 250
            nn.Conv1d(16, 32, kernel_size=7),
            # Lout = 119, given L = 125
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 59, given L = 119
            nn.Conv1d(32, 64, kernel_size=7),
            # Lout = 53, given L = 59
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 26, given L = 53
            nn.Dropout(config.dropout),
            nn.Conv1d(64, 64, kernel_size=7),
            # Lout = 20, given L = 26
            act_fn_by_name[config.activation],
            nn.MaxPool1d(2),
            # Lout = 10, given L = 20
        )
        self.fc_layers = nn.Sequential(
            # length of input = 64 filters * 10 remaining time steps
            nn.Linear(640, 16),
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
        """Receptive field of the CNN in input samples.

        4 conv layers with kernel_size=7, each followed by MaxPool1d(2).
        Pooling doubles the effective stride for subsequent layers, so each
        layer contributes (kernel_size - 1) * 2^i samples of context:
            layer 0: 6 * 1 =  6
            layer 1: 6 * 2 = 12
            layer 2: 6 * 4 = 24
            layer 3: 6 * 8 = 48
        Total: 6 + 12 + 24 + 48 + 1 = 91
        """
        return 91

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
