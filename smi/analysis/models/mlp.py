#!/usr/bin/env python3

"""Windowed multilayer perceptron baseline.

The simplest learned model worth putting in the comparison table. It is
windowed exactly as BarlandCNN is -- a fixed span of samples in, one velocity
out, slid across the signal -- so the number it produces answers
"architecture vs architecture" rather than "this model saw the whole sequence
and the others did not".
"""

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .sliding_window import SlidingWindowModel

logger = logging.getLogger(__name__)

act_fn_by_name = {
    'GELU': nn.GELU,
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'Tanh': nn.Tanh,
}


@dataclass
class MLPConfig:
    """Configuration for the windowed MLP baseline.

    Args:
        window_size: Number of input samples per prediction.
        in_channels: Number of input channels.
        hidden_sizes: Width of each hidden layer, in order.
        activation: Key into ``act_fn_by_name``.
        dropout: Dropout probability applied after each hidden activation.
        window_stride: Hop between successive window starts. A stride below
            window_size makes neighbouring windows overlap, and their
            predictions are averaged.
    """

    window_size: int
    in_channels: int
    hidden_sizes: list[int]
    activation: str
    dropout: float
    window_stride: int

    def __post_init__(self):
        if not self.hidden_sizes:
            raise ValueError('hidden_sizes must contain at least one layer')
        if self.window_stride < 1 or self.window_stride > self.window_size:
            raise ValueError(
                f'window_stride must be in [1, window_size], got '
                f'{self.window_stride} with window_size {self.window_size}'
            )
        if self.activation not in act_fn_by_name:
            raise ValueError(f'Unknown activation: {self.activation}')
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f'dropout must be in [0, 1), got {self.dropout}')


class MLP(SlidingWindowModel):
    """Fully connected network over a flattened fixed-length window.

    Input ``[batch, in_channels, signal_length]`` -> output
    ``[batch, 1, signal_length]``, via the shared sliding-window machinery in
    :class:`SlidingWindowModel`.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config.window_size, config.window_stride)
        self.config = config
        self.in_channels = config.in_channels

        activation = act_fn_by_name[config.activation]
        layers: list[nn.Module] = []
        in_features = config.in_channels * config.window_size
        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(activation())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights(config)

        logger.info(f'Number of parameters in MLP: {self.total_params:,}')

    def _initialize_weights(self, config: MLPConfig) -> None:
        """Initialize linear weights to suit the chosen activation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if config.activation in ('ReLU', 'LeakyReLU'):
                    nn.init.kaiming_normal_(
                        module.weight, mode='fan_in', nonlinearity='relu'
                    )
                else:  # GELU, Tanh and anything else roughly linear at 0
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    @torch.jit.unused
    def receptive_field(self) -> int:
        """Receptive field in input samples.

        Every output position is produced by a dense layer over the whole
        window, so the window is the receptive field exactly.
        """
        return self.window_size

    def _forward_windows(self, x: Tensor) -> Tensor:
        """Run a batch of windows through the dense stack.

        Args:
            x: [n_windows, channels, window_size]

        Returns:
            [n_windows, 1] predictions.
        """
        return self.layers(x.reshape(x.size(0), -1))
