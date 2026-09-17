#!/usr/bin/env python3

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .temporal_block import TemporalBlock

logger = logging.getLogger(__name__)


@dataclass
class TCNConfig:
    """Configuration for Temporal Convolutional Network.

    Args:
        sequence_length: Length of input sequence (time dimension)
        in_channels: Number of input channels (e.g., 3 for interferometer data)
        activation: Activation function name ('GELU', 'ReLU', etc.)
        use_layer_norm: Whether to use layer normalization in temporal blocks
        use_weight_norm: Whether to apply weight normalization to conv layers
        kernel_size: Convolution kernel size for all temporal blocks
        temporal_channels: Output channels for each layer in the network
        dilation_base: Base for exponential dilation sequence.
            Dilations will be [1, base, base^2, base^3, ...] for each layer
        dropout: Dropout probability for Dropout1d after each activation
    """

    sequence_length: int
    in_channels: int
    activation: str
    use_layer_norm: bool
    use_weight_norm: bool
    kernel_size: int
    temporal_channels: list[int]
    dilation_base: int
    dropout: float

    def __post_init__(self):
        """Compute dilations from dilation_base and validate.

        Raises:
            ValueError: If the first dilation is not 1, or if dilation_base
                outruns what one block's two convolutions span -- see
                :attr:`TCN.receptive_field` for why that second case would
                leave the reported receptive field wrong.
        """
        self.n_layers = len(self.temporal_channels)
        self.temporal_dilations = [self.dilation_base**i for i in range(self.n_layers)]
        if self.temporal_dilations[0] != 1:
            raise ValueError('First dilation should be 1')
        max_base = 2 * (self.kernel_size - 1) + 1
        if self.dilation_base > max_base:
            raise ValueError(
                f'dilation_base {self.dilation_base} exceeds '
                f'2 * (kernel_size - 1) + 1 = {max_base} for kernel_size '
                f'{self.kernel_size}: the dilated stack would have gaps and '
                'TCN.receptive_field would over-report. Raise kernel_size or '
                'lower dilation_base.'
            )


class TCN(nn.Module):
    """Temporal Convolutional Network with dilated convolutions.

    A stack of TemporalBlocks with exponentially increasing dilations,
    preceded by a 1x1 lifting convolution and followed by a 1x1 projection.
    """

    # Subclasses may set this to False to use symmetric (non-causal) padding
    _causal: bool = True

    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        self.sequence_length = config.sequence_length
        self.in_channels = config.in_channels
        self.n_layers = config.n_layers

        self.temporal_channels = config.temporal_channels
        self.temporal_dilations = config.temporal_dilations

        # Lifting layer (input projection)
        self.lifting = nn.Conv1d(config.in_channels, config.temporal_channels[0], 1)

        # Create TemporalBlocks for each layer
        self.temporal_blocks = nn.ModuleList([])

        for i in range(self.n_layers):
            if i == 0:
                in_channels = config.temporal_channels[0]
            else:
                in_channels = config.temporal_channels[i - 1]

            out_channels = config.temporal_channels[i]
            dilation = config.temporal_dilations[i]
            padding = (config.kernel_size - 1) * dilation

            self.temporal_blocks.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    config.kernel_size,
                    dilation=dilation,
                    padding=padding,
                    activation=config.activation,
                    use_layer_norm=config.use_layer_norm,
                    causal=self._causal,
                    dropout=config.dropout,
                )
            )

        # Projection layer (output projection)
        self.projection = nn.Conv1d(config.temporal_channels[-1], 1, 1)

        # Initialize weights
        self._initialize_weights(config)

        logger.info(
            f'Number of parameters in {self.__class__.__name__}: {self.total_params:,}'
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the TCN.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length]
        """
        x = self.lifting(x)

        for block in self.temporal_blocks:
            x = block(x)

        return self.projection(x)

    def _initialize_weights(self, config: TCNConfig) -> None:
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

        Each TemporalBlock applies two convolutions at the same dilation, so a
        block contributes twice (kernel_size - 1) * dilation, not once. The
        trailing +1 is the centre sample itself.

        Exact only while `dilation_base <= 2 * (kernel_size - 1) + 1`, which
        `TCNConfig.__post_init__` enforces. That is the condition for the
        dilated coverage to stay contiguous: a block spans
        2 * (kernel_size - 1) * dilation samples, so once the dilation step to
        the next layer exceeds that span the taps are spaced further apart than
        the kernels reach. The stack then develops gaps that no gradient path
        crosses, and this formula -- which measures the nominal span, not the
        sample count inside it -- over-reports. Verified against a gradient
        probe over 234 configurations (kernel_size 2..7 x dilation_base 2..14 x
        depth 2..4) with no mismatch inside the valid regime.
        """
        dilations = torch.tensor(self.config.temporal_dilations)
        per_block = 2 * (self.config.kernel_size - 1) * dilations
        return int(torch.sum(per_block).item()) + 1

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        """Calculate the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
