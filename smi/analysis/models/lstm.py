#!/usr/bin/env python3

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    """Configuration for LSTM model.

    A standard stacked LSTM that processes multi-channel interferometer sequences
    and outputs a single-channel velocity/displacement prediction at each time step.

    Args:
        sequence_length: Length of input sequence (time dimension)
        in_channels: Number of input channels (e.g., 3 for interferometer data)
        hidden_size: Number of features in the LSTM hidden state
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability between LSTM layers (applied when num_layers > 1)
        bidirectional: If True, use a bidirectional LSTM (non-causal)
    """

    sequence_length: int
    in_channels: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool

    def __post_init__(self):
        if self.num_layers < 1:
            raise ValueError(f'num_layers must be >= 1, got {self.num_layers}')
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f'dropout must be in [0, 1), got {self.dropout}')


class LSTM(nn.Module):
    """Stacked LSTM for temporal sequence-to-sequence prediction.

    Takes multi-channel input of shape [batch, in_channels, seq_len] and produces
    output of shape [batch, 1, seq_len], matching the interface expected by LitModule.

    The LSTM processes the sequence along the time axis with all input channels
    presented as features at each time step.
    """

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.in_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        lstm_out_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.projection = nn.Linear(lstm_out_size, 1)

        logger.info(f'Number of parameters in LSTM: {self.total_params:,}')

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the LSTM.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Tensor of shape [batch_size, 1, sequence_length] containing
            predicted velocity
        """
        # x: [batch, in_channels, seq_len] -> [batch, seq_len, in_channels]
        x = x.permute(0, 2, 1)

        # lstm_out: [batch, seq_len, hidden_size * num_directions]
        lstm_out, _ = self.lstm(x)

        # project to 1 output per time step: [batch, seq_len, 1]
        out = self.projection(lstm_out)

        # [batch, 1, seq_len]
        return out.permute(0, 2, 1)

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        """Calculate the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
