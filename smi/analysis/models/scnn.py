#!/usr/bin/env python3

from dataclasses import dataclass

from .tcn import TCN, TCNConfig


@dataclass
class SCNNConfig(TCNConfig):
    """Configuration for the non-causal Sequence CNN (SCNN).

    SCNN is identical to TCN except that every TemporalBlock uses symmetric
    (non-causal) padding, giving the model a symmetric receptive field and
    access to future context (lookahead).  Intended as a direct comparison to
    TCN to measure the benefit of causal constraints.

    All parameters are inherited from TCNConfig:
        sequence_length, in_channels, activation, use_layer_norm, use_weight_norm,
        kernel_size, temporal_channels, dilation_base
    """


class SCNN(TCN):
    """Non-causal Sequence CNN with symmetric receptive field.

    Identical architecture to TCN but uses symmetric padding in every
    TemporalBlock, giving the model access to future context (lookahead).
    """

    _causal: bool = False
