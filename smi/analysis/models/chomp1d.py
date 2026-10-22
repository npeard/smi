#!/usr/bin/env python3

from torch import Tensor, nn


class Chomp1d(nn.Module):
    """Remove padding at the end of the sequence."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, : -self.chomp_size].contiguous()
