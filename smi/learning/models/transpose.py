#!/usr/bin/env python3

from torch import Tensor, nn


class Transpose(nn.Module):
    """Simple transpose module used to swap channel and sequence dimensions
    in TemporalBlock.
    """

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)
