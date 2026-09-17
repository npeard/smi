#!/usr/bin/env python3

"""Shared sliding-window scaffolding for fixed-context regressors.

Some architectures here (BarlandCNN, MLP) consume a fixed-length window and
emit one scalar, yet must satisfy the same
``[batch, channels, length] -> [batch, 1, length]`` contract as the
sequence-to-sequence models. The unfold/fold machinery that bridges the two is
identical for all of them, so it lives here once: subclasses supply only the
per-window head.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class SlidingWindowModel(nn.Module):
    """Applies a fixed-window head across a full sequence.

    Subclasses implement :meth:`_forward_windows`, which maps a batch of
    ``[n_windows, channels, window_size]`` windows to ``[n_windows, 1]``
    predictions. This base class handles extraction, batching and overlap
    averaging.

    Attributes:
        window_size: Number of input samples each prediction is based on.
        window_stride: Hop between successive window starts.
    """

    def __init__(self, window_size: int, window_stride: int):
        super().__init__()
        # Plain ints, not the dataclass config: TorchScript cannot type a
        # dataclass, so forward() must never reach through one.
        self.window_size = window_size
        self.window_stride = window_stride

    def _forward_windows(self, x: Tensor) -> Tensor:
        """Map a batch of windows to one scalar each.

        Args:
            x: [n_windows, channels, window_size]

        Returns:
            [n_windows, 1] predictions.
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """Slide a fixed context window over the full input sequence.

        All windows are extracted simultaneously via functional.unfold and
        processed in a single batched forward pass. Each window produces one
        scalar prediction that is broadcast across all window_size positions it
        covers. Where windows overlap, predictions are averaged
        (functional.fold + count normalization).

        Padding is added on the right only so that the last window is complete
        even when signal_length is not divisible by window_stride.

        Args:
            x: [batch, channels, signal_length]

        Returns:
            [batch, 1, signal_length]
        """
        batch_size, in_channels, signal_length = x.shape
        window_size = self.window_size
        window_stride = self.window_stride

        # Right-pad so an integer number of complete windows covers the signal.
        # num_windows = ceil((signal_length - window_size) / window_stride) + 1
        num_windows = math.ceil(max(signal_length - window_size, 0) / window_stride) + 1
        padded_length = (num_windows - 1) * window_stride + window_size
        pad_right = padded_length - signal_length
        padded = functional.pad(x, (0, pad_right), mode='constant', value=0.0)

        # Extract all strided windows at once via functional.unfold.
        # Result: [batch, in_channels * window_size, num_windows]
        windows = functional.unfold(
            padded.unsqueeze(2), kernel_size=(1, window_size), stride=(1, window_stride)
        )

        # Reshape to [batch * num_windows, in_channels, window_size]
        windows = windows.permute(0, 2, 1).reshape(
            batch_size * num_windows, in_channels, window_size
        )

        # Single batched forward pass -> [batch * num_windows, 1]
        preds = self._forward_windows(windows)

        # Broadcast each scalar prediction over its full window span via
        # functional.fold with kernel_size=window_size. functional.fold sums
        # contributions where windows overlap; dividing by the count averages
        # them.
        # preds: [batch * num_windows, 1] -> [batch, 1 * window_size, num_windows]
        preds = preds.reshape(batch_size, 1, num_windows).expand(
            batch_size, window_size, num_windows
        )

        folded = functional.fold(
            preds,
            output_size=(1, padded_length),
            kernel_size=(1, window_size),
            stride=(1, window_stride),
        )  # [batch, 1, 1, padded_length]

        ones = torch.ones(
            batch_size, window_size, num_windows, device=x.device, dtype=x.dtype
        )
        counts = functional.fold(
            ones,
            output_size=(1, padded_length),
            kernel_size=(1, window_size),
            stride=(1, window_stride),
        )  # [batch, 1, 1, padded_length]

        averaged = folded / counts.clamp(min=1)

        # Remove the height dim and trim to exact signal_length
        return averaged.squeeze(2)[:, :, :signal_length]  # [batch, 1, signal_length]

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
