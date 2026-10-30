#!/usr/bin/env python3
"""Neural operator for per-sample velocity regression, written from scratch.

Our own module, not the ``neuraloperator`` library. Both variants share one
skeleton: a pointwise lift to ``width`` channels, ``n_layers`` operator layers
``h <- GELU(spectral(h) + W h)`` with ``W`` a 1x1 linear path, and a
pointwise two-layer projection to one output channel. Only the spectral
operator differs.

* Offline: a 1-D Fourier neural operator layer. The input is transformed
  over its whole length, the lowest ``modes`` Fourier modes are mixed across
  channels by learned complex weights, and the rest are dropped. Every output
  therefore depends on every input sample: the field is the input, not a
  window, so the model is listed as unbounded context (like the
  bidirectional LSTM) and matched on parameter budget only. Windowing the
  transform to bound the field was not taken: it is not the FNO, and the
  window seams would need their own stitching.
* Causal: truncated modes are non-causal by construction, so the operator
  is instead a learned kernel on lags 0..kernel_len - 1, applied as a long
  causal convolution by FFT, zero-padded to at least L + kernel_len - 1 (no
  circular wrap). The kernel is the inverse real DFT, on a kernel_len-point
  grid, of ``modes`` complex coefficients per channel pair: a band-limited
  kernel, so the parameter count is set by ``modes`` and not by the kernel
  length. Its frequency response at the grid bins is those coefficients, the
  same meaning the offline weights have.

The two variants have identical parameter counts at equal ``width``,
``n_layers`` and ``modes``.
"""

import logging
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

logger = logging.getLogger(__name__)


@dataclass
class NeuralOperatorConfig:
    """Configuration for the neural operator.

    Args:
        sequence_length: Nominal input length (unused by the network, which
            accepts any length; kept for parity with the other configs).
        in_channels: Number of input channels.
        width: Channels of the lifted representation.
        n_layers: Operator layers.
        modes: Complex Fourier coefficients per channel pair and layer: the
            retained modes of the input (offline) or of the kernel (causal).
        kernel_len: Kernel support in samples, lags 0..kernel_len - 1, when
            causal; None when offline, whose operator spans the whole input.
        causal: True for the causal-convolution operator.
    """

    sequence_length: int
    in_channels: int
    width: int
    n_layers: int
    modes: int
    kernel_len: int | None
    causal: bool

    def __post_init__(self):
        """Validate what the operator and its receptive field rely on.

        Raises:
            ValueError: If ``kernel_len`` disagrees with ``causal``, if the
                depth or modes are below one, or if a causal kernel grid has
                fewer than ``modes`` non-Nyquist bins (the excess coefficients
                would never reach the kernel).
        """
        if self.n_layers < 1 or self.modes < 1:
            raise ValueError('n_layers and modes must both be >= 1')
        if self.causal != (self.kernel_len is not None):
            raise ValueError(
                'kernel_len must be set exactly when causal (got '
                f'kernel_len={self.kernel_len}, causal={self.causal})'
            )
        if self.kernel_len is not None and self.modes > (self.kernel_len + 1) // 2:
            raise ValueError(
                f'modes {self.modes} exceeds the {(self.kernel_len + 1) // 2} '
                f'non-Nyquist bins of a {self.kernel_len}-lag kernel'
            )


def _fast_fft_size(n: int) -> int:
    """Smallest power of two >= n.

    Any size >= L + kernel_len - 1 gives the linear convolution; the minimum
    is often a product of large primes (4096 + 21 = 23 * 179), which made the
    causal step 10x the offline one on MPS.
    """
    size = 1
    while size < n:
        size *= 2
    return size


class SpectralConv1d(nn.Module):
    """Truncated-mode spectral convolution over the whole input (FNO layer).

    Weights are stored as real tensors, and mode 0 has no imaginary part: a
    real signal's DC coefficient is real, so its imaginary weight would only
    ever be discarded by the inverse transform -- a parameter counted in the
    budget that cannot learn.
    """

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        # Unit gain per retained mode at init: channel mixing sums `width`
        # terms, so each weight has variance 1 / width.
        scale = 1.0 / math.sqrt(width)
        self.real = nn.Parameter(scale * torch.randn(width, width, modes))
        self.imag = nn.Parameter(scale * torch.randn(width, width, modes - 1))

    def weights(self) -> Tensor:
        """Complex weights [in, out, modes], complex64 for float32 storage."""
        return torch.complex(self.real, functional.pad(self.imag, (1, 0)))

    def forward(self, x: Tensor) -> Tensor:
        """Map [batch, width, length] to the same shape.

        An input shorter than ``2 * modes - 2`` samples has only ``L // 2 + 1``
        frequency bins, so the retained modes are silently truncated to those.
        The bins past them do not exist, so nothing is lost and the shape and
        finiteness contract holds for any length; the weights for the missing
        modes simply get no gradient.
        """
        length = x.shape[-1]
        modes = min(self.modes, length // 2 + 1)
        x_hat = torch.fft.rfft(x)[..., :modes]
        y_hat = torch.einsum('bim,iom->bom', x_hat, self.weights()[..., :modes])
        # irfft zero-fills the truncated modes up to length // 2 + 1.
        return torch.fft.irfft(y_hat, n=length)


class CausalSpectralConv1d(SpectralConv1d):
    """Causal convolution with a band-limited kernel on lags 0..kernel_len - 1.

    Any FFT size >= L + kernel_len - 1 makes the circular product equal the
    linear convolution, so the kernel's tail never wraps onto the first
    samples and output t sees only inputs t - kernel_len + 1 .. t. That is
    exact in exact arithmetic; in float32 the transform spreads round-off of
    ~1e-7 of the peak to every sample, the floor the tests' gradient and step
    probes allow for.
    """

    def __init__(self, width: int, modes: int, kernel_len: int):
        super().__init__(width, modes)
        self.kernel_len = kernel_len

    def kernel(self) -> Tensor:
        """Real kernel [in, out, kernel_len], indexed by lag."""
        return torch.fft.irfft(self.weights(), n=self.kernel_len)

    def forward(self, x: Tensor) -> Tensor:
        """Map [batch, width, length] to the same shape, causally."""
        length = x.shape[-1]
        n = _fast_fft_size(length + self.kernel_len - 1)
        x_hat = torch.fft.rfft(x, n=n)
        k_hat = torch.fft.rfft(self.kernel(), n=n)
        y = torch.fft.irfft(torch.einsum('bif,iof->bof', x_hat, k_hat), n=n)
        return y[..., :length]


class NeuralOperatorLayer(nn.Module):
    """One operator layer: GELU(spectral(h) + W h)."""

    def __init__(self, width: int, modes: int, kernel_len: int | None):
        super().__init__()
        self.spectral = (
            SpectralConv1d(width, modes)
            if kernel_len is None
            else CausalSpectralConv1d(width, modes, kernel_len)
        )
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Map [batch, width, length] to the same shape."""
        return functional.gelu(self.spectral(x) + self.pointwise(x))


class NeuralOperator(nn.Module):
    """Lift, stacked operator layers, project; see the module docstring."""

    def __init__(self, config: NeuralOperatorConfig):
        super().__init__()
        self.config = config
        w = config.width
        self.lift = nn.Conv1d(config.in_channels, w, 1)
        self.layers = nn.ModuleList(
            [
                NeuralOperatorLayer(w, config.modes, config.kernel_len)
                for _ in range(config.n_layers)
            ]
        )
        self.projection = nn.Sequential(
            nn.Conv1d(w, w, 1), nn.GELU(), nn.Conv1d(w, 1, 1)
        )

        logger.info(
            f'Number of parameters in {self.__class__.__name__}: {self.total_params:,}'
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map [batch, in_channels, length] to [batch, 1, length]."""
        h = self.lift(x)
        for layer in self.layers:
            h = layer(h)
        return self.projection(h)

    @property
    @torch.jit.unused
    def receptive_field(self) -> int | None:
        """One-sided lookback in input samples when causal; None when offline.

        Causal: each layer's kernel covers lags 0..K - 1 and every other
        operation is pointwise, so n_layers stacked layers reach back
        n_layers (K - 1) samples, n_layers (K - 1) + 1 inclusive of t -- 85
        for the matched causal configs (4 layers, K = 22). The kernel span of
        the whole network, not of one layer, is what equals the matched
        lookback.

        Offline: None, unbounded -- the spectral convolution spans the whole
        input, so the measured field is the input length (and grows with it).
        ``tests/test_receptive_field.py`` checks both against a gradient probe.
        """
        c = self.config
        if c.kernel_len is None:
            return None
        return c.n_layers * (c.kernel_len - 1) + 1

    @property
    @torch.jit.unused
    def total_params(self) -> int:
        """Calculate the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
