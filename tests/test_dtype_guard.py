"""The float64 guard must fire on a deliberate float64 op and pass on float32."""

import pytest
import torch

from tests._dtype_guard import no_float64


def test_guard_passes_float32():
    with no_float64():
        torch.fft.fft(torch.ones(8)) * 2


def test_guard_fires_on_float64_op():
    with pytest.raises(AssertionError, match='float64'), no_float64():
        torch.ones(4).to(torch.float64)


def test_guard_fires_on_complex128_intermediate():
    x = torch.ones(4, dtype=torch.float64)
    with pytest.raises(AssertionError, match='complex128'), no_float64():
        torch.fft.fft(x).to(torch.complex64)
