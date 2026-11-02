"""The chunked Mamba scan matches the recurrence it replaces, forward and backward."""

import pytest
import torch

from smi.learning.models.mamba import selective_scan
from tests._devices import devices
from tests._dtype_guard import no_float64

# float32 eps is 1.19e-7. The sequential float32 loop and the chunked scan
# both land within 3 eps (of the largest magnitude) of a float64 evaluation at
# every length here, forward and for every gradient; 8 eps leaves room for
# the chunked scan's reassociation over log_32(L) levels while any real
# defect (a carry shifted by one block, a dropped padding step) is O(1).
_TOL = 8 * torch.finfo(torch.float32).eps


def _reference_scan(u, delta, A, B, C, D):
    """The recurrence h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t, stepwise."""
    h = torch.zeros(u.shape[0], u.shape[1], A.shape[1], dtype=u.dtype)
    ys = []
    for t in range(u.shape[-1]):
        a_t = torch.exp(delta[:, :, t, None] * A)
        h = a_t * h + delta[:, :, t, None] * B[:, None, :, t] * u[:, :, t, None]
        ys.append((C[:, None, :, t] * h).sum(-1))
    return torch.stack(ys, -1) + D[:, None] * u


def _inputs(length: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(length)
    batch, d_model, d_state = 2, 8, 4
    return [
        torch.randn(batch, d_model, length, generator=g),
        torch.nn.functional.softplus(
            torch.randn(batch, d_model, length, generator=g) - 2
        ),
        -torch.arange(1, d_state + 1, dtype=torch.float32).expand(d_model, -1),
        torch.randn(batch, d_state, length, generator=g),
        torch.randn(batch, d_state, length, generator=g),
        torch.randn(d_model, generator=g),
    ]


def _value_and_grads(fn, xs, weight):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    y = fn(*xs)
    (y * weight).sum().backward()
    return [y.detach().cpu()] + [x.grad.cpu() for x in xs]


@pytest.mark.parametrize('device', devices())
# 7: one partial block; 32/33: exactly one block, and one sample past it;
# 1000: padded, two levels; 4096: three levels.
@pytest.mark.parametrize('length', [7, 32, 33, 256, 1000, 4096])
def test_chunked_scan_matches_recurrence(length: int, device: str):
    xs = _inputs(length)
    weight = torch.randn(2, 8, length, generator=torch.Generator().manual_seed(1))
    expected = _value_and_grads(
        _reference_scan, [x.double() for x in xs], weight.double()
    )
    with no_float64():
        actual = _value_and_grads(
            selective_scan, [x.to(device) for x in xs], weight.to(device)
        )
    names = ['y', 'du', 'ddelta', 'dA', 'dB', 'dC', 'dD']
    for name, got, want in zip(names, actual, expected, strict=True):
        assert got.dtype == torch.float32
        err = (got.double() - want).abs().max() / want.abs().max()
        assert err <= _TOL, f'{name}: max error {err:.2e} of scale > {_TOL:.2e}'
