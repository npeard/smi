"""Test helper: fail on any float64/complex128 tensor produced by a torch op.

MPS has no float64, and a final ``.to(float32)`` hides promotion that happened
inside a function, so checking only outputs is not enough. A dispatch mode sees
every op's result, including intermediates.
"""

from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

_FORBIDDEN = (torch.float64, torch.complex128)


class _NoFloat64Mode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(out):
            if isinstance(leaf, torch.Tensor) and leaf.dtype in _FORBIDDEN:
                raise AssertionError(f'{func} produced a {leaf.dtype} tensor')
        return out


@contextmanager
def no_float64():
    """Raise AssertionError naming the first op that outputs float64/complex128."""
    with _NoFloat64Mode():
        yield
