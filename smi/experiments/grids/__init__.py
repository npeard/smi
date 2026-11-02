"""The shipped experiment grids, one module per grid, each exporting ``GRID``.

A grid's name is its module name: the runner CLI takes it, and its
checkpoints live under ``runs/grids/<name>/``.
"""

import difflib
import importlib
import pkgutil
from collections.abc import Mapping
from types import MappingProxyType

from smi.experiments.grid import Grid

GRIDS: Mapping[str, Grid] = MappingProxyType(
    {
        m.name: importlib.import_module(f'{__name__}.{m.name}').GRID
        for m in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name)
    }
)


def grid(name: str) -> Grid:
    """The shipped grid ``name``.

    Raises:
        KeyError: For an unknown name, listing the closest known ones.
    """
    if name not in GRIDS:
        near = difflib.get_close_matches(name, GRIDS, n=5, cutoff=0.4)
        raise KeyError(f'unknown grid {name!r}; closest: {near}')
    return GRIDS[name]
