"""Feature registry for per-shot waveform data.

Each acquisition "shot" is a dict of raw channel arrays (``RP1_CH1``,
``RP1_CH2``, ``RP2_CH1``, ``RP2_CH2``) plus the ``sample_rate``. A *feature* is a
named transform from that raw shot to a 1-D float32 array of the same length,
tagged as an ``'input'`` (fed to the model) or a ``'target'`` (supervision
signal).

This mirrors the registry pattern from the coffee project
(``register_feature`` decorator + module-level ``default_registry``), adapted
from scalar tabular rows to per-shot arrays. There is no complex feature
engineering yet; the current features are identity-on-channel (inputs) and
physics transforms (targets). When tabular/scalar EDA features are added later
they can be expressed as Polars ``pl.Expr`` over a per-shot DataFrame -- Polars
is a project dependency for exactly that EDA path -- but the training-time path
here uses plain callables, which is also what the TorchScript ``FeatureMap``
(``analysis/models/base.py``) will mirror for packaging.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

# A raw shot: channel name -> 1-D array, plus a 'sample_rate' float entry.
RawShot = dict[str, object]
ComputeFn = Callable[[RawShot], np.ndarray]
FeatureKind = Literal['input', 'target']


@dataclass
class FeatureSpec:
    """Specification for a registered feature.

    Attributes:
        name: Unique feature name.
        kind: 'input' (model input) or 'target' (supervision signal).
        compute_fn: Maps a raw shot to a 1-D float32 array.
    """

    name: str
    kind: FeatureKind
    compute_fn: ComputeFn


class FeatureRegistry:
    """Central registry mapping feature names to compute functions."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureSpec] = {}

    def register(self, name: str, kind: FeatureKind, compute_fn: ComputeFn) -> None:
        """Register (or replace) a feature."""
        if kind not in ('input', 'target'):
            raise ValueError(f"kind must be 'input' or 'target', got '{kind}'")
        self._features[name] = FeatureSpec(name=name, kind=kind, compute_fn=compute_fn)

    def get(self, name: str) -> FeatureSpec:
        """Return the spec for ``name`` or raise KeyError."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not registered")
        return self._features[name]

    @property
    def available_features(self) -> list[str]:
        """All registered feature names, in registration order."""
        return list(self._features.keys())

    def _names_by_kind(self, kind: FeatureKind) -> list[str]:
        return [n for n, s in self._features.items() if s.kind == kind]

    def input_features(self) -> list[str]:
        """Names of all registered input features, in registration order."""
        return self._names_by_kind('input')

    def target_features(self) -> list[str]:
        """Names of all registered target features, in registration order."""
        return self._names_by_kind('target')

    def compute(self, raw: RawShot, names: list[str]) -> dict[str, np.ndarray]:
        """Compute the requested features for one raw shot.

        Args:
            raw: Raw shot dict (channel arrays + 'sample_rate').
            names: Feature names to compute.

        Returns:
            Mapping name -> 1-D float32 array.
        """
        out: dict[str, np.ndarray] = {}
        for name in names:
            spec = self.get(name)
            out[name] = np.asarray(spec.compute_fn(raw), dtype=np.float32)
        return out


# Module-level default registry; populated by importing ``features.py``.
default_registry = FeatureRegistry()


def register_feature(name: str, kind: FeatureKind) -> Callable[[ComputeFn], ComputeFn]:
    """Decorator registering a compute function on the default registry.

    Args:
        name: Unique feature name.
        kind: 'input' or 'target'.
    """

    def decorator(fn: ComputeFn) -> ComputeFn:
        default_registry.register(name, kind, fn)
        return fn

    return decorator
