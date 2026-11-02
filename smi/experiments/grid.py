"""Typed experiment-grid definitions; the grids themselves live in ``grids/``.

A grid expands to cells ``family x channel subset x sim x seed`` (plus one
optional classical oracle per ``sim x subset``); see
:func:`smi.experiments.runner.expand_grid`. Construction validates what a
typo could otherwise defer to mid-run: every family names a
:data:`~smi.learning.configs.MODEL_CONFIGS` entry, and every ``override`` or
``sweep`` field is a :class:`~smi.theory.forward.SimConfig` field.
"""

import dataclasses
import difflib
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from smi.learning.configs import model_config
from smi.theory.forward import SimConfig

_SIM_FIELDS = frozenset(f.name for f in dataclasses.fields(SimConfig))


@dataclass(frozen=True)
class Family:
    """A trained family: its cell-key label and its model-registry entry.

    ``name`` is part of every cell key (and so of checkpoint file names), so
    one ``config`` trained under different budgets gets distinct names, e.g.
    ``tcn-44k-T90``.
    """

    name: str
    config: str


@dataclass(frozen=True)
class Sim:
    """One ``sims`` entry, expanded per channel subset.

    ``preset`` picks the :class:`SimConfig` constructor; ``override`` replaces
    fields after it; ``sweep`` ``(field, values)`` expands into one sim per
    value, named ``<name>-<field>=<value>`` (one field: a Cartesian product
    would hide cell counts). ``sign_invariant`` ``None`` takes the grid's.
    """

    name: str
    preset: Literal['nominal', 'noise_free']
    feedback_c: float
    alpha: float
    override: Mapping[str, Any] = field(default_factory=dict)
    sweep: tuple[str, tuple[Any, ...]] | None = None
    sign_invariant: bool | None = None

    def __post_init__(self) -> None:
        if self.preset not in ('nominal', 'noise_free'):
            raise ValueError(f'sim {self.name!r}: unknown preset {self.preset!r}')
        named = [*self.override, *([self.sweep[0]] if self.sweep else [])]
        for name in named:
            if name not in _SIM_FIELDS:
                near = difflib.get_close_matches(name, _SIM_FIELDS, n=3)
                raise ValueError(
                    f'sim {self.name!r}: {name!r} is not a SimConfig field; '
                    f'closest: {near}'
                )


@dataclass(frozen=True)
class Synthetic:
    """The grid's synthetic-training budget.

    A model whose ``requires_full_length`` is set trains on full shots
    (``train_crop_length`` ``None``) whatever the grid says.
    """

    steps_per_epoch: int
    val_steps: int
    train_crop_length: int | None


@dataclass(frozen=True)
class EvalSet:
    """The frozen test set every cell of the grid is scored on."""

    seed: int
    n_shots: int


@dataclass(frozen=True)
class Grid:
    """An experiment grid.

    ``sign_invariant`` is the loss default for every sim that does not set its
    own (Michelson sims need it). ``training`` shallowly overrides each
    family's ``training`` block (device, epoch cap); ``convergence.monitor``
    is derived by the runner from the sim's sign invariance and the family's
    target, so one family serves Michelson and SMI sims.
    """

    families: tuple[Family, ...]
    channel_subsets: tuple[tuple[float, ...], ...]
    sims: tuple[Sim, ...]
    seeds: tuple[int, ...]
    oracle: bool
    sign_invariant: bool
    synthetic: Synthetic
    training: Mapping[str, Any]
    test_set: EvalSet

    def __post_init__(self) -> None:
        for family in self.families:
            model_config(family.config)  # an unknown name fails here, not mid-run


# Shared by the shipped grids.
RIG_WAVELENGTHS_UM = (0.635, 0.6748, 0.515)
#: Every non-empty subset of the rig lasers, singles first, in channel order.
ALL_SUBSETS = tuple(
    subset
    for r in range(1, len(RIG_WAVELENGTHS_UM) + 1)
    for subset in itertools.combinations(RIG_WAVELENGTHS_UM, r)
)
#: Every family at the 44k budget (Task 19), as ``matched/<stem>``.
STEMS_44K = (
    'tcn-44k',
    'scnn-44k',
    'barland-44k',
    'tcan-44k',
    'lstm-44k',
    'lstm-bidir-44k',
    'mamba-44k',
    'mlp-44k',
    'patchtransformer-causal-44k',
    'patchtransformer-offline-44k',
    'unettransformer-causal-44k',
    'unettransformer-offline-44k',
    'neuraloperator-causal-44k',
    'neuraloperator-offline-44k',
)
#: The family schedule's budget (30 epochs, T_0 = 30 unless overridden).
FAMILY_BUDGET = Synthetic(steps_per_epoch=200, val_steps=20, train_crop_length=4096)
TEST_SET = EvalSet(seed=1000, n_shots=64)
MPS = {'accelerator': 'mps', 'devices': 1, 'precision': '32-true'}


def tcn_44k(name: str) -> tuple[Family]:
    """The TCN-44k family alone, labelled ``name`` in its cell keys."""
    return (Family(name, 'matched/tcn-44k'),)


#: The noise-free Michelson sim; it cannot show direction, so sign-invariant.
MICHELSON = Sim(
    'michelson', 'noise_free', feedback_c=0.0, alpha=0.0, sign_invariant=True
)
