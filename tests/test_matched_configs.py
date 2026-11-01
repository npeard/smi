"""Matched-budget configs: every family at equal parameters and equal context.

The Q1-Q3 experiments compare architecture families, so a gap in capacity or
lookback between configs would be read as a gap between architectures. The
legacy ``*-matched*.yaml`` files (57k Barland budget) are not covered here.
"""

import math
from pathlib import Path

import pytest
import torch
import yaml

from smi.core.paths import CONFIGS_DIR
from smi.learning.models.barland_cnn import conv_position_field
from smi.learning.models.base import Model
from smi.learning.models.factory import MODEL_REGISTRY, count_parameters, create_model
from smi.learning.models.sliding_window import SlidingWindowModel
from smi.learning.training_interface import MIN_PATIENCE_CHECKS, TrainingConfig
from smi.theory.forward import SimConfig
from tests.test_receptive_field import measure_worst_phase_field, phase_period

MATCHED = CONFIGS_DIR / 'models' / 'matched'
BUDGETS = {'19k': 19_329, '44k': 44_000}
BAND = 0.10
CONFIGS = sorted(p for k in BUDGETS for p in MATCHED.glob(f'*-{k}.yaml'))
#: The receptive field must sit well inside the training crop, or the model is
#: never trained on a window longer than its own context.
MARGIN = 8


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _expected_crop(hparams: dict) -> int | None:
    """Training window the config must declare: the full shot only for a model
    that requires it (lstm-bidir is unbounded too but shift-invariant, so it
    crops like the rest).
    """
    return None if create_model(hparams).requires_full_length else 4096


def _field(model_hparams: dict) -> int | None:
    """Receptive field of the built model; None when it is unbounded."""
    return create_model(model_hparams).receptive_field


# One-sided lookback every causal convolutional/attention family is held to
# (the patch Transformer's worst phase; see its receptive_field).
CAUSAL_FIELD = _field(_load(MATCHED / 'tcn-44k.yaml')['model'])
# Symmetric field every offline family is held to: the field of one position
# of Barland's conv stack (91 samples), because its depth and kernels are fixed by
# the published network while the others can be tuned to it. It is not
# Barland's receptive_field: the dense head and the overlap average widen that
# (to 498 for barland-44k), as they widen the MLP's, so the two sliding-window
# families are held only to the crop margin. Offline families match it within
# FIELD_BAND, not exactly: SCNN needs an odd kernel (symmetric padding cannot
# preserve length otherwise), which puts its fields at 85 or 169, never 91.
FIELD_BAND = 0.10
OFFLINE_FIELD = conv_position_field(
    len(_load(MATCHED / 'barland-44k.yaml')['model']['channels'])
)


def _wrapped_parameters(model_hparams: dict) -> int:
    """Trainable parameters of the model the training path builds.

    The training path wraps the network in ``Model``, which adds only
    normalization buffers; the identity stats here stand in for the sim's.
    """
    wrapped = Model(
        create_model(model_hparams),
        input_mean=torch.zeros(model_hparams['in_channels']),
        input_std=torch.ones(model_hparams['in_channels']),
        output_mean=0.0,
        output_std=1.0,
    )
    return sum(p.numel() for p in wrapped.parameters() if p.requires_grad)


def test_every_registry_family_has_both_budgets():
    assert {_load(p)['model']['type'] for p in CONFIGS} == set(MODEL_REGISTRY)
    stems = {p.stem.rsplit('-', 1)[0] for p in CONFIGS}
    assert {p.stem for p in CONFIGS} == {f'{s}-{k}' for s in stems for k in BUDGETS}


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_parameter_count_within_band(path):
    hparams = _load(path)['model']
    target = BUDGETS[path.stem.rsplit('-', 1)[1]]
    wrapped = _wrapped_parameters(hparams)
    assert wrapped == count_parameters(hparams)
    assert abs(wrapped / target - 1) <= BAND, f'{path.stem}: {wrapped} vs {target}'


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_context_rule_follows_the_column(path):
    config = _load(path)
    hparams = config['model']
    model = create_model(hparams)
    field = model.receptive_field
    if field is None:
        # No finite lookback (recurrent state, whole-input spectral
        # convolution), so only parameters match.
        assert config['synthetic']['train_crop_length'] == _expected_crop(hparams)
        return
    if model.causal:
        assert field == CAUSAL_FIELD
    elif not isinstance(model, SlidingWindowModel):
        assert abs(field / OFFLINE_FIELD - 1) <= FIELD_BAND
    crop = config['synthetic']['train_crop_length']
    assert field * MARGIN <= crop


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_config_loads_and_follows_training_rules(path):
    config = TrainingConfig.from_yaml(str(path))
    training = config.training_config
    assert training['precision'] == '32-true'
    assert training['check_val_every_n_epoch'] == 1
    convergence = training['convergence']
    assert {
        'monitor',
        'patience_epochs',
        'min_rel_delta',
        'min_epochs',
    } <= convergence.keys()
    checks = math.ceil(
        convergence['patience_epochs'] / training['check_val_every_n_epoch']
    )
    assert checks >= MIN_PATIENCE_CHECKS
    assert config.loss_config['sign_invariant'] is False
    synthetic = config.synthetic_config
    assert synthetic is not None
    assert synthetic['use_synthetic_training'] is True
    assert synthetic['train_crop_length'] == _expected_crop(config.model_config)
    # SimConfig has no defaults, so constructing it proves the block is complete.
    sim = SimConfig(**synthetic['sim'])
    assert len(sim.wavelengths_um) == config.model_config['in_channels']
    assert create_model(config.model_config) is not None


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_forward_preserves_length(path):
    hparams = _load(path)['model']
    model = create_model(hparams).eval()
    x = torch.randn(1, hparams['in_channels'], 512)
    with torch.no_grad():
        assert model(x).shape == (1, 1, 512)


# Unbounded fields are excluded: the whole input (tests/test_receptive_field.py
# checks the offline neural operator directly). So are windowed families,
# which a zero-input probe under-counts at Barland's tied max-pools; they are
# probed at a random input there.
BOUNDED_FIELD = [
    p
    for p in CONFIGS
    if (model := create_model(_load(p)['model'])).receptive_field is not None
    and not isinstance(model, SlidingWindowModel)
]


@pytest.mark.parametrize('path', BOUNDED_FIELD, ids=lambda p: p.stem)
def test_measured_field_agrees_with_property(path):
    """The gradient probe is ground truth; the property is only a formula."""
    hparams = _load(path)['model']
    torch.manual_seed(0)  # the NeuralOperator floor is relative to the init
    model = create_model(hparams)
    measured, sees_future = measure_worst_phase_field(
        model,
        hparams['in_channels'],
        length=2048,
        period=phase_period(hparams),
        rel_tol=model.probe_rel_tol,
    )
    assert measured == model.receptive_field
    assert sees_future == (not model.causal)
