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

from smi.analysis.metrics import count_parameters
from smi.analysis.models.base import Model
from smi.analysis.models.factory import MODEL_REGISTRY, create_model, is_causal
from smi.analysis.training_interface import MIN_PATIENCE_CHECKS, TrainingConfig
from smi.synthetic.forward import SimConfig

MATCHED = (
    Path(__file__).parents[1] / 'smi' / 'analysis' / 'models' / 'configs' / 'matched'
)
BUDGETS = {'19k': 19_329, '44k': 44_000}
BAND = 0.10
CONFIGS = sorted(p for k in BUDGETS for p in MATCHED.glob(f'*-{k}.yaml'))
#: The receptive field must sit well inside the training crop, or the model is
#: never trained on a window longer than its own context.
MARGIN = 8


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _field(model_hparams: dict) -> int | None:
    """Receptive field of the built model; None when it is unbounded."""
    return getattr(create_model(model_hparams), 'receptive_field', None)


# One-sided lookback every causal convolutional/attention family is held to.
CAUSAL_FIELD = _field(_load(MATCHED / 'tcn-44k.yaml')['model'])
# Symmetric field every offline family is held to: Barland's, because its
# depth and kernels are fixed by the published network while the others can be
# tuned to it.
OFFLINE_FIELD = _field(_load(MATCHED / 'barland-44k.yaml')['model'])


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
    field = _field(hparams)
    if hparams.get('receptive_field') == 'unbounded':
        # Recurrent state has no finite lookback, so only parameters match.
        assert field is None
        return
    assert field is not None
    expected = CAUSAL_FIELD if is_causal(hparams) else OFFLINE_FIELD
    assert field == expected
    crop = config['synthetic']['train_crop_length']
    assert field * MARGIN <= crop


@pytest.mark.parametrize('path', CONFIGS, ids=lambda p: p.stem)
def test_config_loads_and_follows_training_rules(path):
    config = TrainingConfig.from_yaml(str(path))
    training = config.training_config
    assert training['precision'] == '32-true'
    assert training['check_val_every_n_epoch'] == 1
    convergence = training['convergence']
    assert {'monitor', 'patience_epochs', 'min_rel_delta'} <= convergence.keys()
    checks = math.ceil(
        convergence['patience_epochs'] / training['check_val_every_n_epoch']
    )
    assert checks >= MIN_PATIENCE_CHECKS
    assert config.loss_config['sign_invariant'] is False
    synthetic = config.synthetic_config
    assert synthetic is not None
    assert synthetic['use_synthetic_training'] is True
    assert synthetic['train_crop_length'] == 4096
    # SimConfig has no defaults, so constructing it proves the block is complete.
    sim = SimConfig(**synthetic['sim'])
    assert len(sim.wavelengths_um) == config.model_config['in_channels']
    assert create_model(config.model_config) is not None
