"""Precision policy: no float64 in any registered model; precision is explicit."""

import pytest
import torch

from smi.analysis.models.base import Model
from smi.analysis.models.factory import MODEL_REGISTRY, create_model
from smi.analysis.training_interface import TrainingConfig, TrainingInterface
from tests._dtype_guard import no_float64
from tests.test_torchscript import MODEL_CONFIGS

# Mamba's pure-torch scan is slow, so keep every sequence short.
_MAX_LEN = 256

_CONFIG_BY_TYPE = {hp['type']: (hp, n) for _, hp, n in MODEL_CONFIGS}


def _devices() -> list[str]:
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    return devices


def test_every_registered_model_has_a_precision_fixture():
    assert set(MODEL_REGISTRY) <= set(_CONFIG_BY_TYPE)


@pytest.mark.parametrize('device', _devices())
@pytest.mark.parametrize('model_type', sorted(MODEL_REGISTRY))
def test_no_float64_in_any_registered_model(model_type: str, device: str):
    hparams, length = _CONFIG_BY_TYPE[model_type]
    try:
        inner = create_model(hparams)
    except ImportError as exc:
        pytest.skip(f'optional dependency missing for {model_type}: {exc}')
    model = Model.identity(inner, hparams['in_channels']).to(device)
    x = torch.randn(2, hparams['in_channels'], min(length, _MAX_LEN), device=device)
    with no_float64():
        out = model(x)
        out.float().square().mean().backward()
    for p in model.parameters():
        assert p.dtype == torch.float32
        assert p.grad is None or p.grad.dtype == torch.float32


def test_missing_precision_names_the_field():
    cfg = TrainingConfig(
        model_config={},
        training_config={
            'accelerator': 'cpu',
            'devices': 1,
            'use_logging': False,
            'convergence': {
                'monitor': 'val/velocity_rmse_um_per_ms',
                'patience_epochs': 1,
                'min_rel_delta': 0.01,
            },
        },
        loss_config={'sign_invariant': False},
        data_config={},
    )
    iface = TrainingInterface.__new__(TrainingInterface)
    iface.config = cfg
    with pytest.raises(KeyError, match='precision'):
        iface.setup_trainer()
