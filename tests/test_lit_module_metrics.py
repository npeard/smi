"""The Lightning module must log RMSE, not only losses.

Training previously logged only loss values, so no run produced a number
comparable to the analytic baseline's RMSE without a separate evaluation pass.
"""

import torch

from smi.analysis.lit_module import LitModule

IN_CHANNELS = 3
SEQ_LEN = 1024

# Smallest TCN the factory will build, in the hyperparameter dict shape
# tests/test_torchscript.py uses.
MODEL_HPARAMS = dict(
    type='TCN',
    sequence_length=SEQ_LEN,
    in_channels=IN_CHANNELS,
    activation='GELU',
    use_layer_norm=False,
    use_weight_norm=False,
    kernel_size=3,
    temporal_channels=[4, 4],
    dilation_base=2,
    dropout=0.0,
)


def _module(target: str = 'velocity') -> LitModule:
    return LitModule(
        model_hparams=dict(MODEL_HPARAMS),
        optimizer_hparams={'name': 'Adam', 'lr': 1e-3, 'weight_decay': 0.0},
        scheduler_hparams={},
        loss_hparams={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 1.0,
        },
        training_hparams={'target': target},
        data_hparams={},
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch shaped as VelocityDataset yields it: 2-D targets."""
    return (
        torch.randn(2, IN_CHANNELS, SEQ_LEN),
        torch.randn(2, SEQ_LEN),
        torch.randn(2, SEQ_LEN),
    )


def test_validation_logs_velocity_rmse(monkeypatch):
    """val/velocity_rmse_um_per_ms appears among logged keys."""
    logged: dict[str, float] = {}
    monkeypatch.setattr(
        LitModule, 'log', lambda _self, name, value, **_kw: logged.update({name: value})
    )
    _module().validation_step(_batch(), 0)
    assert 'val/velocity_rmse_um_per_ms' in logged


def test_test_step_logs_velocity_rmse(monkeypatch):
    logged: dict[str, float] = {}
    monkeypatch.setattr(
        LitModule, 'log', lambda _self, name, value, **_kw: logged.update({name: value})
    )
    _module().test_step(_batch(), 0)
    assert 'test/velocity_rmse_um_per_ms' in logged


def test_logged_rmse_is_the_square_root_of_the_velocity_loss(monkeypatch):
    """The two must describe the same quantity, or the units are not um/ms."""
    logged: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(
        LitModule, 'log', lambda _self, name, value, **_kw: logged.update({name: value})
    )
    _module().validation_step(_batch(), 0)
    assert torch.allclose(
        logged['val/velocity_rmse_um_per_ms'],
        torch.sqrt(logged['val/velocity_loss']),
        atol=1e-6,
    )
