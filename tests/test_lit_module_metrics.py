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


def _module(target: str = 'velocity', sign_invariant: bool = False) -> LitModule:
    return LitModule(
        model_hparams=dict(MODEL_HPARAMS),
        optimizer_hparams={'name': 'Adam', 'lr': 1e-3, 'weight_decay': 0.0},
        scheduler_hparams={},
        loss_hparams={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 1.0,
            'sign_invariant': sign_invariant,
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


def test_validation_logs_signed_and_sign_invariant_rmse(monkeypatch):
    """Both RMSEs are logged so curves show a sign collapse; flipped target
    leaves the sign-invariant one unchanged but not the signed one.
    """
    logged: dict[str, float] = {}
    monkeypatch.setattr(
        LitModule, 'log', lambda _self, name, value, **_kw: logged.update({name: value})
    )
    module = _module()
    signals, velocity, displacement = _batch()
    module.validation_step((signals, velocity, displacement), 0)
    signed, invariant = (
        logged['val/velocity_rmse_um_per_ms'],
        logged['val/velocity_rmse_sign_invariant_um_per_ms'],
    )
    assert invariant <= signed
    module.validation_step((signals, -velocity, -displacement), 0)
    assert logged['val/velocity_rmse_um_per_ms'] != signed
    torch.testing.assert_close(
        logged['val/velocity_rmse_sign_invariant_um_per_ms'], invariant
    )


def test_sign_invariant_loss_equal_for_flipped_target():
    """Flipping a shot's whole target (velocity and displacement) cannot change
    the loss when the loss is sign invariant, but does when it is signed.
    """
    module = _module(sign_invariant=True)
    prediction, velocity, displacement = torch.randn(2, SEQ_LEN), *_batch()[1:]
    flip = torch.tensor([[1.0], [-1.0]])
    kept = module.loss_function(prediction, velocity, displacement.clone())
    flipped = module.loss_function(
        prediction, flip * velocity, flip * displacement.clone()
    )
    for key in ('velocity', 'displacement', 'total'):
        torch.testing.assert_close(kept[key], flipped[key])
    signed = _module(sign_invariant=False)
    assert not torch.allclose(
        signed.loss_function(prediction, velocity, displacement.clone())['velocity'],
        signed.loss_function(prediction, flip * velocity, flip * displacement.clone())[
            'velocity'
        ],
    )


def test_sign_invariant_velocity_loss_is_the_square_of_the_invariant_rmse(monkeypatch):
    """With the flag on, val/velocity_loss is sign invariant while the signed
    RMSE stays signed, so the two RMSE keys differ exactly when the sign matters.
    """
    logged: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(
        LitModule, 'log', lambda _self, name, value, **_kw: logged.update({name: value})
    )
    _module(sign_invariant=True).validation_step(_batch(), 0)
    torch.testing.assert_close(
        torch.sqrt(logged['val/velocity_loss']),
        logged['val/velocity_rmse_sign_invariant_um_per_ms'],
    )
