"""Convergence-based stopping: relative plateau criterion and the converged flag.

The research report's noise-free ceilings must come from runs trained to a
plateau, and every record must say whether its run got there.
"""

import math
from pathlib import Path

import pytest
import torch
import yaml
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason

from smi.analysis.lit_module import LitModule
from smi.analysis.training_interface import (
    RelativeEarlyStopping,
    TrainingConfig,
    TrainingInterface,
    convergence_monitor,
)
from tests._helpers import script_log
from tests.test_synthetic_train_cli import CONFIG

SIGNED = 'val/velocity_rmse_um_per_ms'
INVARIANT = 'val/velocity_rmse_sign_invariant_um_per_ms'


# Calls the private hook that RelativeEarlyStopping overrides (see its comment
# for the lightning 2.6.x dependency); the real loop is exercised below.
def _feed(callback: RelativeEarlyStopping, values: list[float]) -> list[bool]:
    return [callback._evaluate_stopping_criteria(torch.tensor(v))[0] for v in values]


def test_improvement_below_relative_delta_counts_as_no_improvement():
    cb = RelativeEarlyStopping(
        monitor=SIGNED, min_rel_delta=0.1, patience=2, min_epochs=0
    )
    # 1.0 -> 0.95 is a 5% gain (< 10%): not an improvement, so two waits stop.
    assert _feed(cb, [1.0, 0.95, 0.94]) == [False, False, True]


def test_improvement_above_relative_delta_resets_patience():
    cb = RelativeEarlyStopping(
        monitor=SIGNED, min_rel_delta=0.1, patience=2, min_epochs=0
    )
    assert _feed(cb, [1.0, 0.95, 0.8, 0.79]) == [False, False, False, False]
    assert cb.best_score == pytest.approx(0.8)


def test_relative_delta_scales_with_the_best_value():
    # A fixed 0.05 absolute gain is 50% of 0.1 but 5% of 1.0.
    small = RelativeEarlyStopping(
        monitor=SIGNED, min_rel_delta=0.1, patience=1, min_epochs=0
    )
    assert _feed(small, [0.1, 0.05]) == [False, False]
    large = RelativeEarlyStopping(
        monitor=SIGNED, min_rel_delta=0.1, patience=1, min_epochs=0
    )
    assert _feed(large, [1.0, 0.95]) == [False, True]


@pytest.mark.parametrize(('flag', 'key'), [(False, SIGNED), (True, INVARIANT)])
def test_monitor_follows_loss_sign_invariance(flag, key):
    assert convergence_monitor({'sign_invariant': flag}) == key


def _config(monitor: str, *, sign_invariant: bool, max_epochs: int) -> TrainingConfig:
    return TrainingConfig(
        model_config={
            'type': 'TCN',
            'sequence_length': 64,
            'activation': 'GELU',
            'use_layer_norm': False,
            'use_weight_norm': False,
            'kernel_size': 3,
            'temporal_channels': [4, 4],
            'dilation_base': 2,
            'dropout': 0.0,
        },
        training_config={
            'max_epochs': max_epochs,
            'warmup_epochs': 0,
            'T_0': 10,
            'T_mult': 1,
            'batch_size': 2,
            'optimizer': 'Adam',
            'weight_decay': 0.0,
            'momentum': 0.9,
            'learning_rate': 0.0,  # frozen weights: validation RMSE is flat
            'eta_min': 0.0,
            'accelerator': 'cpu',
            'precision': '32-true',
            'devices': 1,
            'use_logging': False,
            'check_val_every_n_epoch': 1,
            'target': 'velocity',
            'convergence': {
                'monitor': monitor,
                'patience_epochs': 5,
                'min_rel_delta': 0.01,
                'min_epochs': 0,
            },
        },
        data_config={},
        loss_config={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 0.0,
            'sign_invariant': sign_invariant,
        },
        synthetic_config={
            'use_synthetic_training': True,
            'steps_per_epoch': 2,
            'val_steps': 1,
            'train_crop_length': None,
            'sim': _sim(),
        },
    )


def _sim() -> dict:
    return yaml.safe_load(CONFIG.read_text())['synthetic']['sim']


def _run(config: TrainingConfig) -> TrainingInterface:
    interface = TrainingInterface(config=config)
    interface.train()
    return interface


@pytest.fixture
def beats_zero_predictor(monkeypatch):
    script_log(monkeypatch, None, zero=lambda v: v * 1e6)


def test_plateau_stops_early_and_reports_converged(beats_zero_predictor):
    # Flat validation RMSE with fixed validation data is not exactly constant
    # under a synthetic generator, so use a plateau patience that stops the
    # run well before max_epochs, and assert the flag, not a specific epoch.
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['convergence']['min_rel_delta'] = 0.5
    interface = _run(config)
    assert interface.converged is True
    assert interface.trainer.current_epoch < 40


def test_hitting_max_epochs_reports_not_converged():
    config = _config(SIGNED, sign_invariant=False, max_epochs=2)
    config.training_config['convergence']['patience_epochs'] = 100
    interface = _run(config)
    assert interface.converged is False
    assert interface.trainer.current_epoch == 2


def test_patience_counts_epochs_not_validation_checks():
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['check_val_every_n_epoch'] = 3
    config.training_config['convergence'].update(min_rel_delta=0.5, patience_epochs=15)
    interface = TrainingInterface(config=config)
    assert interface.convergence_callback.patience == 5


def test_patience_stops_after_that_many_epochs_at_a_coarser_cadence(
    beats_zero_predictor,
):
    # Frozen weights: epoch 1 is the only improvement (the first value always
    # is), so the 5 checks of patience at cadence 2 end the run at epoch 10.
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['check_val_every_n_epoch'] = 2
    config.training_config['convergence'].update(min_rel_delta=0.5, patience_epochs=10)
    interface = _run(config)
    assert interface.converged is True
    assert 10 <= interface.trainer.current_epoch <= 14


def test_patience_spanning_too_few_checks_is_rejected():
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['check_val_every_n_epoch'] = 5
    config.training_config['convergence']['patience_epochs'] = 10
    with pytest.raises(
        ValueError, match=r'patience_epochs=10.*check_val_every_n_epoch=5'
    ):
        TrainingInterface(config=config)


def test_missing_check_val_cadence_is_an_error():
    config = _config(SIGNED, sign_invariant=False, max_epochs=1)
    del config.training_config['check_val_every_n_epoch']
    with pytest.raises(KeyError, match='check_val_every_n_epoch'):
        TrainingInterface(config=config)


def test_non_finite_monitor_is_not_converged():
    interface = TrainingInterface(
        config=_config(SIGNED, sign_invariant=False, max_epochs=1)
    )
    interface.convergence_callback._evaluate_stopping_criteria(torch.tensor(math.nan))
    assert interface.converged is False


def test_second_train_does_not_report_a_stale_converged():
    config = _config(SIGNED, sign_invariant=False, max_epochs=1)
    config.training_config['convergence']['patience_epochs'] = 100
    interface = TrainingInterface(config=config)
    interface.convergence_callback.stopping_reason = (
        EarlyStoppingReason.PATIENCE_EXHAUSTED
    )
    assert interface.train() is False


def test_monitor_not_matching_loss_sign_invariance_is_rejected():
    with pytest.raises(ValueError, match=INVARIANT):
        _run(_config(SIGNED, sign_invariant=True, max_epochs=1))


@pytest.mark.parametrize(('flag', 'key'), [(False, SIGNED), (True, INVARIANT)])
def test_best_checkpoint_is_tracked_without_logging(flag, key):
    interface = _run(_config(key, sign_invariant=flag, max_epochs=2))
    assert interface.checkpoint_callback.monitor == key
    assert Path(interface.best_model_path).exists()
    assert interface.epochs_run == 2
    assert math.isfinite(interface.best_monitor)
    assert interface.best_monitor <= interface.final_monitor


def test_missing_min_epochs_is_an_error():
    config = _config(SIGNED, sign_invariant=False, max_epochs=1)
    del config.training_config['convergence']['min_epochs']
    with pytest.raises(KeyError, match='min_epochs'):
        TrainingInterface(config=config)


def test_min_epochs_reaches_the_convergence_callback():
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['convergence']['min_epochs'] = 7
    assert TrainingInterface(config=config).convergence_callback.min_epochs == 7


def test_plateau_waits_for_min_epochs(beats_zero_predictor):
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['convergence'].update(min_rel_delta=0.5, min_epochs=12)
    interface = _run(config)
    assert interface.converged is True
    # Patience (5 checks) runs out inside the grace and is withdrawn; the first
    # check past it, at epoch 12 (0-based), stops the run: 13 epochs run.
    assert interface.epochs_run == 13


def test_plateau_waits_for_min_epochs_when_checking_on_validation_end(
    beats_zero_predictor,
):
    # At cadence 2 Lightning checks in on_validation_end, not on_train_epoch_end.
    # Validation runs at epochs 1, 3, ...; patience runs out inside the grace,
    # so the first check past it, at epoch 13, stops the run.
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['check_val_every_n_epoch'] = 2
    config.training_config['convergence'].update(
        min_rel_delta=0.5, patience_epochs=10, min_epochs=12
    )
    interface = _run(config)
    assert interface.converged is True
    assert interface.epochs_run == 14


def test_stall_inside_grace_period_does_not_stop_a_run_that_escapes(monkeypatch):
    # Flat for 8 epochs (patience 5 runs out at epoch 5, inside the grace),
    # then 10% better per epoch to epoch 20, then flat: the run must keep
    # training past min_epochs and stop on the later plateau.
    def monitor(epoch: int) -> float:
        return 0.9 ** (min(max(epoch, 7), 20) - 7)

    script_log(monkeypatch, monitor, zero=lambda v: v * 1e6)
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['convergence']['min_epochs'] = 10
    interface = _run(config)
    assert interface.converged is True
    assert interface.best_monitor == pytest.approx(0.9**13)
    assert 25 <= interface.epochs_run < 40


def test_grace_leaves_a_run_that_plateaus_after_it_unchanged(monkeypatch):
    # Best at epoch 8, then worse, as after a cosine warm restart: without a
    # grace patience (5) stops it at epoch 13, past a grace of 10. The best and
    # the patience count from inside the grace must carry over, so the run
    # stops on the same epoch, which is why min_epochs leaves every
    # non-stalled S3/S4 row as it was.
    script_log(monkeypatch, lambda e: 0.9**e if e <= 8 else 1.0, zero=lambda v: v * 1e6)
    runs = []
    for min_epochs in (0, 10):
        config = _config(SIGNED, sign_invariant=False, max_epochs=40)
        config.training_config['convergence']['min_epochs'] = min_epochs
        runs.append(_run(config))
    assert runs[0].epochs_run == runs[1].epochs_run
    assert runs[0].best_monitor == runs[1].best_monitor


def test_run_that_never_beats_the_zero_predictor_is_not_converged(monkeypatch):
    # A zero-output model: the monitored RMSE is the zero predictor's itself.
    monkeypatch.setattr(
        LitModule, 'velocity_from_output', lambda _, p: torch.zeros_like(p)
    )
    interface = _run(_config(SIGNED, sign_invariant=False, max_epochs=40))
    stopping_reason = interface.convergence_callback.stopping_reason
    assert stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED
    assert interface.best_monitor == pytest.approx(interface.zero_predictor_monitor)
    assert interface.converged is False


@pytest.mark.parametrize(('monitor', 'converged'), [(0.995, False), (0.98, True)])
def test_converged_needs_min_rel_delta_below_the_zero_predictor(
    monkeypatch, monitor, converged
):
    # Zero predictor scripted to 1.0, min_rel_delta=0.01: 0.5% below it is
    # still a stall, 2% below it is a real plateau.
    script_log(monkeypatch, lambda _: monitor, zero=lambda _: torch.tensor(1.0))
    interface = _run(_config(SIGNED, sign_invariant=False, max_epochs=40))
    assert interface.zero_predictor_monitor == pytest.approx(1.0)
    stopping_reason = interface.convergence_callback.stopping_reason
    assert stopping_reason == EarlyStoppingReason.PATIENCE_EXHAUSTED
    assert interface.converged is converged


def test_non_finite_monitor_inside_grace_still_stops_the_run(monkeypatch):
    # The grace withdraws patience stops only: a NaN at epoch 5 ends the run
    # there, as it did before the grace existed, even though the monitor is
    # finite again later and patience would otherwise run out past the grace.
    script_log(
        monkeypatch, lambda e: math.nan if e == 5 else 1.0, zero=lambda v: v * 1e6
    )
    config = _config(SIGNED, sign_invariant=False, max_epochs=40)
    config.training_config['convergence']['min_epochs'] = 10
    interface = _run(config)
    stopping_reason = interface.convergence_callback.stopping_reason
    assert stopping_reason == EarlyStoppingReason.NON_FINITE_METRIC
    assert interface.epochs_run == 6
    assert interface.converged is False
