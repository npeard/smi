"""Builders shared across test modules: a minimal run record, a scripted log."""

from collections.abc import Callable

import torch

from smi.experiments.results_artifact import RunRecord
from smi.learning.lit_module import LitModule
from smi.learning.training_interface import SIGNED_RMSE_KEY, ZERO_PREDICTOR_RMSE_KEY


def run_record(**overrides) -> RunRecord:
    """A complete RunRecord of a small causal TCN, with ``overrides`` applied."""
    base = dict(
        model_type='TCN',
        config={'kernel_size': 5},
        total_params=57000,
        receptive_field=253,
        causal=True,
        seed=0,
        device='NVIDIA GeForce RTX 3090 Ti',
        dataset='free-space',
        metrics={'velocity_rmse_um_per_ms': {'mean': 0.4, 'median': 0.38}},
    )
    base.update(overrides)
    return RunRecord(**base)


def script_log(
    monkeypatch,
    monitor: Callable[[int], float] | None,
    zero: Callable[[torch.Tensor], torch.Tensor] | None,
) -> None:
    """Script the logged signed monitor by epoch and the zero-predictor value.

    The frozen, untrained tiny TCN need not beat a zero predictor, so a test
    about plateau timing scales the zero-predictor value far above the monitor.
    ``None`` leaves that value as logged.
    """
    real_log = LitModule.log

    def log(self, name, value, *args, **kwargs):
        if name == SIGNED_RMSE_KEY and monitor is not None:
            value = torch.tensor(monitor(self.trainer.current_epoch))
        if name == ZERO_PREDICTOR_RMSE_KEY and zero is not None:
            value = zero(value)
        return real_log(self, name, value, *args, **kwargs)

    monkeypatch.setattr(LitModule, 'log', log)
