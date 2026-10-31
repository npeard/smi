"""Tests for the report-side helpers shared by the synthetic figures."""

import pytest
from _figcommon import check_grace_protocol

from smi.analysis.results_artifact import RunRecord


def _record(**overrides) -> RunRecord:
    base = dict(
        model_type='TCN',
        config={'kernel_size': 5},
        total_params=57000,
        receptive_field=253,
        causal=True,
        seed=0,
        device='cpu',
        dataset='synthetic',
        metrics={},
    )
    base.update(overrides)
    return RunRecord(**base)


def _trained(cell_key: str, epochs_run: int, **overrides) -> RunRecord:
    return _record(
        cell_key=cell_key, epochs_run=epochs_run, max_epochs=100, **overrides
    )


def test_pre_grace_record_stopped_inside_the_grace_is_rejected():
    # No zero_predictor_monitor: trained before the grace, so a stop at epoch
    # 11 would now be withdrawn and the run could have escaped.
    with pytest.raises(ValueError, match='old-a0'):
        check_grace_protocol(
            [_trained('old-a0', 11)], min_epochs=30, expected_collapses=frozenset()
        )


def test_named_expected_collapse_is_accepted():
    check_grace_protocol(
        [_trained('old-a0', 11)], min_epochs=30, expected_collapses={'old-a0'}
    )


@pytest.mark.parametrize(
    'record',
    [
        _trained('past-grace', 31),
        _trained('new-protocol', 11, zero_predictor_monitor=0.5),
        _record(cell_key='capped', epochs_run=20, max_epochs=20),
        _record(cell_key='oracle'),
    ],
)
def test_records_the_grace_would_not_change_are_accepted(record):
    # Past the grace, written under it, stopped by the epoch cap rather than
    # patience, or never trained: the grace changes none of these.
    check_grace_protocol([record], min_epochs=30, expected_collapses=frozenset())
