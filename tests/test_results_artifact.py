"""Tests for the learned-results artifact."""

import json
from dataclasses import asdict

import pytest

from smi.analysis.results_artifact import (
    SCHEMA_VERSION,
    RunRecord,
    check_grace_protocol,
    read_results,
    write_results,
)


def _record(**overrides) -> RunRecord:
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


def test_round_trip_preserves_records(tmp_path):
    path = tmp_path / 'learned_results.json'
    records = [
        _record(best_monitor=0.3, zero_predictor_monitor=0.5185),
        _record(model_type='SCNN', causal=False),
    ]
    write_results(path, records)
    assert read_results(path) == records


def test_artifact_records_schema_version(tmp_path):
    path = tmp_path / 'learned_results.json'
    write_results(path, [_record()])
    assert json.loads(path.read_text())['schema_version'] == SCHEMA_VERSION


def test_datasets_are_keyed_like_the_baseline_artifact(tmp_path):
    """Same dataset keys as baseline_results.json, so figures index both alike."""
    path = tmp_path / 'learned_results.json'
    write_results(path, [_record(dataset='free-space'), _record(dataset='mmfiber')])
    assert set(json.loads(path.read_text())['datasets']) == {'free-space', 'mmfiber'}


def test_causality_flag_survives_the_round_trip(tmp_path):
    """Causal must not be droppable: the comparison depends on it."""
    path = tmp_path / 'learned_results.json'
    write_results(path, [_record(model_type='SCNN', causal=False)])
    assert read_results(path)[0].causal is False


def test_write_is_byte_stable_across_runs(tmp_path):
    """A diff should mean the numbers moved, not that dict ordering did."""
    records = [_record(dataset='mmfiber'), _record(dataset='free-space')]
    first = tmp_path / 'a.json'
    second = tmp_path / 'b.json'
    write_results(first, records)
    write_results(second, list(reversed(records)))
    assert first.read_bytes() == second.read_bytes()


def test_legacy_sees_future_records_still_read(tmp_path):
    """learned_results.json stores sees_future only; old runner records carry a
    since-dropped checkpoint field.
    """
    entry = {**asdict(_record()), 'checkpoint': 'best'}
    del entry['causal']
    entry['sees_future'] = True
    path = tmp_path / 'legacy.json'
    path.write_text(
        json.dumps({'schema_version': SCHEMA_VERSION, 'datasets': {'d': [entry]}})
    )
    (record,) = read_results(path)
    assert record.causal is False
    assert 'sees_future' not in asdict(record)


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
