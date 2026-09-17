"""Tests for the learned-results artifact."""

import json

from smi.analysis.results_artifact import (
    SCHEMA_VERSION,
    RunRecord,
    read_results,
    write_results,
)


def _record(**overrides) -> RunRecord:
    base = dict(
        model_type='TCN',
        config={'kernel_size': 5},
        total_params=57000,
        receptive_field=253,
        sees_future=False,
        seed=0,
        device='NVIDIA GeForce RTX 3090 Ti',
        dataset='free-space',
        metrics={'velocity_rmse_um_per_ms': {'mean': 0.4, 'median': 0.38}},
    )
    base.update(overrides)
    return RunRecord(**base)


def test_round_trip_preserves_records(tmp_path):
    path = tmp_path / 'learned_results.json'
    records = [_record(), _record(model_type='SCNN', sees_future=True)]
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
    """sees_future must not be droppable: the comparison depends on it."""
    path = tmp_path / 'learned_results.json'
    write_results(path, [_record(model_type='SCNN', sees_future=True)])
    assert read_results(path)[0].sees_future is True


def test_write_is_byte_stable_across_runs(tmp_path):
    """A diff should mean the numbers moved, not that dict ordering did."""
    records = [_record(dataset='mmfiber'), _record(dataset='free-space')]
    first = tmp_path / 'a.json'
    second = tmp_path / 'b.json'
    write_results(first, records)
    write_results(second, list(reversed(records)))
    assert first.read_bytes() == second.read_bytes()
