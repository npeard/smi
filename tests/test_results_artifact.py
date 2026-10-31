"""Tests for the learned-results artifact."""

import json
from dataclasses import asdict

from smi.analysis.results_artifact import SCHEMA_VERSION, read_results, write_results
from tests._helpers import run_record


def test_round_trip_preserves_records(tmp_path):
    path = tmp_path / 'learned_results.json'
    records = [
        run_record(best_monitor=0.3, zero_predictor_monitor=0.5185),
        run_record(model_type='SCNN', causal=False),
    ]
    write_results(path, records)
    assert read_results(path) == records


def test_artifact_records_schema_version(tmp_path):
    path = tmp_path / 'learned_results.json'
    write_results(path, [run_record()])
    assert json.loads(path.read_text())['schema_version'] == SCHEMA_VERSION


def test_datasets_are_keyed_like_the_baseline_artifact(tmp_path):
    """Same dataset keys as baseline_results.json, so figures index both alike."""
    path = tmp_path / 'learned_results.json'
    write_results(
        path, [run_record(dataset='free-space'), run_record(dataset='mmfiber')]
    )
    assert set(json.loads(path.read_text())['datasets']) == {'free-space', 'mmfiber'}


def test_causality_flag_survives_the_round_trip(tmp_path):
    """Causal must not be droppable: the comparison depends on it."""
    path = tmp_path / 'learned_results.json'
    write_results(path, [run_record(model_type='SCNN', causal=False)])
    assert read_results(path)[0].causal is False


def test_write_is_byte_stable_across_runs(tmp_path):
    """A diff should mean the numbers moved, not that dict ordering did."""
    records = [run_record(dataset='mmfiber'), run_record(dataset='free-space')]
    first = tmp_path / 'a.json'
    second = tmp_path / 'b.json'
    write_results(first, records)
    write_results(second, list(reversed(records)))
    assert first.read_bytes() == second.read_bytes()


def test_legacy_sees_future_records_still_read(tmp_path):
    """learned_results.json stores sees_future only; old runner records carry a
    since-dropped checkpoint field.
    """
    entry = {**asdict(run_record()), 'checkpoint': 'best'}
    del entry['causal']
    entry['sees_future'] = True
    path = tmp_path / 'legacy.json'
    path.write_text(
        json.dumps({'schema_version': SCHEMA_VERSION, 'datasets': {'d': [entry]}})
    )
    (record,) = read_results(path)
    assert record.causal is False
    assert 'sees_future' not in asdict(record)
