"""Tests for the report-side helpers shared by the synthetic figures."""

from pathlib import Path

import pytest
import yaml
from _figcommon import (
    MATCHED_LENGTH_GRID,
    NOISE_LADDER_GRID,
    check_grace_protocol,
    grace_by_cell,
    noise_ladder,
)

from smi.experiments.results_artifact import RunRecord
from smi.experiments.runner import expand_grid
from tests._helpers import run_record


def _trained(cell_key: str, epochs_run: int, **overrides) -> RunRecord:
    return run_record(
        cell_key=cell_key, epochs_run=epochs_run, max_epochs=100, **overrides
    )


def test_pre_grace_record_stopped_inside_the_grace_is_rejected():
    # No zero_predictor_monitor: trained before the grace, so a stop at epoch
    # 11 would now be withdrawn and the run could have escaped.
    with pytest.raises(ValueError, match='old-a0'):
        check_grace_protocol(
            [_trained('old-a0', 11)],
            min_epochs={'old-a0': 30},
            expected_collapses=frozenset(),
        )


def test_each_record_is_held_to_its_own_cells_grace():
    with pytest.raises(ValueError, match='long-grace'):
        check_grace_protocol(
            [_trained('short-grace', 20), _trained('long-grace', 20)],
            min_epochs={'short-grace': 10, 'long-grace': 40},
            expected_collapses=frozenset(),
        )


def test_legacy_record_of_no_grid_cell_is_rejected():
    with pytest.raises(ValueError, match='orphan'):
        check_grace_protocol(
            [_trained('orphan', 50)], min_epochs={}, expected_collapses=frozenset()
        )


def test_named_expected_collapse_is_accepted():
    check_grace_protocol(
        [_trained('old-a0', 11)],
        min_epochs={'old-a0': 30},
        expected_collapses={'old-a0'},
    )


@pytest.mark.parametrize(
    'record',
    [
        _trained('past-grace', 31),
        _trained('new-protocol', 11, zero_predictor_monitor=0.5),
        run_record(cell_key='capped', epochs_run=20, max_epochs=20),
        run_record(cell_key='oracle'),
    ],
)
def test_records_the_grace_would_not_change_are_accepted(record):
    # Past the grace, written under it, stopped by the epoch cap rather than
    # patience, or never trained: the grace changes none of these.
    check_grace_protocol(
        [record], min_epochs={record.cell_key: 30}, expected_collapses=frozenset()
    )


def test_grace_comes_from_each_grids_resolved_training_config():
    """Every trained cell of every shipped grid has a grace, read not restated."""
    grace = grace_by_cell()
    assert grace['tcn-44k-E60__ch635-675-515__ladder-base__seed0'] == 30
    assert all(isinstance(v, int) and v > 0 for v in grace.values())


def test_record_run_under_another_sim_config_names_its_cell():
    _, (first, *_) = expand_grid(NOISE_LADDER_GRID)
    stale = run_record(cell_key=first.key, sim_config_hash='not-the-grid-hash')
    with pytest.raises(ValueError, match=first.key):
        noise_ladder({first.key: stale})


def test_matched_length_grid_repeats_the_ladders_phase_drift_sweep():
    """plot_noise_ladder pairs the 60-epoch cells to ladder rows by sweep point."""

    def sweep(grid: Path) -> dict:
        (entry,) = [
            s
            for s in yaml.safe_load(grid.read_text())['sims']
            if s['name'] == 'ladder-phasedrift'
        ]
        return entry

    assert sweep(MATCHED_LENGTH_GRID) == sweep(NOISE_LADDER_GRID)
