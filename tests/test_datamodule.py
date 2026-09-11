"""Tests for VelocityDataModule (real-data path)."""

from pathlib import Path

import pytest

from smi.analysis.datamodule import VelocityDataModule

DATASET = (
    Path(__file__).parent.parent / 'smi' / 'analysis' / 'data' / 'circuit-noise-600.h5'
)
SIGNAL_LENGTH = 16384
NUM_PD_CHANNELS = 3
BATCH_SIZE = 4

pytestmark = pytest.mark.skipif(
    not DATASET.exists(),
    reason=f'dataset {DATASET.name} not present (gitignored test data)',
)


def _make_dm() -> VelocityDataModule:
    dm = VelocityDataModule(
        dataset_path=str(DATASET),
        split_ratios=(80, 10, 10),
        batch_size=BATCH_SIZE,
        num_workers=0,
        num_pd_channels=NUM_PD_CHANNELS,
    )
    dm.setup()
    return dm


def test_split_sizes_are_80_10_10():
    """600 shots split 80/10/10 -> 480/60/60."""
    dm = _make_dm()
    assert len(dm.train_dataset) == 480
    assert len(dm.val_dataset) == 60
    assert len(dm.test_dataset) == 60


def test_batch_shapes():
    """A training batch is (signals[B,C,L], velocity[B,L], displacement[B,L])."""
    dm = _make_dm()
    signals, velocity, displacement = next(iter(dm.train_dataloader()))
    assert signals.shape == (BATCH_SIZE, NUM_PD_CHANNELS, SIGNAL_LENGTH)
    assert velocity.shape == (BATCH_SIZE, SIGNAL_LENGTH)
    assert displacement.shape == (BATCH_SIZE, SIGNAL_LENGTH)


def test_split_is_deterministic():
    """Same seed -> identical split indices across instances."""
    dm1 = _make_dm()
    dm2 = _make_dm()
    assert dm1.train_dataset.indices == dm2.train_dataset.indices
