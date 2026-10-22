"""Composition tests: a Ray Tune trial that trains a vmap-ensemble inner loop.

Covers the synthetic ensemble module end to end (metric reporting) and the
``run_search`` -> ensemble routing. CPU-only, tiny sizes; the same code path runs
on the GPU rig.
"""

from pathlib import Path

import lightning as lightning_module
import pytest
from torch.utils.data import DataLoader

# The composition driver imports ray at module top; skip the whole module if Ray
# is unavailable (it is installed in all test envs, but this keeps CI robust).
pytest.importorskip('ray')

from smi.analysis.ensemble import SyntheticEnsembleModule
from smi.analysis.synthetic_lit_module import SyntheticIndexDataset
from smi.analysis.tune_search import SEARCH_METRIC, run_search

CONFIG = (
    Path(__file__).parent.parent
    / 'smi'
    / 'analysis'
    / 'models'
    / 'configs'
    / 'tune-ensemble-example.yaml'
)

MODEL_HPARAMS = {
    'type': 'TCN',
    'sequence_length': 128,
    'activation': 'GELU',
    'use_layer_norm': True,
    'use_weight_norm': False,
    'dropout': 0.0,
    'kernel_size': 3,
    'temporal_channels': [8, 8],
    'dilation_base': 2,
    'in_channels': 3,
}
WAVELENGTHS_NM = [635, 674.8, 515]


def test_synthetic_ensemble_reports_search_metric():
    """A 1-epoch CPU fit of the synthetic ensemble reports val/total_unweighted_loss."""
    module = SyntheticEnsembleModule(
        dict(MODEL_HPARAMS),
        seeds=[0, 1, 2],
        wavelengths_nm=WAVELENGTHS_NM,
        target='velocity',
        per_member_lr=[1e-3, 5e-4, 1e-4],
    )
    batch_size = 4
    train_loader = DataLoader(
        SyntheticIndexDataset(2 * batch_size), batch_size=batch_size
    )
    val_loader = DataLoader(
        SyntheticIndexDataset(2 * batch_size), batch_size=batch_size
    )
    trainer = lightning_module.Trainer(
        max_epochs=1,
        accelerator='cpu',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        check_val_every_n_epoch=1,
    )
    trainer.fit(module, train_loader, val_loader)

    metric = trainer.callback_metrics.get('val/total_unweighted_loss')
    assert metric is not None, 'ensemble must report the shared search metric'
    assert metric.isfinite().all()
    # Best-member tracking populated during validation.
    assert module.best_member_idx is not None


def test_run_search_routes_to_ensemble():
    """run_search on an ensemble config trains the vmap-ensemble, returns a result."""
    best = run_search(str(CONFIG), num_samples=1, gpu_fraction=0.0, grace_period=1)
    assert best is not None
    assert isinstance(best.metrics, dict)
    assert SEARCH_METRIC in best.metrics
