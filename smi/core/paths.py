"""Repository locations of configs, results, data and checkpoints.

Resolved from this file rather than the working directory, so every entry
point finds the same artifacts. The layout assumes an editable install from a
repo checkout, which is the only supported setup.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / 'configs'
RESULTS_DIR = REPO_ROOT / 'results'
DATA_DIR = REPO_ROOT / 'data'
FROZEN_DIR = DATA_DIR / 'frozen'
# Dataset label -> HDF5 file under DATA_DIR, keyed as baseline_results.json
# keys them.
DATASETS: dict[str, str] = {
    'free-space': 'free-space-synchro_10k.h5',
    'mmfiber': 'mmfiber-synchro_10k.h5',
}
# Gitignored and created on first use, so absent from a fresh clone.
CHECKPOINTS_DIR = REPO_ROOT / 'checkpoints'
