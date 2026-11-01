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
# Gitignored and created on first use, so absent from a fresh clone.
CHECKPOINTS_DIR = REPO_ROOT / 'checkpoints'
