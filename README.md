# Neural Networks for Self-Mixing Interferometry

[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-orange.svg)](https://pytorch.org/)

## Overview

Self-mixing interferometry (SMI) offers a cost-effective and optically simple alternative to Michelson interferometry. However, adapting SMI for standard vibrometry remains challenging because analytical signal processing is brittle and sensitive to laser feedback conditions. Furthermore, achieving true portability for field deployment is limited by severe phase noise introduced in optical fibers.

This repository implements an optical and computational framework that overcomes these limitations, using a multispectral SMI array constructed from standard laboratory components. By leveraging convolutional neural networks, our signal processing pipeline reconstructs mechanical displacement with an RMSE below 200 nm directly through multimode fiber. This synergy of frugal optics and deep learning yields an accessible and precise vibrometry tool uniquely suited for hidden or physically constrained environments.

## Project layout

The importable package is `smi/`:

- `smi/analysis/` - ML pipeline: `models/` (architectures + the `Model`
  normalization wrapper in `base.py`), `features/` (Polars `FeatureRegistry` and
  version-controlled normalization stats), `datamodule.py`, `datasets.py`,
  `lit_module.py`, `training_interface.py`.
- `smi/synthetic/` - physics simulation (coil driver, interferometers, waveform).
- `smi/redpitaya/` - Red Pitaya hardware control and acquisition.
- `report/` - Typst slides and reports, built from matplotlib figures through a
  Snakemake DAG (`pixi run report`). See the `report/` section in `CLAUDE.md`.
- `tests/`, `notebooks/` (marimo apps), `scripts/`, `docs/`.

Supported platforms are `win-64` and `linux-64`.

## Quick Start for Contributors

This project uses [Pixi](https://pixi.prefix.dev) for environment and task
management (lockfile-backed, reproducible). Do not use a separate venv or system
Python -- always go through `pixi`.

1. Clone the repository:
   ```bash
   git clone https://github.com/npeard/smi.git
   cd smi
   ```

2. Install the environment (solves and materializes the editable install under `.pixi/`):
   ```bash
   pixi install
   ```

3. Install pre-commit hooks (one-time):
   ```bash
   pixi run pre-commit install
   ```

4. Notebooks in `notebooks/` are [Marimo](https://marimo.io/) notebooks (`.py`
   files); open them with `pixi run marimo edit notebooks/<name>.py` or via the
   Marimo VSCode extension.

5. Create a feature branch, make changes, and run the tooling through Pixi:
   ```bash
   pixi run format     # ruff format + autofix
   pixi run lint       # ruff check
   pixi run typecheck  # ty
   pixi run test       # pytest
   pixi run precommit  # all pre-commit hooks
   pixi run all        # format, lint, ascii, typecheck, test in sequence
   pixi run report     # build the Typst slides and reports
   ```

6. Commit, push, and open a Pull Request on GitHub.

## Running the Main Script

Run `smi/main.py` through Pixi. It provides two execution modes.

### Mode 1: Training a New Model

Train a neural network model using a YAML configuration file:

```bash
pixi run python -m smi.main --config smi/analysis/models/configs/tcn-config.yaml
```

**Arguments:**
- `--config`: Path to YAML configuration file (configs live in `smi/analysis/models/configs/`)
- `--verbosity`: Set logging level (choices: DEBUG, INFO, WARNING, ERROR; default: INFO)

A config may set list-valued hyperparameters to launch a grid search. Set the
`synthetic` section to train on data generated on-device instead of from HDF5.

### Mode 2: Acquiring Real Data from Red Pitaya

Acquire real experimental data from Red Pitaya hardware for training or testing:

```bash
pixi run python -m smi.main --acquire_dataset --num_samples 10000 --dataset_name experimental-data.h5
```

**Arguments:**
- `--acquire_dataset`: Flag to enable dataset acquisition mode
- `--num_samples`: Number of samples to acquire (required with `--acquire_dataset`)
- `--dataset_name`: Filename for the acquired dataset (required with `--acquire_dataset`)

**Note:** The Red Pitaya connection uses default settings configured in the `RedPitayaManager`. Acquired data is saved to `smi/analysis/data/`.

### Evaluating a Trained Model

To visualize predictions, residuals, and input gradient attributions from a trained checkpoint, use the interactive Marimo notebook:

```bash
pixi run marimo edit notebooks/predictions.py
```

Set the checkpoint path and dataset path (or `"synthetic"`) in the UI controls at the top of the notebook.

## Data storage

Datasets are stored as HDF5 (gzip-4, chunked one shot per chunk), which a
one-off benchmark found to be the best balance of compression and random-shot
read speed for this workload; see `docs/data-storage-evaluation.md`.
