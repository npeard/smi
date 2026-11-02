# Neural Networks for Self-Mixing Interferometry

[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2B-orange.svg)](https://pytorch.org/)

## Overview

Self-mixing interferometry (SMI) offers a cost-effective and optically simple alternative to Michelson interferometry. However, adapting SMI for standard vibrometry remains challenging because analytical signal processing is brittle and sensitive to laser feedback conditions. Furthermore, achieving true portability for field deployment is limited by severe phase noise introduced in optical fibers.

This repository implements an optical and computational framework that overcomes these limitations, using a multispectral SMI array constructed from standard laboratory components. By leveraging convolutional neural networks, our signal processing pipeline reconstructs mechanical displacement with an RMSE below 200 nm directly through multimode fiber. This synergy of frugal optics and deep learning yields an accessible and precise vibrometry tool uniquely suited for hidden or physically constrained environments.

## Project layout

The importable package is `smi/`:

- `smi/core/` - acquisition config, kinematics, metrics and `paths`.
- `smi/theory/` - physics simulation (coil driver, interferometers, waveform,
  forward model) and estimation bounds.
- `smi/learning/` - models (architectures + the `Model` normalization wrapper in
  `base.py`), the named training configs (`configs.py`), `lit_module.py`,
  `training_interface.py`.
- `smi/data/` - `features/` (Polars `FeatureRegistry` and version-controlled
  normalization stats), `datamodule.py`, `datasets.py`.
- `smi/estimators/` - classical estimators (forward-model fits, Viterbi
  decode, gain sweep, tabular baselines).
- `smi/experiments/` - grid runner, the experiment grids (`grids/`),
  evaluation and baseline sweeps.
- `smi/acquisition/` - SMI's side of the rig, built on `stemlab`.
- `scripts/` - command-line entry points (training, acquisition, sweeps).
- `report/results/`, `data/`, `runs/` - committed result JSON, raw and frozen
  data (ignored), run outputs (ignored).
- `packages/redpitaya/` - `stemlab`, the standalone Red Pitaya AWG and
  acquisition library (its own package; see its README).
- `report/` - Typst reports, built from matplotlib figures through a
  Snakemake DAG (`pixi run report`). See the `report/` section in `CLAUDE.md`.
- `tests/`, `notebooks/` (marimo apps), `docs/`.

Supported platforms are `win-64`, `linux-64` and `osx-arm64`.

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
   files); open them with `pixi run -e dev marimo edit notebooks/<name>.py` or via the
   Marimo VSCode extension.

5. Create a feature branch, make changes, and run the tooling through Pixi:
   ```bash
   pixi run format     # ruff format + autofix
   pixi run lint       # ruff check
   pixi run typecheck  # ty
   pixi run test       # pytest
   pixi run precommit  # all pre-commit hooks
   pixi run all        # format, lint, ascii, typecheck, test in sequence
   pixi run report     # build the Typst reports
   ```

6. Commit, push, and open a Pull Request on GitHub.

## Running the scripts

Command-line entry points live in `scripts/`; each takes `--help`.

### Training a New Model

Train a neural network model from a named config in `smi/learning/configs.py`:

```bash
pixi run python scripts/train.py --config tcn-config
```

**Arguments:**
- `--config`: Registry name of the config (e.g. `matched/tcn-44k`, `tcn-config`)
- `--verbosity`: Set logging level (choices: DEBUG, INFO, WARNING, ERROR; default: INFO)

Set the `synthetic` section of the config to train on data generated on-device
instead of from HDF5.

### Acquiring Real Data from Red Pitaya

Acquire real experimental data from the Red Pitaya boards:

```bash
pixi run python scripts/acquire.py --num_samples 10000 --dataset_name experimental-data.h5
```

**Arguments:**
- `--num_samples`: Number of shots to acquire
- `--dataset_name`: Filename for the acquired dataset, saved under `data/`

Board hosts and timing constants are in `smi/acquisition/rig.py`.

### Other scripts

`run_experiments.py` (grid runner: `pixi run experiments <grid name>`),
`run_baselines.py`, `gain_sweep.py`, `q1_binned.py`, `calibrate_sim.py`,
`delay_check.py`, `compute_norm_stats.py`, `inspect_data.py`,
`run_baseline_sweep.py` and `benchmark_loading.py` (GPU).

### Evaluating a Trained Model

To visualize predictions, residuals, and input gradient attributions from a trained checkpoint, use the interactive Marimo notebook:

```bash
pixi run -e dev marimo edit notebooks/predictions.py
```

Set the checkpoint path and dataset path (or `"synthetic"`) in the UI controls at the top of the notebook.

## Data storage

Datasets are stored as HDF5 (gzip-4, chunked one shot per chunk), which a
one-off benchmark found to be the best balance of compression and random-shot
read speed for this workload; see `docs/data-storage-evaluation.md`.
