# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

See `~/.claude/CLAUDE.md` (from `~/Documents/agent-config`) for
cross-project workflow, verification, and design preferences -- the numbered
workflow spine, the coding standards, and the agent-evolution rules all live
there. This file covers only what is specific to smi and does not restate the
master.

For everything else, see:

- **`README.md`** -- project intro, quickstart, install.
- **`docs/data-storage-evaluation.md`** -- why HDF5, and the alternatives measured.
- **`QUESTIONS.md`** -- the research questions the report answers.
- **`TASKS.md`** -- the live task list this project works from.

## Core development principles

The master's "reuse before writing new" means, here: add a feature to the
existing `FeatureRegistry`, an architecture under `learning/models/`, a config
to `learning/configs.py`, a grid to `experiments/grids/`, or a figure to
`report/`, rather than building a parallel path beside one of them.

## Development Commands

Use [Pixi](https://pixi.prefix.dev) for all development tasks. Pixi manages an
isolated Python interpreter and lockfile-backed dependencies under `.pixi/`, so
always invoke Python and tools through `pixi run` (or from inside `pixi shell`)
-- never use system Python or a separate venv.

- `pixi run all` - Run full pipeline (format, lint, ascii, typecheck, test)
- `pixi run format` - Format code with ruff and auto-fix issues
- `pixi run lint` - Check code style with ruff
- `pixi run ascii` - Fail on any non-ASCII character in `*.py`/`*.md`
- `pixi run typecheck` - Type-check with ty
- `pixi run test` - Run pytest test suite
- `pixi run spell` - Run codespell for spell checking
- `pixi run precommit` - Run all pre-commit hooks
- `pixi run -e test pytest` - Run pytest in the minimal `test` environment (CI-equivalent)
- `pixi run report` - Build the Typst report (see `report/` below)
- `pixi run report-dry` / `report-clean` / `report-only` (report.typ)
- `pixi run benchmark-loading` - Data-loading throughput benchmark (GPU; not in `all`)

Workflow-spine tasks (see the master guidance's numbered spine):

- `pixi run preflight` - Step 0: is this project fit to start work in?
  `--with-tests` when starting fresh work; `--check-updates` weekly, not per
  session (one network round trip per hook repo). Runs agent-config's pinned
  `preflight` command, which checks the repo it runs in.
- `pixi run friction` - Recurring-friction miner, for step 7 and the `reflect`
  skill's detection half.
- `pixi run toolgaps` - Missing standard tooling, plus an inventory of
  agent-config assets available to adopt.

`friction` and `toolgaps` run agent-config's scripts from the sibling checkout
`../agent-config` rather than copies (why: the comment on them in
`pyproject.toml`); without that checkout only these two tasks fail.

CI (`.github/workflows/pytest.yml`) runs three jobs -- `test`, `lint`
(lint + ascii) and `typecheck` -- because `pixi run all`'s other gates were
previously enforced only by pre-commit hooks, which `--no-verify` bypasses and
a fresh clone does not install. `format` is deliberately not a CI job: it
rewrites files.

Environments: `default` (library), `dev` (= dev + test features; tooling), `test`
(CI-equivalent), `report` (typst + snakemake + figure deps). `ruff`/`codespell`
live in the default env so the `format`/`lint`/`spell` tasks use the pinned,
lockfile-backed tools (not whatever is on PATH).

For installation and setup:

```bash
pixi install                  # Solve and materialize the default environment (editable install)
pixi run pre-commit install   # One-time: set up git hooks
```

Tasks are defined in `[tool.pixi.tasks]` in `pyproject.toml`; dependencies live
in `[tool.pixi.*]` feature/environment tables. The lockfile `pixi.lock` is
committed for reproducibility.

### Environment constraints (three live ones, all easy to break)

**Platforms are `win-64`, `linux-64` and `osx-arm64`.** macOS Apple silicon
is back for MPS work (torch from the default PyPI wheel, torch's libomp linked into the env for
lightgbm); Intel Mac is unsupported. In `pyproject.toml` the
osx-arm64 entry is named `macos-14-arm64` so it can declare macOS 14, which
the torch>=2.14 wheels require; its target tables use that name. The stack is
`torch>=2.14` / `numpy>=2.5`.

**Keep OpenMP runtimes from colliding; the numeric stack comes from PyPI.** `numpy`,
`numba`, `matplotlib` and `h5py` resolve from PyPI (via
`[project.dependencies]`), *not* conda. Conda-forge numpy links
`libomp.dll` while the PyPI CUDA torch wheel bundles Intel's
`libiomp5md.dll`; loading both aborts the process with "OMP: Error #15",
which previously killed the test suite inside `np.corrcoef`. Do not move
these to `[tool.pixi.dependencies]`, and do not reach for
`KMP_DUPLICATE_LIB_OK=TRUE` -- that flag's own error text warns it can
silently produce incorrect results, which is disqualifying here. Of the
runtime libraries only `polars` and `typst` come from conda; typst is a
self-contained Rust binary with no BLAS linkage.

On macOS torch and scikit-learn each bundle a libomp and coexist, but
lightgbm's wheel needs one on the rpath: a second, conda `llvm-openmp` copy
deadlocks or segfaults threaded fits (OMP Error #179), so
`scripts/link-torch-libomp.sh` (pixi activation) links torch's own libomp
into the env instead. `tests/test_platform.py` guards this by running
threaded torch, lightgbm and scikit-learn work in one subprocess, in either
import order.

**Torch comes from the `cu130` PyPI index** because the default win-64
wheel is CPU-only, and because cu124 publishes no cp312/win-64 wheel past
torch 2.6.0 -- reaching 2.14 means CUDA 13.0. The index carries cp314 win/linux wheels only
through torch 2.14.1 (2.15.0 is on cu132), so those platforms resolve 2.14.1
while macOS resolves 2.15.0. CUDA 13 drops Maxwell/Pascal/
Volta but keeps sm_75, so both cards here (RTX 3090 Ti sm_86, Quadro RTX 4000
sm_75) still run. Expect `torch.cuda.is_available()` to be true; if it is
not, the index pin in the `win-64`/`linux-64` `pypi-dependencies` target tables
is the thing to check.

## Package layout

The importable package is `smi/` (distribution name `smi`). Top-level repo
layout: `smi/` (library), `scripts/` (CLIs), `data/`,
`runs/`, `packages/redpitaya/`, `report/`, `tests/`, `notebooks/`
(marimo apps), `docs/`.

Inside `smi/`, lowest layer first. Layering is enforced by
`[tool.importlinter]` via `tests/test_import_contracts.py`.

- `core/` - bottom layer, imports nothing else in `smi`: acquisition config,
  kinematics, metrics, and `paths` (`REPO_ROOT`, `RESULTS_DIR`,
  `DATA_DIR`, `FROZEN_DIR`, `RUNS_DIR`, `GRID_RUNS_DIR`, `TRAIN_RUNS_DIR`; every artifact path comes from
  here), and `cli` (shared `add_verbosity` / `setup_logging` for scripts).
- `theory/` - physics: coil driver, interferometers, waveform, forward model
  (`simulate_batch`) and the Fisher/Bayesian `bounds`; may import only
  `smi.core`.
- `data/` - data conventions (`schema`: HDF5 keys, attrs, `write_shot` /
  `read_shots`, the one legacy sample-rate fallback; `frozen`: seeded synthetic
  test sets in the real-data HDF5 schema), `datasets.py`, `datamodule.py`
  (`VelocityDataModule`), and `features/` - Polars-based `FeatureRegistry`
  (`registry.py`), feature definitions (`features.py`; `register_feature`
  auto-registers on the module-level `default_registry`), and the **generated,
  version-controlled** per-feature mean/std (`normalization.py` - do not
  hand-edit; regenerate with `scripts/compute_norm_stats.py`). May import only
  `smi.core` and `smi.theory`.
- `estimators/` - array-in/array-out classical estimators: `phase_fit`
  (method 1 forward-model fits, windowed/envelope/joint-gain variants, phase
  drift), `inverse` (method 2 Viterbi decode), `gain_sweep` (gain, delay and
  fringe-rate curves), `tabular` (LightGBM/Ridge). May import only `smi.core`
  and `smi.theory`; never sees a file or a `FeatureRegistry`.
- `learning/` - `models/` (one module per architecture, built by
  `factory.py`; `base.py` defines `Model(nn.Module)`, the normalization
  wrapper, and `FeatureMap`), `lit_module.py`, `synthetic_lit_module.py`,
  `training_interface.py`. Does not import estimators, experiments or
  acquisition.
- `experiments/` - dataset-level runs: the grid `runner`, `evaluate`,
  `results_artifact`, `baselines`, `gain_sweep`, `q1_binned`, `calibrate_sim`,
  `delay_check`. Imports anything in `smi` except `smi.acquisition`.
- `acquisition/` - SMI's side of the rig (`rig`: hosts and shot constants;
  `shots`: the shot loop and HDF5 writing). The only `smi` package that imports
  `stemlab`; it and `experiments` do not import each other.

Outside `smi/`:

- `packages/redpitaya/` is the standalone `stemlab` library (SCPI transport,
  board/daisy-chain manager, arbitrary-waveform generation, raw acquisition, a
  generic live view). It is its own distribution, installed editable via
  `[tool.pixi.pypi-dependencies]`, and must never import `smi` (import-linter
  contract); its README states the API and `stemlab.testing.FakeTransport`
  stands in for hardware in tests.
- `scripts/` - CLIs, run as `pixi run python scripts/<name>.py` (every one
  takes `--help`). Argument-parsing wrappers over plain library functions
  (`run_baselines`, `gain_sweep`, `run_experiments`, `q1_binned`,
  `calibrate_sim`, `delay_check`), plus self-contained drivers (`train`, `acquire`,
  `compute_norm_stats`, `inspect_data`, `benchmark_loading`,
  `run_baseline_sweep`). `train.py` must not import `smi.acquisition`.
  `scripts/` is not a package; tests load a script by path.
- `smi/learning/configs.py` - the named training configs (`MODEL_CONFIGS`,
  `model_config(name)`); `smi/experiments/grids/` - the experiment grids, one
  module per grid exporting a typed `GRID` (`smi/experiments/grid.py`), whose
  families name registry entries. Run one by module name:
  `pixi run experiments q3_s1_families`. An edit to either moves the run
  fingerprints of the cells using it, so resume treats those cells as new.
- `report/results/` - committed result JSON. `data/` - raw HDF5 (ignored) and
  `data/frozen/`. `runs/` - gitignored: `runs/grids/<grid>/<cell-key>.ckpt` from
  the runner, `runs/train/<run-name>/` from `TrainingInterface`.

## Normalization & model wrapping

Normalization stats are NOT computed in the dataset. Instead:

1. `scripts/compute_norm_stats.py` points at an HDF5 dataset, evaluates the
   `FeatureRegistry` (inputs = photodiode channels; targets = velocity,
   displacement), and writes per-feature mean/std to `features/normalization.py`.
2. `Model(nn.Module)` wraps an inner architecture and bakes those stats into
   `register_buffer`s. Its `forward` is raw-units in / raw-units out: raw input
   -> `FeatureMap` (identity for now) -> input-normalize -> inner model ->
   output de-normalize. So the internal network sees zero-mean/unit-variance
   features while the rest of the codebase (loss, plotting) stays in raw physical
   units. `LitModule` does the wrapping; `create_model` still returns the bare
   inner model.

Models are TorchScript-scriptable (`Model.to_torchscript()`), verified by
`tests/test_torchscript.py`. TorchScript is for packaging the trained model.
HDF5 (gzip-4, chunked per shot) is the chosen storage format - see `docs/data-storage-evaluation.md`.

## Hyperparameter / architecture search

There is no hyperparameter-search machinery (Ray Tune and the vmap ensemble
were deleted). `scripts/train.py --config <name>` runs one training run from
one registry config. Sweeps are experiment grids (`pixi run experiments
<grid>`), which run each cell as its own fingerprinted, resumable run.
`SyntheticLitModule` draws its batches from `smi.theory.forward.simulate_batch`
and bakes `normalization_stats`.

## Reports (`report/`)

Typst documents built from matplotlib figures through a Snakemake DAG, in the
spirit of the `dispersion-engineering` paper repo but with Typst in place of
LaTeX. Typst is a conda package, so unlike a TeX engine the whole toolchain is
lockfile-reproducible with no manual install.

- One document, `report.typ`, organized by the questions of `QUESTIONS.md`:
  introduction, analytic foundation (identifiability, Fisher information and
  Bayesian bounds, computed in its figure scripts from `smi.theory.bounds`),
  methods, then one section per question in three tiers -- analytic,
  synthetic (`report/results/synthetic_results.json`), real data (the
  committed `baseline_results.json` / `learned_results.json`). No figure
  reads the raw acquisition HDF5, so it builds on a fresh clone.
- `figures/plot_<name>.py` - one script per figure, each taking `--output
  <path>` and writing exactly that path. Shared style lives in
  `figures/_figcommon.py`; scripts import it rather than setting `rcParams`.
- `build/` - gitignored: figure PDFs and the compiled documents.

Adding a figure takes three coordinated edits: drop `figures/plot_<name>.py`
on disk, add `<name>` to `FIGURES` in the `Snakefile`, and list it under the
documents that show it in `DOCUMENT_FIGURES`. `tests/test_report_build.py`
fails if those drift apart in either direction.

One build-time invariant worth knowing, because it is not obvious:

- **Figures must be deterministic.** Seed any RNG. The DAG's caching is only
  trustworthy if a rebuild of unchanged inputs is a no-op.

Snakemake comes from PyPI: it has no conda-forge win-64 build. Local
`--cores` execution is what is exercised here; the cluster executors are the
part with real Windows gaps.

## Working preferences

These override defaults; follow them unless explicitly told otherwise.

### Verification policy

- **Always run `pixi run format`** after code changes. It is the verification
  entry point -- do not call `ruff check` separately as its own step.
- **Run the full suite (`pixi run -e dev python -m pytest`) when you touch
  `learning/models/`, `data/features/`, or `theory/`** -- the
  normalization stats, the TorchScript scriptability check and the model
  forward tests are all coupled through those. It takes about 235 seconds on osx-arm64.
- **Otherwise** run the single most relevant test file. Scope tests to the
  code you changed rather than surfacing unrelated pre-existing failures.
- **GPU claims need GPU evidence.** `torch.cuda.synchronize()` around any
  timed region, warm up before measuring, and report a median over repeats.
  An unsynchronized CUDA timing is fiction. Pin the device explicitly: this
  box has an RTX 3090 Ti (24 GB) and a Quadro RTX 4000 (8 GB), and which one
  you land on changes the answer.
- **Document builds are verified by building.** `pixi run report` from a
  clean `build/`, then again to confirm the second run is a no-op.

## Code Quality

- Ruff formatting and linting with Google-style docstrings
- Type hints required (Python 3.14)
- Pre-commit hooks: ruff, ty, check-ascii, codespell, nbstripout, standard checks
- Spell checking with codespell

### ASCII-only source convention

**Never use Unicode characters in source files (this file included).** Math
notation in docstrings, comments, and string literals must be written in
**plain text or LaTeX**, not Unicode glyphs. Use `rho` (or `\rho`), `psi`,
`tau`, `sum_n`, `<psi|k>`, `A (x) B`, `->`, `<=`, `+/-`, `d^2`, `A^T`
instead of the corresponding Greek letters, angle brackets, arrows, and
relation glyphs. This keeps source grep-able, diff-able, and free of
homoglyph ambiguity.

Enforcement: agent-config's `check-ascii` (run via `pixi run ascii`, part of
`pixi run all`, and the pinned `check-ascii` pre-commit hook) fails on any non-ASCII
codepoint in `*.py`/`*.md`. Ruff's `RUF001/2/3` additionally flag the
confusable subset.
