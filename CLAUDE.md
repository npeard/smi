# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

See `~/.claude/CLAUDE.md` (from `~/Documents/Projects/claude-config`) for
cross-project workflow, verification, and design preferences -- the numbered
workflow spine, the coding standards, and the agent-evolution rules all live
there. This file covers only what is specific to smi and does not restate the
master.

For everything else, see:

- **`README.md`** -- project intro, quickstart, install.
- **`docs/data-storage-evaluation.md`** -- why HDF5, and the alternatives measured.
- **`docs/hyperparameter-search-pipeline-design.md`** -- the two-level search design.
- **`TASKS.md`** -- the live task list this project works from.

## Core development principles

See the master CLAUDE.md's "Core philosophy" for the general rule (every line
carries maintenance cost; write the smallest clear amount; reuse before writing
new). In this repo that means: add a feature to the existing `FeatureRegistry`,
an architecture under `analysis/models/`, or a figure to `report/`, rather than
building a parallel path beside one of them.


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
- `pixi run report` - Build the Typst documents (see `report/` below)
- `pixi run report-dry` / `report-clean` / `report-slides` / `report-baseline`
- `pixi run benchmark-loading` - Data-loading throughput benchmark (GPU; not in `all`)

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

**Platforms are `win-64` and `linux-64`.** macOS was dropped deliberately.
The old `numpy<2` / `torch<2.3` pins existed only to keep an Intel Mac
working and are gone; the stack is now `torch>=2.5` / `numpy>=2`.

**One OpenMP runtime, so the numeric stack comes from PyPI.** `numpy`,
`numba`, `matplotlib` and `h5py` resolve from PyPI (via
`[project.dependencies]`), *not* conda. Conda-forge numpy links
`libomp.dll` while the PyPI CUDA torch wheel bundles Intel's
`libiomp5md.dll`; loading both aborts the process with "OMP: Error #15",
which previously killed the test suite inside `np.corrcoef`. Do not move
these to `[tool.pixi.dependencies]`, and do not reach for
`KMP_DUPLICATE_LIB_OK=TRUE` -- that flag's own error text warns it can
silently produce incorrect results, which is disqualifying here. Only
`typst` and `polars` come from conda; typst is a self-contained Rust
binary with no BLAS linkage.

**Torch comes from the `cu124` PyPI index** because the default win-64
wheel is CPU-only. Expect `torch.cuda.is_available()` to be true; if it is
not, the index pin in `[tool.pixi.pypi-dependencies]` is the thing to check.

**Important**: Always run `pixi run format` first when encountering linting/formatting issues before making manual edits. This auto-fixes most formatting problems and saves time.

## Package layout

The importable package is `smi/` (distribution name `smi`). Top-level repo
layout: `smi/` (library), `tests/`, `notebooks/` (marimo apps), `scripts/`
(`check_ascii.py`), `docs/`.

Inside `smi/`:

- `analysis/` - ML pipeline.
  - `features/` - Polars-based `FeatureRegistry` (`registry.py`), feature
    definitions (`features.py`; `register_feature` decorator auto-registers on
    the module-level `default_registry`), the stats script (`compute_norm_stats.py`),
    and the **generated, version-controlled** per-feature mean/std
    (`normalization.py` - do not hand-edit; regenerate with the script).
  - `models/` - architectures (`tcn`, `scnn`, `tcan`, `lstm`, `mamba`,
    `barland_cnn`) plus `base.py`, which defines `Model(nn.Module)` (the
    normalization wrapper) and `FeatureMap`.
  - `datamodule.py` - `VelocityDataModule` (LightningDataModule).
  - `datasets.py`, `lit_module.py`, `synthetic_lit_module.py`, `training_interface.py`.
- `synthetic/` - physics simulation (coil driver, interferometers, waveform);
  was `acquisition/simulations/`.
- `redpitaya/` - hardware control (manager, scpi, config); was
  `acquisition/redpitaya/`. Imports from `smi.synthetic`; import its manager via
  `from smi.redpitaya.manager import RedPitayaManager` (not re-exported from the
  package `__init__`, to avoid an init-time import cycle).

## Normalization & model wrapping

Normalization stats are NOT computed in the dataset. Instead:

1. `compute_norm_stats.py` points at an HDF5 dataset, evaluates the
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
`tests/test_torchscript.py`. `torch.compile` remains the training-speed path;
TorchScript is for packaging the trained model. HDF5 (gzip-4, chunked per shot)
is the chosen storage format - see `docs/data-storage-evaluation.md`.

## Hyperparameter / architecture search

Two-level search (design: `docs/hyperparameter-search-pipeline-design.md`):

- **Ray Tune** (`analysis/tune_search.py`, `main.py --search`) replaces the old
  YAML grid: list-valued YAML fields become `tune.choice` dimensions; trials run
  via `train_func` (reusing `TrainingInterface`) with an ASHA scheduler and
  fractional-GPU packing. `TrainingConfig.from_yaml` now returns one config.
- **vmap-ensemble** (`analysis/ensemble.py`) trains K same-architecture models on
  one shared on-GPU minibatch via `torch.func` for efficient within-architecture
  sweeps; `generate_synthetic_batch` (in `synthetic_lit_module.py`) is shared.

Deps live in the `tune` pixi feature (`ray[tune]`, `optuna`), included in all
test-running envs so the search path is CPU-tested in CI as well as on the GPU rig.

## Reports and slides (`report/`)

Typst documents built from matplotlib figures through a Snakemake DAG, in the
spirit of the `dispersion-engineering` paper repo but with Typst in place of
LaTeX. Typst is a conda package, so unlike a TeX engine the whole toolchain is
lockfile-reproducible with no manual install.

- `slides.typ` - research-presentation deck. `lib/theme.typ` makes each level-1
  heading a new page.
- `baseline.typ` - the standalone non-deep-learning baseline report.
- `figures/plot_<name>.py` - one script per figure, each taking `--output
  <path>` and writing exactly that path. Shared style lives in
  `figures/_figcommon.py`; scripts import it rather than setting `rcParams`.
- `build/` - gitignored: figure PDFs and the compiled documents.

Adding a figure takes three coordinated edits: drop `figures/plot_<name>.py`
on disk, add `<name>` to `FIGURES` in the `Snakefile`, and list it under the
documents that show it in `DOCUMENT_FIGURES`. `tests/test_report_build.py`
fails if those drift apart in either direction.

Two build-time invariants worth knowing, because neither is obvious:

- **Figures must be deterministic.** Seed any RNG. The DAG's caching is only
  trustworthy if a rebuild of unchanged inputs is a no-op.
- **A slide must fit one page.** Typst silently continues an overlong slide
  onto another page, so `scripts/check_slide_pages.py` compares headings to
  PDF pages and fails the build. If it fires, shrink the content or the
  figure -- do not raise the page budget.

Snakemake comes from PyPI: it has no conda-forge win-64 build. Local
`--cores` execution is what is exercised here; the cluster executors are the
part with real Windows gaps.

## Working preferences

These override defaults; follow them unless explicitly told otherwise.

### Verification policy

- **Always run `pixi run format`** after code changes. It is the verification
  entry point -- do not call `ruff check` separately as its own step.
- **Run the full suite (`pixi run -e dev python -m pytest`) when you touch
  `analysis/models/`, `analysis/features/`, or `synthetic/`** -- the
  normalization stats, the TorchScript scriptability check and the model
  forward tests are all coupled through those. It takes about 95 seconds.
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
- Type hints required (Python 3.12+)
- Pre-commit hooks: ruff, ty, ascii-only, codespell, nbstripout, standard checks
- Spell checking with codespell

### ASCII-only source convention

**Never use Unicode characters in source files (this file included).** Math
notation in docstrings, comments, and string literals must be written in
**plain text or LaTeX**, not Unicode glyphs. Use `rho` (or `\rho`), `psi`,
`tau`, `sum_n`, `<psi|k>`, `A (x) B`, `->`, `<=`, `+/-`, `d^2`, `A^T`
instead of the corresponding Greek letters, angle brackets, arrows, and
relation glyphs. This keeps source grep-able, diff-able, and free of
homoglyph ambiguity.

Enforcement: `scripts/check_ascii.py` (run via `pixi run ascii`, part of
`pixi run all`, and the `ascii-only` pre-commit hook) fails on any non-ASCII
codepoint in `*.py`/`*.md`. Ruff's `RUF001/2/3` additionally flag the
confusable subset.
