# Hyperparameter / Architecture Search Pipeline (Ray Tune + vmap)

Design for a two-level model search pipeline that replaces the current YAML
Cartesian-product grid search. Status: design (not yet implemented).

## 1. Context and goals

Today, `TrainingConfig.from_yaml` (in `smi/analysis/training_interface.py`)
expands list-valued YAML fields into a Cartesian product of configs and shuffles
them; `TrainingInterface` then runs them one at a time. This does not scale, has
no early stopping, and does not exploit that our models are small (~100k params)
or that synthetic data is generated cheaply on the GPU.

Two concrete workflows must be served:

1. **Broad architecture search on synthetic data.** Synthetic signals are
   generated directly on the GPU in minibatches (`SyntheticLitModule`), so data
   is cheap and effectively unlimited. We want to scan many architectures
   (TCN / SCNN / TCAN / LSTM / Mamba and their structural knobs) quickly and
   cheaply to brainstorm and downselect.
2. **Focused architecture + hyperparameter search on real data.** After
   downselecting architectures from (1), run as many configurations as fit on a
   single 24 GB CUDA GPU against real experimental HDF5 data
   (`VelocityDataModule`), now weighted toward hyperparameter tuning of the few
   surviving architectures.

Constraints:
- **Single 24 GB GPU**, on a separate CUDA rig. This dev machine is an Intel Mac
  locked to torch 2.2 with no CUDA, so every path must be CPU-runnable (tiny
  sizes) for tests; GPU is exercised only on the rig.
- Reuse the existing pieces: `LitModule` (loss, optimizer, scheduler), the
  `Model` normalization wrapper + `create_model` factory, `VelocityDataModule`,
  `SyntheticLitModule` on-GPU generation, and the `WandbLogger` / `ModelCheckpoint`
  wiring in `TrainingInterface.setup_trainer`.

## 2. Strategy: two complementary levels

Two parallelization strategies fit different search axes; we use both.

- **Outer level - Ray Tune (independent trials).** Varies anything, including
  architecture, across separate trial processes. Fractional-GPU packing runs
  several trials per GPU; an ASHA scheduler early-stops weak configs; a search
  algorithm (grid/random first, Optuna later) chooses configs. This is the
  general engine and the replacement for the YAML grid.
- **Inner level - vmap-ensemble (one process, shared data).** Within a single
  process/GPU, train K models of the *same* architecture as one batched
  computation via `torch.func`, all consuming the *same* on-GPU-generated
  minibatch. This is maximally GPU-efficient for many tiny models and is the
  efficient way to fan out the within-architecture axis (seed/init, learning
  rate, loss weights, dropout). It cannot vary architecture.

They compose: a Ray trial fixes an architecture, then optionally runs a
vmap-ensemble over the within-architecture axis inside that trial.

## 3. Single-GPU (24 GB) packing model

- Ray fractional GPU: `tune.with_resources(train_func, {"CPU": n, "GPU": frac})`
  packs ~`1/frac` trials per GPU and sets `CUDA_VISIBLE_DEVICES`. Memory is NOT
  isolated, so concurrent models must fit; cap with `max_concurrent_trials`.
- **The real limit for tiny models is CUDA context, not weights.** Each trial is
  a separate process holding a ~0.3-0.6 GB CUDA context, so a handful of 100k-param
  models can still consume several GB just in contexts. Practical guidance:
  pack a modest number of fractional-GPU trials (e.g. 8-16) and let ASHA cycle
  configs through, rather than chasing very small `frac`.
- For same-architecture fans, the **vmap-ensemble packs far more models** (one
  context, one data generation, one set of kernel launches), so prefer it over
  many fractional-GPU trials on that axis.
- Exposed knobs: `gpu_fraction`, `max_concurrent_trials`, ensemble size `K`.

## 4. Components and integration points

1. **Search-config schema** (replaces the YAML grid). A config expressing a Ray
   search space per section (model/training/loss/data/synthetic) using
   `tune.choice`, `tune.loguniform`, etc. Retire the list-expansion +
   `random.shuffle` in `TrainingConfig`; a single sampled config still maps onto
   the existing `*_hparams` dicts that `LitModule` consumes.
2. **Tune driver** (new, e.g. `smi/analysis/tune_search.py`, plus a `main.py`
   mode). Builds the search space, the `train_func`, an `ASHAScheduler`
   (`max_t`, `grace_period`), `tune.Tuner(param_space, tune_config=TuneConfig(
   metric="val/total_unweighted_loss", mode="min", scheduler=...),
   run_config=...)`, and per-trial resources. Extracts and reports the best
   config(s).
3. **`train_func(config)`**. Instantiates the data path (`VelocityDataModule`
   for real data, or the synthetic `LitModule` path), builds `LitModule` (which
   already wraps the inner model via the `Model` normalization wrapper), and a
   Lightning `Trainer` whose callbacks include Ray's `TuneReportCheckpointCallback`
   so the `self.log("val/total_unweighted_loss", ...)` metrics flow to Tune. This
   reuses almost all of the current `TrainingInterface` logic.
4. **vmap-ensemble module** (new, e.g. `smi/analysis/ensemble.py`). An
   `EnsembleModule` that holds K models of one architecture and runs them via
   `params, buffers = stack_module_state(models)`,
   `base = deepcopy(models[0]).to('meta')`,
   `fmodel(p,b,x) = functional_call(base, (p,b), (x,))`,
   `vmap(fmodel, in_dims=(0,0,None))(params, buffers, shared_batch)` -> `[K, B, 1, L]`.
   Per-model losses are computed by broadcasting the existing loss over the K
   dimension; per-model learning rates / loss weights are stacked scalars. The
   optimizer updates the stacked params (a functional optimizer over the stacked
   tensors, or N param-groups). Wrap the ensemble as a single `LightningModule`
   whose `training_step` performs the vmap, so logging/checkpointing/selection of
   the best member are preserved.
5. **Synthetic batch sharing.** Refactor `SyntheticLitModule._generate_synthetic_batch`
   so the ensemble's `training_step` generates one batch on the GPU and feeds it
   to all K members (the `in_dims=(0,0,None)` broadcast above). On-GPU generation
   is unchanged otherwise.
6. **Logging / selection.** Keep `WandbLogger`; each Ray trial logs its run, and
   the ensemble logs per-member metrics under one run. Best config/member is read
   from Tune results and exported via the existing `Model.to_torchscript()`.

## 5. The two workflows, concretely

**Workflow 1 - synthetic broad architecture search.**
Ray Tune over the architecture space (model type + structural knobs), synthetic
on-GPU data, short per-trial budget, ASHA, fractional GPU. Because synthetic data
is cheap, throughput is high. Optionally each trial trains a small vmap-ensemble
over seeds to denoise architecture comparisons. Output: a ranked shortlist of
architectures.

**Workflow 2 - real-data focused hyperparameter search.**
Take the shortlisted architectures and run Ray Tune over hyperparameters (lr,
weight decay, schedule, loss weights, dropout, ...) against real HDF5 data via
`VelocityDataModule`, packing as many trials as fit on 24 GB and using ASHA to
kill weak configs early. vmap-ensembles fan out seed/lr per architecture. Output:
best hyperparameters per architecture, plus exported TorchScript models.

## 6. Reproducibility, checkpointing, selection

- Deterministic seeds per trial and per ensemble member (the ensemble's distinct
  per-member inits are exactly what `stack_module_state` captures).
- ASHA + `TuneReportCheckpointCallback` for checkpoint-aware early stopping;
  best-config extraction from the Tuner result grid; persist to the Ray results
  dir and W&B.

## 7. Testing (dev Mac, no CUDA)

- Every path CPU-runnable with tiny `K`, short `sequence_length`, 1-2 epochs.
- Unit tests: vmapped ensemble forward equals the looped per-model forward
  (`allclose`); shared-input `in_dims` shape correctness; search-space
  construction from config; a `train_func` smoke test (1 trial, CPU, 1 step).
- Mark GPU-throughput tests as CUDA-only/skipped on the Mac. All tests run under
  the existing thread-pinned `tests/conftest.py`.

## 8. Dependencies

- Add `ray[tune]` (and optionally `optuna`) in a dedicated `tune` pixi feature /
  environment so it stays out of the default/library env. Ray's CPU path runs on
  the Mac for tests; GPU packing runs on the rig. Pin Ray/Lightning/W&B versions
  known-compatible on the rig.

## 9. Phasing

- **Phase 1 (highest value):** Ray Tune replaces the YAML grid - single-model
  trials, ASHA, fractional GPU, both data paths, best-config extraction.
- **Phase 2:** `EnsembleModule` + synthetic batch sharing; usable standalone and
  as a Ray trial's inner loop.
- **Phase 3:** Optuna search, result reporting, automated best-model TorchScript
  export.

## 10. Risks and open questions

- CUDA-context overhead caps Ray concurrency for tiny models (mitigate with vmap
  for same-architecture fans and a concurrency cap).
- vmap per-model optimizer / per-model lr plumbing is the main new complexity.
  Buffers under vmap (e.g. BatchNorm running stats) need care; our models lean on
  LayerNorm, which is simpler - verify per architecture. `torch.func` +
  `torch.compile` interaction should be checked on the rig.
- Whether to wrap the ensemble as a `LightningModule` (keeps logging/checkpoint)
  or a custom loop (simpler) - leaning LightningModule.
- Ray + Lightning + W&B version compatibility on the rig.
- DataLoader `num_workers` on the rig (Linux fork is fine; the macOS segfault is
  dev-box-only).

## 11. Implementation status

- **Phase 1 (done):** `smi/analysis/tune_search.py` (`build_param_space`,
  `train_func`, `run_search` with ASHA + fractional GPU), the YAML grid removed
  from `training_interface.py` (`from_yaml` returns a single config), a `--search`
  mode in `main.py`, `configs/tune-example.yaml`, and CPU smoke tests in
  `tests/test_tune_search.py`.
- **Phase 2 (done):** `smi/analysis/ensemble.py` (`EnsembleModule`: shared-batch
  vmap forward, per-member loss, shared- or per-member-lr optimization,
  best-member export), a reusable `generate_synthetic_batch` extracted from
  `SyntheticLitModule`, and `tests/test_ensemble.py` (incl. vmap==looped allclose).
- **Composition (done):** a Ray Tune trial whose config has an `ensemble` section
  trains a vmap-ensemble inner loop instead of a single model -- `train_func`
  routes to `_train_ensemble`, which builds a `SyntheticEnsembleModule` (shared
  on-GPU synthetic batches) or a real-data `EnsembleModule` (`VelocityDataModule`)
  and reports the best member's `val/total_unweighted_loss` so ASHA compares it
  with single-model trials. See `configs/tune-ensemble-example.yaml`.
- **Optuna (done):** `run_search(search_alg='optuna')` (and `main.py
  --search_alg optuna`) uses Ray's `OptunaSearch` (TPE) over the same space
  instead of random sampling.
- **Best-model export (done):** single-model trials checkpoint each validation;
  `export_best_model(best_result, path)` (and `main.py --export_best PATH`) loads
  the best trial's checkpoint and writes a TorchScript artifact (baking out
  training-time weight parametrizations so it scripts cleanly). Ensemble trials
  are not checkpointed; use `EnsembleModule.best_member_model` for those.
