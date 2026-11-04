# Self-Mix Interferometry Tasks

The research questions are in `QUESTIONS.md`; `report/report.typ` answers
them, one section each. Question numbers below follow QUESTIONS.md. The
current spec is `docs/superpowers/specs/2026-11-02-unified-report-design.md`
(gitignored; results R1-R8), which keeps the phases P4-P6 and infrastructure
I1-I2 of `2026-10-29-synthetic-research-program-design.md`.

---

## 1. Done: code organization and separation of concerns (branch `restructure`)

`smi.analysis` is gone. The package is layered `core` < `theory` < `data` /
`estimators` < `learning` < `experiments`, with `acquisition` beside
`experiments`; `tests/test_import_contracts.py` enforces the direction.

- [x] **R0. Design spec.** Target packages and import direction fixed, enforced
  by `[tool.importlinter]` contracts (including "only `smi.acquisition` imports
  `stemlab`", written over all of `smi` so a new module cannot slip through).
- [x] **R1. Shared core.** `smi.core` holds acquisition config, kinematics and
  metrics; the simulator receives acquisition config instead of importing the
  hardware config.
- [x] **R2. Data layer.** `smi.data` with one HDF5 `schema` module; datasets no
  longer import the analytic decode.
- [x] **R3. Split `analysis/`.** `estimators`, `learning`, `experiments` and
  `theory`; `baseline_fit` and `gain_sweep` split into estimator, sweep and CLI.
- [x] **R4. Scripts and artifacts out of the package.** CLIs in `scripts/`;
  `report/results/`, `data/frozen/`, `runs/` at the repo root (configs and
  grids became Python registries in `smi/`),
  resolved through `smi.core.paths`; Snakefile and figure scripts follow.
- [x] **R5. Dead paths decided.** Ray Tune, the vmap ensemble, VICReg and the
  `torch.compile` branch deleted; `LitModule` now requires `target`;
  `run_training.bat` is gone.
- **Constraint held: behavior-preserving.** Suite green, grid fingerprints
  unchanged, `synthetic_values.json` byte-identical.

Carried-over cleanup:

- [x] `notebooks/causal_inductive_bias_test.ipynb` (imported a nonexistent
  `models.utcn`) deleted.
- [x] The `matched/` and `displacement/` entries of `smi/learning/configs.py` keep a stale `data_dir`
  (`./analysis/data`) because it is hashed into run fingerprints; callers pass
  `dataset_path` explicitly. Retire it together with a fingerprint-golden
  regeneration.
- [x] Argument parsing moved out of the library `main`s (`run_baselines`,
  `gain_sweep`, ...) into `scripts/`; tests call the plain functions.

---

## 2. Next: acquisition and data handling (spec I1)

- [ ] **Red Pitaya AWG and acquisition API as its own repo.** The seam is
  built: `packages/redpitaya` (`stemlab`) is standalone and SMI's side is
  `smi.acquisition`; what remains is moving it to its own repo. It sends an
  arbitrary sample array on a channel at a sample rate with a trigger
  configuration, acquires N samples on channels, daisy-chains boards, and
  returns arrays plus metadata. It knows nothing about SMI waveforms ("one
  purpose per module").
  - SMI keeps waveform design (`Waveform`, `CoilDriver`), the HDF5 schema,
    and SMI-specific views.
  - Its own spec and plan, after R1 has passed constants in rather than
    importing them.
- [ ] **NiceGUI (FastAPI) GUI** replacing the ~800-line Matplotlib GUI in
  `manager.py`. Try a reducer-style state store as an experiment, judged on
  whether it removes concrete hardware-callback and UI races.
- [ ] **Continuous acquisition.** Evaluate continuous streaming (the old
  Julia RedPitayaDAQServer item) against the new API, rather than porting
  separately.
- [ ] **Data handling for new acquisitions.**
  - Paired free-space/fiber shots of the same trajectory (Q3.3).
  - An orthogonal-polarization PD channel (Q3.2).
  - A dichroic monitor color (Q3.1).
  - Each extends the HDF5 schema; design the schema change with R2.
  - I1 must land before any of these acquisitions.

---

## 3. Then: resume the research program

Report follow-ups, from the PI's round-2 review:
- [ ] **PI review of the regenerated report** (`report/build/report.pdf`).
- [ ] **Extra seeds where the figures-only reader judged an answer only
  suggestive.**
  - A4: seeds 1-2 of TCN and TCAN at C = 0.2, about 80 min, to settle "TCAN
    best on velocity".
  - A3: seeds of 515 nm alone and the best pair at the nominal 30 dB point,
    about 6 cells, to settle the 7-10% nominal gain.
- [ ] **Trim panels the reader found beside the point** (PI's call):
  - Fig 2's alpha = 5 series;
  - Fig 5's nominal-noise panel;
  - Fig 7's flat knob panels;
  - one of Figs 6c and 10c;
  - Fig 10b;
  - Fig 11a's middle families;
  - Fig 12a.

Open science questions the synthetic run raised:
- [ ] **Per-frequency SNR test of the velocity-target rationale.** Fig 5
  shows the 1/f^2 weighting but not that the trace lacks low-frequency
  information.
- [ ] **Channel subsets at the nominal point with phase drift removed**, to
  test whether drift masks the multi-channel gain.
- [ ] **Why one Michelson channel recovers nothing** (a local ambiguity
  section 1 leaves open), and why the single-channel oracle also fails.
- [ ] **The LSTM and offline-neural-operator stalls.** Displacement
  retargeting did not rescue them (S1d). Remaining levers: initialization,
  learning rate, a velocity auxiliary loss. Mamba trains on velocity and was
  not retargeted.

Phases:
- [x] **P4: fiber simulator** (spec 4.3). A time-varying transmission
  matrix over N modes x 2 polarizations, wavelength-dependent, emitting the
  feedback coupling kappa(t) and the orthogonal-polarization power P_perp(t).
  It serves Q3.1-Q3.4. Own spec and plan.
- [x] **Q3: fiber phase scrambling.** Q3.1: monitor color, settled by Fisher
  condition number and RMSE. Q3.3: matched free-space/fiber denoiser. Q3.2: the
  P_perp side channel, corr(P_perp, |kappa|^2) against mode count.
- [x] **P5: flow matching.** Q3.3: fiber -> free-space trace. Q3.4: signal
  -> trajectory posterior, settled by coverage, CRPS and posterior-mean RMSE
  against a deep ensemble.
  Answered on `report-tightening` with a conditioning encoder: whole-shot
  context drives the gain, the flow adds calibrated bands; denoising does not
  beat the fiber regressor.

Follow-ups the Q3 runs raised:
- [ ] **Retrain the fiber arms.** Every q3_fiber net sits at the zero
  predictor, even on a static fiber, and UNetTransformer never converged.
  Levers: a curriculum from the static fiber, visibility augmentation, a
  longer budget.
- [ ] **A working classical decode at the nominal point** (C = 0.2,
  alpha = 3), to calibrate whether a static fiber keeps the information. The
  inverse fit scores 1.25 there against the zero predictor's 0.4575.
- [x] **The flow posteriors are sign-symmetric.** Fixed by the conditioning
  encoder with an auxiliary regression loss (the encoder-free field ignored
  its signals); the posterior mean takes the right sign on 95% of free-space
  shots (`q3_4_posterior_sign_enc`).
- [ ] **A whole-shot-context regressor as the fair comparator.** Train
  UNetTransformer and TCAN with a full-shot field and no crop. The encoder
  head alone (66,867-sample field, 44k params) scores 0.098 against TCAN's
  0.200 at the nominal point, so context, not family, may set every Q2
  ranking; the flow's gain over a matched whole-shot regressor is the open
  number.
- [ ] **Fading mechanism after H1-H4 rejected.** Candidates: per-shot gain
  normalization in training, amplitude-dependent fringe-shape learning
  (single-color fading already tested).
- [ ] **Rerun the Q3.1/Q3.2 monitor and P_perp arms at a low-fading point**
  (e.g. modes-1 or modes-2 of q3_fiber_sweep) where TCAN learns, since at the
  combined point every arm sits at the zero predictor.
- [ ] **Denoiser state.** The encoder denoise chain scores 0.376 against the
  fiber TCAN's 0.358 (both near zero); draws not plateaued at 32 (2.7%).
  Retry only against a fiber regressor that learns.
- [ ] **Slow figure tests.** The report-build and example-shot tests rebuild
  figures; split a fast structural subset from the full build.
- [ ] **Per-shot std in `RunRecord`**, so paired SEs need no side file
  (`report/results/per_shot_rmse.json` today).

- [ ] **P6: unblinding.**
  - Calibrate the simulator on real data (`scripts/calibrate_sim.py`).
  - Score the pre-registered predictions (commits d2b5fbc, c4707de) against
    the real acquisitions.
  - Resolve the CoilDriver drive-scale quirk (the code gives ~350x the
    published 0.375 um/ms; the simulator anchors to the published value).
  - Revisit the nominal values anchored to published numbers.
- [ ] **The R4 full ladder.** Trimmed to the nominal and two stressed rungs
  for time; q2_r4_noise ran about 3 h against an estimated 9.
- [ ] **Tabular q3_noise_free LightGBM cells** were never run.
- [ ] **An offset tolerance for the Q1.2 sign evidence.**
- [ ] **UNetTransformer's 18-49% seed spread on Q1.**
- [ ] **I2: the Google Slides deck to Typst slides** (connector auth needed;
  location to be chosen).

---

## 4. Carried over (real-data work, still open)

- [ ] **`normalization.py` provenance** (partial). Provenance is recorded;
  still open is whether the stats should come from a committed fixed
  reference subset rather than whichever 10,000-shot file is on disk. Fold
  into R2.
- [ ] **SCNN / TCN receptive-field study.** Sweep the field and the
  lookahead, against the mmfiber/free-space generalization gap. Field
  dominates capacity: field-29 TCNs score ~0.20 at 28k and 36k parameters,
  against 0.1401 for a 19k model at field 85.
- [ ] **Feature engineering for the fiber/free-space gap.** Learned models
  degrade 2.0-2.6x on fiber; the analytic decode is flat at 1.01x.
- [ ] **Exclude out-of-distribution amplitude frames.** Count them, and
  assess whether they hurt training.
- [ ] **Scaling study** (RMSE against dataset size and against parameter
  count). The harness exists. Barland & Gustave 2021 bound the useful
  parameter range near 10^5.
- [ ] **A third seed for the learned-baseline sweep.** The 15% SCNN-over-TCN
  margin at 44k on free space is about one seed spread.
- [ ] **Cavity reflectance notebook: Fisher information** and the dynamic
  range / Fisher-information tradeoff.
- [ ] **PCB for current supply and active modulation.** Evaluate, and find
  an optics hardware consultant. (The dichroic reflector item is now Q3.1.)

---

## 5. Tooling

- [ ] **ruff PLR0917** (too many positional arguments, 49 sites) is
  ignored beside PLR0913. Fix it with keyword-only signatures during R3, or
  keep the ignore deliberately.
- [ ] **Snakemake ILP scheduler** fails on Apple Silicon without Rosetta.
  Preflight suggests `scheduler: greedy` in a profile.
- [ ] **Restore the HDF5 figures through a committed raw-data extract**
  (the free-space/fiber spectrum and the other real-data figures out of the
  DAG).
- [ ] **Snakemake does not rebuild deleted figure outputs** while
  `report.pdf` is present (found on `research-questions`).
- [ ] **agent-config Windows CI** (left for now). The `report` env resolves
  Python 3.15, where `immutables` has no Windows wheel.

---
