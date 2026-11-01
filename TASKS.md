# Self-Mix Interferometry Tasks

The research program is specified in
`docs/superpowers/specs/2026-10-29-synthetic-research-program-design.md`
(gitignored; Q1-Q5, phases P0-P6, infrastructure I1-I2). P0-P3 are done; the
synthetic report is `report/synthetic.typ`. Order from here: integrate the
synthetic branch, restructure the code base, restructure acquisition and data
handling, then resume experiments (P4-P6).

---

## 1. Now: code organization and separation of concerns

`smi/analysis/` holds 20 top-level modules plus `models/`, `features/`,
`baselines/`, data artifacts and experiment grids, and it mixes the data
layer, analytic estimators, learned models, the experiment harness, one-off
scripts and acquisition. The import graph shows layering violations, not
just a flat directory:

- **The simulator depends on analysis.** `synthetic/frozen.py` imports
  `analysis.baseline_fit`.
- **The simulator and analysis depend on hardware config.**
  `synthetic/forward.py`, `synthetic/waveform.py`, `analysis/lit_module.py`,
  `datasets.py`, `ensemble.py` and `calibrate_sim.py` import
  `redpitaya.redpitaya_config` for sample-rate and buffer constants.
  `waveform.py` <-> `RedPitayaConfig` is the cycle the spec's I1 calls out.
- **Kinematics live in the coil driver.** `integrate_velocity` and
  `derivative_displacement` are imported from `synthetic.coil_driver` by
  `metrics`, `lit_module`, `evaluate`, `ensemble`, `datasets`, `baseline_fit`,
  `gain_sweep`, `delay_check` and `benchmark_loading`.
- **Metrics depend on models.** `analysis/metrics.py` imports
  `models.factory`.
- **The data layer depends on an estimator.** `analysis/datasets.py`
  imports `baseline_fit`.
- **Acquisition lives in analysis.** `analysis/generate_data.py` drives
  `redpitaya.manager`.
- **Two modules do many jobs.** `baseline_fit.py` (1876 lines) and
  `gain_sweep.py` (1376) each combine an estimator, a sweep and a CLI.
- **Artifacts live inside the package.** These include
  `smi/analysis/data/*.json`, `smi/analysis/experiments/*.yaml`,
  `smi/analysis/models/checkpoints/` and `smi/analysis/models/configs/`.

Tasks:

- [ ] **R0. Design spec** (`superpowers:brainstorming`; architectural). Fix
  the target packages and the allowed import direction between them, and
  enforce that direction with a test (e.g. `import-linter` contracts) so it
  cannot erode again.
  - A starting proposal, not a decision: `smi/core` (acquisition constants,
    units, kinematics, metrics), `smi/acquisition` (Red Pitaya, until I1
    moves it out), `smi/data` (HDF5 schema, datasets, datamodule, frozen
    sets, features and normalization), `smi/synthetic` (simulator),
    `smi/estimators` (analytic decode, gain sweep, oracle fit, tabular
    baselines), `smi/learning` (models, Lightning modules, training
    interface), `smi/experiments` (runner, evaluate, results artifact),
    `smi/theory` (bounds).
  - Settle also where grids, configs, results and checkpoints live, and
    which modules are scripts rather than library code (`inspect_data`,
    `delay_check`, `benchmark_loading`, `q1_binned`, `calibrate_sim`).
- [ ] **R1. Shared core first.** Move acquisition constants out of
  `RedPitayaConfig` into a config the simulator receives rather than imports,
  and kinematics out of `CoilDriver`. Break `synthetic -> analysis` and
  `metrics -> models`.
- [ ] **R2. Data layer.** Datasets, datamodule, frozen sets, features and
  normalization go behind one HDF5 schema module. Decouple datasets from the
  analytic decode.
- [ ] **R3. Split `analysis/`** into estimators, learning, experiments and
  theory, per R0. Break `baseline_fit.py` and `gain_sweep.py` into estimator,
  sweep and CLI.
- [ ] **R4. Scripts and artifacts out of the package.** Move CLIs to an
  entry-point or `scripts/` layer, and data, grids and checkpoints to
  top-level data directories. Update the report's Snakefile inputs.
- [ ] **R5. Decide dead or unused paths, then delete or keep deliberately:**
  - Ray Tune and the vmap ensemble (`tune_search.py`, `ensemble.py`,
    `main.py --search`; about 1k lines, used by no reported result);
  - the `torch.compile` branch, which never executes (gated on
    `sys.version_info < (3, 12)` while `requires-python >= 3.12`);
  - `LitModule`'s silent fallback to `target='velocity'` when the key is
    missing (a hidden default);
  - `run_training.bat`.
- **Constraint for R1-R5: behavior-preserving.** The full suite stays
  green, `tests/golden/grid_fingerprints.json` stays unchanged (resume must
  still match every recorded cell), and `report/build/synthetic_values.json`
  stays byte-identical. Each phase is reviewed before the next.

---

## 2. Next: acquisition and data handling (spec I1)

- [ ] **Red Pitaya AWG and acquisition API as its own repo.** It sends an
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
  - Paired free-space/fiber shots of the same trajectory (Q4b, Q5.1).
  - An orthogonal-polarization PD channel (Q4c).
  - A dichroic monitor color (Q4a).
  - Each extends the HDF5 schema; design the schema change with R2.
  - I1 must land before any of these acquisitions.

---

## 3. Then: resume the research program

Report follow-ups, from the PI's round-2 review:
- [ ] **PI review of the regenerated report** (`report/build/synthetic.pdf`).
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
- [ ] **P4: fiber simulator** (spec 4.3). A time-varying transmission
  matrix over N modes x 2 polarizations, wavelength-dependent, emitting the
  feedback coupling kappa(t) and the orthogonal-polarization power P_perp(t).
  It serves Q4a/b/c and Q5.1/5.2. Own spec and plan.
- [ ] **Q4: fiber phase scrambling.** 4a: monitor color, settled by Fisher
  condition number and RMSE. 4b: matched free-space/fiber denoiser. 4c: the
  P_perp side channel, corr(P_perp, |kappa|^2) against mode count.
- [ ] **P5: Q5 flow matching.** 5.1: fiber -> free-space trace. 5.2: signal
  -> trajectory posterior, settled by coverage, CRPS and posterior-mean RMSE
  against a deep ensemble.
- [ ] **P6: unblinding.**
  - Calibrate the simulator on real data (`smi/analysis/calibrate_sim.py`).
  - Score the pre-registered predictions (commits d2b5fbc, c4707de) against
    the real acquisitions.
  - Resolve the CoilDriver drive-scale quirk (the code gives ~350x the
    published 0.375 um/ms; the simulator anchors to the published value).
  - Revisit the nominal values anchored to published numbers.
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
  an optics hardware consultant. (The dichroic reflector item is now Q4a.)

---

## 5. Tooling

- [ ] **ruff PLR0917** (too many positional arguments, 49 sites) is
  ignored beside PLR0913. Fix it with keyword-only signatures during R3, or
  keep the ignore deliberately.
- [ ] **Snakemake ILP scheduler** fails on Apple Silicon without Rosetta.
  Preflight suggests `scheduler: greedy` in a profile.
- [ ] **agent-config Windows CI** (left for now). The `report` env resolves
  Python 3.15, where `immutables` has no Windows wheel.

---
