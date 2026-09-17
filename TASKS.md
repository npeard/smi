# Self-Mix Interferometry Tasks

---

## Infrastructure & Data Pipeline

- [ ] **`normalization.py` was regenerated from a different dataset** -- PARTIAL. Provenance is now recorded in the generated header (source file, size in bytes, shot count, selection), so regeneration is reproducible. The stat VALUES were deliberately left unchanged, because moving them would re-scale every model and break comparability with the analytic baseline artifact. Still open: whether the stats should come from a committed fixed reference subset rather than all 10,000 shots of whichever file is on disk.
- [ ] **Investigate whether using Ray Tune fractional device allocation** can help with synthetic data parameter sweeps. Generating synthetic data on GPU is fast and we can generate small amounts of data at the same time, so we could use Ray to train many models at once with different hyperparameters and select the best one.
- [x] **Check for signal delay offset between velocity and PD channels** -- DONE. Verdict: no correctable delay. All six dataset/channel pairs land within 8 samples (<= 16 us) of zero lag. The two channels clearing the correlation threshold (free-space RP2_CH2 peak 0.732, RP2_CH1 peak 0.398) report 0 and -1 samples. No correction applied or needed.
---

## Model Architecture

- [x] **Remove conditional downsampling in `TemporalBlock`** -- DONE. The residual now always routes through a 1x1 conv. Reference config `[16,16,32,32,64,64]` k=5 moved 98,305 -> 104,065 parameters.
- [x] **Correct `receptive_field` calculation** -- DONE. Each TemporalBlock applies two convolutions at one dilation, so the old formula reported half the true field. Now `2*(k-1)*sum(dilations)+1`, verified against a gradient probe that measures the field empirically rather than restating the formula. TCAN needed a `-1` so its encoder+decoder sum does not double-count the shared centre sample.
- [x] **Model baselines** -- DONE. On free-space: MLP 0.1275, LightGBM 0.2483, Ridge 0.3837 um/ms, against the analytic decode's 0.5611. All three beat the analytic baseline; all three lose to the TCN's 0.1116. The MLP evaluates at window stride 1, so its margin over LightGBM measures what windowed context buys over trees on the same features.

---

## Experiments & Benchmarks

- [x] **Baseline: SCNN vs TCN vs Barland at matched parameter count** -- DONE. The ported Barland 2021 architecture is the strongest model on both acquisitions: free-space 0.0814 against SCNN-44k 0.0953 and TCN-44k 0.1116, and mmfiber 0.1643 against 0.2471 and 0.2540. That is 27% and 35% ahead of the repository's causal TCN at a comparable budget. Barland and the MLP are evaluated at window strides 8 and 1; at the original stride 256 both were measuring their own output discretization (oracle floor 0.1866 on free-space) rather than their architecture. See `report/report.typ` section 9.
- [ ] **SCNN / TCN receptive field study** -- sweep receptive field size and lookahead window; quantify effect on mmfiber/freespace generalization gap. Motivated by the measured finding that field dominates capacity: the field-29 TCNs score ~0.20 on free-space at both 27,985 and 35,585 parameters, while a 19,329-parameter model at field 85 scores 0.1401.
- [ ] **Feature engineering for fiber/freespace gap** -- investigate whether any simple hand-crafted features dramatically close the mmfiber/freespace distribution shift. Now better motivated: the measured degradation is 2.28x for the TCN and 2.59x for the SCNN, while the analytic decode is flat at 1.01x. No learned model is exempt -- Barland degrades 2.02x despite winning both datasets -- so the gap is a property of resolving fine fringe structure, not of one architecture.
- [ ] **Exclude OOD amplitude frames from dataset** -- identify and count bad frames with raw signal amplitude out of distribution; assess whether they hurt training or force the model to exploit multi-channel information
- [ ] **Scaling study** (RMSE against dataset size and against parameter count) -- deferred from the learned-baselines session by decision, not by failure. Most of the harness now exists (`smi/analysis/run_baseline_sweep.py`, `smi/analysis/results_artifact.py`, `smi/analysis/metrics.py`). Note that Barland & Gustave 2021 report that going beyond ~10^5 parameters "does not lead to better training and may even lead to worse predictions", which bounds how far up the parameter axis is worth sweeping.
- [ ] **A third seed for the learned-baseline sweep.** The sweep ran 2 seeds. The 15% SCNN-over-TCN margin at 44k on free-space is about one seed-spread wide and should not be leaned on until a third seed settles it. Every cross-architecture conclusion is many spreads wide and is already safe.
- [ ] **`torch.compile` has never executed.** It is gated on `sys.version_info < (3, 12)` in `lit_module.py` while `requires-python` is `>=3.12`, so the condition is always false. Every sweep record carries `compiled: false`. Decide whether to enable it deliberately and measure, or remove the dead branch.

---

## Analysis & Presentation

- [ ] **Cavity reflectance notebook: Fisher information** -- enhance notebook to compute Fisher information and characterize the dynamic range / FI tradeoff
- [ ] **Block diagram visualization of network** -- set up a clean block diagram rendering of the model architecture
- [ ] **Incorporate smi paper repo into Typst report**

---

## Hardware

- [ ] **Dichroic reflector for reference wavelength to deconvolve phase disruptions from fiber** Might be simpler passive approach to removing phase distortion from the fiber compared to modulation techniques. Test in simulation first.
- [ ] **PCB design for current supply / active modulation** -- evaluate whether a custom PCB improves sensing performance and scales to more diodes; find an optics hardware expert to consult
- [ ] **Continuous data acquisition via Julia / Red Pitaya DAQ Server** -- evaluate whether continuous acquisition is worth porting to Julia: [RedPitayaDAQServer docs](https://tknopp.github.io/RedPitayaDAQServer/dev/scpi.html)
