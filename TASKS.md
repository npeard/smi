# Self-Mix Interferometry Tasks

---

## Infrastructure & Data Pipeline

- [ ] **`normalization.py` was regenerated from a different dataset** The stats were previously generated from `circuit-noise-600.h5` (600 shots), which no longer exists in `smi/analysis/data/`. They are now generated from `free-space-synchro_10k.h5` (10000 shots), so the three `pd_*` input stats and `displacement` moved as well as `velocity` -- a deliberate, sanctioned change, but it means any checkpoint trained before this point has different input scaling baked in and is not directly comparable. Worth deciding whether the stats should come from a fixed, committed reference subset so that regenerating them is reproducible rather than dependent on which acquisitions happen to be on disk.
- [ ] **Investigate whether using Ray Tune fractional device allocation** can help with synthetic data parameter sweeps. Generating synthetic data on GPU is fast and we can generate small amounts of data at the same time, so we could use Ray to train many models at once with different hyperparameters and select the best one.
- [ ] **Check for signal delay offset between velocity and PD channels** -- if large enough, offset corrupts supervision signal and degrades prediction quality
---

## Model Architecture

- [ ] **Remove conditional downsampling in `TemporalBlock`** -- always route through MLP; also evaluate whether stacking two same-channel temporal blocks in a row is useful or whether every block should vary channel count
- [ ] **Correct `receptive_field` calculation** -- current formula does not account for the fact that each `TemporalBlock` contains TWO convolutional layers
- [ ] **Model baselines** Compare TCNs to simpler models to justify the expense of the modeling approach. How well do an MLP and LightGBM tree perform? (The *non-learned* baseline is done -- see "Establishing baselines" below. This item is the learned-but-simpler comparison, still open.)

---

## Experiments & Benchmarks

- [ ] **Baseline: SCNN vs TCN vs Barland at matched parameter count** -- direct apples-to-apples comparison
- [ ] **SCNN / TCN receptive field study** -- sweep receptive field size and lookahead window; quantify effect on mmfiber/freespace generalization gap
- [ ] **Feature engineering for fiber/freespace gap** -- investigate whether any simple hand-crafted features dramatically close the mmfiber/freespace distribution shift
- [ ] **Exclude OOD amplitude frames from dataset** -- identify and count bad frames with raw signal amplitude out of distribution; assess whether they hurt training or force the model to exploit multi-channel information

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
