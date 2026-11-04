# Research questions

The questions `report/report.typ` answers, one major section each. Each is
answered first analytically, then on synthetic data, then on real data. Each
tier makes a prediction for the next, and a robust failure of that prediction
is a result rather than a defect. This file changes only when a result
invalidates a question or raises a new one; `TASKS.md` is the working log.

## Q1. Multiple channels

Do multiple interferometer channels improve trajectory estimation, and by how
much?

- **Q1.1** Does accuracy improve monotonically with channel count, and what
  does one more interferometer buy?
- **Q1.2** Is an array of distinct colors better than an array of one color?

## Q2. Estimators

Which estimator retrieves displacement and velocity trajectories with the
highest fidelity?

- **Q2.1** What is each model's accuracy floor on noiseless synthetic
  trajectories, ridge regression and LightGBM included?
  - **Q2.1.1** Does the self-mixing nonlinearity help or hurt estimation
    accuracy?
- **Q2.2** As noise is turned on, which model stays most robust?
- **Q2.3** A model frozen after calibration may be moved to a new site. Does
  it still estimate trajectories under a changed coupling, wavelength
  (operating temperature) or phase offset?

## Q3. Multimode fiber

How fast does trajectory accuracy degrade as each multimode-fiber impairment
(static coupling loss, speckle drift, path drift) turns on, and why? In
simulation static coupling loss dominates, so the channel questions are posed
against it.

- **Q3.1** Where the fringes survive the coupling loss, does a dichroic
  monitor color that sees only the fiber improve retrieval by decorrelating
  the path drift from the other channels?
- **Q3.2** Does the orthogonal-polarization intensity, a fade indicator,
  recover the accuracy lost to static coupling loss?
- **Q3.3** Given paired free-space and fiber measurements of the same
  trajectories, can a flow-matching model denoise the fiber signals?
- **Q3.4** Using conditional flow matching for the trajectory posterior, does
  fiber scrambling broaden the posterior uniformly or split it into discrete
  trajectories?

## Q4. Sagnac self-mixing accelerometer

A common-path Sagnac loop cancels reciprocal fiber phase in its conventional
readout, but read through the laser's self-mixing the reciprocal phase returns
as the feedback phase. Does the self-mixing nonlinearity still enhance its
acceleration sensing? (Analytic only.)
