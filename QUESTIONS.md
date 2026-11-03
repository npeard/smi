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

Coupling through multimode fiber scrambles phase, coupling and polarization
through its path length and modal dispersion. How can trajectories measured
through fiber be retrieved most faithfully?

- **Q3.1** A dichroic at the distal face can return one color that sees only
  the fiber. Does that fiber-only channel improve retrieval by decorrelating
  the fiber phase from the other channels?
- **Q3.2** A polarizing beamsplitter can send the returned light scrambled
  into the orthogonal polarization to its own photodiode. Does that intensity
  carry information that aids trajectory estimation?
- **Q3.3** Given paired free-space and fiber measurements of the same
  trajectories, can a flow-matching model denoise the fiber signals?
- **Q3.4** Using conditional flow matching for the trajectory posterior, does
  fiber scrambling broaden the posterior uniformly or split it into discrete
  trajectories?

## Q4. Sagnac self-mixing accelerometer

A common-path Sagnac self-mixing interferometer should be insensitive to fiber
noise, and so buildable on single-mode fiber with no user-facing ports. Does the
self-mixing nonlinearity enhance its acceleration sensing? (Analytic only.)
