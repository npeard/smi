// Standalone report: what a non-learned baseline achieves on this data.
//
// Build:  pixi run report-baseline  (or, from report/: snakemake build/baseline.pdf)
//
// STATUS: skeleton. The section structure and the argument each section owes
// the reader are fixed here; the measured numbers and the figures that carry
// them are filled in once the baseline-fit results artifact exists. Anything
// still unwritten is wrapped in `todo`, which renders as a visible marker so
// an unfinished section cannot be mistaken for a finished one.
//
// ASCII only, per the repo convention.

#import "lib/theme.typ": fig, report-theme, takeaway, todo

#show: report-theme.with(
  title: "A non-learned baseline for velocity estimation",
  subtitle: "What the Michelson model alone recovers, and what it does not",
  author: "Nolan Peard",
  date: "Draft",
)

= Why this document exists

A velocity RMSE from a neural network is uninterpretable on its own. It needs
a number beside it from a method that does not learn, computed on the same
shots and reported in the same units. This document produces that number and
says precisely what it does and does not establish.

The question it answers is a methodological one that came up while planning
the baseline: should each shot be scored by its *likelihood* under a Michelson
model, or should the model actually be *fit* to each shot? Section 2 argues
these are the same computation, which is why the simple option is also the
rigorous one.

= Likelihood versus fit

#todo[
  The crux argument. Under additive Gaussian residuals with fixed variance,
  the negative log-likelihood of a shot is the sum of squared residuals up to
  an additive constant and a positive scale factor. So "evaluate the
  likelihood under the model" and "fit the model and report MSE" rank shots
  identically and are, to within constants, the same number. Spell that out
  with the algebra, then state the practical consequence: fit in closed form,
  report MSE, and the likelihood interpretation comes for free.
]

= The two measurements

== Method 1: forward-model residual

Displacement is known from the drive voltage, so the interference signal it
predicts is known up to a per-shot amplitude and phase. Both enter linearly
once the cosine is expanded into its cosine and sine components, so the fit is
a two-parameter least-squares solve with a closed form -- no optimizer, no
initial guess, no convergence to check.

The residual RMSE this leaves is the honest floor: it is what remains after
the model has been given the right answer for the displacement.

#todo[
  Report residual RMSE per channel for both acquisitions, with N stated.
]

== Method 2: multi-wavelength inverse

The actual baseline. Given only the three photodiode signals, unwrap phase
using the wavelength diversity to resolve fringe ambiguity, then differentiate
for velocity. Scored in the same units the network is scored in.

#todo[
  Report velocity RMSE for free space and mm fiber, with N stated, alongside
  the method-1 floor from the same shots.
]

#takeaway[
  Method 2 is the number that goes beside the network. Method 1 is what tells
  you how to read it: a poor method-2 result with a good method-1 residual
  means the inverse is the weak link, whereas both being poor points at the
  data or the calibration.
]

= Results

#todo[
  Results table -- method, acquisition, N, RMSE -- generated from the results
  artifact, plus a figure showing a representative shot with the true and
  recovered velocity overlaid, and the residual distribution across shots.
]

= Free space versus multimode fiber

The fiber path is the deployment-relevant one and the harder one. The raw data
already shows why: the fiber acquisition carries substantially less
high-frequency content than free space in the same channel.

#fig(
  "freespace_vs_fiber",
  width: 100%,
  caption: [
    635 nm photodiode. (a) one shot from each acquisition, DC removed.
    (b) mean amplitude spectrum over N = 64 shots. Mode scrambling in the
    fiber low-passes the fringe signal before it reaches the detector.
  ],
)

#todo[
  Quantify what that spectral loss costs the inverse: the method-2 RMSE
  difference between the two acquisitions, and whether method 1 degrades by a
  comparable factor (data-limited) or not (inverse-limited).
]

= Scope

#todo[
  State plainly what the baseline establishes -- a floor a learned model must
  beat to be worth its complexity -- and what it does not: it is not an upper
  bound on what classical processing can do, since a more careful inverse or a
  per-shot calibration could do better, and it says nothing about latency or
  about behaviour outside this waveform family.
]
