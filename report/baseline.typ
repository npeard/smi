// Standalone report: what a non-learned baseline achieves on this data.
//
// Build:  pixi run report-baseline  (or, from report/: snakemake build/baseline.pdf)
//
// Every number in this document comes from smi/analysis/data/baseline_results.json,
// written by smi.analysis.baseline_fit. Nothing here is estimated or recalled:
// if a claim needs a number the artifact does not contain, the claim does not
// get made. Figures read that same artifact and recompute nothing.
//
// ASCII only, per the repo convention.

#import "lib/theme.typ": fig, report-theme, takeaway

#show: report-theme.with(
  title: "A non-learned baseline for velocity estimation",
  subtitle: "What the Michelson model alone recovers, and what it does not",
  author: "Nolan Peard",
  date: "Draft",
)

= Why this document exists

A velocity RMSE from a neural network is uninterpretable on its own. It needs
a number beside it from a method that does not learn, computed on the same
shots and in a unit the two can actually be compared in. This document
produces that number and says precisely what it does and does not establish.

The question it answers is a methodological one that came up while planning
the baseline: should each shot be scored by its *likelihood* under a Michelson
model, or should the model actually be *fit* to each shot? Section 2 argues
these are the same computation, which is why the simple option is also the
rigorous one.

The result is worse than expected, and in an informative way. The Michelson
model does not explain this data well even when it is handed the true
displacement, so the baseline's error is mostly a statement about the model
and the data rather than about the estimator.

= Likelihood versus fit

Write one shot's samples as $y_n$ and the model's prediction as
$f_n (theta)$, with $theta$ the parameters being fit. Take the residuals to be
independent Gaussians of fixed variance $sigma^2$ -- the standard assumption,
and the one under which "likelihood" is even defined here. The likelihood of
the shot is then

$ L(theta) = product_(n=1)^N 1 / sqrt(2 pi sigma^2)
  exp(- (y_n - f_n (theta))^2 / (2 sigma^2)), $

and its negative logarithm is

$ - ln L(theta) = 1 / (2 sigma^2) sum_(n=1)^N (y_n - f_n (theta))^2
  + N / 2 ln(2 pi sigma^2). $

The second term does not contain $theta$ or the data, and $1 \/ 2 sigma^2$ is
a positive constant. So the negative log-likelihood is the summed squared
residual up to an additive constant and a positive scale -- which is to say,
up to nothing that can change an ordering. Maximizing the likelihood over
$theta$ and minimizing MSE over $theta$ return the same $theta$; ranking shots
by likelihood and ranking them by MSE produce the same ranking.

#takeaway[
  *Recommendation: fit each shot in closed form and report MSE.* The two
  options in the original question are not two options. The fit is where the
  work is, MSE is what it leaves behind, and the likelihood reading comes free
  with it -- so take the route that has a closed form and no convergence to
  check. Per-shot MSE is reported throughout this document as an RMSE, and as
  an $R^2$ where a scale-free version is more readable.
]

The one thing the equivalence does hide is $sigma^2$. Treating it as fixed and
known is what makes MSE and likelihood interchangeable; if the noise level
genuinely varied shot to shot and that variation mattered, the constant term
would stop being constant. Nothing in what follows depends on that
refinement.

= The two measurements

Both run over N = 200 shots from each acquisition, decoded on a GPU
(`cuda:0`), 16384 samples per shot at 488.28 kHz. Shot-level metrics are
summarized by their median unless stated otherwise; section 5 explains why the
median and not the mean.

== Method 1: forward-model residual

Displacement is known from the drive voltage, so the interference signal it
predicts is known up to a per-shot amplitude and phase. Both enter linearly
once the cosine is expanded into its cosine and sine components, so the fit is
a two-parameter least-squares solve with a closed form -- no optimizer, no
initial guess, no convergence to check. It is a global optimum by
construction.

The residual this leaves is the honest floor: it is what remains after the
model has been given the right answer for the displacement. No inverse,
learned or otherwise, can do better than a model that already knows the
answer.

== Method 2: multi-wavelength inverse

The actual baseline. Given only the three photodiode signals, resolve fringe
ambiguity using the wavelength diversity across 635, 675 and 515 nm, decode
displacement on a 1601-point grid, then differentiate for velocity. Scored on
the same quantity the network predicts, in physical units -- see section 4.2
on which velocity unit to compare against a training log.

The overall sign of the displacement is not scored. A cosine is even, so
$d(t)$ and $-d(t)$ produce identical fringes; the sign is genuinely
unobservable and penalizing it would measure nothing.

= Results

#fig(
  "baseline_summary",
  width: 100%,
  caption: [
    N = 200 shots per acquisition. *Left:* method 1's explained variance per
    channel, with displacement known. *Middle:* method 2's displacement error
    beside the RMS of the displacement it is trying to recover. *Right:* the
    same error normalized by that RMS; the dashed line is the error a
    predictor that always outputs zero would achieve.
  ],
)

== Method 1: the model does not explain the data

Median $R^2$ per channel, over N = 200 shots, with the true displacement
supplied:

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header([acquisition], [635 nm], [675 nm], [515 nm]),
  [free space], [0.199], [0.236], [0.089],
  [mm fiber], [0.376], [0.188], [0.265],
))

Between 9% and 38% of the variance, depending on channel and acquisition.
Equivalently, the residual RMSE is 0.79 to 0.95 times the signal's own RMS in
every one of the six channel/acquisition combinations: fitting the model
removes only a small fraction of the signal.

This is the headline finding, and it is not a result about an estimator. The
displacement was not estimated here -- it was known. A Michelson model with
free amplitude and phase, handed the correct displacement, still fails to
account for most of what the photodiodes recorded.

== Method 2: the baseline number

#align(center, table(
  columns: 6,
  align: (left, right, right, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header(
    [acquisition],
    [disp. RMSE (um)],
    [vel. RMSE (um/s)],
    [vel. RMSE (um/ms)],
    [disp. NRMSE],
    [corr.],
  ),
  [free space], [0.433], [488.1], [0.488], [1.10], [0.53],
  [mm fiber], [0.462], [550.2], [0.550], [1.18], [0.42],
))

Medians over N = 200. For scale, the true displacement RMS is 0.351 um (free
space) and 0.375 um (mm fiber), and the true velocity RMS is 348 and
369 um/s. The NRMSE column divides each shot's displacement error by the
standard deviation of that shot's true displacement, so *NRMSE above 1 means
the decode is worse than predicting zero* -- and both acquisitions sit just
above 1. The displacement correlations, 0.53 and 0.42, say the decode is not
noise: it tracks the real motion partway and then loses it.

*Velocity is tabulated twice on purpose.* The results artifact stores um/s,
but `LitModule.loss_function` multiplies both prediction and target by 1e-3
before taking the MSE, so a velocity loss logged during training is in um/ms.
Comparing this baseline to a logged network number without noticing that is a
factor of 1000. The repository has an open task to settle on one convention;
until it does, the um/ms column is the one to place beside a training log.

#takeaway[
  Read the two together. Method 2's NRMSE near 1.1 could mean a weak inverse
  or it could mean the data does not match the model. Method 1 settles it:
  with displacement *known*, the model still explains under 40% of the
  variance. The error is dominated by model/data mismatch, not by the
  estimator.
]

The synthetic control is what justifies blaming the model rather than the
decoder. The same decoder, run on signals genuinely generated by the
Michelson forward model, recovers
displacement to a median NRMSE below 0.06 with at least 75% of shots under
0.10 -- a factor of roughly twenty better than it manages on real shots. That
bound is asserted by the test suite (`tests/test_baseline_fit.py`) and so is
re-checked on every run, rather than being a number remembered from a session.
So the decoder does its job; what fails is the agreement between model and
data.

== The amplitude trend

#fig(
  "baseline_amplitude_trend",
  width: 100%,
  caption: [
    Method-1 median $R^2$ binned by the drive's peak-to-peak displacement
    excursion, per channel, N = 200 shots per acquisition. Bin populations are
    marked. Every channel in both acquisitions collapses above 2 um; the free
    space panel falls monotonically across all four bins, while the two fiber
    channels that peak in the 0.3-1 um bin do not.
  ],
)

Splitting the method-1 fits by how far the speaker actually moved shows the
failure is strongly amplitude dependent. In free space the trend is monotone
across all four bins in every channel: 635 nm falls 0.567, 0.569, 0.203,
0.032 from the smallest bin to the largest; 675 nm falls 0.849, 0.712, 0.236,
0.030; 515 nm falls 0.443, 0.322, 0.089, 0.013.

The fiber acquisition is not monotone across all four bins -- 635 nm and
515 nm both peak in the 0.3-1 um bin rather than the smallest one (0.440 then
0.628, and 0.361 then 0.707) -- but the large-excursion collapse is
identical: every channel in both acquisitions drops to $R^2$ between 0.03 and
0.11 above 2 um, from several times that below 1 um. The smallest bin is also
the thinnest (n = 20 and 23 against 66 to 72 in the largest), so the
non-monotonicity at the low end rests on fewer shots than the collapse at the
high end does.

What is consistent across all six combinations is the direction: the model
describes small excursions and fails on large ones. That is the shape of an
error that accumulates with the number of fringes crossed, which points at the
displacement calibration -- a gain or transfer-function error in the
drive-to-displacement conversion would grow with excursion exactly this way,
since the phase error is proportional to the displacement error. It is also
consistent with per-shot phase drift, which has more time to matter across a
longer sweep.

Both are testable and neither has been tested. The trend is stated here as the
most actionable thing the baseline found, not as a diagnosis.

= How to read these numbers

Three things would mislead a reader who took the artifact at face value.

*Use the medians.* The mean `displacement_nrmse` is 4.58 (free space) and 4.80
(mm fiber), around four times the medians of 1.10 and 1.18. The means are
artifacts of the normalization, not evidence of worse performance: NRMSE
divides by each shot's true displacement RMS, and the quietest shots in these
subsets have a true displacement RMS of 0.003 um (free space) and 0.001 um
(mm fiber). A near-motionless shot puts a near-zero number in the denominator
and produces an NRMSE in the hundreds -- the maxima are 262 and 267 -- which
drags the mean without describing any typical shot. Every headline number in
this document is a median for that reason.

*The synthetic control is noiseless.* It has no measurement noise and no
within-shot phase drift, so it establishes that the decoder is sound on clean
fringes generated by the model it assumes. It does not establish
noise-robustness. Some unknown part of the real-data gap could still be
estimator sensitivity to noise rather than model mismatch, and this data
cannot separate those two.

*The fiber gap does not appear here.* Free space and mm fiber are comparable
at this level of performance -- 0.433 against 0.462 um displacement RMSE,
1.10 against 1.18 NRMSE, and method-1 $R^2$ that is actually higher for the
fiber in two channels of three. Whatever makes the fiber path harder for a
learned model is not visible in the baseline. The band-limiting below is real
in the raw data; it is not costing the baseline anything measurable, because
the baseline is limited by something larger.

= Free space versus multimode fiber

The fiber path is the deployment-relevant one and the expected-harder one. The
raw data does differ: the fiber acquisition carries less high-frequency
content than free space in the same channel.

#fig(
  "freespace_vs_fiber",
  width: 100%,
  caption: [
    635 nm photodiode. (a) one shot from each acquisition, DC removed.
    (b) mean amplitude spectrum over N = 64 shots. Mode scrambling in the
    fiber low-passes the fringe signal before it reaches the detector.
  ],
)

That spectral difference does not translate into a baseline difference, per
the previous section. Both acquisitions are limited by model/data mismatch,
which is large enough to hide whatever the band-limiting costs. If the fiber
gap is real for a learned model, the baseline is not sensitive enough to see
it, and a comparison at this level should not be cited as evidence either way.

= Scope

*What this establishes.* A floor. Method 2's displacement RMSE of 0.433 um
(free space) and 0.462 um (mm fiber), and velocity RMSE of 488 and 550 um/s
-- equivalently 0.488 and 0.550 um/ms -- over N = 200 shots, are what a
closed-form classical inverse achieves with no training. A learned model has to beat these to justify its complexity, and
because the NRMSE sits just above 1, it has to beat predicting zero as well --
a lower bar than it sounds, and one that should be checked rather than
assumed. Method 1 further establishes that the Michelson forward model, as
currently calibrated, accounts for under 40% of the recorded variance even
with displacement known.

*What this does not establish.* It is not an upper bound on classical
processing: a better inverse, a per-shot calibration, or a corrected
drive-to-displacement transfer function could all do better, and the amplitude
trend suggests the last of those is worth trying first. It says nothing about
latency, and nothing about waveform families outside the one these
acquisitions sweep. It does not prove the decoder is noise-robust, only that
it is correct on clean model-generated fringes. And it does not speak to the
free-space/fiber gap, which is invisible at this error level.

Improving the inverse is therefore the wrong next step. The thing to chase is
why a model that already knows the displacement cannot fit the data, since
both of these numbers depend on that gap.
