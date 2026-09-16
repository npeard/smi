// Standalone report: what a non-learned baseline achieves on this data.
//
// Build:  pixi run report-baseline  (or, from report/: snakemake build/baseline.pdf)
//
// Every number in this document comes from smi/analysis/data/baseline_results.json
// (method 1 and method 2, written by smi.analysis.baseline_fit) or from
// smi/analysis/data/gain_sweep_results.json (the gain, delay and fringe-rate
// sweeps, written by smi.analysis.gain_sweep). Nothing here is estimated or
// recalled: if a claim needs a number the artifacts do not contain, the claim
// does not get made. Figures read the first artifact and recompute nothing.
//
// The two methods run at different shot counts -- method 1 at N = 1000, method 2
// at N = 200, because method 2's Viterbi decode costs ~5.85 s per shot. State
// which N a table was produced at, every time.
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

An earlier draft of this document reported that the Michelson model explains
only 9 to 38% of the photodiode variance even when handed the true
displacement, and concluded that the model and the data disagree badly. That
conclusion was an artifact of how the model was fit, and section 4 retracts
it. The baseline's own error numbers stand; the diagnosis of what limits them
has changed.

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

The equivalence hides $sigma^2$, and it also hides the choice of *what
$theta$ is*. Treating the noise as fixed and known is what makes MSE and
likelihood interchangeable. Deciding how many parameters the fit gets, and
over how long a stretch of the record, is a separate decision that the
equivalence says nothing about -- and it is the decision the earlier draft got
wrong.

= The two measurements

Both run on shots from each acquisition at 16384 samples per shot, 488.28 kHz,
decoded on a GPU (`cuda:0`). They do *not* run on the same number of shots.
Method 1 is a closed-form least-squares solve, about a minute for 1000 shots
on a CPU, so it runs at N = 1000. Method 2's Viterbi decode costs about
5.85 s per shot, so it stays at N = 200. Every table below names its own N;
none of them can be read under the other's. Shot-level metrics are summarized
by their median unless stated otherwise; section 6 explains why the median and
not the mean.

== Method 1: forward-model residual

Displacement is known from the drive voltage, so the interference signal it
predicts is known up to an amplitude and phase. Both enter linearly once the
cosine is expanded into its cosine and sine components, so the fit is a
three-parameter least-squares solve -- cosine, sine, constant -- with a closed
form. No optimizer, no initial guess, no convergence to check, and a global
optimum by construction.

The question the earlier draft did not ask is *over what interval*. Fitting
one amplitude, phase and offset across the whole 33.6 ms record assumes those
three quantities are constant for 33.6 ms. In this data they are not, and
section 5 shows what that assumption costs. The fit is therefore reported two
ways: globally, and refit independently in windows of $W = 1024$ samples
(2.1 ms) with the residual and total variance pooled across windows into one
$R^2$ per shot. Variances are pooled rather than per-window $R^2$ values
averaged, and the pooled denominator uses the whole record's mean, so the
windowed number is directly comparable to the global one.

A shorter window fits more of anything, so a windowed $R^2$ reported alone is
not a measurement. Every windowed number below therefore appears with a
*scrambled-theta control*: the same window count and the same three parameters
per window, but with the displacement taken from a different shot. The control
is how much $R^2$ the fit buys from its own flexibility. What the model earns
is the gap between the two.

== Method 2: multi-wavelength inverse

The actual baseline. Given only the three photodiode signals, resolve fringe
ambiguity using the wavelength diversity across 635, 675 and 515 nm, decode
displacement on a 1601-point grid, then differentiate for velocity. Scored on
the same quantity the network predicts, in physical units: velocity in um/ms
and displacement in um, which is what the repository stores and what the
training loss is computed in.

The overall sign of the displacement is not scored. A cosine is even, so
$d(t)$ and $-d(t)$ produce identical fringes; the sign is genuinely
unobservable and penalizing it would measure nothing.

= Results

#fig(
  "baseline_summary",
  width: 100%,
  caption: [
    *Left:* method 1's explained variance per channel, with displacement
    known, N = 1000 shots per acquisition. Bars are the windowed refit at
    $W = 1024$, the dash below each bar is the scrambled-theta control it has
    to clear, and the hollow marker is the global single-window fit. *Middle:*
    method 2's displacement error beside the RMS of the displacement it is
    trying to recover, N = 200. *Right:* the same error normalized by that
    RMS; the dashed line is the error a predictor that always outputs zero
    would achieve.
  ],
)

== Method 2: the baseline number

#align(center, table(
  columns: 5,
  align: (left, right, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header(
    [acquisition],
    [disp. RMSE (um)],
    [vel. RMSE (um/ms)],
    [disp. NRMSE],
    [corr.],
  ),
  [free space], [0.433], [0.488], [1.10], [0.53],
  [mm fiber], [0.462], [0.550], [1.18], [0.42],
))

Medians over N = 200. For scale, the true displacement RMS is 0.362 um (free
space) and 0.379 um (mm fiber), and the true velocity RMS is 0.373 and
0.377 um/ms. The NRMSE column divides each shot's displacement error by the
standard deviation of that shot's true displacement, so NRMSE above 1 means
the decode is worse than predicting zero -- and both acquisitions sit just
above 1. The displacement correlations, 0.53 and 0.42, say the decode is not
noise: it tracks the real motion partway and then loses it.

*These numbers are not retracted.* They are a valid statement about this
inverse on this data, and they are the floor a learned model has to beat. What
has changed is the explanation of why the floor sits where it does.

== Method 1: the fit was wrong, then the model is imperfect

Median $R^2$ per channel over N = 1000 shots, with the true displacement
supplied, at the live coil-driver gain. Global is one fit across 33.6 ms;
windowed is refit per 1024 samples; control is the same windowed fit with
another shot's displacement.

#align(center, table(
  columns: 5,
  align: (left, left, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header([acquisition], [channel], [global], [windowed], [control]),
  [free space], [635 nm], [0.171], [0.715], [0.298],
  [free space], [675 nm], [0.176], [0.629], [0.330],
  [free space], [515 nm], [0.080], [0.422], [0.238],
  [mm fiber], [635 nm], [0.479], [0.863], [0.332],
  [mm fiber], [675 nm], [0.166], [0.602], [0.315],
  [mm fiber], [515 nm], [0.303], [0.730], [0.252],
))

*The global column is retracted as a statement about the model.* It is a
statement about a three-parameter fit held fixed for 33.6 ms. Refitting the
same three parameters every 2.1 ms raises the same shots to 0.42-0.86 without
giving the model any new freedom about the displacement, which was known
throughout and identical in both columns.

Read the windowed column against the control, never alone. The gaps, in the
row order above, are 0.42, 0.30, 0.18, 0.53, 0.29 and 0.48. That is what the
model earns over its own flexibility, and it is what supports the claim that
the model describes this data at all. The windowed figures on their own would
overstate agreement by 0.24 to 0.33.

$W = 1024$ is chosen by that gap rather than by taste. On the
largest-excursion cohort the control rises from 0.006-0.009 at $W = 16384$ to
0.185-0.274 at $W = 1024$ to 0.569-0.681 at $W = 256$, while the real value
rises more slowly; $W = 1024$ is where the difference is widest. A $W = 256$
number is mostly fitting capacity and must never be quoted as agreement.

*Free-space 515 nm is the weak case and should be named as one.* Its gap is
0.18, the narrowest of the six, and its global $R^2$ of 0.080 is the lowest.
Averaging the three free-space channels would report a gap near 0.30 and hide
it. 515 nm crosses the most fringes per micron of any of the three, which is
the ordering one would expect if the residual is a phase effect, but this
document does not have a measurement that pins that.

== What the fits look like on real shots

Every number above is a median. A median $R^2$ of 0.715 does not say whether
the fit follows individual fringes or is only correlated with them, so
@fig-traces draws three shots and the fits themselves.

#fig(
  "baseline_shot_traces",
  width: 100%,
  caption: [
    Method 1 on three free-space 635 nm shots, chosen by a rule rather than by
    eye: over the same N = 1000 shots the table above was computed on, these
    are the shots nearest the 10th, 50th and 95th percentile of displacement
    peak-to-peak, which resolves to shots 203, 90 and 365. *Left:* the
    drive-implied displacement for the whole 33.6 ms record, with the zoom
    window shaded. *Right:* the photodiode trace inside that window, with the
    global fit and the $W = 1024$ refit over it. The boxed numbers are
    whole-shot values from the same functions that produced the table, not
    read off the slice shown. The control is a median over 16 partner shots.
  ],
) <fig-traces>

The global fit is the dotted line, and the three panels are the collapse the
old report described, one shot at a time. At 0.29 um peak-to-peak it holds a
plausible amplitude and phase for the window shown and scores 0.69. At 1.51 um
it is about half the measured amplitude and drifting out of step with it,
at 0.15. At 3.74 um it has given up: 0.09, with an amplitude that no longer
matches anything on the panel. What moves between the three is not the data's
agreement with the model but how far the phase wanders inside one record --
1.36, 1.81 and 2.73 fringes -- and a single fixed phase cannot follow any of
that.

The windowed refit is the solid line. At low and medium excursion it tracks
the measured trace fringe for fringe: the medium panel has three full fringes
in 1.5 ms and the refit finds all three, with peaks and zero crossings landing
where the data puts them. That is what a windowed $R^2$ of 0.93 and 0.66 looks
like, against controls of 0.55 and 0.29.

The high-excursion panel is the weak one and is shown as such. The refit is
still roughly in step near 11.6 and 12.7 ms, but through the middle of the
window -- around 12.0 to 12.5 ms, where the mirror is moving fastest -- it
runs nearly flat while the measured signal turns over three sharp fringes. The
fitted phase has slipped against the real one within a single window. Its
0.43, against a 0.26 control, is a gap of 0.17: still above the control, and
well below what the same estimator achieves on the two smaller shots.

This is the residual amplitude dependence of section 5, visible rather than
inferred, and the panel also shows why a gain correction does not fix it:
where the refit and the measurement are in step, their amplitudes agree.
The mismatch is in phase, not in gain. The sharp flat-topped fringe
peaks are a property of the fringe shape at high fringe rate, not detector
clipping -- only 1 shot in 400 reaches the channel maximum, and the fraction
of samples near a shot's extremes does not grow with excursion.

Three shots out of a thousand cannot establish a trend, and none is claimed
here; the binned trend in section 5 does that over all of them. What these
panels settle is the narrower question the medians leave open, which is
whether the windowed fit follows real fringes or merely correlates with
them. At low and medium excursion it follows them.

#takeaway[
  Method 1 no longer says "the model does not explain this data". It says the
  global fit was the wrong estimator, and that once refit per window the model
  clears its own scrambled control by 0.18 to 0.53 depending on channel. The
  remaining question is what the gap *fails* to close at large excursion, which
  is section 5.
]

The synthetic control is what keeps the blame off the decoder. The same
decoder, run on signals genuinely generated by the Michelson forward model,
recovers displacement to a median NRMSE below 0.06 with at least 75% of shots
under 0.10 -- roughly twenty times better than it manages on real shots. That
bound is asserted by the test suite (`tests/test_baseline_fit.py`) and so is
re-checked on every run, rather than being a number remembered from a session.

= What limits the fit

#fig(
  "baseline_amplitude_trend",
  width: 100%,
  caption: [
    Method-1 median $R^2$ binned by the drive's peak-to-peak displacement
    excursion, per channel and per acquisition, N = 1000 shots each. The
    dotted line is the global fit, the solid line the windowed refit at
    $W = 1024$, the dashed line the scrambled-theta control, and the shaded
    band the gap between the last two. The global fit collapses toward zero
    above 2 um in all six panels; the windowed fit does not.
  ],
)

Binning the method-1 fits by how far the speaker actually moved separates
three effects that the single global number confounded. The next three
subsections take them in the order they matter.

== The global fit was the wrong estimator

The global curve falls from 0.453-0.730 in the smallest bin to
0.014-0.047 above 3 um, and does so in all six channel/acquisition
combinations. The windowed curve over the same shots stays at 0.265-0.777 in
that top bin, against controls of 0.186-0.280. Nothing about the model or the
displacement changed between those two curves; only the interval over which
amplitude and phase were held fixed.

The fringe phase and the fringe contrast both wander within a shot, and those
are the two quantities the global fit holds constant. Within-shot drift of 1
to 2.7 fringes has been measured on this data and is recorded in
`smi/analysis/baseline_fit.py`; that magnitude lives there rather than in the
results artifacts, so this document cites it and does not tabulate it. For
scale, at the median true velocity RMS of 0.373 and 0.377 um/ms a
1024-sample window sweeps 2.3 to 3.1 fringes depending on wavelength, and the
whole 33.6 ms record sweeps 33 times as many. A single phase cannot describe a
record whose fringe phase wanders by a fringe or more, and a single amplitude
cannot describe one whose contrast varies, so a three-parameter global fit
leaves most of the variance unexplained. That also accounts for the excursion
dependence without any extra assumption: a longer sweep gives the drift more
time to accumulate.

== A real gain error in the drive-to-displacement conversion, secondary

Displacement is derived from the drive voltage through a calibrated transfer
function whose scalar gain is `k = 14.553`. Sweeping a multiplier on that gain
over 101 values from 10 to 35 at N = 1000 gives a sharply peaked median $R^2$,
and the peak is not at the live value:

#align(center, table(
  columns: 5,
  align: (left, left, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header(
    [acquisition],
    [channel],
    [best `k`],
    [vs live],
    [global $R^2$, live -> best],
  ),
  [free space], [635 nm], [16.50], [+13.4%], [0.162 -> 0.462],
  [free space], [675 nm], [16.25], [+11.7%], [0.171 -> 0.346],
  [free space], [515 nm], [16.00], [+9.9%], [0.077 -> 0.176],
  [mm fiber], [635 nm], [15.50], [+6.5%], [0.468 -> 0.613],
  [mm fiber], [675 nm], [15.75], [+8.2%], [0.160 -> 0.230],
  [mm fiber], [515 nm], [15.25], [+4.8%], [0.291 -> 0.396],
))

So the displacement targets are about 5 to 13% too small. That is a real
error and worth fixing, and it is the honest version of the caveat that
targets are derived rather than measured: everything downstream inherits this
calibration's accuracy, and its accuracy is now measured at the 5-13% level
rather than assumed.

It is secondary to the estimator problem, for two reasons the sweep itself
supplies. First, the correction it buys shrinks once the estimator is right:
windowed $R^2$ moves only 0.712 -> 0.767, 0.626 -> 0.651 and 0.422 -> 0.435
in free space, and 0.860 -> 0.883, 0.600 -> 0.617 and 0.726 -> 0.746 in the
fiber. Second, the best gain is *not* consistent between the two
acquisitions -- 16.50 against 15.50 at 635 nm, 16.25 against 15.75 at 675 nm,
16.00 against 15.25 at 515 nm, a disagreement of 3.2 to 6.5%. Drive-to-
displacement gain is a property of the speaker, not of the optical path, so a
scalar `k` that lands in two different places on two acquisitions is absorbing
something other than a pure gain error. The gain step is 0.25, which is 1.7%
of the live value, so the disagreement is wider than the sweep's resolution.

For that reason the calibration constant in `smi/synthetic/calib_params.py`
has deliberately *not* been re-tuned here. Changing it would shift every
training target, and a constant the two acquisitions disagree about by up to
6.5% is not settled enough to shift them on.

The same sweep does refute the value the 2024 calibration implied,
`k = 33.42`, which is kept commented out in that file under the note
"Incorrect fringe counts in 2024 calibration?". At that gain the median $R^2$
is 0.012 to 0.029 across the six channel/acquisition combinations, which is
noise. The doubt in the comment was warranted.

== A residual amplitude dependence that is not a scale error

Even with the gain fixed, the largest excursions are fit worse than the
smallest. The evidence that this residual is not a scale error is that giving
each shot its *own* optimal gain does not remove it. Fitting one gain per shot
at N = 1000 -- strictly more freedom than any global recalibration can have --
leaves the free-space above-3 um bin at median $R^2$ 0.143 (635 nm), 0.094
(675 nm) and 0.052 (515 nm). The best single global gain does no better there:
0.122, 0.076 and 0.020. A scale error would be absorbed entirely by a per-shot
gain; this is not.

The per-shot fits also come with a warning about how to read them. Each shot's
gain is chosen by its own best $R^2$, and at some excursions a gain of roughly
twice the true one re-aligns the fringe pattern by a whole fringe order and
scores well. On the 675 nm fiber channel, 249 of 1000 shots pick a gain
between 29.5 and 32.5, about twice the live value, and 334 pick a gain above
25 altogether. Free-space 515 nm does the same in 125 shots. Those fits inflate
that channel's per-shot median $R^2$ of 0.336, so it should not be read as
evidence that the fiber 675 nm channel is fit better than its global 0.166
suggests. The median gains themselves are unaffected -- 15.75 with an
interquartile range from 15.50 to 30.50 on that channel, against a clean
15.25-15.50 on the others -- but the quartile spread is the tell.

== What this section rules out

Four other candidates could explain an excursion-dependent residual. Each was
checked against the data and none survives. They are listed with the check
that killed them, so the next reader finds evidence rather than a claim.

- *A time-delay misalignment between drive and photodiode.* Shifting the
  predicted signal against the recorded one over $plus.minus 8$ samples
  produces no interior peak in any of the six channel/acquisition
  combinations: the curve is highest at the sweep's edge and declines
  monotonically inward. A real delay would peak at a nonzero shift and fall
  off on both sides. The verdict is taken from whether the curve peaks at all,
  not from where the argmax lands, because an argmax at an endpoint is what a
  monotonic curve returns whether or not there is a delay.
- *Aliasing of the fringe rate.* Peak fringe rate at the live gain runs
  9.9 kHz (675 nm free space) to 13.6 kHz (515 nm fiber) against a Nyquist
  frequency of 244 kHz, i.e. 18 to 25 times headroom. The fringes are
  comfortably sampled.
- *A missing DC term.* The fit includes a constant, globally and per window,
  so a per-record or per-window offset is already absorbed.
- *The $4 pi d \/ lambda$ phase factor and the resonator parameters.* A wrong
  factor of two in the phase, or a mis-set $f_0$ or $Q$, would appear in the
  gain sweep as a peak at a different gain or as no peak at all. The sweep
  peaks sharply within 13% of the live gain in every channel, and the
  wavelength-independence of the best gain is what a mechanical displacement
  error looks like and what an optical-phase error does not.

= How to read these numbers

Four things would mislead a reader who took the artifacts at face value.

*Use the medians.* The mean `displacement_nrmse` is 4.58 (free space) and 4.80
(mm fiber), around four times the medians of 1.10 and 1.18. The means are
artifacts of the normalization, not evidence of worse performance: NRMSE
divides by each shot's true displacement RMS, and the quietest shots in these
subsets have a true displacement RMS of 0.002 um (free space) and 0.001 um
(mm fiber). A near-motionless shot puts a near-zero number in the denominator
and produces an NRMSE in the hundreds -- the maxima are 262 and 267 -- which
drags the mean without describing any typical shot. Every headline number in
this document is a median for that reason.

*Never quote a windowed $R^2$ without its control.* At $W = 1024$ the controls
run 0.238 to 0.332 overall and 0.186 to 0.280 in the largest-excursion bin.
Those are not negligible, and a windowed figure reported alone credits the
model with them.

*The synthetic control is noiseless.* It has no measurement noise and no
within-shot phase drift, so it establishes that the decoder is sound on clean
fringes generated by the model it assumes. It does not establish
noise-robustness. Some unknown part of the real-data gap could still be
estimator sensitivity to noise rather than model mismatch, and this data
cannot separate those two.

*The fiber gap does not appear here.* Free space and mm fiber are comparable
at this level of performance -- 0.433 against 0.462 um displacement RMSE,
1.10 against 1.18 NRMSE. Method 1 favors the fiber in two windowed channels of
three (0.863 against 0.715 at 635 nm, 0.730 against 0.422 at 515 nm, and
0.602 against 0.629 at 675 nm). Whatever makes the fiber path harder for a
learned model is not visible in the baseline. The band-limiting below is real
in the raw data; it is not costing the baseline anything measurable.

= Free space versus multimode fiber

The fiber path is the deployment-relevant one and the expected-harder one. The
raw data does differ: the fiber acquisition carries less high-frequency
content than free space in the same channel.

#fig(
  "freespace_vs_fiber",
  width: 60%,
  caption: [
    635 nm photodiode, mean amplitude spectrum over N = 1024 shots per
    acquisition. Mode scrambling in the fiber low-passes the fringe signal
    before it reaches the detector.

    A single-shot overlay of the two files is not shown: the drive waveforms
    were generated by separate, unseeded random draws, so a shared shot index
    does not mean a shared waveform. Shot 7's drive, for example, correlates
    at only +0.116 between the two files with a 5.5x RMS mismatch (0.0269 V
    free space, 0.1475 V fiber) -- comparing that pair by shot number would
    compare two unrelated realizations as though they were matched. The mean
    spectrum avoids this: it is a distribution-level statistic over many
    shots of each acquisition, so it is valid evidence about the two
    acquisitions even though no individual shot in one file corresponds to
    any shot in the other.
  ],
)

That spectral difference does not translate into a baseline difference, per
the previous section. Both acquisitions are limited by the same things, and a
comparison at this level should not be cited as evidence either way about the
fiber gap.

= Scope

*What this establishes.* A floor. Method 2's displacement RMSE of 0.433 um
(free space) and 0.462 um (mm fiber), and velocity RMSE of 0.488 and
0.550 um/ms, over N = 200 shots, are what a closed-form classical inverse
achieves with no training. A learned model has to beat these to justify its
complexity, and because the NRMSE sits just above 1, it has to beat predicting
zero as well -- a lower bar than it sounds, and one that should be checked
rather than assumed.

Method 1 establishes two further things, at N = 1000. The Michelson forward
model does describe this data when it is fit over intervals short enough for
its amplitude and phase to be constant: windowed $R^2$ of 0.42 to 0.86 against
scrambled controls of 0.24 to 0.33. And the drive-to-displacement calibration
carries a real gain error of 5 to 13%, so the supervision targets are
systematically small by about that much.

*What this does not establish.* It is not an upper bound on classical
processing: a better inverse, or a corrected transfer function, could do
better. It says nothing about latency, and nothing about waveform families
outside the one these acquisitions sweep. It does not prove the decoder is
noise-robust, only that it is correct on clean model-generated fringes. It
does not speak to the free-space/fiber gap, which is invisible at this error
level. It does not give a corrected value for `k`: the two acquisitions
disagree by 3.2 to 6.5%, so the sweep locates an error without resolving it.
And the residual amplitude dependence that survives a per-shot gain fit is
unexplained -- measured, bounded, and open.

The next steps are the inverse and the calibration, in that order. Improving
the inverse is worth doing: method 2's NRMSE near 1.1 is now the binding
number, because the model it inverts is no longer the thing that fails. Then
settle `k`, which needs an experiment that can explain why two acquisitions of
the same speaker prefer gains 3.2 to 6.5% apart.
