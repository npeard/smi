// Standalone report: what a non-learned baseline achieves on this data, and
// what ten trained models achieve beside it.
//
// Build:  pixi run report-only  (or, from report/: snakemake build/report.pdf)
//
// Every number in this document comes from smi/analysis/data/baseline_results.json
// (method 1 and method 2, written by smi.analysis.baseline_fit), from
// smi/analysis/data/gain_sweep_results.json (the gain, delay and fringe-rate
// sweeps, written by smi.analysis.gain_sweep), or from
// smi/analysis/data/learned_results.json (the trained models, written by
// smi.analysis.run_baseline_sweep). Nothing here is estimated or
// recalled: if a claim needs a number the artifacts do not contain, the claim
// does not get made. Figures read the artifacts and recompute nothing.
//
// The two methods run at different shot counts -- method 1 at N = 1000, method 2
// at N = 200, because method 2's Viterbi decode costs ~5.85 s per shot. State
// which N a table was produced at, every time.
//
// ASCII only, per the repo convention.

#import "lib/theme.typ": fig, report-theme, takeaway

#show: report-theme.with(
  title: "Velocity estimation from self-mixing fringes",
  subtitle: "What the Michelson model recovers, and what ten trained models recover beside it",
  author: "Nolan Peard",
  date: "Draft",
)

= Why this document exists

The project's goal is an estimator that recovers a mirror's velocity from
three self-mixing photodiode channels, accurately enough and causally enough
to run against live hardware. This document is where the candidates for that
estimator are scored against each other on one footing.

A velocity RMSE from a neural network is uninterpretable on its own. It needs
a number beside it from a method that does not learn, computed on the same
shots and in a unit the two can actually be compared in. Sections 2 to 7
produce that number -- the Michelson analytic baseline -- and say precisely
what it does and does not establish. Section 9 puts ten trained models beside
it, and section 8 places both against the two published networks this work
started from.

The second half changes how to read the first. The analytic baseline explains
this data badly, which sections 4 to 6 measure and diagnose. Section 9 then
finds that every learned model clears it, including a ridge regression with
274 coefficients. That completes the earlier argument quantitatively instead
of contradicting it: what it measures is how weak the physical model is as
fitted here, and the network results inherit that reading.

The methodological question the baseline half answers came up while planning
it: should each shot be scored by its *likelihood* under a Michelson
model, or should the model actually be *fit* to each shot? Section 2 argues
these are the same computation, which is why the simple option is also the
rigorous one.

It also answers the question that decides what a fit means: over how long a
stretch of the record may the fit's amplitude and phase be held constant? In
this data those two quantities drift within a single shot, so the answer
changes what the same model, on the same shots, appears to explain. Sections 4
and 5 measure that and separate it from the parts of the residual no choice of
window reaches.

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
equivalence says nothing about. It is also the decision that moves the numbers
most, which is why section 3.1 states the interval before section 4 reports
anything fit over one.

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

What the solve does not settle is *over what interval*. Fitting one amplitude,
phase and offset across the whole 33.6 ms record assumes those three
quantities are constant for 33.6 ms. In this data they are not, and section 5
shows what that assumption costs. The fit is therefore reported two ways:
globally, and refit independently in windows of $W$ samples with the residual
and total variance pooled across windows into one $R^2$ per shot. Variances
are pooled rather than per-window $R^2$ values averaged, and the pooled
denominator uses the whole record's mean, so the windowed number is directly
comparable to the global one.

$W$ is *fringe-matched*: it is resolved for each shot and each channel so that
one window spans about two fringes, clipped to $[256, 4096]$ samples. A fixed
$W$ cannot serve both ends of this data, because a window holding a third of a
fringe on the quietest shots holds a dozen on the loudest, and a measure whose
window means something different in each excursion bin cannot be read across
bins. This is the estimator behind every headline table and figure here, and
the window each one came from is recorded per channel in the results
artifact. Where a number is instead taken at a fixed $W$ -- the window sweep
of section 4.2, the amplitude diagnosis of section 4.3 and the gain sweep of
section 5.2, each for a reason stated where it appears -- the text says so.

A shorter window fits more of anything, so a windowed $R^2$ reported alone is
not a measurement. Every windowed number below therefore appears with a
*scrambled-theta control*: the same $W$, the same window count and the same
three parameters per window, but with the displacement taken from a different
shot. Matching $W$ between the fit and its control is what makes the pair a
comparison -- a fit with more fitting freedom than its control would be
scored on the freedom. The control is how much $R^2$ the fit buys from that
flexibility, and what the model earns is the gap between the two.

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
    known, N = 1000 shots per acquisition. Bars are the windowed refit at a
    fringe-matched $W$, the dash below each bar is the scrambled-theta control
    it has to clear at that same $W$, and the hollow marker is the global
    single-window fit. Read the bar against its dash, not its height. *Middle:*
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
    [Pearson r (disp.)],
  ),
  [free space], [0.433], [0.488], [1.10], [0.53],
  [mm fiber], [0.462], [0.550], [1.18], [0.42],
))

Medians over N = 200. Both RMSEs are larger than the motion they measure:

#align(center, table(
  columns: 3,
  align: (left, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header(
    [acquisition],
    [true disp. RMS (um)],
    [true vel. RMS (um/ms)],
  ),
  [free space], [0.362], [0.373],
  [mm fiber], [0.379], [0.377],
))

The NRMSE column divides each shot's displacement error by the standard
deviation of that shot's own true displacement, so it says what the error is
worth against the motion that shot contained. At NRMSE 1 the decode is as
accurate as a predictor that always outputs zero. Both acquisitions sit just
above 1, so on these shots the zero predictor is the more accurate of the two
by 10 to 18%. A learned model that merely beat the decode's RMSE would still
be worse than a constant, which makes the constant, not the decode, the bar
that has to be checked.

=== $R^2$ and Pearson r are not interchangeable

Two statistics appear in this document and neither substitutes for the other.

/ $R^2$: the fraction of a signal's variance a fit explains, computed from a
  sum of squared residuals. It can go negative, and it is sensitive to *scale*:
  a fit at the wrong amplitude scores badly. Used from section 4.2 onward.
/ Pearson r (disp.): the linear correlation between the decoded and the true
  displacement, the last column of the table above. It is scale-free, so it
  would score a decode that tracked the right shape at the wrong amplitude as
  highly as a perfect one.

Together they say more than either does alone. Pearson r of 0.53 and 0.42 puts
the decode well clear of noise: it follows the real motion partway before
losing it. The NRMSE just above 1 says that what it follows is not yet worth
more than a constant. Both readings are true of the same decode, and a decode
that recovered the shape at the wrong amplitude would look exactly like this.

=== The limits of this measurement

This is one inverse on these two acquisitions. The decode resolves fringe
ambiguity across the three wavelengths on a 1601-point displacement grid and
differentiates for velocity; a finer grid, a different fringe-ambiguity
strategy, or a smoother velocity estimate would each give a different number.
The RMSEs above are a floor a learned model must clear. They are not a bound
on classical processing. The sign caveat of section 3.2 applies: the overall
sign of $d(t)$ is unobservable and is not scored.

The decoder itself is sound on fringes the Michelson model generated: run on
synthetic signals it recovers displacement to a median NRMSE below 0.06, with
at least 75% of shots under 0.10, roughly twenty times better than it manages
here. That bound is asserted by the test suite (`tests/test_baseline_fit.py`)
and so is rechecked on every run. Those synthetic fringes carry no measurement
noise and no within-shot phase drift, so the check establishes that the
decoder is correct, not that it is noise-robust; section 6 returns to what
that leaves open.

== Method 1: does the Michelson model describe this data

Method 1 is handed the true displacement, so what it fails to explain is the
model's own disagreement with the measurement rather than any inverse's error.
The measurement is the *gap*: the windowed $R^2$ minus the scrambled-theta
control run at the same $W$. That difference is what the model earns over its
own fitting flexibility. It is the column to read, because the windowed column
alone credits the model with whatever a short window can fit for free, and
that amount varies by channel. Medians over N = 1000 shots, at the live
coil-driver gain:

#align(center, table(
  columns: 7,
  align: (left, left, right, right, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header(
    [acquisition],
    [channel],
    [global $R^2$],
    [windowed $R^2$],
    [control $R^2$],
    [*gap*],
    [median $W$],
  ),
  [free space], [635 nm], [0.171], [0.726], [0.309], [*0.42*], [1042],
  [free space], [675 nm], [0.176], [0.628], [0.328], [*0.30*], [1108],
  [free space], [515 nm], [0.080], [0.503], [0.290], [*0.21*], [846],
  [mm fiber], [635 nm], [0.479], [0.872], [0.335], [*0.54*], [1028],
  [mm fiber], [675 nm], [0.166], [0.557], [0.283], [*0.27*], [1092],
  [mm fiber], [515 nm], [0.303], [0.781], [0.314], [*0.47*], [834],
))

Global is one fit across the whole 33.6 ms record. Windowed is the same three
parameters refit per fringe-matched window. Control is that windowed fit run
again against another shot's displacement at the same $W$. Gap is windowed
minus control. The last column is the median $W$ the channel resolved to; the
spread behind that median is wide, since $W$ tracks each shot's fringe rate --
free-space 635 nm runs 688 to 2286 samples between its quartiles and reaches
the clip at 4096 on the quietest shots.

The model clears its control on all six, by 0.21 to 0.54. The global column is
a property of an estimator that holds amplitude and phase fixed for 33.6 ms,
and it is in the table as the contrast that motivates windowing at all.
Between the two columns the model gains nothing about the displacement: it is
known throughout and identical in both. Only the interval over which amplitude
and phase are held constant differs, and on these shots that interval is worth
0.39 to 0.55 in $R^2$. Section 5 measures how the difference behaves against
excursion, which is where it comes from.

=== Why a control is mandatory

A shorter window fits more of anything. Three free parameters per window,
enough windows, and a cosine at the wrong phase will still track a photodiode
trace some of the way -- which is why the windowed column read alone would
credit the model with 0.28 to 0.34 of $R^2$ it did not earn. Those are the
control values, measured by running the identical fit against a displacement
that belongs to a different shot. On free-space 515 nm that is more than half
the windowed number. No windowed $R^2$ appears anywhere in this document
without its control beside it.

The control has to run at the fit's own $W$. Fringe-matching resolves $W$ per
shot, so the fit's freedom varies from shot to shot; a control pinned at some
other length measures a different estimator, and the difference between the
two would mix agreement with fitting capacity in unknown proportion.

=== How the window length is chosen

$W$ spans about two fringes, clipped to $[256, 4096]$ samples, and both bounds
are set by what the control does rather than by preference. Measured on the
free-space above-3 um cohort, the control climbs from 0.006-0.010 at
$W = 16384$ through 0.19-0.25 at $W = 1024$ to 0.56-0.71 at $W = 256$, while
the real fit rises more slowly over the same range. At $W = 256$ the control is
about two thirds of the real fit: a number quoted from there is mostly fitting
capacity, and must never be read as agreement. That is the lower clip. At the
upper clip the opposite failure appears -- four windows in a record leave the
within-shot phase drift the windowing exists to absorb largely unabsorbed, and
both curves fall toward zero together. The clip brackets the range where the
gap is worth reading. These window-sweep numbers are recorded in
`smi/analysis/baseline_fit.py` beside the clip constants they justify, rather
than in the results artifacts.

=== Per-channel caveats

*Free-space 515 nm is the weak case.* It has the narrowest gap of the six,
0.21, and the lowest global $R^2$ of the six, 0.080. Averaging the three
free-space channels would report a gap near 0.31 and hide it, so every
table in this document reports the channels separately. 515 nm crosses the most
fringes per micron of the three wavelengths, which is the ordering to expect if
the residual is a phase effect; this document has no measurement that pins that
down.

*The mm-fiber 675 nm channel is the one to read carefully elsewhere.* Nothing
in the table above is affected, because both columns there use the known
displacement. But where a gain is fit per shot -- section 5.3 -- that channel
picks a gain near twice the true one in 249 of 1000 shots, which re-aligns the
fringe pattern by a whole fringe order and scores well without describing the
motion. Its per-shot median $R^2$ of 0.336 is inflated by those aliases and is
not evidence that the channel is fit better than the table says.

== What the fits look like on real shots

Every number above is a median over a thousand shots. A median $R^2$ does not
say whether the fit follows individual fringes or merely happens to land close
to them on average, so @fig-traces draws three shots and the fits themselves.

#fig(
  "baseline_shot_traces",
  width: 100%,
  caption: [
    Method 1 on all three free-space channels, for three shots chosen by a
    rule rather than by eye: over the same N = 1000 shots the table above was
    computed on, these are the shots nearest the 10th, 50th and 95th
    percentile of displacement peak-to-peak, which resolves to shots 203, 90
    and 365. *Top row:* the drive-implied displacement for the whole 33.6 ms
    record, with the zoom window marked. *Lower rows:* one channel each, with
    the photodiode trace inside that window, the per-channel fit, and the
    joint fit at one shared displacement gain $g$. The panel titles carry
    whole-shot values from the same functions that produced the table, not
    numbers read off the slice shown; the control is a median over 16 partner
    shots at the same window length as the fit beside it.
  ],
) <fig-traces>

These are the estimator of section 4.2 applied shot by shot, so the panel
titles state the $W$ each shot resolved to: on 515 nm, 4096 at low excursion
down to 519 at high. A loud shot gets a shorter window because it crosses more
fringes in the same time, which is what keeps one window's worth of
accumulated phase error comparable between the three columns.

Read down a column and the three channels agree about what is happening, which
is itself the point: they are three independent detectors watching one mirror.
At 0.29 um peak-to-peak all three follow the single slow fringe in the zoom,
scoring 0.79, 0.91 and 0.65 against controls of 0.11, 0.25 and 0.26. At 1.51 um
the 635 nm and 675 nm panels turn over three full fringes in 1.5 ms and the fit
finds all three, with peaks and zero crossings landing where the data puts
them: 0.68 and 0.72 against 0.29 and 0.33.

At 3.74 um the fits are still above their controls on every channel but the
margin has narrowed, and the panels show where it goes. On 635 nm and 675 nm
the fit tracks the fringes it finds but misses the tallest peaks, and on
515 nm -- which crosses the most fringes per micron of the three -- it follows
the envelope while the measured signal turns over fringes the fit does not
reproduce at all. That is the 515 nm weakness of the table, drawn rather than
tabulated.

The joint fit is the dashed line, and its shared gain is printed above each
column: $g = 1.028$, $1.095$ and $1.036$ for the three shots. Those bracket
the 1.10 to 1.13 the independent gain sweep of section 5 found by refitting
each channel separately over a grid, which is a real cross-check -- two
different estimators of the same mechanical scale, one using three
wavelengths at once and one using each alone. The joint fit gains little over
the per-channel fits here (0.70 against 0.67 on high-band 635 nm, and within
0.01 elsewhere), for a reason the panels make visible: with amplitude and
phase free in every window there is already enough freedom to absorb a gain
error, so constraining the gain buys less than it would if the phase were
held.

*The fitted amplitude is biased low, and that bias is a symptom rather than a
separate defect.* At fixed $W = 1024$ the ratio of fitted amplitude to the
window's own amplitude ($sqrt(2)$ times its standard deviation) has median
0.586 on high-band 635 nm and sits below 1 in 94% of windows. A one-directional
bias, not mistracking, which would scatter about 1. The cause is in the
estimator's algebra rather than in the detector: a least-squares amplitude
attenuates toward zero when the regressor is wrong, because projecting the
data onto a cosine whose phase is off by $epsilon$ recovers $A cos epsilon$
and not $A$. Average that over a window in which the phase error wanders and
the fitted amplitude comes out small. So the low amplitude measures the phase
error. Fit and data agree in amplitude locally, wherever they are in step; the
whole-window amplitude runs low precisely where they do not stay in step. A
gain correction cannot fix that, because there is no gain error to fix.

The sharp flat-topped fringe peaks are a property of the fringe shape at high
fringe rate, not detector clipping -- only 1 shot in 400 reaches the channel
maximum, and the fraction of samples near a shot's extremes does not grow with
excursion.

Three shots out of a thousand cannot establish a trend, and none is claimed
here; the binned trend in section 5 does that over all of them. What these
panels settle is the narrower question the medians leave open, which is
whether the windowed fit follows real fringes or only tracks their general
shape without matching them fringe for fringe. At low and medium excursion,
on 635 nm and 675 nm, it follows them.

=== Why the fringe-matched window and not a smooth envelope

If the low amplitude is a phase-error symptom, then anything that reduces the
phase error inside one fit should recover amplitude. Two ways to give the fit
that room, each measured against its own control:

/ fringe-matched $W$: scale the window by the shot's own fringe rate so every
  window spans about two fringes, clipped to $[256, 4096]$. At a fixed $W$ a
  quiet shot's window holds a third of a fringe and a loud one's holds a
  dozen, so the phase has wildly different room to wander in each.
/ slowly-varying envelope: keep $W = 1024$ and let the amplitude and phase
  vary as a low-order polynomial in $t$ within each window, instead of
  stepping discontinuously at every window edge.

Both add fitting freedom, so both are scored against a scrambled-theta control
recomputed *under that same freedom*. A fit with more parameters compared
against a control with fewer measures the parameters: on the comparison below,
scoring both variants against the fixed window's own control would have made
the envelope the clear winner on raw $R^2$ and the fringe-matched window the
loser, which is the reverse of what a matched control shows.

#figure(
  table(
    columns: 5,
    align: (left, right, right, right, right),
    table.header(
      [variant],
      [$W$],
      [$R^2$],
      [control $R^2$],
      [gap],
    ),
    [fixed window], [1024], [0.431], [0.258], [0.174],
    [fringe-matched], [640], [0.672], [0.376], [*0.296*],
    [envelope, degree 2], [1024], [0.750], [0.474], [0.275],
  ),
  caption: [
    The three variants on the high-excursion shot (365), free-space 635 nm,
    each with its own scrambled-theta control at its own window length and
    parameter count. The gap is the only column that compares across rows.
  ],
)

The envelope has the highest raw $R^2$ and is the weaker variant anyway. Its
control rises almost as fast as it does, which is what a freedom that fits
anything looks like, and on the low-excursion shot the effect is stark: the
envelope's $R^2$ goes from 0.927 to 0.964 while its control goes from 0.547 to
0.838, so the gap *falls* from 0.380 to 0.125. Most of what the envelope adds
is capacity, not agreement. It also does nothing about the phase error, since
it only lets the estimator describe a varying amplitude rather than find the
right phase.

The fringe-matched window is the variant that earns its freedom, and it is the
estimator every table in this document uses. It widens the high-excursion gap
from the fixed window's 0.174 to 0.296, and it recovers the amplitude the
fixed window was losing: the median amplitude ratio on that shot rises from
0.586 to 0.883, with the fraction of windows below 1 falling from 94% to 81%.
Amplitude recovered by shortening a window is the mechanism confirming itself,
since nothing in the shortening touches amplitude directly.

The improvement is not uniform. On the low-excursion shot the rule asks for
$W = 4096$, the clip's upper bound, and four windows in a record is few enough
that the phase drift the windowing exists to absorb starts coming back: $R^2$
there is 0.794 against the fixed window's 0.927. The control falls much
further over the same change, 0.547 to 0.111, so the gap still widens -- but
the raw $R^2$ at the top of the clip is the one place where a shorter fixed
window scores higher, and reading those two numbers without their controls
would invert the conclusion.

#takeaway[
  Method 1 says the Michelson model describes this data when its amplitude and
  phase are fit over intervals short enough for them to be constant: refit per
  fringe-matched window, the model clears its own scrambled control by
  0.21 to 0.54 depending on channel. The fitted amplitude running low at large
  excursion is a consequence of the same phase error rather than a separate
  gain problem, and matching the window to the fringe rate recovers it from
  0.586 to 0.883 of the local amplitude. What the gap *fails* to close at
  large excursion is section 5.
]

= What limits the fit

#fig(
  "baseline_amplitude_trend",
  width: 100%,
  caption: [
    Method-1 median $R^2$ binned by the drive's peak-to-peak displacement
    excursion, per channel and per acquisition, N = 1000 shots each. The
    dotted line is the global fit, the solid line the windowed refit at a
    fringe-matched $W$, the dashed line the scrambled-theta control at that
    same $W$, and the shaded band the gap between the last two. Because $W$
    tracks each shot's fringe rate it shortens as excursion grows: each panel
    states the median and interquartile range of the $W$ its own numbers came
    from. The global fit collapses toward zero above 2 um in all six panels;
    the windowed fit does not.
  ],
)

Binning the method-1 fits by how far the speaker actually moved separates
three effects that any aggregate median blends together. The next three
subsections take them in the order they matter.

== The interval the fit holds its parameters over

The global curve falls from 0.453-0.730 in the smallest bin to
0.014-0.047 above 3 um, and does so in all six channel/acquisition
combinations. The windowed curve over the same shots stays at 0.510-0.867 in
that top bin, against controls of 0.365-0.400. Nothing about the model or the
displacement differs between those two curves; only the interval over which
amplitude and phase are held fixed.

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

It is secondary to the interval the fit spans, for two reasons the sweep
itself supplies. First, the correction it buys shrinks once amplitude and
phase are refit per window: windowed $R^2$ moves only 0.712 -> 0.767,
0.626 -> 0.651 and 0.422 -> 0.435
in free space, and 0.860 -> 0.883, 0.600 -> 0.617 and 0.726 -> 0.746 in the
fiber. The sweep's windowed measure is taken at a fixed $W = 1024$ rather than
fringe-matched, and has to be: the swept gain scales the displacement, so it
scales the fringe rate, and a window matched to that rate would change length
along the very axis the sweep varies. A curve whose estimator moves with its
x-axis locates nothing. These six numbers are therefore the sweep's own
measure and are not comparable to the fringe-matched tables of section 4.2;
what they establish is the *size* of the gain's effect, and that is the claim
being made. Second, the best gain is *not* consistent between the two
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

*Never quote a windowed $R^2$ without its control.* The controls run
0.28 to 0.34 overall and 0.37 to 0.40 in the largest-excursion bin. Those are
not negligible, and a windowed figure reported alone credits the model with
them.

*The synthetic control is noiseless.* It has no measurement noise and no
within-shot phase drift, so it establishes that the decoder is sound on clean
fringes generated by the model it assumes. It does not establish
noise-robustness. Some unknown part of the real-data gap could still be
estimator sensitivity to noise rather than model mismatch, and this data
cannot separate those two.

*The fiber gap does not appear here.* Free space and mm fiber are comparable
at this level of performance -- 0.433 against 0.462 um displacement RMSE,
1.10 against 1.18 NRMSE. Method 1 favors the fiber on two of the three
channels by gap over control (0.54 against 0.42 at 635 nm and 0.47 against
0.21 at 515 nm, with free space ahead only at 675 nm, 0.30 against 0.27).
Whatever makes the fiber path harder for a
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

= Prior art: two published networks

Two papers from the same group have already put a convolutional network on
this problem, and both are the starting point for what section 9 trains.

@barland2021cnn trains a single-channel network on self-mixing fringes and
reports a Pearson correlation of 0.90 between the recovered and the true
displacement, at 57,153 parameters. @matha2023multichannel extends the
approach to three channels at 207,905 parameters, reporting a correlation of
0.96 and a velocity RMSE of 0.13 um/ms; its best single channel reaches 0.97
and 0.11 um/ms.

`smi/analysis/models/barland_cnn.py` is a port of the 2021 architecture. At
one input channel it comes to 57,153 trainable parameters, an exact match
with the total the paper states, which is the check that establishes the port
is faithful rather than merely similar. At the three channels this data
supplies it comes to 57,377, and that is the configuration section 9 trains.

*Those reported numbers are not comparable to this document's.* They were
measured on different apparatus, with training displacements drawn from eight
harmonic frequencies between 53 and 93 Hz, against targets calibrated on that
rig. Section 9's Barland row is this repository's port of their architecture
on this repository's data. It is evidence about the architecture; it is not
their result, and the two should not be differenced.

= Learned models against the analytic baseline

Ten configurations, two seeds each, on both acquisitions: 40 runs, no
failures. Every run trained from scratch for 20 epochs on one RTX 3090 Ti,
eager rather than compiled, 1 h 51 min in total. The tabular baselines
(ridge, LightGBM) fit on CPU. The epoch cap was chosen to fit roughly two GPU
hours and is not a convergence criterion, which is the first thing to know
about every absolute number below.

#fig(
  "model_comparison",
  width: 100%,
  caption: [
    Velocity RMSE for every configuration trained, one panel per acquisition,
    against the analytic decode of section 3.2 drawn as the dashed line. Bars
    are the mean over two seeds and the whisker spans the seed-to-seed range,
    which two seeds give and a standard error does not. Row order is the
    free-space ranking and is held fixed in the right panel, so where the
    fiber ordering differs it reads as crossing rather than as a second
    unrelated chart. Hatched bars label the models that see samples after the
    timestamp they predict; only the TCN rows do not.
  ],
) <fig-models>

== What every configuration scored

Velocity RMSE in um/ms and displacement RMSE in um, each the mean over the
two seeds, from `smi/analysis/data/learned_results.json`. Every row carries
its parameter count, its receptive field in samples, and whether it reads
samples after the timestamp it predicts. The last three rows are the models
that trained worst and are here for the same reason as the rest: a table that
reports only the winners cannot be used to decide what to try next.

#align(center, table(
  columns: 8,
  align: (left, right, right, center, right, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header(
    [model],
    [params],
    [field],
    [causal],
    [vel. FS],
    [vel. MMF],
    [disp. FS],
    [disp. MMF],
  ),
  [Barland], [57,377], [91], [no], [*0.0814*], [*0.1643*], [0.1345], [0.3897],
  [SCNN], [44,049], [85], [no], [0.0953], [0.2471], [0.0945], [0.3287],
  [TCN], [44,049], [85], [*yes*], [0.1116], [0.2540], [0.1137], [0.3285],
  [SCNN], [19,329], [85], [no], [0.1125], [0.2713], [0.1293], [0.3778],
  [MLP], [51,329], [256], [no], [0.1275], [0.2547], [0.2571], [0.4993],
  [TCN], [19,329], [85], [*yes*], [0.1401], [0.2788], [0.1443], [0.3772],
  [TCN], [35,585], [29], [*yes*], [0.2021], [0.3136], [0.1921], [0.4107],
  [TCN], [27,985], [29], [*yes*], [0.2096], [0.3203], [0.2290], [0.4253],
  [LightGBM], [200], [91], [no], [0.2483], [0.3240], [0.3827], [0.4808],
  [Ridge], [274], [91], [no], [0.3837], [0.3431], [0.6061], [0.5133],
  [analytic], [--], [--], [no], [0.5611], [0.5647], [0.5051], [0.4833],
))

FS is free space, MMF multimode fiber. The analytic row is method 2 of
section 3.2, the same decode the dashed line in @fig-models marks; the means
are quoted here because the learned rows are means, where section 4.1's table
quotes medians of the same decode.

Across all twenty runs per acquisition the mean velocity RMSE is 0.1712
(free space) and 0.2771 (mm fiber). The best single run is 0.0799 and 0.1632.
Those are different claims and neither substitutes for the other: the mean
says what a configuration drawn from this set achieves on average, and the
best says what the set's most fortunate seed of its best architecture
achieved once.

== Every learned model clears the analytic baseline

On both acquisitions, all ten configurations score a lower velocity RMSE than
the analytic decode. The weakest, a ridge regression over a 91-sample window
with 274 coefficients, comes in at 0.3837 against the decode's 0.5611 in free
space and 0.3431 against 0.5647 in the fiber.

A linear fit over a 91-sample window is not a powerful estimator, so its
beating the multi-wavelength inverse says more about the inverse than about
the regression. Sections 4 to 6 already found that the Michelson model, given
the true displacement, leaves most of the variance unexplained unless its
amplitude and phase are refit every couple of fringes, and that the decode's
NRMSE sits just above 1. These runs measure the consequence. What remains
open after them is why the physical model does this badly, not whether
networks help.

One qualification belongs immediately beside the comparison. The analytic
decode is causal by construction and so is the TCN. The other eight rows are
not: SCNN, Barland, the MLP, LightGBM and Ridge all label each window at its
centre, so they read samples after the timestamp they predict. The sentence
"every learned model beats the analytic baseline" is true, and eight of the
ten do it with lookahead the decode and the TCN are denied.

The MLP row is the useful one for reading the three non-deep rows against
each other. At window stride 1 it steps its 256-sample window one sample at a
time, so the comparison against LightGBM and Ridge holds the features and the
evaluation grid fixed and varies only the estimator: 0.1275 um/ms against
LightGBM's 0.2483 and Ridge's 0.3837 in free space. What separates them is
how much of the window each can actually use: a boosted-tree ensemble over 91
raw samples lands at 1.95x the error of a two-layer network over 256.

One thing not to read into that. Stride 1 does not make the MLP a pointwise
map: 256 windows still overlap every output position, and the value reported
there is their average. It is closer to a kernel-256 convolution with a
learned nonlinear kernel and a stride-1 output than to a per-sample
regression, and its 256-sample receptive field in the table is the honest
description.

== Causality costs less than the architecture literature would suggest

The TCN and the SCNN at 44,049 parameters are a controlled pair. Both span 85
samples, both have the same parameter count, and they differ in where that
span sits relative to the predicted timestamp -- verified by gradient probe,
the TCN spanning $[-84, 0]$ samples and the SCNN $[-42, +42]$.

In free space the SCNN scores 0.0953 against the TCN's 0.1116, a 15% margin
for the non-causal model. *That margin is about one seed-spread wide* -- the
SCNN's two seeds differ by 0.0129 at this configuration -- so it establishes
the direction of the effect and not its size. What it does support is the
weaker and more useful statement: giving up lookahead entirely costs
something on the order of a seed, not a factor.

The comparison against the published architecture runs the other way, and it
is wider than any seed spread. In free space the Barland port at 57,377
parameters scores 0.0814 against the causal TCN's 0.1116 at 44,049 -- a 27%
lower velocity RMSE -- and behind fiber 0.1643 against 0.2540, 35% lower. The
seed spreads there are 0.0028 and 0.0004, an order below the free-space gap
and two below the fiber one. Part of that margin is lookahead the TCN is
denied; the SCNN pair above bounds how much lookahead alone is worth at about
one seed, so most of it is the architecture.

*This does not establish that the published network is non-causal.* Both
papers were searched for language about causality, centering or lookahead and
neither resolves where in its 256-sample window the output timestamp sits.
The 2021 paper's remark that the network can be used in real time is a
throughput claim, and its figures show offline reconstruction of recorded
traces. The causality conclusion here rests entirely on the TCN-versus-SCNN
pair above, which is this repository's own controlled comparison. What the
Barland row measures is the architecture as ported, run centered, on this
data.

== The published architecture wins on both acquisitions

The port of @barland2021cnn is the strongest of the ten configurations on
both datasets: 0.0814 um/ms in free space against the best dilated stack's
0.0953, and 0.1643 behind fiber against 0.2471. It is the only model that
places first twice, and the margin over the repository's own causal TCN is
27% in free space and 35% in the fiber. There is no dataset-dependent
recommendation to make here. A 2021 architecture with a 256-sample window and
aggressive max-pooling, ported and retrained on this data, is simply better
than the dilated-convolution stack this repository built, at a comparable
parameter budget.

That is not the answer this section was built to find, and it changes what
section 10 should sweep: the open question is which part of the Barland stack
does the work -- the wider window, the pooling, or the fully connected head --
not whether the dilated stack can be tuned past it.

== Measuring an architecture requires an output resolution finer than the score

`SlidingWindowModel` labels each window at its centre and averages the
predictions that overlap a given output sample. Run at a stride equal to the
window, nothing overlaps: the model emits one constant per 256-sample block,
so a 16,384-sample shot carries only 64 distinct values. That is a resolution
limit, not a modelling limit, and it has an oracle: the RMSE reached by a
predictor that emits the exact per-block mean of the true velocity. On
free space that floor is 0.1866 um/ms.

The Barland and MLP configurations were first run at stride 256 and scored
0.1962 and 0.2409. Barland's number sits within 5% of the floor, which means
the run reported almost nothing about the architecture -- a perfect estimator
constrained to the same output grid would have scored 0.1866. Both were
re-run at a stride where the floor cannot bind: Barland at 8, floor 0.0065,
and the MLP at 1, floor 0.0000. The tabular baselines moved to evaluation
stride 1 for the same reason. The TCN and SCNN rows emit one prediction per
input sample and were never affected; their numbers are byte-identical across
the two rounds.

The lesson generalizes past this table. A windowed model's score is bounded
below by its output discretization, and that bound is cheap to compute --
block-average the truth and take the RMSE -- so it should be computed before
any comparison is drawn, not after one turns out to be surprising. A score
close to its own floor is a measurement of the evaluation grid.

== The generalization gap is the finding

Ranking by how much each model degrades from free space to fiber is close to
inverting the free-space ranking, which is itself the result:

#align(center, table(
  columns: 4,
  align: (left, right, right, right),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header([model], [vel. FS], [vel. MMF], [MMF / FS]),
  [Ridge (274)], [0.3837], [0.3431], [*0.89*],
  [analytic decode], [0.5611], [0.5647], [1.01],
  [LightGBM (200)], [0.2483], [0.3240], [1.30],
  [TCN (28k, field 29)], [0.2096], [0.3203], [1.53],
  [TCN (36k, field 29)], [0.2021], [0.3136], [1.55],
  [TCN (19k, field 85)], [0.1401], [0.2788], [1.99],
  [MLP (51k)], [0.1275], [0.2547], [2.00],
  [Barland (57k)], [0.0814], [0.1643], [2.02],
  [TCN (44k, field 85)], [0.1116], [0.2540], [*2.28*],
  [SCNN (44k, field 85)], [0.0953], [0.2471], [*2.59*],
))

Every learned model degrades behind fiber; the analytic decode does not. That
much is unambiguous, and it is the finding. The ordering is also roughly
monotonic against the free-space ranking -- the models resolving the finest
fringe structure lose the most -- with the four best free-space models
degrading between 2.0x and 2.6x while the decode holds flat at 1.01x and the
ridge regression improves slightly, at 0.89x.

Each end of that has a reading the data supports. Ridge improving is what a
model extracting only a coarse relationship should do when speckle destroys
the fine structure it was never using; the coarse relationship survives. At
the other end, a stack that has learned the detailed fringe shape in free
space has learned something mode scrambling takes away.

The Barland port is not an exception to this. It degrades 2.02x, squarely
inside the band its free-space score predicts, and it wins behind fiber by
starting further ahead rather than by holding its ground. Its 256-sample
window and heavy pooling buy accuracy on both acquisitions, not robustness
across them.

Section 7 found no free-space/fiber gap in the analytic baseline at all, and
these ratios say why that was not reassuring: the gap is invisible to an
estimator that is not resolving the structure the fiber destroys. It appears
as soon as one is.

== Receptive field dominates parameter count

Two TCN configurations run at a 29-sample receptive field, at 27,985 and
35,585 parameters. They score 0.2096 and 0.2021 in free space: 27% more
capacity bought 0.0075 um/ms, which is within the smaller model's own seed
spread of 0.0160. A TCN at 19,329 parameters and an 85-sample field scores
0.1401 -- better than either, with 46% fewer parameters than the larger
field-29 model.

So on this data, over this range, widening the receptive field buys what
adding parameters at a fixed field does not. That is the cheapest thing the
next session can act on, and section 10 lists the sweep it implies.

== What limits these numbers

Eight things bound what the section above establishes, each of them a threat
to a conclusion a reader would otherwise draw.

*Only the TCN is causal.* Eight of the ten configurations label each window
at its centre and so read samples after the timestamp they predict. Any
comparison that crosses the hatch boundary in @fig-models compares estimators
solving different problems.

*Neither published paper states whether its network is causal*, so the
causality claim is scoped to this repository's own TCN-versus-SCNN pair. See
section 9.3.

*Cross-apparatus RMSE is indicative only.* Section 8's reported numbers came
off different hardware with harmonic training displacements. The Barland row
in these tables is a port run here, not a reproduction of their measurement.

*Two seeds is thin.* The 15% SCNN-over-TCN margin in free space is about one
seed-spread wide and is not leaned on anywhere above. Every cross-architecture
conclusion drawn here -- the 27% and 35% Barland-over-TCN gaps, the
receptive-field result, the generalization ratios -- is many spreads wide.
Which of the two any given claim is has been stated beside it.

*The runs are budget-limited, not converged.* Twenty epochs, chosen to fit
about two GPU hours. The absolute RMSEs are therefore pessimistic; what the
section compares is ten models trained under one identical budget, which is
the comparison the numbers support.

*Output stride now differs by row, so a matched parameter budget no longer
implies a matched output resolution.* Barland runs at window stride 8, the
MLP at 1, the tabular baselines evaluate at 1, and the TCN and SCNN emit one
prediction per input sample by construction. Each stride is recorded
per-record in the artifact. The strides were chosen to push each model's
discretization floor well below its score rather than to match each other:
Barland retains a floor of 0.0065 um/ms, 12x below its 0.0814, so the floor
cannot bind -- but it is not zero, and a future comparison drawn at a finer
error scale would have to revisit it.

*Displacement is not measured the same way in the two halves of the table.*
The learned displacement column is obtained by integrating predicted
velocity; the analytic decode fits displacement directly and differentiates
for velocity. They are different estimators of the same quantity, so the
displacement columns compare less cleanly than the velocity ones, and the
velocity column is the one every claim above uses.

*Not every published figure in section 8 came off the page cleanly.* The two
parameter counts, 57,153 and 207,905, are each stated twice in their own
paper, and the first is confirmed independently by this repository's port
measuring exactly 57,153. The correlation and RMSE figures -- 0.90 for
@barland2021cnn, and 0.96, 0.13 um/ms, 0.97 and 0.11 um/ms for
@matha2023multichannel -- were recovered from a text extraction that mangled
the decimal-point glyph, and were reconstructed by cross-referencing passages
where the same value extracted cleanly. They are almost certainly right; a
reader checking one against its source should know which figures are
transcribed and which are reconstructed.

Two further facts about how the runs were produced belong on the record.
Every model trained from scratch: the checkpoint directory was empty, and the
`TemporalBlock` change together with the regenerated normalization statistics
would have invalidated older checkpoints in any case. And at these model
sizes the data loader, not the GPU, was the constraint -- about 42 s per epoch
of loading against 9.6 s of compute, so the dataset had to be preloaded to
fit the budget. Those two figures are the measurement, not a quotation: the
compute side is 19.2 ms per step at batch 16, warmed, with
`torch.cuda.synchronize()` around the timed region and the median taken over
ten steps on the same GPU. The loader headroom reported by
`smi.analysis.benchmark_loading` was measured on a production-size TCN at
batch 32, where training is GPU bound; it does not transfer to 19-44k
parameters, which is why the sweep had to be measured on its own.

= Future directions

In order of what returns most for the least work.

*A scaling study: RMSE against dataset size and against parameter count.*
Deferred from this round, and most of its harness is now in place -- the
sweep runner, the results artifact and the comparison figure all generalize
to more rows. It is also the axis where published guidance exists:
@barland2021cnn reports that going beyond roughly $10^5$ parameters "does not
lead to better training and may even lead to worse predictions", which
brackets how far up the parameter axis is worth sweeping before measuring.

*A third seed.* Cheap, and it settles the one caveat above that a
measurement rather than an argument can settle: whether the SCNN's free-space
margin over the TCN is real.

*A receptive-field sweep.* Section 9.7 found 27% more parameters at a fixed
29-sample field buying nothing, while widening the field at fewer parameters
cut the error by 31%. Only two field lengths have been tried, 29 and 85, so
the axis that moved the numbers most is the one with two points on it.

*Feature engineering for the fiber gap.* Better motivated now than before:
section 9.6 measures the degradation at 2.3x for the best causal model rather
than leaving it as a suspicion, and locates it specifically in the models
that resolve fine fringe structure. That is a target a feature can be
designed against.

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
its amplitude and phase to be constant: windowed $R^2$ of 0.50 to 0.87 against
scrambled controls of 0.28 to 0.34, a gap of 0.21 to 0.54 by channel. And the
drive-to-displacement calibration
carries a real gain error of 5 to 13%, so the supervision targets are
systematically small by about that much.

Section 9 establishes that the floor is clear of every learned model tried,
by a margin that runs from 1.46x (ridge, free space) to 6.9x (the Barland
port, free space); that this repository's port of the published architecture
is the best model measured on both acquisitions, ahead of the causal TCN at
44,049 parameters by 27% in free space and 35% behind fiber; and that the
free-space-to-fiber degradation runs from 0.89x to 2.59x across the ten
configurations, ordered against the free-space ranking.

*What this does not establish.* It is not an upper bound on classical
processing: a better inverse, or a corrected transfer function, could do
better. It says nothing about latency, and nothing about waveform families
outside the one these acquisitions sweep. It does not prove the decoder is
noise-robust, only that it is correct on clean model-generated fringes. The
baseline half does not speak to the free-space/fiber gap, which is invisible
at this error level -- section 9.6 is where that gap becomes measurable, and
it is measured there only for the models trained under this round's budget.
It does not give a corrected value for `k`: the two acquisitions
disagree by 3.2 to 6.5%, so the sweep locates an error without resolving it.
And the residual amplitude dependence that survives a per-shot gain fit is
unexplained -- measured, bounded, and open.

Section 10 lists what to do on the learned side. On the classical side two
items remain and are unaffected by it. Improving the inverse is still worth
doing: method 2's NRMSE near 1.1 is the binding number, and the model it
inverts is not what fails -- method 1 clears its control on every channel
with the displacement supplied, so the loss is in the decode. And `k` still
needs settling, which needs an experiment that can explain why two
acquisitions of the same speaker prefer gains 3.2 to 6.5% apart.

Those two are now lower priority than they were when this document covered
only the baseline. A learned model already clears the inverse by a wide
margin, so the inverse is no longer on the path to a usable estimator; it is
on the path to understanding why the physical model underperforms, which is
a separate and still-open question.

#bibliography("refs.bib")
