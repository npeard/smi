// Research report: velocity retrieval from multi-wavelength self-mixing
// fringes, organized by the questions of QUESTIONS.md. Each question section
// runs in three tiers -- analytic, synthetic, real data -- and states its own
// status in place.
//
// Build:  pixi run report-only  (or, from report/: snakemake build/report.pdf)
//
// Every number in this document comes from a committed artifact or from a
// computation in a committed script:
// - report/results/baseline_results.json (method 1 and method 2, written by
//   smi.experiments.baselines), report/results/gain_sweep_results.json (the
//   gain, delay and fringe-rate sweeps, written by smi.experiments.gain_sweep)
//   and report/results/learned_results.json (the trained models, written by
//   scripts/run_baseline_sweep.py), quoted in the real-data tables;
// - report/results/synthetic_results.json, the experiment runner's records,
//   read by the report/figures/plot_*.py scripts this document includes;
// - smi.theory.bounds, evaluated by the figure scripts
//   report/figures/plot_{fisher_phase,velocity_bounds,sign_information}.py;
// - build/synthetic_values.json, written by report/figures/synthetic_values.py
//   from smi.theory.bounds, learned_results.json, synthetic_results.json and
//   the experiment grids: every value quoted in the prose is read from it
//   below, so a library or artifact change rebuilds the prose rather than
//   leaving it stale.
//
// The two real-data methods run at different shot counts -- method 1 at
// N = 1000, method 2 at N = 200, because method 2's Viterbi decode costs
// ~5.85 s per shot. State which N a table was produced at, every time.
//
// The two figures that read the raw acquisition HDF5 (the method-1 shot traces
// and the free-space/fiber spectrum) are out of the DAG until the data are on
// disk; TASKS.md tracks restoring them from a committed extract.
//
// ASCII only, per the repo convention.

#import "lib/theme.typ": fig, report-theme, takeaway

#show: report-theme.with(
  title: "Velocity estimation from self-mixing fringes",
  author: "Nolan Peard",
  date: "Draft",
)

#set math.equation(numbering: "(1)")
// Floats go to the top or bottom of a page, so a figure that misses the space
// left on one page does not leave that space blank.
#show figure: set figure(placement: auto)

#let values = json("/build/synthetic_values.json")
#let val(key) = values.at(key)

= Introduction <sec-intro>

A diode laser that receives some of its own light back from a target becomes
an interferometer. The returning field perturbs the cavity's phase and gain,
so the output power, read by the photodiode packaged behind the laser,
carries a fringe for every half wavelength of target travel. Lang and
Kobayashi modelled this external-cavity feedback @lang1980feedback, and three
reviews @giuliani2002selfmixing @donati2012selfmixing @taimre2015tutorial
describe the resulting self-mixing interferometry (SMI) as a sensor. It needs
no reference arm, beamsplitter or external detector, but its fringe is not a
cosine. Feedback of strength $C$ skews it toward a sawtooth whose lean gives
the direction of motion, which a plain two-beam interferometer does not
observe.

The established instrument for the same measurement is the laser Doppler
vibrometer, a two-beam interferometer that recovers direction with a
frequency-shifted reference @rothberg2017ldv. Our rig shines three lasers of
different colors onto one mirror. The task is to recover the mirror's
velocity from the three photodiode channels, causally and accurately enough
to run against live hardware.

Learned decoders are the current alternative to fitting the signal model.
Barland and Gustave trained a convolutional network on single-channel
self-mixing fringes @barland2021cnn. Matha et al. extended it to three
channels and showed the sensor keeps tracking when some channels lose signal
@matha2023multichannel, the nearest prior work. Neither scored a
non-learning comparator on the same shots, so this report scores learned and
classical estimators side by side.

Deployment adds a multimode fiber between laser and target. Mode coupling in
the fiber scrambles the transmitted field into speckle @ploschner2015chaos.
We infer that the light fed back to the laser is scrambled too, shot by shot,
so an estimator trained in free space sees fringes it was never shown.

Four questions organize the work: whether extra channels help (Q1), which // math-ok: question label
estimator is most accurate and robust (Q2), how to recover motion through // math-ok: question label
multimode fiber (Q3), and whether a Sagnac self-mixing accelerometer gains // math-ok: question label
from the nonlinearity (Q4). Each is answered analytically, then on simulated // math-ok: question label
data, where the true motion is known, then on the real acquisitions; Q4 stops // math-ok: question label
at the analytic tier. The synthetic program fixes its predictions for the real
data before reading them. Questions without results carry a one-line status in
place. The answers so far:

#takeaway[
  + *Q1.1, channel count: for TCAN each added color helps*, noise-free and at
    the nominal point; the offline UNetTransformer gains nothing its seed
    spread resolves (@fig-q1-count).
  + *Q1.2, colors against a one-color array: quadrature wins only without
    noise*; at the nominal point the two are level.
  + *Q2.1, accuracy floor: only with feedback does the architecture matter.*
    TCAN is best, while the LSTMs and the offline neural operator do not
    train (@fig-families). Context length
    limits these regressors more than family does (@sec-q2-context).
  + *Q2.1.1, self-mixing nonlinearity: weak optical feedback makes the
    direction of motion observable.* Without it a network trained on signed
    velocity predicts zero; with it the error falls steadily with feedback
    strength (@fig-exit-check, @fig-feedback).
  + *Q2.2, noise robustness: per-laser phase drift dominates at the nominal
    operating point.* Removing it cuts the error more than any electronics
    knob (@fig-loo).
  + *Q2, real data: every learned model clears the analytic baseline*, which // math-ok: question label
    does worse than predicting zero; the port of the Barland network @barland2021cnn is best on both
    acquisitions (@fig-models).
  + *Q3, fiber degradation: static coupling loss costs the most*, against an // math-ok: question label
    analytic reference that ranks it free; path drift costs little and speckle
    drift nothing (@sec-q3-degradation).
  + *Q3.3 and Q3.4, flow matching: the encoder head's whole-shot context
    drives the gain.* Alone it halves TCAN's error at matched size; sampling
    the flow adds calibrated bands and a small further gain, while
    denoising fiber traces does not help (@sec-q3-posterior).
  + *Q3, multimode fiber, real data: every neural model degrades behind // math-ok: question label
    fiber*, while the analytic decode holds (@sec-learned-gap).
]

= Analytic foundation <sec-foundation>

Before training any network we ask what the signals determine at all: whether
they fix the direction of motion, and how precisely they fix the displacement.
The answers say which errors a learned estimator could remove and which no
estimator can.

== From a Michelson interferometer to self-mixing <sec-model>

A mirror moves along a trajectory $d(t)$, and we want its velocity
$v(t) = dot(d)(t)$, the quantity the networks output. Each laser forms its own
interferometer with the mirror, so the channels record the same $d(t)$ through
different wavelengths. In a Michelson interferometer channel $k$ reads
$
  y_k (t) = a_k (t) [1 + V_k (t) cos theta_k (t)] - 1 + n_k (t),
  quad theta_k (t) = (4 pi d(t)) / lambda_k + phi_k .
$ <eq-michelson>
The optical phase $theta_k$ carries the motion, one fringe per $lambda_k \/ 2$
of travel. The offset $phi_k$ is set by the unknown static path length and
drawn uniformly on $[0, 2 pi)$ per shot. The amplitude $a_k$ and visibility
$0 < V_k <= 1$ scale the fringe without carrying motion, and $n_k$ is detector
noise.

In self-mixing the mirror's light re-enters the laser cavity and pulls the
laser's phase, which distorts the fringe. The signal generalizes
@eq-michelson by replacing the cosine with a fringe function $F$,
$
  y_k (t) = a_k (t) [1 + V_k (t) F(theta_k (t))] - 1 + n_k (t),
  quad F(theta) = cos phi_F (theta) ,
$ <eq-signal>
where the phase $phi_F$ at which the laser oscillates solves the excess-phase
equation
$
  phi_F = theta - C sin(phi_F + beta), quad beta = arctan alpha .
$ <eq-excess>
The feedback strength $C$ says how strongly the returning light pulls the
phase, and $C = 0$ recovers the Michelson cosine. The linewidth enhancement
factor $alpha$ couples gain to phase and sets the skew $beta$. In the
weak-feedback regime $0 <= C < 1$ the right side of @eq-excess has slope at
least $1 - C > 0$ in $phi_F$, so $F$ is single-valued.

== The Michelson sign is unobservable <sec-sign>

If the signals do not determine which way the mirror moves, a network asked
for signed velocity should give up. For the Michelson model they do not. The
map $d -> -d$, $phi_k -> -phi_k$ sends every $theta_k$ to $-theta_k$, the
cosine is even, and the uniform law of $phi_k$ is unchanged, so a trajectory
and its mirror image have the same likelihood,
$
  p(y | d) = integral p(y | d, phi) p(phi) dif phi
  = integral p(y | -d, -phi) p(-phi) dif phi = p(y | -d) .
$ <eq-sign-likelihood>
The simulated drive has independent uniform line phases, so $p(d) = p(-d)$,
the posterior is symmetric, and the least-squares estimate is
$
  E[v(t) | y] = 0 .
$ <eq-sign-zero>
To ask instead whether a network recovers the motion up to the unobservable
sign, we score each shot with the sign-invariant loss
$
  cal(L)_"SI" = min_(epsilon in {plus.minus 1}) "MSE"(epsilon hat(v), v) .
$ <eq-sign-invariant>
@fig-exit-check tests both predictions with one network trained three ways,
each pair of runs differing in the loss or the fringe. The runs are short, so
only their ordering is evidence.

#fig(
  "exit_check",
  width: 62%,
  caption: [
    Without feedback a signed-loss network learns to predict zero; with
    feedback it recovers the motion and its sign. Bars: test velocity NRMSE,
    each shot's RMSE over its velocity std, so predicting zero scores 1
    (dashed), of a causal temporal convolutional network (TCN) on three
    noise-free channels, trained at most #val("exit_max_epochs") epochs; x
    labels give the training loss and colour the scoring, where
    sign-invariant takes the better of $plus.minus hat(v)$ per shot.
    Michelson: $C = 0$; self-mixing (SMI): $C = 0.2$, $alpha = 3$.
  ],
) <fig-exit-check>

== Feedback breaks the symmetry <sec-feedback>

Feedback tilts the fringe toward a sawtooth whose steep edge faces the
direction of motion. The question is when this breaks the symmetry of
@sec-sign, and how much sign evidence one shot carries.

The symmetry survives feedback when $alpha = 0$: then $beta = 0$, negating
@eq-excess shows $phi_F (-theta) = -phi_F (theta)$, $F$ stays even, and
@eq-sign-likelihood holds for every $C$. For $C > 0$ and $alpha != 0$ the
fringe's slow and steep edges swap with the direction of motion.

With one channel the break is incomplete, because a slightly warped phase
reproduces the fringe exactly:
$
  theta'' = phi_F + C sin(phi_F - beta) quad => quad
  F(-theta'') = F(theta), quad theta'' - theta = -2 C sin beta cos phi_F .
$ <eq-warp>
Only the motion prior's penalty on that warp separates the two hypotheses.
With two or more channels each warp depends on its own $phi_F$ and
$lambda_k$, so no single warped trajectory mirrors every channel.

To size the evidence we hold the trajectory fixed and compare the true signal
with the closest sign-flipped one,
$
  tilde(y)_k (t) = m + b F(-(4 pi d(t)) / lambda_k + phi'), quad b >= 0 ,
$ <eq-flipped>
free in offset, gain and phase, with the gain non-negative because photodiode
polarity is known. The Kullback-Leibler divergence between the two noise
distributions, for $L$ white-noise samples of std $sigma$ and uniform phase
coverage, is to leading order in $C$
$
  I_"sign" = sum_k (L C^2 E[V_k^2] alpha^2) / (4 sigma^2 (1 + alpha^2)) .
$ <eq-sign-info>
Freeing the trajectory can only lower $I_"sign"$, so it is an upper bound.

#fig(
  "sign_information",
  width: 62%,
  caption: [
    One noisy shot carries far more than one nat of sign evidence across weak
    feedback, growing as $C^2$. Upper bound on the expected log-likelihood
    ratio of the true signal over the closest sign-flipped one, for one
    16384-sample shot at SNR 30 dB, $V tilde U(0.5, 1)$, mean over 8 shots;
    $K = 1$ is the 635 nm channel and $K = 3$ all three. Markers: computed;
    lines: @eq-sign-info.
  ],
) <fig-sign>

With the trajectory fixed, no mirror reproduces the signals; this is
numerical, not proven. Whether some other trajectory, for example one that
switches fringe branch at the extrema, mirrors every channel at once is open.

== Fisher information per sample <sec-fisher>

Once the sign is settled, the question is precision. The Fisher information,
whose inverse bounds an unbiased estimator's variance, adds over independent
channels and samples, so it shows what each channel contributes and where it
goes blind. It vanishes at the fringe extrema, and feedback moves it onto the
steep edge.

For channel $k$ at unit amplitude with white Gaussian noise of std $sigma$,
the information about $d$ in one sample is
$
  J_k (theta) = ((4 pi V_k) / (lambda_k sigma))^2 F'(theta)^2, quad
  F'(theta) = - (sin phi_F) / (1 + C cos(phi_F + beta)) ,
$ <eq-fisher>
with $F'$ from implicit differentiation of @eq-excess. $F'^2$ vanishes at the
fringe extrema, where a small motion leaves the signal unchanged
(@fig-fisher). $K$ channels carry $sum_k J_k$, and the $lambda_k^(-2)$ factor
favours the shortest wavelength.

Additivity alone does not remove the blind spots. Equal-wavelength channels
with offsets spaced by $pi \/ K$ would, but the rig's channels differ in
wavelength and their offsets are uncontrolled. Wavelength diversity therefore
raises the average information without guaranteeing it in the worst case, when
all channels near their extrema together.

The bounds below need the information averaged over a uniform fringe phase,
which has a closed form,
$
  Phi(C, alpha) = chevron.l F'^2 chevron.r_theta
  = 1 / (2 s) [1 - cos(2 beta) (1 - s) / (1 + s)], quad s = sqrt(1 - C^2) ,
$ <eq-phase-factor>
obtained by changing variables to $phi_F$ and using two standard averages
over a uniform angle $u$,
$
  chevron.l 1 / (1 + C cos u) chevron.r_u = 1 / s, quad
  chevron.l (cos 2u) / (1 + C cos u) chevron.r_u = (1 - s) / (s (1 + s)) .
$ <eq-phase-integrals>
For $|alpha| > 1$, $cos 2 beta < 0$, so feedback raises the average above
the Michelson value.

#fig(
  "fisher_phase",
  width: 62%,
  caption: [
    Feedback concentrates the information on the steep fringe edge but keeps
    the blind spots at the extrema. The squared fringe slope $F'(theta)^2$
    of @eq-fisher over one fringe at $alpha = 3$. Dashed: its phase averages
    $Phi(C, alpha)$ of @eq-phase-factor, #val("phase_factor_c02") at
    $C = 0.2$ and #val("phase_factor_c07") at $C = 0.7$.
  ],
) <fig-fisher>

== Causal and offline bounds on velocity error <sec-crb>

Velocity is read across many samples, so bounding its error needs a model of
how the motion evolves. Combining @eq-fisher with one gives two Bayesian
Cramer-Rao bounds: causal, for an estimator that sees only past samples, and
offline, for one that sees the whole shot. If a learned estimator sits far
above them, information is not what limits it.

The bounds need a linear motion model, so we fit a Gauss-Markov surrogate to
the simulator's motion. Its state $x_n = (d, v, dot(v))$ evolves as
$
  x_(n+1) = A x_n + w_n, quad w_n tilde cal(N)(0, Q) ,
$ <eq-prior>
with one real pole and one resonance, discretized exactly at the sample rate,
and matches the simulated trajectories' in-band velocity spectrum to
#val("prior_psd_rms_db") dB rms.

Because $theta_k mod 2 pi$ is uniform whatever $d$ is, the average of
@eq-fisher over phase and visibility is exact:
$
  macron(J) = sum_k E[V_k^2] Phi(C, alpha) ((4 pi) / (lambda_k sigma))^2 .
$ <eq-info-avg>
The causal bound is the posterior Cramer-Rao bound @tichavsky1998pcrb, whose
information matrix obeys
$
  J_(n+1) = (Q + A J_n^(-1) A^T)^(-1) + macron(J) c_d c_d^T ,
$ <eq-pcrb>
with $c_d$ selecting $d$: the covariance recursion of a Kalman filter that
measures $d$ with information $macron(J)$ per sample. The offline bound is the
Van Trees bound @vantrees1968detection, the inverse of the block-tridiagonal
prior information plus $macron(J) c_d c_d^T$ per block. Averaging the
information before inverting is safe, because
$
  (E J)^(-1) <= E[J^(-1)] ,
$ <eq-jensen>
so the averaged bound is valid and looser than one conditioned on the phases.

#fig(
  "velocity_bounds",
  width: 62%,
  caption: [
    Extra channels and stronger feedback lower the causal velocity bound far
    less than they raise the information per sample. Stationary RMS velocity
    bounds at SNR 30 dB, $V tilde U(0.5, 1)$, $alpha = 3$; $K = 1$ is 635 nm,
    $K = 2$ adds 674.8 nm, $K = 3$ adds 515 nm. Causal: @eq-pcrb after
    convergence; offline: the smoothing bound at mid-record.
  ],
) <fig-bounds>

The causal bound scales close to the inverse fourth root of $macron(J)$
(@fig-bounds) and sits #val("real_gap") times below the causal TCN's error on
the real free-space test split (@sec-learned-table). Within its own model the
bound is nearly attained by an extended Kalman filter given the true phases,
so that gap measures what the bound leaves out as much as any estimator's
shortfall.

The bounds are exact for a simpler model than the simulator, so they are not
a floor for it. They use the surrogate prior rather than the simulator's
finite sum of drive lines; they know $phi_k$ and $V_k$; they omit drift,
offset, the photodiode low-pass and the ADC; and they are local, blind to the
global sign and the fringe order.

= Methods <sec-methods>

== Notation <sec-notation>

One symbol per quantity, throughout.

#align(center, table(
  columns: 2,
  align: (left, center),
  stroke: 0.5pt + rgb("#d0d0d0"),
  table.header([quantity], [symbol]),
  [optical round-trip phase], [$theta$],
  [fit parameters], [$theta.alt$],
  [channel (color) count], [$K$],
  [samples per shot], [$L$],
  [shots in a set], [$N$],
  [fiber modes per polarization], [$M$],
  [fiber feedback coupling], [$kappa$],
  [fiber dispersion], [$D$],
  [per-laser phase drift], [$sigma_phi$],
  [fiber path drift], [$sigma_p$],
  [process noise covariance], [$Q$],
  [Sagnac coupler power split], [$rho_c$],
  [Sagnac loop amplitude reflectivity], [$rho_r$],
  [Sagnac element phase per displacement], [$eta$],
  [Sagnac modulation depth], [$a_0$],
  [proof-mass quality factor], [$Q_m$],
))

== Data, training and evaluation <sec-methods-synthetic>

Every learned result compares cells that differ in one factor: the feedback,
a noise knob, the channel subset or the model family. That comparison is fair
because data, training budget and scoring are otherwise shared.

The simulator draws each shot's trajectory as a sum of drive lines with
independent uniform phases and a per-shot amplitude, and renders the channels
of @eq-signal. Each imperfection it models is a knob: white noise, ADC bits,
photodiode bandwidth, visibility, DC offset, and amplitude, contrast and phase
drift. The noise-free configuration turns every knob off; the nominal point
sets each to its assumed rig value. Every model maps the channel signals to
velocity at every sample. The reference model is a causal temporal
convolutional network (TCN) of 44k parameters. @tab-splits gives the data
splits and @tab-training the training protocol.

#figure(
  {
    set par(justify: false)
    set text(size: 9pt)
    table(
      columns: (auto, 1.8fr, 1fr),
      align: (left, left, left),
      stroke: 0.5pt + rgb("#d0d0d0"),
      table.header([Split], [How it is generated], [Size]),
      [Training],
      [Simulated fresh for every batch from the cell's seeded random stream, so no
        shot repeats. Each shot is cut to a random #val("m_crop")-sample window;
        the offline neural operator trains on whole shots.],
      [#val("m_batch") shots $times$ #val("m_steps") steps $=$
        #val("m_shots_per_epoch") shots per epoch, #val("m_samples_per_epoch") M
        samples (#val("m_full_samples_per_epoch") M uncropped)],

      [Validation],
      [#val("m_val_batches") batches from a fixed seed (#val("m_val_seed")), cropped
        like training, simulated once and replayed at every epoch. It picks the
        best checkpoint and drives early stopping.],
      [#val("m_val_shots") windows],

      [Test],
      [Frozen sets of whole shots, one per simulator configuration, seed
        #val("m_test_seed"), each identified by a hash of its configuration.
        Every estimator of a configuration scores the same shots.],
      [#val("m_test_shots") shots per set, #val("m_test_sets") sets],
    )
  },
  caption: [
    The three data splits behind every learned result. Training data are never
    reused, so the amount seen grows with the epochs trained (@tab-training).
  ],
) <tab-splits>

#assert(val("m_full_length_is_no"), message: "a family other than the offline neural operator now trains on whole shots")

#figure(
  {
    set par(justify: false)
    set text(size: 9pt)
    table(
      columns: (auto, 1fr),
      align: (left, left),
      stroke: 0.5pt + rgb("#d0d0d0"),
      table.header([Setting], [Value]),
      [Shot], [#val("m_shot_length") samples at 488 kHz (33.6 ms); drive lines
        1 Hz to 1 kHz, amplitude $U(0, 1)$ per shot],
      [Reference TCN field], [#val("q2_tcn_field") most recent samples],
      [Optimizer], [Adam, peak learning rate #val("m_lr"), cosine annealing
        with warm restarts],
      [Noise and channel grids], [$T_0 = 30$ epochs, cap 100;
        #val("m_t30_cells") cells, #val("m_t30_epochs_lo") to
        #val("m_t30_epochs_hi") epochs, #val("m_t30_kshots_lo") k to
        #val("m_t30_kshots_hi") k shots],
      [Feedback curve, families], [$T_0 = 90$ epochs, cap 150;
        #val("m_t90_cells") cells, #val("m_t90_epochs_lo") to
        #val("m_t90_epochs_hi") epochs, #val("m_t90_kshots_lo") k to
        #val("m_t90_kshots_hi") k shots],
      [Early stopping], [no #val("m_min_rel_delta_pct")% validation gain in
        #val("m_patience") epochs, never before epoch #val("m_grace")],
      [$alpha = 0$ controls], [#val("q3_a0_epochs") epochs; cannot learn the
        sign, so the best checkpoint is the first],
      [Seeds], [one per cell unless stated; three noise-free seeds span
        #val("q2_seed_range_pct")% of the mean error],
      [Hardware], [Apple #val("m_device"), float#val("m_precision"), Python
        3.14, torch 2.15.0],
      [Zero predictor], [#val("q1_zero") $mu$m/ms, noise-free, three channels,
        $C = 0.2$],
    )
  },
  caption: [
    Training protocol shared by every synthetic cell. The seed range is the
    resolution for ranking any two cells.
  ],
) <tab-training>

Each error is a shot's RMSE averaged over the test shots: velocity in
$mu$m/ms, displacement in $mu$m. NRMSE divides each shot's RMSE by its
velocity std before averaging, so it weights slow shots more. Displacement
RMSE integrates the predicted velocity and removes each start offset. The
grace period before early stopping lets slow starters leave the zero-output
plateau.

#fig(
  "spectral_mismatch",
  width: 75%,
  caption: [
    Velocity, not displacement, has its power where the fringes do. Hann-windowed
    power spectral densities, each divided by its peak, of the two targets and
    each channel's fringe frequency $2 |v| \/ lambda_k$, averaged over
    #val("spec_shots") shots at $C = 0.2$, $alpha = 3$: (a) the nominal
    point; (b) noise-free, with the same trajectories. Below
    #val("spec_bin_hz") Hz drive content enters a shot only as a trend.
  ],
) <fig-spectral>

The networks predict velocity rather than displacement because velocity
matches the information in the signal (@fig-spectral). Each fringe frequency
follows $|v|$, while displacement weights the velocity spectrum by
$1 \/ f^2$ and puts most of its power at frequencies no shot resolves.

== Likelihood versus fit <sec-likelihood>

Write one shot's $L$ samples as $y_n$ and the model's prediction as
$f_n (theta.alt)$, with $theta.alt$ the fit parameters. With independent
Gaussian residuals of fixed variance $sigma^2$ the likelihood is

$ p(bold(y) | theta.alt) = product_(n=1)^L 1 / sqrt(2 pi sigma^2)
  exp(- (y_n - f_n (theta.alt))^2 / (2 sigma^2)), $

and its negative logarithm is

$ - ln p(bold(y) | theta.alt) = 1 / (2 sigma^2) sum_(n=1)^L (y_n - f_n (theta.alt))^2
  + L / 2 ln(2 pi sigma^2). $

This is the summed squared residual up to a constant and a positive scale, so
maximum likelihood and least squares return the same $theta.alt$ and rank
shots identically.

#takeaway[
  *Recommendation: fit each shot in closed form and report MSE.* Likelihood
  and MSE are one criterion here. Per-shot MSE is reported as an RMSE, and as
  an $R^2$ where a scale-free version is more readable.
]

The equivalence leaves open what $theta.alt$ is: how many parameters the fit
gets and over how long a stretch of the record. That choice moves the numbers
most, so every fit below states its interval.

== The two measurements <sec-measurements>

Both methods run on the same shot format. Method 1 is a closed-form
least-squares solve and runs on five times more shots than method 2, whose
Viterbi decode is slow. Every table names its own N, and shot-level
metrics are medians unless stated.

=== Method 1: forward-model residual <sec-method1>

Displacement is known from the drive voltage, so the interference signal it
predicts is known up to an amplitude and phase. Expanding the cosine makes the
fit a linear least-squares solve in cosine, sine and constant, with a global
optimum and nothing to converge.

The open choice is the interval. The amplitude, phase and offset are not
constant over a whole record in this data, so the fit is reported globally and
refit in windows of $W$ samples, with residual and total variance pooled
across windows into one $R^2$ per shot against the whole record's mean, so the
two are comparable.

$W$ is *fringe-matched*: resolved per shot and channel so that one window
spans about two fringes, within fixed limits. A fixed $W$ would hold a fraction
of a fringe on the quietest shots and many on the loudest, so its meaning would
change across excursion bins. Every headline table and figure uses this
window; the text flags any fixed-$W$ number.

A shorter window fits more of anything, so every windowed $R^2$ appears beside
a *scrambled-displacement control*: the same $W$, window count and parameters,
with the displacement taken from a different shot. The control measures what
the fit buys from flexibility; the model earns the gap between the two.

=== Method 2: multi-wavelength inverse <sec-method2>

Method 2 is the classical baseline. From the three photodiode signals alone it
resolves the fringe ambiguity using the wavelength diversity, decodes
displacement on a grid, and differentiates for velocity. It is scored in the
networks' physical units. The overall sign is not scored, because under the
Michelson model it is unobservable (@sec-sign).

== How to read these numbers <sec-read>

*Statistics.* Every real-data RMSE is the mean over a run's test shots of the
per-shot RMSE; a learned configuration's value is the mean over its two seeds.
The decode's NRMSE and Pearson r, and every method-1 $R^2$, are medians over
shots, because a near-motionless shot makes its NRMSE explode
(@sec-r-method2). The zero predictor's per-shot velocity RMSE is the shot's
true velocity RMS, so its mean is the bar for every mean velocity RMSE.

*Controls.* A windowed $R^2$ is quoted only beside its scrambled-displacement
control at the same $W$ (@sec-method1), because the control alone reaches a
large fraction of the windowed value (@sec-r-method1).

== Multimode fiber model <sec-fiber-model>

Q3 runs the simulator through a model of multimode fiber that carries $M$ // math-ok: question label
spatial modes in two polarizations. The laser launches into one mode, and its
cavity couples back only to that mode. One-way transmission is
$T(t, lambda) = e^(i 2 pi p(t) n(lambda) \/ lambda) U(t, lambda)$: a common
optical-path drift $p(t)$ times a speckle unitary $U = e^(i H)$ whose
Hermitian generator random-walks. The model rests on three assumptions.

1. Reciprocity. Traversed backwards the fiber is $T^T$, so after an ideal
   distal mirror the returned field is $r = T^T T e_0$, and the feedback
   coupling is $kappa = (T^T T)_(00)$.
2. $|kappa| <= 1$ multiplies both $C$ and the fringe visibility; $arg kappa$
   adds a per-color speckle phase $psi_k$.
3. A polarizing beamsplitter sends every orthogonal mode to one photodiode,
   which reads $P_perp = sum_(j=1)^M |r_(M + j)|^2$.

With $n(lambda) \/ n_"ref" = 1 + D (lambda - 1 mu"m")$, channel $k$ sees
$
  theta_k (t) = (4 pi) / lambda_k (d(t) + n p(t)) + psi_k (t)
$ <eq-fiber-phase>
up to the dispersion term. Generators of different colors are correlated over
a Gaussian width in wavelength, so colors far apart see independent speckle.

== Flow-matching models <sec-flow-methods>

Q3.3 and Q3.4 use conditional flow matching: a velocity field integrates
Gaussian noise $x_0$ to a draw $x_1$ of the target (velocity for the
posterior, free-space signals for the denoiser) given the fiber signals. The
field is a global-context UNetTransformer. A flow without an encoder
ignores its input, so every flow here conditions on an encoder. It maps the
signals to features the field reads, and its first channels are fit to
$x_1$ by an auxiliary regression loss, so the features must carry the
target; a toy test (`tests/test_flow.py`) checks that an encoder flow reads
a conditioning the encoder-free one ignores. The encoder is itself a
whole-shot UNetTransformer, so its regression head is also scored alone as a
deterministic regressor. Flows train on full shots, not crops; scores use
the same frozen test sets and evaluation path as the regressors, and an
audit rescores the encoder head and posterior mean on a fresh test seed.
The encoder flows keep the checkpoint with the lowest validation flow loss,
not flow plus regression loss, so the head scored is not its own best.

== Predictions for the real data <sec-real-predictions>

// The free-space causal TCN's test velocity RMSE (@sec-learned-table) and
// the free-space mean true velocity RMS (@sec-r-method2), the zero
// predictor's mean per-shot error: one statistic, the mean over shots.
#let real-tcn-fs = decimal(val("rl_tcn44_v_fs"))
#let real-v-rms-fs = decimal(val("rs_vrms_fs"))
#let real-ratio = calc.round(real-tcn-fs / real-v-rms-fs, digits: 2)

#assert(
  float(val("pred_base_ratio")) < float(real-ratio)
    and float(real-ratio) < float(val("pred_c09_ratio")),
  message: "the real free-space ratio no longer lies between the noise-free base and every nominal-point variant",
)

The synthetic results matter only if they predict something checkable about
the rig, so @tab-real-predictions fixes those predictions before the real data
are read, each with the observation that would refute it. The rig's $C$,
$alpha$, SNR, drift, motion prior and training budget differ from the
simulator's, so absolute errors should not transfer; the predictions are
ratios and orderings against the real-data results (@sec-learned-table).
Those are two-seed means, so P6 scores every falsifier against $s$, the seed // math-ok: stage label
range it measures on the real runs.

#figure(
  align(center, {
    set par(justify: false)
    table(
    columns: (auto, 1.4fr, 1fr, 1.4fr),
    align: (right, left, left, left),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([], [Prediction], [Interval], [Falsified if]),
    [1],
    [The TCN's signed and sign-invariant free-space test RMSE agree],
    [within 10%],
    [sign-invariant is more than 10% below signed],

    [2],
    [Simulator calibration reads feedback on the free-space file],
    [$C >= 0.05$, $alpha >= 0.5$],
    [$C < 0.03$, or $alpha < 0.5$],

    [3],
    [The rig's detector knobs are less degraded than nominal],
    [$C > 0.24$, or SNR above 31 dB, or visibility floor above 0.6],
    [$C <= 0.2$, SNR at most 30 dB and visibility floor at most 0.5],

    [4],
    [A per-laser phase reference, or less drift, cuts the TCN's error],
    [directional only; not scoreable at P6, needs new hardware], // math-ok: stage label
    [drift calibrates below nominal on new hardware, yet the cut is within $s$],

    [5],
    [515 nm is the best single channel],
    [lowest single-channel RMSE],
    [another channel is lower by more than $s$],

    [6],
    [Three channels beat the best single channel],
    [error ratio #val("pred_triple_30") to #val("pred_triple_free")],
    [ratio below $1 - s$, or above #val("pred_triple_free") $(1 + s)$],

    [7],
    [515 nm alone recovers the signed motion],
    [best validation more than #val("q1_stall_tol_pct")% below the zero predictor],
    [within #val("q1_stall_tol_pct")% of the zero predictor],

    [8],
    [TCAN beats the causal TCN],
    [error ratio 1 to #val("q2_tcan_gain")],
    [ratio below $1 - s$],

    [9],
    [Both LSTMs stall under this protocol],
    [best validation within #val("q2_stall_tol_pct")% of the zero predictor],
    [either leaves the plateau],

    [10],
    [No other offline family beats Barland's #val("rl_barland_v_fs") $mu$m/ms],
    [within $s$ of it, or above],
    [one scores more than $s$ below it],
  )}),
  caption: [
    Pre-registered predictions for the free-space acquisition, scored in P6. // math-ok: stage label
    Errors are test velocity RMSE under the real-data protocol of @sec-learned;
    ratios are the single channel's or the TCN's error over the comparator's.
    $s$ is the seed range P6 measures on the real runs at two seeds. // math-ok: stage label
  ],
) <tab-real-predictions>

= Q1: multiple channels <sec-q1> // math-ok: question label

== Q1.1: channel count <sec-q1-count> // math-ok: question label

=== Analytic

Channels with independent phases add information, so the velocity bound falls
as $1 \/ sqrt(sum_k J_k)$ and short wavelengths dominate it
(@tab-q1-count). With one unknown phase per channel the Fisher matrix of
$(v, phi_1, ..., phi_K)$ is block-arrowhead, so the velocity information
profiled over the phases is the sum of the per-channel values, each weighted
by $lambda_k^(-2)$ as in @eq-fisher. Each seed draws one set of trajectories
and phases and reuses it for every $K$ and both $C$, so each set equals the
sum of its single channels exactly and the feedback comparison is paired.
Feedback scales every entry by the phase factor of @eq-phase-factor, as
predicted to within the seed scatter (@tab-q1-checks).

to seed.

#assert(val("ch_additive_exact"), message: "the common-random-number channel sets no longer equal the sum of their singles")

#figure(
  table(
    columns: 5,
    align: (center, center, right, right, right),
    table.header([$K$], [added $lambda$ (nm)], [$J$, $C = 0.2$], [$J$, $C = 0$], [bound ratio]),
    ..val("ch_count_table").map(r => (
      [#r.n], [#r.lam], [#r.info02], [#r.info0], [#r.bound02],
    )).flatten(),
  ),
  caption: [
    Velocity information adds over channels. $J$: profiled per-sample
    velocity information, in $10^7 (mu"m/ms")^(-2)$, of the first $K$ colors of
    the list, mean over seeds. Bound ratio: $sqrt(J_1 \/ J_N)$ at $C = 0.2$, the CRB on velocity
    relative to 635 nm alone. Each row slices one shot draw per seed to its
    first $K$ channels; the R1 grid instead scores each $K$ on its own frozen
    test set, from a shared seed.
  ],
) <tab-q1-count>

#figure(
  table(
    columns: 2,
    align: (left, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([check], [value]),
    [information gain, 785 nm as fourth channel], [#val("ch_info_jump_785_pct")%],
    [information gain, 405 nm as sixth channel], [#val("ch_info_jump_405_pct")%],
    [feedback rise at $C = 0.2$, predicted by $Phi(0.2, 3) \/ Phi(0, 3)$], [#val("ch_fb_pred_pct")%],
    [feedback rise, measured (paired, mean $plus.minus$ SE)], [#val("ch_fb_meas_pct")% $plus.minus$ #val("ch_fb_se_pct")%],
    [single-entry scatter seed to seed], [#val("ch_fb_spread_pct")%],
  ),
  caption: [
    The paired feedback ratio matches the phase-factor prediction. Profiled
    information of @tab-q1-count, noise-free detector at 30 dB SNR,
    $alpha = 3$, #val("ch_fisher_shots") shots, #val("ch_count_seeds") seeds.
  ],
) <tab-q1-checks>

=== Synthetic

#assert(val("r1_tcan_nf_tracks"), message: "noise-free TCAN no longer follows the CRB ratio within 25% at N >= 3")
#assert(val("r1_tcan_nf_n2_beats"), message: "noise-free TCAN's N = 2 ratio no longer beats the CRB ratio by 15%")
#assert(val("r1_tcan_nom_monotone"), message: "nominal TCAN no longer falls at every added channel")
#assert(val("r1_unet_nf_spread_big"), message: "UNetTransformer's noise-free N = 3 seed range is no longer above 30%")
#assert(val("r1_unet_nom_flat"), message: "UNetTransformer's nominal error now varies by more than 25% over N")

Noise-free, TCAN's error falls with channel count about as the bound does; at
the nominal point it keeps falling up to six colors; the offline
UNetTransformer shows no gain that its seeds resolve (@fig-q1-count,
@tab-q1-r1). The R1 grid trains the best causal and the best offline family // math-ok: grid label
of @sec-q2-floor on the first $K$ colors of @tab-q1-count, with feedback,
noise-free and at the nominal point. Each cell has one seed;
the one- and three-color cells have three. All test sets share one motion draw.

Noise-free, TCAN's six-color error ratio tracks the bound's from three colors on,
but single steps do not (@tab-q1-r1). The first two steps depart in opposite
directions, the second color gaining more than the bound allows and the third less. A
ratio of bounds does not cap a ratio of errors, so a single-channel model far
above its own bound would give this pattern; these runs cannot say whether it
is one.

At the nominal point TCAN's error falls with every added color but stays above
the bound ratio. The bound leaves out phase drift, the knob that dominates the
nominal error (@sec-q2-noise), which would produce such a gap; nothing here
tests it.

UNetTransformer's curve cannot be ranked: noise-free its seed range exceeds
any step, and at the nominal point every $K$ is level within that range. The
offline model learns at the nominal point, but extra colors buy it nothing.

Which pair of colors is used does not matter at this resolution (open markers in
@fig-q1-count). The pairs containing the shortest wavelength come out lowest, as the
$lambda^(-2)$ weighting favours, but within the shot standard error, estimated from
the recorded quartiles.

#fig(
  "q1_count",
  caption: [
    Noise-free, TCAN's error falls about as the velocity bound does as colors
    are added; at the nominal point it falls more slowly, and UNetTransformer
    does not resolve a gain at either point. Test velocity RMSE against the
    number $K$ of colors, $C = 0.2$, $alpha = 3$: (a) noise-free, (b) the
    nominal point. Lines: seed means; dots: seeds 0 to 2 at $K = 1, 3$;
    open markers: the other two-color pairs; stars: 635 nm arrays with
    quadrature offsets (@sec-q1-colors). Dotted: the bound ratio of
    @tab-q1-count, scaled to TCAN at $K = 1$.
  ],
) <fig-q1-count>

#figure(
  table(
    columns: 9,
    align: (center, center, right, right, right, right, right, right, right),
    table.header(
      [$K$], [$lambda$ (nm)], [bound], [step (%)],
      [TCAN nf], [step (%)], [TCAN nom], [step (%)], [UNetT nf],
    ),
    ..val("r1_table").map(r => (
      [#r.n], [#r.lam], [#r.bound], [#r.bstep], [#r.tcan_nf], [#r.tcan_nf_step],
      [#r.tcan_nom], [#r.tcan_nom_step], [#r.unet_nf],
    )).flatten(),
  ),
  caption: [
    From $K = 3$ on, TCAN's noise-free error ratio stays within 25% of the
    bound ratio, though single steps do not. Test velocity RMSE ($mu$m/ms), seed
    mean where seeds exist; nf: noise-free; nom: the nominal point; step:
    change from $K - 1$. Bound: the CRB ratio of @tab-q1-count, with its
    step.
  ],
) <tab-q1-r1>

The subset grid below came before R1 and compares against the best single // math-ok: grid label
channel rather than 635 nm.

#assert(val("q1_multi_beats_single"), message: "a multi-channel set no longer beats the best single channel at C = 0.2 by more than the seed range")
#assert(val("q1_515_best"), message: "515 nm is no longer the best single channel at C = 0.2 for both families")
#assert(val("q1_runner_up_tcn_at_edge"), message: "the TCN's 515 nm lead is no longer at the edge of the seed range")
#assert(val("q1_mich_single_at_zero"), message: "a Michelson single channel no longer scores at the zero predictor")
#assert(val("q1_mich_single_converged"), message: "the stall test now marks a Michelson single channel as not trained")
#assert(val("q1_mich_multi_far"), message: "a Michelson multi-channel set now comes near the zero predictor")
#assert(val("q1_oracle_single_above_zero"), message: "the single-channel Michelson oracle no longer scores above the zero predictor")
#assert(val("q1_bin_nf_flat"), message: "noise-free, an error curve now rises toward the 515 nm extremum or departs from flat")
#assert(val("q1_bin_nom_small_rise"), message: "at 30 dB the one- or three-channel error now departs from flat in extremum distance")
#assert(val("q1_bin_gain_not_near"), message: "the noise-free multi-channel gain is now larger near the 515 nm extremum than midway")
#assert(val("q1_bin_v_shape"), message: "the noise-free multi-channel gain by |v| no longer vanishes in the slowest bin and exceeds 1.5 higher up")
#assert(val("q1_bin_v_slow_above_zero"), message: "in the slowest |v| bin a noise-free model now scores at or below the zero predictor")
#assert(val("q1_s4_30_outside"), message: "at 30 dB the multi-channel gain is no longer several times the nominal seed range, or 635 + 674.8 nm now beats 515 nm alone")
#assert(val("q1_s4_oracle_worse"), message: "the nominal-point oracle no longer scores worse than the zero predictor")

Noise-free, a second wavelength cuts the error well below the best single
channel, and on Michelson one channel alone cannot estimate velocity at all;
at the nominal point the gain nearly vanishes (@fig-q1-subsets). @eq-fisher
predicts that a channel goes blind at its fringe extrema, and incommensurate
wavelengths put different channels' extrema at different times, so extra
channels should help most near one channel's extrema and at velocity turning
points. To test this we train the TCN and TCAN, which fuses per-channel TCN
encodings by attention across channels at the same parameter count, on every
subset of the three channels, noise-free and, for the TCN, at the nominal
point across SNR.

#fig(
  "q1_subsets",
  caption: [
    Noise-free a second wavelength cuts the error well below the best single
    channel, 515 nm, and on Michelson one channel recovers nothing; at the
    nominal point the gain nearly vanishes. (a) Noise-free test velocity RMSE of each
    channel subset, one seed per cell, Michelson scored sign-invariant and
    $C = 0.2$ signed, $alpha = 3$; oracle: the classical
    inverse fit, given the true excursion. (b) The TCN at the nominal point
    with SNR varied: the lowest error among single channels, among pairs, and
    all three.
  ],
) <fig-q1-subsets>

The single-channel Michelson failure is unexplained. The global mirror of
@sec-sign cannot cause it, because the sign-invariant score forgives it. One
untested explanation is a local mirror: near $t_0$ the fringe is
$
  cos(psi_0 + (4 pi [d(t) - d(t_0)]) / lambda), quad
  psi_0 = (4 pi d(t_0)) / lambda + phi ,
$ <eq-local-mirror>
unchanged when the local offset $psi_0$ and the displacement increment are
both negated, so a windowed estimator cannot fix the sign locally. The
whole-shot oracle also fails, which no window argument explains.

#fig(
  "q1_binned",
  caption: [
    The multi-channel gain does not concentrate at the fringe extrema but
    grows with speed, vanishing in the slowest $|v|$ decile (c). Pooled test RMSE of the TCN ($C = 0.2$, $alpha = 3$, one seed)
    in #val("q1_bin_width")-fringe bins of the distance from $phi_F$ to the
    nearest extremum, of the 515 nm channel (solid) and of the channel
    farthest from one (dotted): (a) noise-free, (b) the nominal point at
    30 dB. (c) The same in $|v|$ deciles.
  ],
) <fig-q1-binned>

The extremum prediction fails (@fig-q1-binned): the gain does not concentrate
at the extrema, and it grows with speed, against the turning-point clause.
One untested reading is that an estimator seeing a window interpolates across
an extremum, as the prior does in the bounds of @sec-crb.

Against the best single channel, the TCN's gain at the nominal point is small
and comes from pairing it with another channel (#ref(<fig-q1-subsets>)b). The larger
nominal gain of @fig-q1-count is measured against the red channel, the weakest of the
three. The classical oracle decodes under the Michelson model and scores worse
than zero at the nominal point, so it is no reference there. These runs are
one seed each at the shorter budget, and the nominal point ran for the TCN
only.

=== Real data

The real three-channel gain over the best single channel should fall between
the two synthetic limits, noise-free and nominal (prediction 6 of
@tab-real-predictions). The nominal end is within the seed range of parity,
and on another rig @matha2023multichannel found the best single channel ahead
of all three, so a ratio below 1 is a live outcome. R1 adds a prediction // math-ok: grid label
against 635 nm alone: TCAN's three-channel ratio should lie between its
noise-free and nominal values in @tab-q1-r1, and it is falsified if the real
data resolve no three-channel gain over 635 nm.

Predictions 5 and 7 rest on thinner evidence. The shortest wavelength leads in
the $lambda_k^(-2)$ order and in every synthetic cell, but its noise-free TCN
lead is at the edge of the seed range. A real green channel at the zero predictor
would mean feedback too weak for one channel to read the sign.

== Q1.2: several colors against a one-color array <sec-q1-colors> // math-ok: question label

=== Analytic

A one-color array with quadrature offsets reads the sign without feedback;
distinct colors do not, and local precision barely separates the two
(@tab-q1-colors). We compare $K$ colors with independent phases against $K$
copies of one color sharing one unknown phase, with known offsets spread in
quadrature or all zero. Local information follows $sum_k lambda_k^(-2)$. The
sign evidence of @eq-sign-info separates them: without feedback it is zero for
colors and for zero offsets, as @sec-sign requires, while the quadrature
array carries more than weak feedback gives the colors. This does not
contradict the $C^2$ leading order, which holds for channels that each fit
their own mirror phase. With one shared phase and distinct known offsets, no
single mirror phase fits every channel even without feedback; this is quadrature
detection. The evidence is the fixed-trajectory upper bound, and assumes the
offsets known exactly.

#figure(
  table(
    columns: 5,
    align: (center, center, left, right, right),
    table.header([$C$], [$K$], [configuration], [$J$], [sign (nats/shot)]),
    ..val("ch_colors_table").map(r => (
      [#r.c], [#r.n], [#r.arm.replace("_", " ")], [#r.info], [#r.sign],
    )).flatten(),
  ),
  caption: [
    Quadrature offsets, not colors, carry the sign at $C = 0$. Colors: the
    first $K$ of 635, 674.8, 515 nm with independent phases. Array: $K$ times
    635 nm sharing one phase, offsets quadrature-spread or zero. $J$ as in
    @tab-q1-count; same detector, 30 dB, $alpha = 3$. Each arm has its own
    shots from a shared seed, and in the R2 grid its own frozen test set: the
    arms differ in channel set and offsets, so one set cannot serve them.
  ],
) <tab-q1-colors>

=== Synthetic

#assert(val("r2_nf_array_wins"), message: "noise-free, the quadrature array no longer beats distinct colors by 5 times")
#assert(val("r2_nom_parity"), message: "at the nominal point the array and the colors now differ by more than two combined shot SEs")
#assert(val("r2_zero_shared"), message: "the R2 arms no longer share one motion draw")

Noise-free, the one-color quadrature array beats distinct colors several
times over, while at the nominal point the two are level (@tab-q1-r2, stars
in @fig-q1-count). The R2 grids train the R1 families on $K$ copies of one color // math-ok: grid label
sharing one mirror phase with quadrature offsets; the color arm is R1's cells // math-ok: grid label
at the same $K$. Each arm has its own frozen test set, because channel set
and offsets differ, but all share the motion draw.

The noise-free gap is larger than local precision allows, so local precision
is the wrong predictor here. Whether the sign evidence or the shared phase,
one unknown where the colors have $K$, carries the gain is untested. At the
nominal point the arms differ by less than two combined shot standard errors
in all four comparisons, so the nominal knobs erase the array's advantage;
which knob does so is untested.

#figure(
  table(
    columns: 8,
    align: (center, left, right, right, right, right, right, right),
    table.header(
      [$K$], [sim], [TCAN array], [TCAN colors], [ratio], [UNetT array],
      [UNetT colors], [ratio],
    ),
    ..val("r2_table").map(r => (
      [#r.n], [#r.sim], [#r.tcan_arr], [#r.tcan_col], [#r.tcan_ratio],
      [#r.unet_arr], [#r.unet_col], [#r.unet_ratio],
    )).flatten(),
  ),
  caption: [
    Noise-free the quadrature array is an order of magnitude ahead; at the
    nominal point it is level with the colors. Test velocity RMSE ($mu$m/ms),
    seed 0, $C = 0.2$, $alpha = 3$; ratio: colors over array. Each arm has
    its own test set, with the same motion draw in all of them.
  ],
) <tab-q1-r2>

=== Real data

_Status: not run; needs a same-color array on the rig_. The committed
prediction: at matched channel count and visibility, a one-color array with
offsets near quadrature recovers the sign at the rig's weakest feedback, where
the three colors do not, with local precision close to the colors'
(@tab-q1-colors). It is falsified if the array's sign error there is no lower
than the colors', or if collapsing its offsets to zero leaves it unchanged. In
simulation the advantage appears only noise-free (@tab-q1-r2), so near the
nominal point the sign test is the only informative one.

= Q2: which estimator <sec-q2> // math-ok: question label

== Q2.1: accuracy floor on noise-free data <sec-q2-floor> // math-ok: question label

=== Analytic

Noise-free, the floor is zero: as $sigma -> 0$ the information of
@eq-info-avg diverges and both bounds of @sec-crb vanish, so any error on the
noise-free cells of @fig-families is a model limit. At the nominal noise the
best networks sit well above both bounds (@tab-q2-floor), and since the
noise-free bound is lower still, those factors understate the matched-noise
gap.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([], [$C = 0.2$], [Michelson]),
    [causal bound ($mu$m/ms)], [#val("an_c02_causal_bound")], [#val("an_mich_causal_bound")],
    [offline bound ($mu$m/ms)], [#val("an_c02_offline_bound")], [#val("an_mich_offline_bound")],
    [best causal network / causal bound], [#val("an_c02_causal_factor")], [#val("an_mich_causal_factor")],
    [best offline network / offline bound], [#val("an_c02_offline_factor")], [#val("an_mich_offline_factor")],
    [SNR at which the causal bound equals the best causal network (dB)], [#val("an_c02_causal_snr")], [#val("an_mich_causal_snr")],
  ),
  caption: [
    The best noise-free networks sit far above the bounds at the nominal
    #val("an_snr_nominal") dB. Three channels, $V tilde U(0.5, 1)$,
    $alpha = 3$; Michelson bounds are local and compare with the
    sign-invariant score. The best causal network at $C = 0.2$ scores
    $#val("an_c02_causal_net")$ $mu$m/ms.
  ],
) <tab-q2-floor>

Prediction: every trained family sits above the bound at matched noise. A
factor within the margin by which an extended Kalman filter given the true
phases meets the causal bound would put an estimator at the information
limit; a factor well above it marks a model limit. A network below the bounds
would point at what they leave out, such as the drive-line structure.

=== Synthetic

#assert(val("q2_mich_tied"), message: "the Michelson families other than the offline neural operator no longer lie within the seed range")
#assert(val("q2_oracle_beats_all"), message: "a learned family now comes within a factor 1.5 of the Michelson oracle")
#assert(val("q2_tcan_best"), message: "TCAN is no longer the best family at C = 0.2 by more than the seed range")
#assert(val("q2_learned_beat_oracle_smi"), message: "a trained family no longer beats the oracle at C = 0.2")
#assert(val("q2_stalled_named"), message: "the stalled cells are no longer the two LSTMs and the offline neural operator at C = 0.2")
#assert(val("q2_sliding_named"), message: "the sliding-window families are no longer Barland and the MLP")
#assert(val("q2_matched_named"), message: "the fixed-window families are no longer the eight named in the context paragraph")
#assert(val("q2_barland_ties_scnn"), message: "Barland and SCNN no longer tie at C = 0.2")
#assert(val("q2_oracle_worse_than_zero"), message: "the oracle's signed C = 0.2 cell no longer scores worse than the zero predictor")
#assert(val("s1d_smi_stalled"), message: "a displacement-target cell at C = 0.2 now leaves the zero-output plateau")
#assert(val("s1d_mich_trained"), message: "a displacement-target cell on Michelson no longer trains")
#assert(val("s1d_shape"), message: "on Michelson, retargeting no longer helps the LSTMs and costs the neural operator and TCN, or a retargeted cell now reaches the velocity-trained TCN")
#assert(val("s1d_mamba_trained"), message: "Mamba no longer trains on velocity at C = 0.2")

Without feedback no family that trains separates from the TCN, and all sit far
above the classical oracle; with feedback TCAN is best on velocity, while the
LSTMs and the offline neural operator do not train on either target
(@fig-families). Each family is matched to the TCN's parameter count and
trained as the feedback curve was, on noise-free three-channel data: signed
loss with feedback, sign-invariant loss on Michelson. Causal
families see only the past, offline ones both sides. Context is matched for the convolutional,
transformer and causal neural-operator families; the recurrent families (LSTM
and the state-space model Mamba) see unbounded context, the offline neural
operator the whole shot, and the MLP and Barland CNN average overlapping
windows.

#fig(
  "model_families",
  width: 70%,
  caption: [
    Without feedback (Michelson, hollow) every family that trains, except
    the offline neural operator, lies within the #val("q2_seed_range_pct")%
    seed range of the TCN; with feedback TCAN is best on velocity, and both LSTMs and
    the offline neural operator stall, even when retrained on displacement
    (b). (a) Noise-free test velocity RMSE of each family at 44k parameters,
    one seed per cell; stalled: best validation error within
    #val("q2_stall_tol_pct")% of the zero predictor's. The oracle, a classical
    inverse fit given the true excursion, decodes under the Michelson model,
    so its signed $C = 0.2$ cell scores worse than the zero predictor. (b)
    Test displacement RMSE, sign-invariant on Michelson, of each family
    trained on velocity (the cells of (a)) and on displacement.
  ],
) <fig-families>

At default initialization the LSTM's sensitivity to past input decays within
a few samples, while velocity must be read over a fringe period of hundreds;
chrono gate initialization and a larger learning rate did not move it. The
offline neural operator mixes a few global Fourier modes and has no local path
at the fringe frequency.

Retargeting the stalled families to displacement does not make them train
usefully (#ref(<fig-families>)b). With feedback all retargeted cells stall,
and without it the LSTMs train but stay far behind the velocity-trained TCN.
The TCN control cannot represent displacement, which depends on the whole
history, so only the LSTMs and the offline neural operator are informative.
Mamba already trains on velocity and was not retargeted.

#assert(val("disp_lowest_named"), message: "the three lowest families on displacement at C = 0.2 are no longer the two offline transformers and TCAN, tied with each other and below the TCN")

Displacement RMSE ranks the families differently from velocity RMSE
(@fig-displacement). Displacement is what the instrument reports, and it
weights slow error more (@fig-spectral). We rescore the checkpoints and test
shots of @fig-families without retraining; the seed range on displacement is
wider than on velocity, which decides which families are resolved. Why
integration reorders the families is untested.

#fig(
  "displacement_rmse",
  width: 80%,
  caption: [
    Displacement does not keep the families' velocity ranking. With
    feedback, TCAN, best on velocity, ties the offline PatchTransformer and
    UNet-Transformer, all below the TCN. (a) The TCN
    against $C$, as in #ref(<fig-feedback>)a. (b) Displacement against velocity RMSE for every trained
    family of @fig-families, one seed each: hollow on Michelson,
    sign-invariant; filled at $C = 0.2$, signed.
  ],
) <fig-displacement>

Each cell is one seed at one budget, and budgets differ: some cells stopped
before the first restart and some overran into a second cycle. The one gain
well outside the seed range is TCAN over the TCN with feedback; more seeds of
both would test it.


Ridge regression and LightGBM, the tabular estimators of the real-data
comparison, do not approach the networks on any simulation
(@tab-tabular-synthetic). Ridge scores as the zero predictor everywhere;
LightGBM improves with $C$ but stays several times behind the best network.
Both are offline, see three channels through windows labelled at their
centre, and fit training shots seed-disjoint from the networks' frozen test
sets.

// Tabular cells, read straight from the runner's records. The document rule
// does not list synthetic_results.json, but synthetic_values.json depends on
// it, so a changed artifact still rebuilds this table.
// The best network on the same frozen test set, scored the same way.
// (sim name, label, scored sign-invariant): Michelson cannot be scored
// signed, because the sign is unobservable there (@sec-sign).
#figure(
  {
    let syn-records = json("/results/synthetic_results.json").at("datasets").values().flatten()
    let tab-cell(prefix, sim) = syn-records.find(r => (
      r.at("sim_name", default: none) == sim and r.at("cell_key").starts-with(prefix + "__")
    ))
    let tab-metric(r, si) = r.at("metrics").at(
      if si { "velocity_nrmse_sign_invariant" } else { "velocity_nrmse" },
    ).at("mean")
    let tab-best-net(sim, si) = {
      let ds = tab-cell("ridge-w91", sim).at("dataset")
      let nets = syn-records.filter(r => (
        r.at("dataset", default: none) == ds
          and not r.at("cell_key").starts-with("ridge-")
          and not r.at("cell_key").starts-with("lightgbm-")
      ))
      calc.min(..nets.map(r => tab-metric(r, si)))
    }
    let tab-sims = (
      ("michelson", [Michelson, $C = 0$], true),
      ("smi-a3-feedback_c=0.05", [$C = 0.05$], false),
      ("smi-a3-feedback_c=0.1", [$C = 0.1$], false),
      ("smi-a3", [$C = 0.2$], false),
      ("smi-a3-feedback_c=0.4", [$C = 0.4$], false),
      ("smi-a3-feedback_c=0.7", [$C = 0.7$], false),
      ("smi-a3-feedback_c=0.9", [$C = 0.9$], false),
      ("smi-a5", [$C = 0.2$, $alpha = 5$], false),
    )
    let tab-rows = tab-sims.map(((sim, label, si)) => (
      label: label,
      si: si,
      ridge: tab-metric(tab-cell("ridge-w91", sim), si),
      lgbm: tab-metric(tab-cell("lightgbm-w91", sim), si),
      net: tab-best-net(sim, si),
    ))
    let tab-field = tab-cell("ridge-w91", "smi-a3").at("receptive_field")
    let tab-feedback = tab-rows.slice(1, 7)
    let tab-ratios = tab-feedback.slice(2).map(r => r.lgbm / r.net)
    let f2(x) = {
      let (whole, ..frac) = str(calc.round(x, digits: 2)).split(".")
      whole + "." + (frac.at(0, default: "") + "00").slice(0, 2)
    }
    assert(tab-rows.all(r => calc.abs(r.ridge - 1) < 0.01), message: "ridge now scores away from NRMSE 1 on some sim")
    assert(
      tab-feedback.windows(2).all(((a, b)) => b.lgbm < a.lgbm + 0.01),
      message: "LightGBM's NRMSE no longer falls with C",
    )
    assert(tab-rows.all(r => r.lgbm > r.net), message: "LightGBM now matches the best network on some sim")
    set text(size: 9pt)
    table(
      columns: 5,
      align: (left, right, right, right, right),
      stroke: 0.5pt + rgb("#d0d0d0"),
      table.header([sim], [ridge], [LightGBM], [best network], [LightGBM / best]),
      ..tab-rows
        .map(r => (
          r.label + if r.si { [#super[SI]] },
          f2(r.ridge),
          f2(r.lgbm),
          f2(r.net),
          f2(r.lgbm / r.net),
        ))
        .flatten(),
    )
    v(2pt)
    [Windows of 91 samples labelled at their centre, a #tab-field\-sample
      two-sided span; #tab-cell("ridge-w91", "smi-a3").at("config").at("n_train_shots")
      training shots.]
  },
  caption: [
    Ridge scores as the zero predictor and LightGBM trails the best network
    on every sim. Noise-free test velocity NRMSE, three channels,
    $alpha = 3$ unless stated, one seed; best network: the lowest of every
    trained family on the same frozen test set. #super[SI]: sign-invariant
    score. On Michelson the tabular models fit the signed target while the
    networks trained with the sign-invariant loss, so a tabular fit there
    tends to zero (@eq-sign-zero) and its score measures the loss, not the
    model.
  ],
) <tab-tabular-synthetic>

// The Q2 figures are tall; without a flush they float past this heading.
#place.flush()

==== Context length, not family, limits these regressors <sec-q2-context>

Every family above is compared at matched context, under a hundred samples.
That choice hides the larger lever. At the nominal point a regressor with
TCAN's parameter count but a whole-shot field halves TCAN's error
(@tab-q3-flow), a gap wider than any between families here. The real data
point the same way: widening the receptive field bought what extra
parameters did not (@sec-learned-field). The comparison is not yet fair,
because that regressor also trained on full shots rather than crops; a
TCAN or UNetTransformer trained with a whole-shot field is the test.

=== Real data

_Status: measured on both acquisitions in @sec-q2-real; the predictions
below are scored at unblinding_. The Q2 predictions claim orderings only, // math-ok: question label
because synthetic family ranks transfer imperfectly: on real data Barland
beats SCNN (@sec-learned-table), where the synthetic cells tie them
(@fig-families). TCAN's gain over the TCN was measured noise-free, and noise
shrank the Q1 gain, so prediction 8 runs from parity to the noise-free factor. // math-ok: question label
Prediction 9 applies only to the same protocol of default initialization,
signed loss and patience. Mamba, whose feedback cell trains, ran after the
predictions were fixed and has none.

Prediction 10 covers SCNN, the MLP, both transformers, the offline neural
operator and the bidirectional LSTM. It rests on the real ranking, where
Barland leads every configuration on both acquisitions (@sec-learned-table),
not on the synthetic ties; for the four families that ranking omits, it is an
extrapolation.

=== Q2.1.1: does the self-mixing nonlinearity help? <sec-q2-nonlinearity> // math-ok: question label

==== Analytic

Whether the sign can be read at all is @sec-sign and #ref(<sec-feedback>);
how much the error should fall with $C$ follows from @sec-crb. Feedback
enters the per-sample information only through $Phi(C, alpha)$, and
expanding @eq-phase-factor, $Phi - 1 \/ 2 = (C^2 \/ 4)(1 - cos(2 beta) \/ 2)
+ O(C^4)$, quadratic in $C$. The causal bound falls only as
about the fourth root of the information ratio (@tab-q2-c-bounds,
@fig-bounds). At $alpha = 0$ the information still rises, but the sign
evidence is zero (@sec-feedback), so a signed estimator gains nothing.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([], [$alpha = 3$], [$alpha = 0$]),
    [information over Michelson, $2 Phi$, at $C = #val("an_c_lo")$], [#val("an_info_c_lo")], [--],
    [information over Michelson at $C = #val("q3_c_top")$], [#val("an_info_c_hi")], [#val("an_info_c_hi_a0")],
    [causal bound fall to $C = 0.2$], [#val("an_bound_gain_c02")], [--],
    [causal bound fall to $C = #val("q3_c_top")$], [#val("an_bound_gain_c_hi")], [#val("an_bound_gain_c_hi_a0")],
    [measured TCN fall to $C = #val("q3_c_top")$], [#val("q3_gain_top")], [--],
  ),
  caption: [
    Information explains only a small part of the network's fall with
    feedback. Factors relative to Michelson, 30 dB causal bound, three
    channels.
  ],
) <tab-q2-c-bounds>

Prediction: if information per sample set the error, the network's $C$ curve
would barely move at weak feedback and then fall by at most the bound's factor.
The measured curve falls far more (@tab-q2-c-bounds), so most of its fall is
the network closing its distance to the bound (@tab-q2-floor), not a gain in
information.

==== Synthetic

#assert(val("q3_monotone"), message: "the 90-epoch C curve is no longer monotone")
#assert(val("q3_t30_c04_below"), message: "the 30-epoch C = 0.4 inversion is gone")

With feedback a signed estimator reads the direction of motion, and its error
falls steadily with $C$, far below Michelson's even when Michelson is forgiven
the sign (@fig-feedback). @sec-feedback predicts the direction can be read,
but not how fast the error falls or whether a strongly tilted fringe is harder
to decode. So we train the same causal TCN with the signed loss on noise-free
data at six feedback strengths, and compare each with Michelson scored
sign-invariantly, the best a Michelson estimator can do.

#fig(
  "feedback_curve",
  caption: [
    With feedback ($C > 0$, $alpha != 0$) a signed-loss network reads the
    direction of motion, and its error falls steadily with $C$, far below
    Michelson's sign-invariant RMSE. (a) Test velocity RMSE in $mu$m/ms and (b) NRMSE,
    each shot's RMSE over its velocity std so the zero predictor scores 1,
    averaged over shots, of the causal TCN on three noise-free channels at
    $alpha = 3$, one seed; the 30-epoch curve dips at $C = 0.4$ because that
    run trained #val("q3_t30_c04_epochs") epochs, past its first restart. (c)
    Sign-invariant RMSE pooled in $|v|$ deciles for the 90-epoch Michelson,
    $C = 0.2$ and $C = 0.9$ cells.
  ],
) <fig-feedback>

RMSE and NRMSE disagree at weak feedback because they weight different shots.
RMSE pools samples, which fast motion dominates, and Michelson's error grows
with $|v|$ (#ref(<fig-feedback>)c). NRMSE lets slow shots dominate, where the
estimator's own floor sets the score. Signed and sign-invariant RMSE coincide
(#ref(<fig-feedback>)a), so the sign is resolved on nearly every shot.

The curve holds for a fixed budget, and a longer cycle reached a lower error, so convergence is not demonstrated. Within that caveat the
operating target is the strongest weak feedback sampled. On the rig, $C$ is
set by how much light returns into the laser: target reflectivity or
attenuation in the return path, focusing and coupling, and the external
cavity length. Which of these is practical to change is untested.

==== Real data

The published runs already show that the rig breaks the global sign symmetry
of @sec-sign. They trained with a plain signed MSE loss, which on Michelson
data leaves the TCN at the zero predictor (@fig-exit-check). The free-space
TCN instead scores well below the zero predictor's mean per-shot error
(@tab-real-predictions, @sec-learned-table).
Prediction 1 holds the sign resolved on nearly every test shot, as in every
long-budget cell of @sec-q2-nonlinearity, with its interval widened for
noise.

Feedback is one way to break the symmetry. Shot-to-shot repeatable phase
offsets, a position-dependent amplitude or drive crosstalk would also break
it, and the simulator models none of them. Prediction 2 therefore requires
the calibration to read the feedback the sign needs; crosstalk and
photodiode nonlinearity also feed the second fringe harmonic it reads, so it
can pass for the wrong reason. Its threshold is the weakest feedback sampled
in @fig-feedback. The calibration returns nearest-grid estimates with no
uncertainty and under-reads $C$ at strong drift, so its falsifier sits below
that threshold, and a reading in between neither confirms nor falsifies.

== Q2.2: robustness to noise <sec-q2-noise> // math-ok: question label

=== Analytic

The bounds of @sec-crb capture two of the six knobs. White noise enters
@eq-info-avg as $1 \/ sigma^2$, and visibility enters through $E[V^2]$; both
rescale the information. Amplitude drift, contrast drift and DC offset are
absent from the bounds.

Phase drift cannot be added as a change of parameter, because it is a
nuisance process. Channel $k$ sees the phase $4 pi d \/ lambda_k + phi_k (t)$,
so each sample gives $K$ phases for $K + 1$ unknowns. Only the priors separate
$d$ from the drifts: the random walk of each $phi_k$ against the smooth drive
prior on $d$. Extra channels average the drift down but do not remove it. A
bound with drift would need the $phi_k$ as extra random-walk states in
@eq-prior, which we have not built, so we give none.

We therefore predict phase drift to cost the most, because it alone adds an
unknown confounded with $d$. White noise should come second, visibility third,
and the three offset-like knobs last; that last placement is argued, not
computed.

=== Synthetic

#assert(val("noise_ranking"), message: "the noise ladder no longer ranks phase drift, then SNR, then the rest")
#assert(val("loo_e60_outside"), message: "at equal length, removing phase drift no longer cuts the nominal error by more than the nominal seed spread")
#assert(val("r4_phase_first"), message: "phase drift is no longer the costliest single knob for all three R4 networks")
#assert(val("r4_vis_over_snr"), message: "the families ranking visibility above white noise are no longer TCAN, PatchTransformer and LightGBM")

At the nominal operating point, per-laser phase drift dominates the error:
removing it at equal training length cuts the error
more than any electronics knob (@fig-loo). Two designs ask which knob costs
the most, both with the TCN at the nominal feedback. The ladder switches
one knob on at a time from the noise-free base (@fig-noise). The
leave-one-out switches one knob off at a time from the nominal point
(@fig-loo); it is the design a hardware fix asks about, because the knobs
need not add.

#fig(
  "noise_ladder",
  caption: [
    Alone, phase drift costs the most, then white noise. Test velocity RMSE of
    the TCN with one knob on and the rest noise-free, $C = 0.2$, one seed per
    point; each x axis runs toward the worse signal.
  ],
) <fig-noise>

#fig(
  "leave_one_out",
  width: 62%,
  caption: [
    At the nominal point, removing phase drift cuts the error most, at either
    training length; removing an electronics knob barely moves it. Test
    velocity RMSE of the TCN, $C = 0.2$, with the named knob returned to
    noise-free and every other knob at nominal, one seed each.
  ],
) <fig-loo>

Three more families agree that phase drift is the costliest knob, but
visibility costs more than white noise for TCAN and PatchTransformer, against
the prediction (@tab-q2-r4). This grid switches each knob on alone, for the
three best noise-free families of @sec-q2-floor plus LightGBM, one seed each,
with one stressed rung for drift and white noise. UNetTransformer keeps the
predicted order, but its visibility and DC-offset rows trained longer, so we
do not weigh it. That the networks invert the bound's order means the bounds
misstate what a trained network pays for white noise or for visibility; the
data do not say which. LightGBM learns little on any row, so its ranking
carries no information about the knobs.

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    table.header([knob at nominal], [TCN], [TCAN], [UNetT], [PatchT], [LightGBM]),
    ..val("r4_table").map(r => (
      [#r.knob], [#r.tcn], [#r.at("tcan-44k")],
      [#r.at("unettransformer-offline-44k")],
      [#r.at("patchtransformer-offline-44k")], [#r.at("lightgbm-w91")],
    )).flatten(),
  ),
  caption: [
    Phase drift costs the most for every neural network; visibility outranks
    white noise for TCAN and PatchTransformer. Test velocity RMSE ($mu$m/ms)
    with one knob at its nominal value and the rest noise-free, then the two
    stressed rungs and the nominal point, $C = 0.2$, $alpha = 3$, one seed.
    TCN: the noise ladder's nominal-value cell (@fig-noise).
  ],
) <tab-q2-r4>

The lever these runs support is per-laser frequency stability or a per-laser
phase monitor, not common-path stabilization. The simulation draws drift
independently per laser; a drift shared by all channels would read as
displacement and is not simulated. The verdict rests on assumed nominal
values: the drift nominal, taken from the phase wander the method-1 fit leaves
on real data, is a lower bound, and the rest are guesses. Measuring the rig's
drifts and SNR would test it. The causal bound of @sec-crb is no constraint
here, since the noise-free base already sits well above it.

=== Real data

_Status: predictions only; no acquisition varies the noise on purpose. The
rig's measured errors are in #ref(<sec-q2-real>)._ Measured against the zero
predictor, the rig sits well below the synthetic nominal point and below the
no-drift cell, near the high-feedback cell. So drift alone cannot explain the
rig, and longer training does not close the gap. Either the nominal knobs or
the synthetic motion prior overstate the difficulty, and the drift-dominance
verdict may not describe the rig.

Prediction 3 tests the knob half. Each knob must clear nominal by the round
trip's acceptance bar; a reading between nominal and its bar neither confirms
nor falsifies, and an unreturned knob drops out. Drift is not a clause,
because its nominal is a lower bound. If prediction 3 fails, the motion prior
or a noise source the simulator lacks carries the gap. The grid of
@tab-q2-r4 widens prediction 4 to every network: on the rig, phase drift
should cost more than any other knob. Visibility against white noise is not
predicted, because the families disagree.

Prediction 4 needs a per-laser phase monitor or a more stable acquisition,
neither of which exists, so it is directional only. Its ceiling is the
equal-length drift cut of @fig-loo. It is falsified if new hardware reads
$sigma_phi$ below the nominal #val("noise_phase_rad") $"rad s"^(-1\/2)$ and the
TCN's error falls by less than $s$.

== Q2.3: generalization across alignment <sec-q2-alignment> // math-ok: question label

=== Analytic

A wavelength shift common to all channels is exactly a velocity rescale. In
@eq-michelson the phase at scaled wavelength $s lambda_k$, $s = 1 + epsilon$,
equals the phase of trajectory $d \/ s$ at the trained wavelengths. A model
right at nominal therefore returns $v \/ s$ and scores
$
  "RMSE"(epsilon) approx sqrt("RMSE"_0^2 + (epsilon \/ (1 + epsilon))^2 v_"rms"^2) .
$ <eq-shift-scale>
Over the motion prior the scale term overtakes the nominal error only at
$|epsilon| approx #val("shift_tcan_eps_cross")$. Feedback $C$ changes the
fringe shape of @eq-excess, not its rate, so we predict only a minimum at the
trained $C$, steepest toward $C -> 0$, where the sign-carrying asymmetry
vanishes.

The inter-channel phase configuration should not matter. Training draws an
independent phase per channel and shot; pinning the channels to one shared
phase with fixed offsets selects one slice of that distribution. The
shared-phase information bound stays within a few percent of the
independent-phase value for every configuration scored below, equal phases
included, because three wavelengths sweep the channels apart within a shot.
The sign evidence varies more but is large everywhere. We predict that an
efficient estimator's error is configuration-independent and that the
configuration average equals the nominal score.

=== Synthetic

Feedback strength is the knob that matters, as predicted; wavelength shifts do
not (@fig-shift). We score the nominal-trained
TCAN and UNet-Transformer, unchanged, on frozen test sets shifted in
one knob, by mean per-shot velocity RMSE. Both errors are lowest near the
trained $C$ and rise more steeply toward weak feedback. Wavelength shifts,
common or per-channel, stay within the shot error of nominal, consistent with
@eq-shift-scale.

The phase configuration moves the learned errors far more than the bound
allows. Averaged over random shared-phase configurations, each network
recovers its nominal score, as predicted. The spread across configurations,
however, is several per-shot standard errors and tracks neither the
configuration's information nor its sign evidence, so the dependence is the
learned estimators' own. Equal phases are the TCAN's worst case, beyond every
random configuration, on both test draws; we have no explanation for it.

#fig(
  "shift_generalization",
  caption: [
    Feedback and the inter-channel phase configuration move the error;
    wavelength does not. (a) Test feedback $C$; dotted, trained $C$.
    (b) Common (filled) and per-channel (open) wavelength shift; dashed,
    @eq-shift-scale. (c) Shared phase, channel $k$ offset by $k$ steps, on two
    test draws; lines, each draw's nominal score; bands, mean $plus.minus$ 1 sd
    over the random configurations. Bars: per-shot SE.
  ],
) <fig-shift>

=== Real data

_Status: blinded; no shifted real acquisition exists_. The committed
prediction: a frozen model degrades with a realignment that changes feedback
strength, by the synthetic factors of @fig-shift, and not with a diode
temperature change, which @eq-shift-scale bounds. It is falsified if a
temperature-only change of $Delta lambda \/ lambda = 10^(-3)$ raises the real
error by more than #val("shift_falsify_temp_pct")%, or if a realignment that
halves $C$ changes it by less than #val("shift_falsify_realign_pct")%.
== Q2 real data: the analytic baseline and learned models <sec-q2-real> // math-ok: question label

This tier scores three things on the two acquisitions of @sec-measurements:
the multi-wavelength decode, the forward-model fit given the true
displacement, and the learned models. No acquisition varies the noise or the
alignment on purpose, so the noise and alignment questions have no real-data
measurement of their own.

#fig(
  "baseline_summary",
  width: 100%,
  caption: [
    The analytic baseline fails on velocity while the Michelson model, given
    the displacement, still explains part of each channel. *Left:* method 1's
    explained variance per channel, N = 1000 shots per acquisition: bar, the
    windowed refit at a fringe-matched $W$; dash, the scrambled-displacement
    control at that $W$; hollow marker, the global single-window fit. Read
    the bar against its dash. *Middle:* method 2's displacement error beside
    the RMS of the true displacement, N = 200. *Right:* that error normalized
    by the RMS; the dashed line is the zero predictor. Bars are medians over
    shots with interquartile whiskers (@sec-read).
  ],
)

=== Method 2: the baseline number <sec-r-method2>

#figure(
  table(
    columns: 8,
    align: (left, right, right, right, right, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header(
      [acquisition],
      [disp. RMSE ($mu$m)],
      [vel. RMSE ($mu$m/ms)],
      [disp. NRMSE],
      [vel. NRMSE],
      [Pearson r (disp.)],
      [true disp. RMS ($mu$m)],
      [true vel. RMS ($mu$m/ms)],
    ),
    [free space], [#val("rd_d_fs")], [#val("rd_v_fs")], [#val("rd_dnrmse_fs")],
    [#val("rd_vnrmse_fs")], [#val("rd_r_fs")], [#val("rs_drms_fs")],
    [#val("rs_vrms_fs")],
    [mm fiber], [#val("rd_d_mmf")], [#val("rd_v_mmf")], [#val("rd_dnrmse_mmf")],
    [#val("rd_vnrmse_mmf")], [#val("rd_r_mmf")], [#val("rs_drms_mmf")],
    [#val("rs_vrms_mmf")],
  ),
  caption: [
    Method 2 on N = 200 shots per acquisition. RMSEs and true RMS are means
    over shots; NRMSE and r are medians, since a few near-still shots inflate
    the mean NRMSE several-fold (@sec-read). The true velocity RMS is the
    zero predictor's error.
  ],
) <tab-r-method2>

The decode is worse than predicting zero on both acquisitions, by mean RMSE
and by median shot (@tab-r-method2). It is not noise: its displacement
correlates with the true motion, so it follows the shape partway before
losing it. Pearson r is scale-free and $R^2$ is not, which is why the two
can disagree about the same decode. A learned model must therefore clear the
zero predictor, not merely the decode.

This is one inverse, so its error is a floor a learned model must clear, not
a bound on classical processing. On noise-free synthetic Michelson fringes
the same decoder recovers displacement about twenty times more accurately, a
check the test suite reruns. That check has no noise or within-shot drift,
so it shows the decoder is correct, not that it is robust; this data cannot
split the real-data error between noise sensitivity and model mismatch. The
overall sign of $d(t)$ is unobservable and is not scored (@sec-method2).

=== Method 1: does the Michelson model describe this data <sec-r-method1>

Given the true displacement, the Michelson model clears its own fitting
capacity on every channel once amplitude and phase are refit per window. The
measure is the gap: windowed $R^2$ minus a control that runs the same fit
against another shot's displacement at the same $W$. A shorter window fits
anything better, so the windowed $R^2$ alone credits the model with capacity
it did not earn; the control measures that capacity.

#figure(
  table(
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
    [free space], [674.8 nm], [0.176], [0.628], [0.328], [*0.30*], [1108],
    [free space], [515 nm], [0.080], [0.503], [0.290], [*0.21*], [846],
    [mm fiber], [635 nm], [0.479], [0.872], [0.335], [*0.54*], [1028],
    [mm fiber], [674.8 nm], [0.166], [0.557], [0.283], [*0.27*], [1092],
    [mm fiber], [515 nm], [0.303], [0.781], [0.314], [*0.47*], [834],
  ),
  caption: [
    Method 1, medians over N = 1000 shots at the live coil-driver gain.
    Global: one three-parameter fit over the whole record. Windowed: the
    same parameters refit per fringe-matched window. Control: the windowed
    fit against another shot's displacement. The gap is the column to read.
    $W$ tracks each shot's fringe rate, so its spread behind the median is
    wide.
  ],
) <tab-r-method1>

The global and windowed columns share the model and the displacement; only
the interval over which amplitude and phase are held constant differs, and
that interval accounts for most of the explained variance. Free-space 515 nm
is the weakest channel on both global $R^2$ and gap, so channels are always
reported separately.

The window spans about two fringes, clipped at both ends by what the control
does. As $W$ shortens the control climbs toward the fit itself, so the lower
clip keeps the gap from becoming mostly capacity. At the upper clip too few
windows remain to absorb within-shot phase drift, and fit and control fall
toward zero together.

=== What the fits look like on real shots <sec-r-shots>

// Shot-trace figure (figures/plot_baseline_shot_traces.py) is out of the DAG
// until a committed raw-data extract restores it; TASKS.md tracks this.
#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header(
      [shot],
      [peak-to-peak ($mu$m)],
      [635 nm $R^2$ (control)],
      [674.8 nm $R^2$ (control)],
      [515 nm $R^2$ (control)],
      [joint $g$],
    ),
    [203], [0.29], [0.79 (0.11)], [0.91 (0.25)], [0.65 (0.26)], [1.028],
    [90], [1.51], [0.68 (0.29)], [0.72 (0.33)], [--], [1.095],
    [365], [3.74], [0.67 (0.38)], [--], [--], [1.036],
  ),
  caption: [
    On individual shots the windowed fit clears its control on every
    tabulated channel, with a margin that narrows as excursion grows. Three
    free-space shots nearest the 10th, 50th and 95th percentile of
    peak-to-peak displacement: windowed $R^2$ at each shot's own
    fringe-matched $W$, its control, and the shared gain $g$ of a joint
    three-wavelength fit; -- where none was recorded. The trace figure these
    come from needs the raw acquisition files and is not built here.
  ],
) <fig-traces>

Three shots cannot establish a trend. They settle the question the medians
leave open: at low and medium excursion the windowed fit follows individual
fringes, not only their envelope. The joint-fit gains agree with the
independent per-channel gain sweep of @sec-limits, and constraining the gain
jointly adds little, since free amplitude and phase per window already
absorb a gain error. The flat-topped fringe peaks are fringe shape, not
detector clipping: almost no shot reaches the channel maximum.

The fitted amplitude runs low, and that is a symptom of phase error rather
than a gain defect. Projecting the data onto a cosine whose phase is off by
$epsilon$ recovers $A cos epsilon$, not $A$, so a window in which the phase
error wanders returns a small amplitude. Anything that reduces the phase
error within one fit should then recover amplitude, which @tab-r-variants
tests.

#figure(
  table(
    columns: 7,
    align: (left, left, right, right, right, right, right),
    table.header(
      [shot],
      [variant],
      [$W$],
      [$R^2$],
      [control $R^2$],
      [gap],
      [amp. ratio],
    ),
    [365 (high)], [fixed window], [1024], [0.431], [0.258], [0.174], [0.586],
    [365 (high)], [fringe-matched], [640], [0.672], [0.376], [*0.296*], [0.883],
    [365 (high)], [envelope, degree 2], [1024], [0.750], [0.474], [0.275], [--],
    [203 (low)], [fixed window], [1024], [0.927], [0.547], [0.380], [--],
    [203 (low)], [fringe-matched], [4096], [0.794], [0.111], [*0.683*], [--],
    [203 (low)], [envelope, degree 2], [1024], [0.964], [0.838], [0.125], [--],
  ),
  caption: [
    The fringe-matched window earns its extra freedom and the smooth
    envelope does not. Free-space 635 nm; each variant is scored against a
    control recomputed under its own window length and parameter count, so
    only the gap compares across rows. The envelope lets amplitude and phase
    vary as a polynomial within a window. Amp. ratio: median fitted
    amplitude over the window's own ($sqrt(2)$ times its standard
    deviation).
  ],
) <tab-r-variants>

The envelope has the highest raw $R^2$ but a control that rises almost as
fast, which is what freedom that fits anything looks like. The fringe-matched
window widens the gap and recovers the lost amplitude without touching
amplitude directly, which confirms the phase-error mechanism. Every table in
this document uses it. Raw $R^2$ without its control would invert this
ranking on the low-excursion shot.

#takeaway[
  Given the displacement, the Michelson model describes this data only when
  its amplitude and phase are refit over intervals short enough for them to
  be constant. The low fitted amplitude at large excursion is the same phase
  error, not a separate gain problem.
]

=== The baseline shows no fiber gap <sec-baseline-fiber>

Free space and fiber score alike under both methods: the decode's errors
match across acquisitions (@tab-r-method2), and method 1's gap favors the
fiber on two of three channels (@tab-r-method1). The fiber trace carries
less high-frequency content (@sec-q3-real), but at this error level that
costs the baseline nothing measurable.

=== What limits the fit <sec-limits>

#fig(
  "baseline_amplitude_trend",
  width: 100%,
  caption: [
    The global fit collapses with excursion and the windowed fit does not.
    Method-1 median $R^2$ binned by peak-to-peak drive excursion, per channel
    and acquisition, N = 1000 shots each: dotted, global fit; solid,
    windowed refit at a fringe-matched $W$; dashed, its control; shaded, the
    gap. $W$ shortens as excursion grows; each panel states the $W$ its
    numbers came from.
  ],
) <fig-baseline-amplitude-trend>

Binning by excursion separates three effects that aggregate medians blend:
the fit interval, a gain error, and a residual amplitude dependence.

*The fit interval dominates.* Fringe phase and contrast both wander within a
shot, by a fringe or more over a record, and these are exactly what the
global fit holds constant. A longer excursion sweeps more fringes and gives
the drift more room to accumulate, which accounts for the collapse with
excursion without any further assumption (@fig-baseline-amplitude-trend).

*The drive-to-displacement gain is too small, but this is secondary.* The
displacement targets come from the drive voltage through a calibrated gain
$k$, not from a measurement. Sweeping $k$ gives a sharply peaked $R^2$ above
the live value on every channel (@tab-r-gain), so the targets are too small.
Refit per window, the same correction barely moves $R^2$. The best gain also
differs between the two acquisitions by more than the sweep step, although
it is a property of the speaker, not of the optical path. A scalar that lands
in two places is absorbing more than a gain error, so the calibration
constant has not been re-tuned. The sweep does rule out the gain an earlier
calibration implied: there the fit explains nothing.

#figure(
  table(
    columns: 5,
    align: (left, left, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header(
      [acquisition],
      [channel],
      [best $k$ (vs live)],
      [global $R^2$, live -> best],
      [windowed $R^2$, live -> best],
    ),
    [free space], [635 nm], [16.50 (+13.4%)], [0.162 -> 0.462], [0.712 -> 0.767],
    [free space], [674.8 nm], [16.25 (+11.7%)], [0.171 -> 0.346], [0.626 -> 0.651],
    [free space], [515 nm], [16.00 (+9.9%)], [0.077 -> 0.176], [0.422 -> 0.435],
    [mm fiber], [635 nm], [15.50 (+6.5%)], [0.468 -> 0.613], [0.860 -> 0.883],
    [mm fiber], [674.8 nm], [15.75 (+8.2%)], [0.160 -> 0.230], [0.600 -> 0.617],
    [mm fiber], [515 nm], [15.25 (+4.8%)], [0.291 -> 0.396], [0.726 -> 0.746],
  ),
  caption: [
    A gain correction lifts the global fit but barely moves the windowed
    one. Median $R^2$ over a sweep of 101 gains from 10 to 35 (step 0.25),
    N = 1000, live $k = 14.553$. The windowed measure is at fixed
    $W = 1024$: a fringe-matched window would change length with the swept
    gain, so these windowed values are not comparable to @tab-r-method1.
    At the earlier calibration's $k = 33.42$, median $R^2$ is 0.012 to 0.029
    on every channel.
  ],
) <tab-r-gain>

*A residual amplitude dependence remains, and it is not a scale error.*
Giving each shot its own best gain leaves the largest-excursion bin fit
about as poorly as the best global gain does (@tab-r-amp). A scale error
would be absorbed by a per-shot gain. The per-shot gains need care: on some
channels a gain near twice the true one realigns the fringes by a whole
order and scores well. These aliases inflate the per-shot median $R^2$ of
the fiber 674.8 nm channel, so it is not evidence that this channel fits
better than @tab-r-method1 says.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([gain], [635 nm], [674.8 nm], [515 nm]),
    [per-shot best], [0.143], [0.094], [0.052],
    [best global], [0.122], [0.076], [0.020],
  ),
  caption: [
    Even a per-shot gain leaves the large-excursion residual. Global-fit
    median $R^2$ in the free-space bin above 3 $mu$m, N = 1000. On fiber
    674.8 nm, 249 of 1000 shots pick a gain near twice the live value; on
    free-space 515 nm, 125 do.
  ],
) <tab-r-amp>

Four other explanations were checked against the data and ruled out:

- *Drive-photodiode delay.* Shifting the prediction against the record over
  a few samples gives no interior peak on any channel; a real delay would.
- *Fringe-rate aliasing.* Peak fringe rates sit more than an order of
  magnitude below Nyquist.
- *A missing DC term.* The fit already includes a constant, globally and per
  window.
- *The optical phase factor or resonator parameters.* Either error would
  move or remove the gain-sweep peak. The best gain is close to the live one
  and independent of wavelength, which a mechanical displacement error
  produces and an optical-phase error does not.
=== Prior art: two published networks <sec-prior>

Two papers from one group put a convolutional network on this problem, a
single-channel one @barland2021cnn and a three-channel extension
@matha2023multichannel (@tab-prior). We ported the 2021 architecture; at one
input channel the port matches the paper's parameter count exactly, which
establishes that it is faithful rather than merely similar. The published
scores are not comparable to ours. They came from a different rig, with
harmonic training displacements and targets calibrated on that rig, so the
Barland row below is evidence about the architecture, not their result.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([network], [params], [Pearson r], [vel. RMSE ($mu$m/ms)]),
    [@barland2021cnn, 1 channel], [57,153], [0.90], [--],
    [@matha2023multichannel, 3 channels], [207,905], [0.96], [0.13],
    [@matha2023multichannel, best single channel], [--], [0.97], [0.11],
    [this port, 1 channel], [57,153], [--], [--],
    [this port, 3 channels (trained here)], [57,377], [--], [--],
  ),
  caption: [
    Published figures, on their authors' apparatus, beside the parameter
    counts of this repository's port. Training displacements there were
    eight harmonics between 53 and 93 Hz. The correlations and RMSEs were
    recovered from a text extraction that mangled decimal points, by
    cross-referencing passages where the value extracted cleanly; the
    parameter counts are each stated twice in their paper.
  ],
) <tab-prior>

=== Learned models against the analytic baseline <sec-learned>

Every learned configuration beats the analytic decode on both acquisitions,
and the port of the published architecture is best on both. @tab-learned-setup
lists how they were trained.

#figure(
  table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([setting], [value]),
    [configurations], [ten, two seeds each, on both acquisitions (40 runs, no failures)],
    [training], [from scratch, 20 epochs, one RTX 3090 Ti, eager; dataset preloaded],
    [budget], [about two GPU hours in total; the epoch cap is not a convergence criterion],
    [tabular], [ridge and LightGBM on CPU, over a 91-sample window],
    [output stride], [Barland 8, MLP 1, tabular 1; TCN and SCNN one output per sample],
    [score], [mean RMSE over test shots, then over seeds (@sec-read)],
  ),
  caption: [
    Training and evaluation of the learned models. Every absolute number
    below is budget-limited and so pessimistic; the models are compared
    under one identical budget.
  ],
) <tab-learned-setup>

#fig(
  "model_comparison",
  width: 100%,
  caption: [
    Every learned model clears the analytic decode, and Barland leads on
    both acquisitions. Velocity RMSE per configuration, one panel per
    acquisition; the dashed line is the method-2 decode. Bars are the mean
    over two seeds and whiskers the seed-to-seed range. Rows keep the
    free-space order in both panels, so a different fiber ordering reads as
    crossing. Hatched bars read samples after the timestamp they predict;
    only the TCNs do not.
  ],
) <fig-models>

==== What every configuration scored <sec-learned-table>

#let rl(stem) = (
  [#val("rl_" + stem + "_v_fs")],
  [#val("rl_" + stem + "_v_mmf")],
  [#val("rl_" + stem + "_d_fs")],
  [#val("rl_" + stem + "_d_mmf")],
)
#figure(
  table(
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
    [Barland], [#val("rl_barland_params")], [91], [no],
    [*#val("rl_barland_v_fs")*], [*#val("rl_barland_v_mmf")*],
    [#val("rl_barland_d_fs")], [#val("rl_barland_d_mmf")],
    [SCNN], [#val("rl_scnn44_params")], [85], [no], ..rl("scnn44"),
    [TCN], [#val("rl_tcn44_params")], [85], [*yes*], ..rl("tcn44"),
    [SCNN], [#val("rl_scnn19_params")], [85], [no], ..rl("scnn19"),
    [MLP], [#val("rl_mlp_params")], [256], [no], ..rl("mlp"),
    [TCN], [#val("rl_tcn19_params")], [85], [*yes*], ..rl("tcn19"),
    [TCN], [#val("rl_tcn36_params")], [29], [*yes*], ..rl("tcn36"),
    [TCN], [#val("rl_tcn28_params")], [29], [*yes*], ..rl("tcn28"),
    [LightGBM], [#val("rl_lgbm_params")], [91], [no], ..rl("lgbm"),
    [Ridge], [#val("rl_ridge_params")], [91], [no], ..rl("ridge"),
    [analytic], [--], [--], [no], [#val("rd_v_fs")], [#val("rd_v_mmf")],
    [#val("rd_d_fs")], [#val("rd_d_mmf")],
  ),
  caption: [
    Velocity RMSE ($mu$m/ms) and displacement RMSE ($mu$m) for every
    configuration, mean over test shots and two seeds; FS free space, MMF
    multimode fiber. Field is the receptive field in samples. Ridge params
    count coefficients, LightGBM's count trees (of 31 leaves); their
    stride-1 outputs average overlapping windows, so each depends on a
    181-sample two-sided span (@tab-tabular-synthetic). The MLP's output
    likewise averages 256 overlapping windows. Learned displacement
    integrates predicted velocity while the decode fits displacement
    directly, so every claim uses velocity. The analytic row is method 2
    under the same statistic.
  ],
) <tab-learned>

The weakest learned model, a ridge regression over a short window, beats
the multi-wavelength decode, which says more about the decode than about the
regression (@tab-learned). It is a low bar: in free space ridge sits about
level with the zero predictor of @tab-r-method2. Given the true displacement
the Michelson model explains little unless refit every couple of fringes
(@sec-r-method1, @sec-limits); why it fits this badly stays open. The decode
is offline, a whole-shot path search followed by smoothed differentiation,
so the causal TCNs clear it without lookahead.

==== Lookahead is worth about one seed spread <sec-learned-causal>

The TCN and SCNN at equal parameter count are a controlled pair: both span
85 samples, the TCN's ending at the predicted timestamp and the SCNN's
centered on it. The SCNN is ahead in free space by about one seed spread
(@fig-models), so this budget puts the cost of causality on the order of a
seed, not a factor. Barland's lead over the causal TCN is many seed spreads
wide, so most of it is not lookahead. The two stacks differ in window,
pooling, head and parameter count at once, and which carries the margin is
the question the next sweep should target. Neither published paper states
where its output timestamp sits, so the causality conclusion rests on the
TCN-SCNN pair alone.

==== Measuring an architecture requires an output resolution finer than the score

A windowed model's score is bounded below by its output grid. At a stride
equal to the window, the model emits one constant per block, and a predictor
emitting the exact block mean of the true velocity sets a floor. At that
stride Barland scores within a few percent of the floor, which measures the
grid, not the architecture (@tab-stride-floor). The table's rows therefore
run at strides where the floor cannot bind.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([model], [stride], [floor ($mu$m/ms)], [score ($mu$m/ms)]),
    [Barland], [256], [0.1866], [0.1962],
    [MLP], [256], [0.1866], [0.2409],
    [Barland], [8], [0.0065], [#val("rl_barland_v_fs")],
    [MLP], [1], [0.0000], [#val("rl_mlp_v_fs")],
  ),
  caption: [
    At the window's own stride the discretization floor binds; at the
    strides used in @tab-learned it does not. Free space; floor is the RMSE
    of the per-block mean of the true velocity. The stride-256 runs are
    outside the committed artifact.
  ],
) <tab-stride-floor>

==== Fiber degrades every deep network, in proportion to its free-space accuracy <sec-learned-gap>

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([model], [vel. FS], [vel. MMF], [MMF / FS]),
    ..(
      ("Ridge (274)", "ridge"),
      ("analytic decode", "decode"),
      ("LightGBM (200)", "lgbm"),
      ("TCN (28k, field 29)", "tcn28"),
      ("TCN (36k, field 29)", "tcn36"),
      ("TCN (19k, field 85)", "tcn19"),
      ("MLP (51k)", "mlp"),
      ("Barland (57k)", "barland"),
      ("TCN (44k, field 85)", "tcn44"),
      ("SCNN (19k, field 85)", "scnn19"),
      ("SCNN (44k, field 85)", "scnn44"),
    ).map(((label, stem)) => {
      let pre = if stem == "decode" { "rd_" } else { "rl_" + stem + "_" }
      ([#label], [#val(pre + "v_fs")], [#val(pre + "v_mmf")],
        [#val("rl_" + stem + "_ratio")])
    }).flatten(),
  ),
  caption: [
    The more accurate a model is in free space, the more it loses behind
    fiber. Velocity RMSE ($mu$m/ms) on each acquisition and their ratio,
    rows ordered by increasing ratio.
  ],
) <tab-learned-gap>

Every deep network degrades behind fiber, while ridge and the decode do not
(@tab-learned-gap). Across the learned models the ratio falls as the
free-space error rises (Spearman $rho = #val("rl_ratio_spearman")$). One
hypothesis fits both ends: a model that extracts only coarse structure
loses little when speckle scrambles fine structure. Neither the structure
each model uses nor what the fiber removes was measured. Barland degrades
like the other top networks, so it leads through fiber by starting ahead,
not by degrading less. The decode's lack of a fiber gap (@sec-baseline-fiber)
is therefore weak evidence about the fiber.

==== Receptive field dominates parameter count <sec-learned-field>

Between the two field-29 TCNs, extra capacity buys less than the smaller
model's seed spread, while a field-85 TCN with fewer parameters beats both
(@tab-learned). Over this range, widening the receptive field buys what
adding parameters does not; only two field lengths were run.

==== What limits these numbers

- *Causality.* Only the TCNs are causal, so a comparison across the hatch
  boundary of @fig-models compares estimators solving different problems.
- *Two seeds.* The SCNN-over-TCN margin is about one seed spread, and no
  conclusion rests on it; the other comparisons above are at least two spreads
  wide.
- *Budget.* The runs are budget-limited, not converged (@tab-learned-setup).
- *Apparatus.* The published numbers of @sec-prior are indicative only.
= Q3: fiber <sec-q3> // math-ok: question label

Q3 asks how fast accuracy degrades as each fiber impairment turns on, and why. // math-ok: question label
The sub-questions then ask what a monitor color, a $P_perp$ channel or a flow
model recovers.

== Degradation with each fiber impairment <sec-q3-degradation>

=== Analytic

The fiber model (@sec-fiber-model) has three knobs: the mode count, which sets
the static coupling loss $E|kappa|^2$; the speckle drift rate; and the path
drift rate. The linear-Gaussian reference of @sec-q3-monitor, which knows the
fringe order, predicts each knob's cost from the free-space point
(@tab-q3-sweep). A static $|kappa|$ is a fixed per-shot gain: it costs fringe
SNR and $C_"eff" = C |kappa|$ but no information about the phase law, so the
reference predicts static fading free. Path drift competes with the target
in velocity units and is predicted to dominate; speckle drift at these rates
moves the speckle phase by a fraction of a nanometre and is predicted to cost
nothing.

=== Synthetic

#assert(val("fs_fading_unexplained"), message: "a fading hypothesis now predicts the observed cost")
#assert(val("fs_no_sign_flips"), message: "a sweep point now shows whole-shot sign flips")
#assert(val("fs_speckle_free"), message: "speckle drift now costs the network")

TCAN was trained at every point of each knob swept from the one-mode static
fiber with the others off, on paired motion (@fig-q3-sweep). The ranking is
the reverse of the prediction. Static fading costs the most, and by eight
modes TCAN is close to the zero predictor. Path drift costs little until its
fastest rung. Speckle drift costs nothing. Signed and sign-invariant errors
coincide at every point, so no point loses whole-shot sign.

The reference ranks path drift first only in MSE ratio to its own floor,
which is several times below TCAN's. In added MSE the network pays more for
path drift than the reference predicts (@tab-q3-sweep), so drift is not
tolerated; the network's other error dominates its floor instead.

#fig(
  "q3_fiber_sweep",
  width: 92%,
  caption: [
    Static fading costs the most, path drift little and speckle drift
    nothing. TCAN test velocity RMSE as each fiber knob turns on alone from the
    one-mode static fiber; bars: shot SE; hollow: sign-invariant RMSE; dashed:
    free space; solid: the zero predictor.
  ],
) <fig-q3-sweep>

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.header(
      [knob], [value], [ref. FS / fiber MSE], [ref. $Delta$MSE], [TCAN], [sign-inv.],
      [TCAN $Delta$MSE],
    ),
    ..val("fs_table").map(r => (
      [#r.knob], [#r.value], [#r.ref_ratio], [#r.ref_excess], [#r.tcan],
      [#r.tcan_si], [#r.tcan_excess],
    )).flatten(),
  ),
  caption: [
    The reference's ranking reverses in the network. Per knob value ($M$,
    speckle drift in rad/s#super[1/2], path drift in $mu$m/s#super[1/2]): the
    reference's free-space over fiber MSE, its MSE added over the one-mode
    static fiber, TCAN's signed and sign-invariant velocity RMSE ($mu$m/ms),
    and TCAN's added MSE. Added MSE in $10^(-3)$ ($mu$m/ms)#super[2].
  ],
) <tab-q3-sweep>

=== Mechanism

Three mechanisms could make static fading costly, and each predicts a share
of the learned signal retained, $1 - "MSE" \/ "MSE"_0$ relative to the one-mode
point, from runs that change only its knob (@tab-q3-mechanism). The lost
$C_"eff"$ would erase the feedback asymmetry that carries the sign
(@sec-sign): read at the same $C_"eff"$, the noise-free $C$ sweep retains
most of the signal, and so does the nominal SNR ladder read at the same sign
evidence. The lost fringe SNR, read on that ladder at the same SNR loss,
retains nearly all of it. In-shot fading is absent from a static fiber, and
the speckle drift that adds it costs nothing. Each predicts far less loss
than measured, as does their product.

The per-shot errors locate the loss. The added error tracks each shot's
speed, not its fringe depth, and the median shot barely moves while the mean
rises: the network shrinks its fast-shot estimates toward zero whatever each
shot's own fade. Restoring each faded shot's amplitude to the one-mode level
does not recover the one-mode model's accuracy (@tab-q3-mechanism), so it is
not a normalization mismatch. The three colors fade independently. The open
candidate is therefore the loss of agreement between colors, on which the
multicolor estimate of fast motion rests (@sec-q1-colors): each color's
static gain is a different random number. A sweep of mode count on a single
color tests it; if the single-color loss matches the SNR prediction, the
cross-color mechanism holds.

#figure(
  table(
    columns: 10,
    align: (left, right, right, right, right, right, right, right, right, right),
    table.header(
      [$M$], [$C_"eff"$], [SNR loss], [observed], [$C$ sweep], [SNR], [sign ev.],
      [$r$ speed], [$r$ depth], [median],
    ),
    ..val("fs_hyp_table").map(r => (
      [#r.m], [#r.c_eff], [#r.snr_loss], [#r.observed], [#r.h1], [#r.h2],
      [#r.h1s], [#r.r_speed], [#r.r_amp], [#r.median],
    )).flatten(),
  ),
  caption: [
    Every hypothesis retains far more signal than TCAN does. Share of learned
    signal retained relative to $M = 1$: observed, and predicted by the
    noise-free TCN $C$ sweep at $C_"eff"$, by the nominal TCN SNR ladder at
    the SNR loss (dB), and by that ladder at matched sign evidence. The noise
    conditions and families differ, so only the retained shares are compared.
    $r$: correlation over paired shots of the added MSE with shot speed and
    with fringe depth; median: median per-shot RMSE ($mu$m/ms). The one-mode
    model scores #val("fs_probe_raw") on eight-mode shots raw and
    #val("fs_probe_rescaled") rescaled to its training level; the colors'
    fringe depths correlate at most #val("fs_color_corr").
  ],
) <tab-q3-mechanism>

=== Real data

_Status: not run on a characterized fiber_. The rig's learned models learn
through its multimode fiber and degrade by a bounded factor
(@tab-learned-gap), consistent with a fiber in the low-impairment corner of
the sweep: few effective modes or a strong fundamental-mode return. The
committed prediction: on the rig, the fiber penalty grows with the static
coupling loss, measured as the fringe depth through fiber over free space,
and not with the path drift rate. It is falsified if the penalty tracks the
drift rate at fixed fringe depth.

== Q3.1: a fiber-only monitor color <sec-q3-monitor> // math-ok: question label

=== Analytic

With a common path drift and no dispersion, the fiber confounds $d$ with the
drift. In @eq-fiber-phase, $D = 0$ makes $n$ the same for every color, so $d$
and $n p$ enter every channel only through their sum. Adding $delta$ to $d$
and removing $delta \/ n$ from $p$ leaves every signal unchanged. The
likelihood alone cannot separate the two; only their priors can, and only as
far as their spectra differ.

In the simulator they differ widely. The path drift is a random walk, with
$dot(p)$ white, while the drive's power lies in the drive band. A
linear-Gaussian reference with perfect demodulation estimates $d$ from the
sum against those two spectra; the path drift accounts for nearly all of its
fiber loss, and the monitor below recovers most of it (@tab-q3-analytic).

The reference has five limits.
+ It bounds the error in neither direction. Knowing the fringe order is
  optimistic; linear-Gaussian estimation is pessimistic for non-Gaussian
  speckle and the non-stationary chirp.
+ Fading enters only through the shot-averaged $|kappa|^2$, so the reference
  cannot see in-shot fades or cycle slips.
+ The separation rests entirely on the two prior spectra. The thermal and
  mechanical drift of a real fiber is concentrated at low frequency, would
  overlap the drive band, and would push the ratio toward the shared-band
  case below.
+ It ignores $P_perp$, so the $P_perp$ arm scores the same as the fiber
  alone.
+ The path term is common to all colors only because $D = 0$.

When drift and target share one band, the degeneracy is exact. The
constant-parameter model, with $d$ and $p$ one unknown each per shot, is that
case: its Fisher matrix (@sec-fisher) is singular to float precision, and
dispersion lifts it as $D^2$ (@tab-q3-analytic).

A dichroic at the distal face returns a monitor color before the target, so
that channel sees $n p$ alone. If the signal channels measure the sum
$d + n p$ with information $S$ and the monitor measures $n p$ with
information $M$, then $d$ is their difference, with variance $1 \/ S + 1 \/ M$.
In the shared-band case the recovered fraction of the free-space information
is therefore
$
  S_"fiber" / S = M / (S + M) ,
$ <eq-monitor-fraction>
which the Fisher matrix confirms for one and three monitor colors
(@tab-q3-analytic). Both hold at $|kappa| = 1$ and assume the reciprocity
and mode coupling of @sec-fiber-model.

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header([quantity], [value]),
    [reference velocity RMSE, free space / fiber ($mu$m/ms)],
    [#val("dg_ref_fs") / #val("dg_ref_fiber")],
    [free-space over fiber MSE: fiber, monitor, no path term, $P_perp$],
    [#val("dg_ratio_fiber"), #val("dg_ratio_monitor"), #val("dg_ratio_nopath"), #val("dg_ratio_pbs")],
    [Fisher information over free space at $D = 0$],
    [#val("fb_info_d0").mantissa $times 10^(#val("fb_info_d0").exponent)$],
    [condition number at $D = 0$],
    [#val("fb_cond_d0").mantissa $times 10^(#val("fb_cond_d0").exponent)$],
    [information at $D = 10^(-3)$ / $0.1$ $mu"m"^(-1)$],
    [#val("fb_info_dlo").mantissa $times 10^(#val("fb_info_dlo").exponent)$ /
      #val("fb_info_dhi").mantissa $times 10^(#val("fb_info_dhi").exponent)$],
    [log-log slope in $D$], [#val("fb_disp_slope")],
    [monitor fraction, #val("fb_mon_um") nm: @eq-monitor-fraction / Fisher],
    [#val("fb_mon1_pred") / #val("fb_mon1")],
    [monitor fraction, #val("fb_mon3_um") nm: @eq-monitor-fraction / Fisher],
    [#val("fb_mon3_pred") / #val("fb_mon3")],
  ),
  caption: [
    The path drift carries the reference's fiber loss, and the monitor
    fraction of @eq-monitor-fraction matches the Fisher matrix. Reference:
    #val("dg_shots") shots; Fisher: #val("fb_shots") shots.
  ],
) <tab-q3-analytic>

=== Synthetic

#assert(val("r7_all_near_zero"), message: "a fiber arm now scores below 0.9 of the zero predictor")
#assert(val("r7_zero_shared"), message: "the q3_fiber arms no longer share one motion draw")
#assert(val("r7_tcan_all_converged"), message: "a TCAN fiber arm no longer converges")

At the combined point, eight modes with both drifts at their top rungs, the
impairments compound to the zero predictor in every arm, so no arm can show
the monitor's benefit (@fig-q3-fiber, @tab-q3-fiber). Paired against its
four-color control, the monitor moves TCAN's error by
#val("r7_tcan_monitor") $plus.minus$ #val("r7_tcan_monitor_se") $mu$m/ms,
against a fiber cost of #val("r7_tcan_cost"). The static eight-mode
fading alone brings TCAN most of the way there (@sec-q3-degradation). The
monitor arm should be rerun at a mode count where the network still learns.

#fig(
  "q3_fiber",
  width: 70%,
  caption: [
    At the combined point every fiber arm sits at the zero predictor, with or
    without the monitor or $P_perp$. Test velocity RMSE of each q3_fiber arm,
    seed 0, the nominal point; bars: per-cell shot SE; hollow: not converged;
    grey: the zero predictor, shared by all arms.
  ],
) <fig-q3-fiber>

#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.header([arm], [TCAN], [over zero], [epochs], [UNetT], [over zero], [converged]),
    ..val("r7_table").map(r => (
      [#r.arm], [#r.tcan], [#r.tcan_oz], [#r.tcan_ep], [#r.unet], [#r.unet_oz],
      [#r.unet_conv],
    )).flatten(),
  ),
  caption: [
    Every fiber arm scores at the zero predictor. Test velocity RMSE ($mu$m/ms),
    its ratio to the zero predictor, TCAN's epochs, and whether
    UNetTransformer converged (TCAN converged in every arm).
  ],
) <tab-q3-fiber>

=== Real data

_Status: not run; needs new hardware, a distal dichroic and a monitor laser_.
The committed prediction: through fiber, adding the monitor color lowers a
trained estimator's velocity error, by an amount set by how far the real path
drift overlaps the drive band; with full overlap it is the shared-band
fraction of @eq-monitor-fraction. Since static fading dominates the
simulated cost (@sec-q3-degradation), the gain is measurable only where the
fringe depth through fiber is high. It is falsified if adding the monitor
leaves the fiber error unchanged at matched training and fringe depth.

== Q3.2: the orthogonal-polarization channel <sec-q3-polarization> // math-ok: question label

=== Analytic

$P_perp$ is a fade indicator. The fiber is lossless, so
$
  |kappa|^2 <= 1 - P_perp ,
$
with equality only for one mode. For more modes the bound loosens: the
correlation of $P_perp$ with $|kappa|^2$ falls as the inverse square root of
the mode count and its $R^2$ as the inverse (@tab-q3-pperp). The mean
coupling follows the coherent-backscatter value of the circular orthogonal
ensemble,
$
  E|kappa|^2 = 2 / (2M + 1) ,
$
except for one mode, where the model's wrapped eigenphases lack the
ensemble's level repulsion. Since $|kappa|$ scales $C$ and the visibility
(@sec-fiber-model), $P_perp$ tells an estimator how deep the fringes have
faded. The prediction: it helps most at few modes, and its share of the fade
variance falls inversely with the mode count.

#figure(
  table(
    columns: 2,
    align: (left, right),
    table.header([quantity], [value]),
    [$"corr"(P_perp, |kappa|^2)$, $M = 1$], [#val("fb_corr_n1")],
    [$"corr" sqrt(M)$, $M >= 8$], [$-$#val("fb_corr_sqrt_hi") to $-$#val("fb_corr_sqrt_lo")],
    [$R^2 M$, $M >= 8$], [#val("fb_r2n_lo") to #val("fb_r2n_hi")],
    [corr, $R^2$ at $M = 8$], [#val("fb_corr_n8"), #val("fb_r2_n8")],
    [$E|kappa|^2$ at $M = 8$: model / ensemble], [#val("fb_k2_n8") / #val("fb_coe_n8")],
    [$E|kappa|^2$ at $M = 1$: model / ensemble], [#val("fb_k2_n1") / 0.67],
  ),
  caption: [
    $P_perp$ indicates the fade, less well with more modes. Over
    #val("fb_draws") fiber draws.
  ],
) <tab-q3-pperp>

=== Synthetic

At the combined point the $P_perp$ arm sits at the zero predictor with the
rest (@tab-q3-fiber), so it tests nothing. Because fading is the dominant
simulated cost (@sec-q3-degradation), a fade indicator is where the
channel's value should show: the arm should be rerun on the static fiber
sweep, where the network still learns.

=== Real data

_Status: not run; needs new hardware, a polarizing beamsplitter and a
photodiode at the fiber output_. The committed prediction: across shots,
$P_perp$ anticorrelates with the squared fringe depth, with magnitude near
the inverse-square-root law of @tab-q3-pperp for the fiber's mode count, which is
not measured. It is falsified if $P_perp$ is uncorrelated with the fringe
depth.

== Q3.3: flow-matching denoising <sec-q3-denoise> // math-ok: question label

=== Analytic

#assert(val("r8_path_over_fringe"), message: "the path drift over one shot no longer exceeds a fringe")

The fiber operating point is the combined point of
@sec-q3-degradation, with no monitor and $D = 0$. There the degeneracy of
@sec-q3-monitor has a definite size: over one shot the path drift $p(t)$
wanders #val("r8_path_shot_um") $mu$m rms, more than a fringe at every
color. Its sample-to-sample increment, read as a velocity, is white, and it
separates from the drive only as far as the two priors' spectra differ. The
linear-Gaussian reference of @sec-q3-monitor gives the free-space MSE
fraction for these spectra (@tab-q3-analytic); the flow fit measures the
separation without the linear-Gaussian assumption.

A denoiser maps fiber signals $y_f$ to an estimate $hat(y)_0$ of the
free-space signals $y_0$, and a free-space regressor $g$ maps that to
velocity. The chain is a function of
$y_f$. Among all functions of $y_f$ the posterior mean minimizes mean-squared
error, so
$
  "MSE"(g(hat(y)_0 (y_f))) >= "MSE"(E[v | y_f]) ,
$ <eq-bayes-bound>
and the same bound holds for a regressor trained on $y_f$ directly. It does
not order the two pipelines, since both can fall short of $E[v | y_f]$. The
bound is derived; the order of the two pipelines is empirical. The chain has two handicaps. The posterior mean of
$y_0$ averages over the confounded $delta(t)$, which blurs the fringes, so the
free-space regressor sees off-distribution input. A single posterior sample
keeps sharp fringes but carries an arbitrary $delta(t)$.

The prediction: on raw fiber traces, denoise-then-regress does not beat a
regressor trained on fiber; a tie is its best case. A clear win would mean the
fiber-trained regressor sits far from the Bayes estimator, and its capacity,
budget and test-set isolation would need checking before the win is read as a
property of the fiber. What the denoiser can add instead is a calibrated
spread over $y_0$.

=== Synthetic

#assert(not val("fl_den_beats_fiber"), message: "the denoise chain now beats the fiber-trained regressor")
#assert(val("fl_den_steps_plateau"), message: "the denoise step count no longer plateaus")

The prediction holds: denoise-then-regress does not beat the fiber-trained
TCAN. On the fiber test set the chain's error, #val("fl_den_chain") $mu$m/ms,
is above the fiber-trained TCAN's (@tab-q3-flow), even with the
conditioning encoder that makes the posterior flows read their signals
(@sec-flow-methods). The test is weak, because that regressor sits at the
zero predictor (@sec-q3-monitor), so both pipelines are near the floor. The
step count has converged; the draw count has not, so more draws would move
the chain's score slightly, not past the regressor. The whole-shot encoder of
@sec-q3-posterior, which clears the regressor directly, is the better route.

=== Real data

_Status: not run; needs paired free-space and fiber recordings of the same
motion, a new acquisition_. The committed prediction: a free-space regressor
on denoised fiber traces does not beat a regressor trained on fiber traces.
It is falsified if, at matched training, the chain's velocity error is lower
by more than $s$, the two-seed range of @tab-real-predictions.

== Q3.4: the shape of the fiber posterior <sec-q3-posterior> // math-ok: question label

=== Analytic

Modes of the posterior follow the symmetries of the likelihood. The null
direction $delta(t)$ of @sec-q3-monitor is a continuum, and the priors along
it, a random walk for $p$ and a smooth drive for $d$, are unimodal. The fiber
should therefore broaden the posterior over $v(t)$ smoothly along that
direction, by up to the path-drift increment scale, without adding discrete
modes. The one discrete symmetry is the sign map of
@sec-sign, which the fiber preserves, because $psi_k$ and $p$ have symmetric
priors under negation. Feedback breaks it with evidence that grows as $C^2$
(@sec-feedback). The fiber lowers the fringe depth by $|kappa|$, with
$E|kappa|^2 approx #val("r8_coe")$ at $M = #val("r8_modes")$
(@sec-q3-polarization), which lowers that evidence. So at a fixed $C$, fiber
shots should be bimodal, $v$ against $-v$, more often than free-space shots.

The fiber flows use a global-context backbone, with attention over every
bottleneck token of a shot and a receptive field longer than the shot. Path
drift and the sign pair are shot-global, and a local field would draw
marginals coherent only within its window. The prediction: the fraction of
multimodal posteriors per time sample rises in fiber only through the sign
pair, with no excess beyond it; per-sample spread rises; 90% bands stay
calibrated. More than two modes at many time samples, or modes local in time,
would falsify it and point at fringe-order ambiguity under fading, which this
analysis does not predict.

=== Synthetic

#assert(float(val("fl_head_gain")) >= 1.8, message: "the encoder head no longer roughly halves TCAN's free-space error")

Whole-shot context, not the flow, drives the gain. The encoder head alone,
a deterministic regressor with TCAN's parameter count but a field longer
than the shot, cuts TCAN's free-space error #val("fl_head_gain")-fold
(@tab-q3-flow). Sampling the flow on top cuts it by a further
#val("fl_flow_gain_pct")%, and the posterior is calibrated where TCAN's seed
ensemble is not (@fig-q3-flow). In free space neither estimator memorized
its test set: both score the same on a fresh seed.

The sign is resolved: on #val("fl_sign_fs")% of free-space shots most
posterior draws take the right direction, consistent with a likelihood
whose only discrete symmetry the feedback breaks. Through fiber the flow
and the head clear TCAN, which sits at the zero predictor, but both score
worse on the fresh set than on the test set, so the fiber margin is smaller
than the test set shows. The bands stay calibrated there. The fair comparison this leaves open is a regressor
trained with a whole-shot field, not the cropped 85-sample one.

#figure(
  {
    set par(justify: false)
    set text(size: 9pt)
    let row(k) = (
      [#val("fl_" + k + "_label")],
      [#val("fl_" + k + "_field")],
      [#val("fl_" + k + "_params")],
      [#val("fl_" + k + "_fs")],
      [#val("fl_" + k + "_fs_fresh")],
      [#val("fl_" + k + "_coverage_fs")],
      [#val("fl_" + k + "_crps_fs")],
      [#val("fl_" + k + "_fiber")],
      [#val("fl_" + k + "_fiber_fresh")],
      [#val("fl_" + k + "_coverage_fiber")],
    )
    table(
      columns: 10,
      align: (left,) + (right,) * 9,
      table.header(
        table.cell(rowspan: 2)[estimator],
        table.cell(rowspan: 2)[field],
        table.cell(rowspan: 2)[params],
        table.cell(colspan: 4)[free space],
        table.cell(colspan: 3)[fiber],
        [RMSE], [fresh], [cov.], [CRPS], [RMSE], [fresh], [cov.],
      ),
      ..("tcan", "unet", "ensemble", "head", "flow").map(row).flatten(),
    )
  },
  caption: [
    Encoder flow against the regressors, #val("fl_shots") test shots per
    arm. RMSE and CRPS in $mu$m/ms; field in samples; fresh: the same
    estimator on an unseen seed-2000 test set; cov.: fraction of the truth
    inside the 90% band. The regressors train on 4096-sample crops, the
    encoder and flow on full shots. The flow's RMSE and fresh columns come
    from one 16-draw audit; its coverage and CRPS from 64 draws.
  ],
) <tab-q3-flow>

#fig(
  "q3_flow",
  caption: [
    Velocity RMSE (left) and 90% band coverage (right) of
    @tab-q3-flow's estimators. Look at the gap between the cropped-field
    regressors and the encoder head, which is most of the gain, and at
    the fiber markers moving right on the fresh set.
  ],
) <fig-q3-flow>

=== Real data

_Status: not run; needs paired free-space and fiber recordings of the same
motion, a new acquisition_. The committed prediction: through fiber, the
posterior over $v(t)$ widens smoothly, with no modes beyond the sign pair.
It is falsified by more than two modes at many time samples, or by modes
local in time.

== Real-data context: free space versus multimode fiber <sec-q3-real>

The fiber path is the deployment-relevant one and the expected-harder one. The
raw data does differ: the fiber acquisition carries less high-frequency
content than free space in the same channel.

// Spectrum figure (figures/plot_freespace_vs_fiber.py): out of the DAG until a
// committed raw-data extract restores it; its caption is in git history.

_Status: the 635 nm mean amplitude spectrum behind this comparison needs the
raw acquisition files and is not shown in this build_.

That spectral difference does not translate into a baseline difference
(@sec-baseline-fiber), so the baseline is not evidence either way about the
fiber gap.

The simulated fiber now has a degradation curve per impairment
(@sec-q3-degradation), and the rig's fiber penalty (@sec-learned-gap) falls in
its low-impairment corner. Which corner the rig occupies awaits a measurement
of its fringe depth through fiber over free space; every other prediction
outside this section is for free space.

= Q4: Sagnac accelerometer <sec-q4> // math-ok: question label

Q4 asks whether a Sagnac loop read by self-mixing would give a common-path, // math-ok: question label
fiber-noise-insensitive accelerometer. It would not be insensitive: the loop
returns the nonreciprocal phase in the amplitude and the reciprocal fiber
phase in the round-trip phase $phi_0$, which takes the place of $theta$ in
@eq-excess. Whether that drift matters depends on its bandwidth against the
signal band, which we do not model.

=== Analytic

A coupler of power split $rho_c$ closed on a loop returns to the laser
$
  E_r = 2 i sqrt(rho_c (1 - rho_c)) cos(Delta phi \/ 2) e^(i Sigma \/ 2), quad
  Sigma = phi_"CW" + phi_"CCW", quad Delta phi = phi_"CW" - phi_"CCW"
$ <eq-loop-mirror>
for any $rho_c$ @mortimore1988loop, because each beam takes one bar and one
cross pass; we check it numerically by summing the two beam fields. The
nonreciprocal phase sets only the amplitude reflectivity $rho_r$, the cosine
term. Self-mixing reads the round-trip phase of the
return @lang1980feedback @taimre2015tutorial, which contains $Sigma \/ 2$ and
the lead fiber in full. A conventional Sagnac readout at the second coupler
port is insensitive to reciprocal fiber noise; the self-mixing readout is not.

The output encodes the velocity of an element placed off the loop midpoint,
not the acceleration of the loop. An element with phase $eta x(t)$, crossed
by the two beams $tau$ apart, gives $Delta phi = eta tau v$, exactly for
quadratic $x(t)$. Rotation is nonreciprocal @post1967sagnac. Uniform linear
acceleration of a rigid loop is not: the Sagnac delay is proportional to
the loop integral of the metric's time-space components @rizzi2004rotating
@tartaglia1998sagnac, which vanish in the Rindler metric; that step is
our own and untested. A proof mass of resonance $omega_0$ and quality factor
$Q_m$ makes $Delta phi$ follow acceleration, with gain
$eta tau Q_m \/ omega_0$, only near resonance. Off-center strain of the coil
may act as a distributed proof mass; we have no measurement of its size.

With feedback $C_0$ at full reflection, both feedback and modulation depth
scale with $rho_r$:
$
  phi_F = phi_0 - C_0 rho_r sin(phi_F + beta), quad y = a_0 rho_r cos phi_F + n .
$ <eq-sagnac-signal>
A quarter-wave bias $Delta phi_b$ is needed, since
$partial rho_r \/ partial Delta phi$ vanishes without one. Strong feedback
$C_0$ above the single-mode limit is allowed because $C_"eff" = C_0 rho_r$,
but $C_"eff"$ must stay below that limit over the whole swing, or the laser
turns multistable @lang1980feedback. With $phi_0$ known and
$g = partial Delta phi \/ partial a$, the acceleration Fisher information has
the form of @eq-fisher with prefactor
$
  a_0 g sin(Delta phi_b \/ 2) \/ (2 sigma)
$
and $F'$ replaced by $partial (y \/ a_0) \/ partial rho_r$. We call its ratio
to its value without feedback the feedback gain.

Feedback raises that information little unless $phi_0$ is held at its best
value (@fig-sagnac, @tab-sagnac). The phase-averaged gain falls below the
conventional phase readout's at the same $C$ (@eq-phase-factor),
because $Delta phi$ enters the amplitude, not the phase, and the loop bias
also lowers the feedback to $C_"eff"$. The best-phase gain is a genie bound:
holding $phi_0$ there is the fiber-phase stabilization the premise hoped to
avoid. Sign needs no feedback asymmetry, because the bias makes the response
signed.

The averaged gain is larger without linewidth enhancement than at the rig's
$alpha$, because of where the feedback phase dwells. Over a uniform $phi_0$,
$phi_F$ dwells longest near $phi_F = -beta$, where $cos(phi_F + beta)$ peaks.
Without linewidth enhancement that is where the slope
$partial (y \/ a_0) \/ partial rho_r$ peaks; at the rig's $alpha$ the slope
there is small.

#figure(
  table(
    columns: 6,
    table.header(
      [$C_0$ ($C_"eff"$)],
      [swing (rad)],
      ..val("sagnac_table_alpha").map(a => [$alpha = #a$]),
      [$2 Phi(C_"eff", 3)$],
    ),
    ..val("sagnac_table").map(r => ([#r.c0], [#r.swing], ..r.cells.map(c => [#c]), [#r.smi])).flatten(),
  ),
  caption: [
    The phase-averaged gain stays below the conventional readout at the same
    $C_"eff"$; only the best phase exceeds it. Feedback gains, phase-averaged /
    best phase, bias $pi \/ 2$. Swing: for $C_0 > 1$, the largest
    $|Delta phi - Delta phi_b|$ keeping $C_0 rho_r < 1$; blank rows are
    unbounded.
  ],
) <tab-sagnac>

#fig(
  "sagnac_fisher",
  width: 62%,
  caption: [
    Only a held best phase buys a large feedback gain, and it diverges as
    $C_"eff" -> 1$. Acceleration Fisher at $C_0$ over $C_0 = 0$, bias
    $pi \/ 2$: best phase $phi_0$ (solid), phase-averaged (dashed).
    $alpha = 5$ (@tab-sagnac) overlaps $alpha = 3$ and is not drawn.
  ],
) <fig-sagnac>

=== Synthetic

_Status: not run; Q4 is analytic only by design, a PI decision_. // math-ok: question label

=== Real data

_Status: not built; needs a bench fiber Sagnac loop read by self-mixing_. The
committed prediction, from @eq-loop-mirror: with the nonreciprocal phase
swept by a known bias, the modulation amplitude follows
$|cos(Delta phi \/ 2)|$, and with a phase modulator in the lead fiber the
fringe phase follows the reciprocal phase in full. It is falsified if the
amplitude does not vanish at $Delta phi = pi$, or if a reciprocal lead-fiber
phase step leaves the fringe phase unmoved.


= Scope and future directions <sec-closing>

== What this report establishes <sec-scope>

*Q1, channels.* On simulated data, for TCAN each added color lowers the
error noise-free and at the nominal point (@sec-q1-count); a quadrature
one-color array beats distinct colors only without noise
(@sec-q1-colors). The real-data tier is a committed prediction, not yet
scored.

*Q2, estimators.* On simulated data the architecture matters only with
feedback, and feedback is what makes the sign of the motion observable
(@sec-q2-floor, @sec-q2-nonlinearity); context length matters more than
family (@sec-q2-context); at the nominal point phase drift
dominates every other imperfection (@sec-q2-noise). On the rig
(@sec-q2-real), the closed-form decode scores worse than predicting zero,
every learned model tried clears it, and the Barland port is best on both
acquisitions. With the displacement supplied, the Michelson model describes
the data when its amplitude and phase are refit per fringe-matched window.
The drive-to-displacement calibration carries a gain error of several
percent, so the supervision targets are systematically small.

*Q3, fiber.* Static coupling loss dominates the simulated fiber's cost, by a
mechanism that neither the lost SNR nor the lost feedback accounts for
(@sec-q3-degradation); the impairments together reach the zero predictor, so
Q3.1 and Q3.2 are untested there. A whole-shot-context regressor, alone or
as the encoder of a flow posterior, halves the free-space error of the
cropped-field regressors and clears the zero predictor through fiber; the
flow adds calibrated bands (@sec-q3-posterior). Denoising fiber traces
before a free-space regressor does not help. On the rig
every learned model degrades behind fiber, more so the better it is in free
space, while the decode holds (@sec-learned-gap).

*Q4, Sagnac.* A self-mixing readout of a Sagnac loop is not insensitive to
reciprocal fiber phase, and its feedback gain is small unless that phase is
held at its best value (@sec-q4).

== What it does not establish

The decode is not an upper bound on classical processing: a better inverse,
or a corrected transfer function, could do better. Nothing here speaks to
latency or to waveform families outside the one these acquisitions sweep.
The decoder is shown correct on clean model-generated fringes, not
noise-robust. The baseline cannot resolve the free-space/fiber gap, and the
learned ratios measure it only under a short, two-seed budget. The
calibration gain is located, not corrected, because the two acquisitions
disagree on it. The residual amplitude dependence that survives a per-shot
gain fit is bounded but unexplained. No real-data prediction has been scored,
and Q4 has no measurement.

== Future directions <sec-future>

In order of what returns most for the least work.

*A third seed on the real sweep*, to settle whether the SCNN's free-space
margin over the TCN is real.

*A receptive-field sweep.* Receptive field moved the real-data errors more
than parameter count (@sec-learned-field), yet that axis has two points.

*A scaling study of error against dataset size and parameter count.*
@barland2021cnn reports that going beyond roughly $10^5$ parameters "does
not lead to better training and may even lead to worse predictions"; a sweep
through and past that size would test the claim on this data.

*A whole-shot-context regressor*, TCAN or UNetTransformer trained with a
full-shot field, as the fair comparator for the flow posterior and for
every family ranking of Q2.

*Retrain the simulated fiber arms*, with a curriculum from the static fiber,
visibility augmentation or a longer budget, so that Q3.1 and Q3.2 get an
answer; or rerun their arms at a low-fading point where the model learns.

*Feature engineering for the fiber gap*, aimed at the models that degrade
most (@sec-learned-gap).

*Unblind the real data.* Calibrate the simulator on the real files, score the
pre-registered predictions of @tab-real-predictions, and resolve the coil
driver's drive-scale discrepancy against the published value.

*Close the open synthetic items:* extra seeds where an ordering sits near the
seed range, the LSTM and offline-neural-operator stalls, the full noise
ladder, and an offset tolerance for the Q1.2 sign evidence.

*Two classical items.* Improve the inverse: method 1 clears its control with
the displacement supplied, so the loss is in the decode, not the model it
inverts. And explain why two acquisitions of the same speaker prefer
different calibration gains. Neither is on the path to a usable estimator;
both explain why the physical model underperforms.
= Appendix: example shots <sec-appendix-shots>

#let shots = json("/results/example_shots.json")
#let decimate = shots.at("decimate")
#let real-status = shots.at("real").at("status")
#let real-dataset = shots.at("real").at("dataset")

Each figure shows one synthetic test shot (seed 1000, nominal rig point:
$C = 0.2$, $alpha = 3$, three colors) chosen by displacement peak-to-peak at
the 10th, 50th and 90th percentile of the free-space test set, beside the
same motion through the combined-point multimode fiber arm
(@sec-q3-monitor). Rows share time: inputs,
velocity, displacement; the shaded strip is the full-rate zoom below it.
Traces above the zoom are subsampled by #decimate. The flow
curve is the encoder flow's posterior mean with its 90% band; the encoder
head is the same flow's whole-shot regressor alone (@sec-flow-methods).
Predicted displacement is integrated predicted velocity.

#fig(
  "example_shots_low",
  caption: [
    Low excursion. Look at whether any model tracks the sub-fringe motion at
    all, and whether the flow band holds the target.
  ],
) <fig-shots-low>

#fig(
  "example_shots_mid",
  caption: [
    Mid excursion. Compare the drift of integrated displacement between the
    cropped-field networks and the whole-shot encoder head and flow, and the
    fiber column against free space.
  ],
) <fig-shots-mid>

#fig(
  "example_shots_high",
  caption: [
    High excursion. Compare the zoomed fringes in free space and in fiber,
    and how closely each model's velocity follows the target through them
    in each column.
  ],
) <fig-shots-high>

*Real data:* #if real-status == "present" [
  the same layout on three real free-space shots, scored by the
  free-space-trained checkpoints.
] else [
  blinded; the panels build once the free-space acquisition is unblinded.
]

#fig(
  "example_shots_sweep",
  caption: [
    The same three shots through a static fiber of one mode (left) and of
    eight modes (right), each scored by TCAN trained on that fiber. Look at
    the fringe depth in the zoom and at TCAN's prediction thinning to spikes
    as the modes multiply.
  ],
) <fig-shots-sweep>

#fig(
  "example_shots_real",
  caption: [Real free-space shots, low to high excursion, same rows as above.],
) <fig-shots-real>

// The appendix figures float; flush them so none lands in the bibliography.
#place.flush()

#bibliography("refs.bib")
