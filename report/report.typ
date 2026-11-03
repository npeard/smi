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
carries fringes at every $lambda \/ 2$ of target travel. Lang and Kobayashi
modelled this external-cavity feedback @lang1980feedback, and three reviews
@giuliani2002selfmixing @donati2012selfmixing @taimre2015tutorial describe
the resulting self-mixing interferometry (SMI) as a sensor. It
needs no reference arm, beamsplitter or external detector, but its fringe is
not a cosine. Feedback of strength $C$ skews
it toward a sawtooth whose lean gives the direction of motion, which a
plain two-beam interferometer does not observe.

The established instrument for the same measurement is the laser Doppler
vibrometer, a two-beam interferometer that recovers direction with a
frequency-shifted reference @rothberg2017ldv. Our rig shines three lasers, at 635, 674.8 and 515 nm, onto
one mirror. The task is to recover the mirror's velocity from the three
photodiode channels, causally and accurately enough to run against live
hardware.

Learned decoders are the current alternative to fitting the signal model.
Barland and Gustave trained a convolutional network on single-channel
self-mixing fringes @barland2021cnn. Matha et al. extended it to three
channels and showed the sensor keeps tracking when some channels lose signal
entirely @matha2023multichannel, the nearest prior work to Q1. Both were
developed on the authors' own rigs, with no non-learning comparator scored on
the same shots, so this report scores learned and classical estimators side
by side.

Deployment adds a multimode fiber between laser and target. Mode coupling in
the fiber scrambles the transmitted field into speckle @ploschner2015chaos.
We infer that the light fed back to the laser is scrambled too, shot by shot,
so an estimator trained in free space sees fringes it was never shown.

Four questions organize the work, as listed in `QUESTIONS.md`: whether extra
channels help (Q1), which estimator is most accurate and robust (Q2), how to
recover motion through multimode fiber (Q3), and whether a Sagnac self-mixing
accelerometer gains from the nonlinearity (Q4). Each question is answered
analytically, then on simulated data, where the true motion is known, then on
the real acquisitions; Q4 stops at the analytic tier. The synthetic program fixes its predictions for the
real data before reading them. Questions without results carry a one-line
status in place. The answers so far:

#takeaway[
  + *Q1.1, channel count: a second wavelength helps only without noise.*
    Noise-free it cuts the error #val("q1_gain_tcn") to #val("q1_gain_tcan")
    times below the best single channel, and on Michelson one channel alone
    recovers nothing. At the nominal point the gain is at most
    #val("q1_s4_gain_max_pct")% (@fig-q1-subsets).
  + *Q2.1, accuracy floor: only with feedback does the architecture matter.*
    There TCAN is best on velocity, #val("q2_tcan_gain") times below the TCN
    in one seed, while the LSTMs and the offline neural operator do not
    train, even when retargeted to displacement (@fig-families).
  + *Q2.1.1, self-mixing nonlinearity: weak optical feedback makes the direction of motion
    observable.* Without it, as in a plain Michelson interferometer, a network
    trained on signed velocity predicts zero (@fig-exit-check). With it the
    error falls steadily with feedback strength, to #val("q3_gain_top") times
    below the Michelson error up to sign (@fig-feedback).
  + *Q2.2, noise robustness: at the nominal operating point, per-laser phase
    drift dominates.* At the nominal point every simulated imperfection, or
    knob, sits at its assumed rig value. Removing drift there cuts the error
    #val("loo_e60_cut")% at equal training length, while no electronics knob
    moves it more than #val("loo_pdbw_cut")% (@fig-noise, @fig-loo).
  + *Q2, estimators, real data: every learned model clears the analytic
    baseline.* The multi-wavelength Michelson decode's median velocity NRMSE
    is #val("real_decode_nrmse"), above the 1 that predicting zero scores, and
    even a #val("real_ridge_params")-coefficient ridge regression beats it;
    the port of the Barland network @barland2021cnn is best on both
    acquisitions (@fig-models).
  + *Q3, multimode fiber, real data: every neural model degrades behind
    fiber.* The four best free-space networks lose
    #val("real_top4_ratio_lo") to #val("real_top4_ratio_hi") times. The decode
    holds at #val("real_decode_ratio") times, one ratio over
    #val("real_decode_shots") shots, and ridge improves to
    #val("real_ridge_ratio") (@sec-learned-gap).
]

= Analytic foundation <sec-foundation>

Before training any network we ask what the signals determine at all: whether
they fix the direction of motion, and how precisely they fix the displacement.
The answers say which errors a learned estimator could remove and which no
estimator can.

== From a Michelson interferometer to self-mixing <sec-model>

We measure motion by interferometry. A mirror moves along a trajectory $d(t)$,
and we want to recover its velocity $v(t) = dot(d)(t)$, which is also the
quantity the networks are trained to output. Each laser forms its own
interferometer with the same mirror, so the three photodiode channels record
the same $d(t)$ through three different wavelengths. In the familiar Michelson
interferometer, channel $k$ reads
$
  y_k (t) = a_k (t) [1 + V_k (t) cos theta_k (t)] - 1 + n_k (t),
  quad theta_k (t) = (4 pi d(t)) / lambda_k + phi_k .
$ <eq-michelson>
The optical phase $theta_k$ is what carries the motion: the round trip to the
mirror adds $4 pi d \/ lambda_k$, so one fringe passes for every $lambda_k \/ 2$
of travel. The phase offset $phi_k$ is set by the static path length, which we
do not know, so it is drawn uniformly on $[0, 2 pi)$ for every shot. The
amplitude $a_k$ (optical power, normalized to 1) and the visibility
$0 < V_k <= 1$ (fringe contrast) scale the fringe without carrying motion, the
$-1$ centers the signal, and $n_k$ is detector noise. Only $d(t)$ is wanted;
the rest are nuisance or noise.

In self-mixing the mirror's light re-enters the laser cavity, and the laser
itself becomes the interferometer. The returning light pulls the laser's
phase, which distorts the fringe. The self-mixing signal generalizes
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
factor $alpha$ of the laser, typically 3 to 5 for diode lasers, couples its
gain to its phase and sets the skew $beta$. We stay in the weak-feedback
regime $0 <= C < 1$. There the right side of @eq-excess, as a map of
$phi_F$, has slope at least $1 - C > 0$, so each $theta$ has exactly one root
and $F$ is single-valued.

== The Michelson sign is unobservable <sec-sign>

The first question is whether the signals determine which way the mirror
moves. If they do not, no estimator can learn the sign, and a network asked
for signed velocity should give up. For the Michelson model they do not. The
map $d -> -d$, $phi_k -> -phi_k$ sends every $theta_k$ to $-theta_k$, and the
cosine is even, so every channel of @eq-michelson is unchanged. The uniform
law of $phi_k$ is also unchanged by $phi_k -> -phi_k$ modulo $2 pi$, so
integrating out the offsets gives the same likelihood for a trajectory and its
mirror image,
$
  p(y | d) = integral p(y | d, phi) p(phi) dif phi
  = integral p(y | -d, -phi) p(-phi) dif phi = p(y | -d) .
$ <eq-sign-likelihood>
The simulated drive is a sum of lines with independent uniform phases, so the
trajectory prior is symmetric too, $p(d) = p(-d)$. The posterior is then
symmetric, and the estimate with least mean squared error is
$
  E[v(t) | y] = 0 .
$ <eq-sign-zero>
A network trained with signed mean squared error on Michelson data should
therefore collapse toward zero. To ask instead whether it recovers the motion
up to the unobservable sign, we score each shot with the sign-invariant loss
$
  cal(L)_"SI" = min_(epsilon in {plus.minus 1}) "MSE"(epsilon hat(v), v) .
$ <eq-sign-invariant>
@fig-exit-check tests both predictions with one network trained three ways,
so each pair of runs differs in one factor: the loss or the fringe. The runs
train at most #val("exit_max_epochs") epochs, so only their ordering is
evidence.

#fig(
  "exit_check",
  width: 62%,
  caption: [
    Without feedback a signed-loss network learns to predict zero; with
    feedback it recovers the motion and its sign. Bars: test velocity NRMSE,
    each shot's RMSE over its velocity std, so predicting zero scores 1
    (dashed), of a causal temporal convolutional network (TCN) on three
    noise-free channels; x labels give the training loss and colour the
    scoring, where sign-invariant takes the better of $plus.minus hat(v)$ per
    shot. Michelson: feedback strength $C = 0$; self-mixing (SMI): $C = 0.2$
    and linewidth enhancement factor $alpha = 3$.
  ],
) <fig-exit-check>

== Feedback breaks the symmetry <sec-feedback>

Feedback tilts the fringe toward a sawtooth whose steep edge faces the
direction of motion, which is how self-mixing sensors read direction. This
subsection asks when the tilt breaks the symmetry of @sec-sign, and how much
evidence about the sign one shot carries.

The symmetry survives feedback when $alpha = 0$. Then $beta = 0$, and negating
@eq-excess shows that $-phi_F$ solves it at $-theta$. The root is unique, so
$phi_F (-theta) = -phi_F (theta)$, $F$ is still even, and
@eq-sign-likelihood holds for every $C$. For $C > 0$ and $alpha != 0$ the
fringe's slow and steep edges swap with the direction of motion, and the
symmetry breaks.

With a single channel the break is incomplete, because a slightly warped
phase undoes it. The warped phase below satisfies @eq-excess at $-theta''$
with root $-phi_F$, so it reproduces the fringe exactly:
$
  theta'' = phi_F + C sin(phi_F - beta) quad => quad
  F(-theta'') = F(theta), quad theta'' - theta = -2 C sin beta cos phi_F .
$ <eq-warp>
A sign-flipped trajectory, displaced by a small warp that follows the fringe,
therefore reproduces one channel without error. Only the motion prior's
penalty on that warp separates the two hypotheses. With two or more channels
each warp depends on its own $phi_F$ and $lambda_k$, so no single warped
trajectory mirrors every channel.

To size the evidence we hold the trajectory fixed and ask how well the true
signal can be told from the closest sign-flipped one,
$
  tilde(y)_k (t) = m + b F(-(4 pi d(t)) / lambda_k + phi'), quad b >= 0 .
$ <eq-flipped>
The flipped signal may choose its offset $m$, gain $b$ and phase $phi'$
freely; the gain stays non-negative because photodiode polarity is known. The
measure is the Kullback-Leibler divergence between the noise distributions of
$y_k$ and $tilde(y)_k$, in nats. For $L$ samples per shot, white noise of
std $sigma$ and uniform phase coverage, the leading order in $C$ is
$
  I_"sign" = sum_k (L C^2 E[V_k^2] alpha^2) / (4 sigma^2 (1 + alpha^2)) .
$ <eq-sign-info>
Freeing the trajectory can only lower $I_"sign"$, so it is an upper bound.

#fig(
  "sign_information",
  width: 62%,
  caption: [
    One noisy shot carries far more than one nat of sign evidence across weak
    feedback, growing as $C^2$. Upper bound on the sign evidence, the expected
    log-likelihood ratio in nats of the true signal over the closest
    sign-flipped one, for one 16384-sample shot at SNR 30 dB, visibility
    $V tilde U(0.5, 1)$, mean over 8 shots; $K = 1$ is the 635 nm channel and
    $K = 3$ all three. Markers: computed; lines: the leading order,
    @eq-sign-info.
  ],
) <fig-sign>

With the trajectory held fixed, no mirror reproduces the signals; this is
numerical, not proven. Whether some other trajectory mirrors every channel at
once, for example one that switches fringe branch at the extrema, is an open
gap.

== Fisher information per sample <sec-fisher>

Once the sign is settled, the question becomes precision: how tightly does
one sample pin down the displacement? The standard measure is the Fisher
information, the expected curvature of the log-likelihood in the unknown. Its
inverse is the smallest variance an unbiased estimator can reach (the
Cramer-Rao bound). It suits this problem because it adds over independent
channels and samples, so it says directly what each channel contributes and
where on the fringe a channel goes blind. This subsection shows that the
information vanishes at the fringe extrema and that feedback moves it onto the
steep edge.

For channel $k$ at unit amplitude with white Gaussian noise of std $sigma$,
the Fisher information about $d$ in one sample is
$
  J_k (theta) = ((4 pi V_k) / (lambda_k sigma))^2 F'(theta)^2, quad
  F'(theta) = - (sin phi_F) / (1 + C cos(phi_F + beta)) ,
$ <eq-fisher>
with $F'$ from implicit differentiation of @eq-excess. The prefactor is the
fringe's phase-to-signal gain over the noise. The factor $F'^2$ vanishes where
$sin phi_F = 0$, at the fringe extrema, where a small motion leaves the signal
unchanged (@fig-fisher). The channel noises are independent, so $K$ channels
together carry $sum_k J_k$. Because $J_k$ scales as $lambda_k^(-2)$, the
515 nm channel carries $(0.635 \/ 0.515)^2 = 1.52$ times the information of
the 635 nm channel at equal visibility and noise.

Additivity alone does not remove the blind spots. Equal-wavelength Michelson
channels with offsets spaced by $pi \/ K$ would remove them, since for $K >= 2$ their
$sin^2 theta$ terms sum to $K \/ 2$ at every phase, but the rig's channels
differ in wavelength and their offsets are not controlled. Different
wavelengths usually put the extrema of different channels at different
displacements, yet nothing prevents all channels from nearing their extrema
together, so wavelength diversity raises the average information through the
$lambda_k^(-2)$ factor without guaranteeing it in the worst case.

The bounds below need the information averaged over a uniform fringe phase.
That average has a closed form,
$
  Phi(C, alpha) = chevron.l F'^2 chevron.r_theta
  = 1 / (2 s) [1 - cos(2 beta) (1 - s) / (1 + s)], quad s = sqrt(1 - C^2) .
$ <eq-phase-factor>
Changing variables from $theta$ to $phi_F$, which is monotone for $C < 1$,
reduces it to two standard averages over a uniform angle $u$,
$
  chevron.l 1 / (1 + C cos u) chevron.r_u = 1 / s, quad
  chevron.l (cos 2u) / (1 + C cos u) chevron.r_u = (1 - s) / (s (1 + s)) .
$ <eq-phase-integrals>
The Michelson limit is $Phi = 1 \/ 2$. For $|alpha| > 1$, $cos 2 beta < 0$, so
feedback raises the average: at $alpha = 3$, $Phi = #val("phase_factor_c02")$
at $C = 0.2$ and #val("phase_factor_c07") at $C = 0.7$.

#fig(
  "fisher_phase",
  width: 62%,
  caption: [
    Feedback concentrates the information on the steep fringe edge but keeps
    the blind spots at the extrema. The squared fringe slope $F'(theta)^2$,
    the phase-dependent factor of @eq-fisher, over one fringe at
    $alpha = 3$. Dashed: its phase averages $Phi(C, alpha)$ of
    @eq-phase-factor.
  ],
) <fig-fisher>

== Causal and offline bounds on velocity error <sec-crb>

Information per sample does not bound the velocity error by itself, because
velocity is read across many samples and depends on how the motion evolves.
Combining @eq-fisher with a model of that evolution gives two Bayesian
Cramer-Rao bounds: a causal one, for an estimator that sees only past samples,
and an offline one, for an estimator that sees the whole shot. Their role here
is a floor: if a learned estimator sits far above them, information is not
what limits it.

The bounds need a linear motion model, so we fit a Gauss-Markov surrogate to
the simulator's motion. Its state $x_n = (d, v, dot(v))$ at sample $n$ evolves
as
$
  x_(n+1) = A x_n + w_n, quad w_n tilde cal(N)(0, Q) ,
$ <eq-prior>
driven by white noise $w_n$ through one real pole and one resonance, and
discretized exactly at the sample rate. The fit matches the in-band velocity
line powers of 256 seeded trajectories to #val("prior_psd_rms_db") dB rms and
reproduces their velocity and displacement variances.

Each sample adds the same information about $d$ on average. Because
$theta_k mod 2 pi$ is uniform whatever $d$ is, the average of @eq-fisher over
phase and visibility is exact, with no Monte Carlo:
$
  macron(J) = sum_k E[V_k^2] Phi(C, alpha) ((4 pi) / (lambda_k sigma))^2 .
$ <eq-info-avg>
The causal bound is the posterior Cramer-Rao bound of @tichavsky1998pcrb. For
the linear-Gaussian model of @eq-prior its information matrix $J_n$ obeys
$
  J_(n+1) = (Q + A J_n^(-1) A^T)^(-1) + macron(J) c_d c_d^T ,
$ <eq-pcrb>
where $c_d$ selects $d$ from the state. This is the covariance recursion of a
Kalman filter that measures $d$ linearly with information $macron(J)$ per
sample. The offline bound is the Bayesian (Van Trees) bound
@vantrees1968detection: the block-tridiagonal prior information plus
$macron(J) c_d c_d^T$ on each block, whose inverse diagonal blocks are the
surrogate's smoother covariances. Averaging the information before inverting
it is safe, because by Jensen's inequality
$
  (E J)^(-1) <= E[J^(-1)] ,
$ <eq-jensen>
so the averaged bound is valid and looser than one conditioned on the phases.

#fig(
  "velocity_bounds",
  width: 62%,
  caption: [
    Three channels multiply the information per sample by
    #val("info_ratio_k3_k1") but the causal velocity bound only by a factor
    #val("causal_ratio_k3_k1"), and raising $C$ from 0 to 0.9 multiplies it by
    #val("causal_ratio_c09_c0"). Stationary RMS velocity bounds at SNR
    30 dB, $V tilde U(0.5, 1)$, $alpha = 3$, for $K$ channels: $K = 1$ is
    635 nm, $K = 2$ adds 674.8 nm and $K = 3$ adds 515 nm. Causal (past samples
    only): @eq-pcrb after convergence; offline (whole shot): the
    smoothing bound at mid-record.
  ],
) <fig-bounds>

The causal bound scales close to the inverse fourth root of $macron(J)$
(@fig-bounds). It sits #val("real_gap") times below the causal TCN's error of
#val("real_tcn_rmse") $mu$m/ms on the real free-space test split
(@sec-learned-table). Within its own model the bound is nearly attained: in
`tests/test_bounds.py` an extended Kalman filter given the true phases must
score within twice the causal bound. So that gap measures what the bound
leaves out as much as any estimator's shortfall.

The bounds are exact for a simpler model than the simulator, so they are not
a floor for it. They use the fitted surrogate rather than the simulator's
finite sum of drive lines, which a line-aware estimator could exploit. They
know $phi_k$ and $V_k$ and set $a_k = 1$. They omit amplitude, contrast and
phase drift, DC offset, the photodiode low-pass and the ADC. And they are
local: they see neither the global sign nor the fringe order, so at $C = 0$ or
$alpha = 0$ they compare only with a sign-invariant error.

= Methods <sec-methods>

== Data, training and evaluation <sec-methods-synthetic>

Every learned result below compares cells that differ in one factor: the
feedback, a noise knob, the channel subset or the model family. That
comparison is fair only because the data, the training budget and the scoring
are otherwise shared. This section states them.

The simulator draws each shot's trajectory as a sum of drive lines from 1 Hz
to 1 kHz with independent uniform phases and an amplitude drawn per shot from
$U(0, 1)$. It renders the three channels of @eq-signal over
#val("m_shot_length") samples at 488 kHz, a 33.6 ms shot. Each imperfection
the simulator models is a knob: white noise, ADC bits, photodiode bandwidth,
visibility, DC offset, and amplitude, contrast and phase drift. The noise-free
configuration turns every knob off, and the nominal point sets each to its
assumed rig value. Every model maps the three channel
signals to velocity at every sample. The reference model is a causal temporal
convolutional network (TCN) of 44k parameters that sees the
#val("q2_tcn_field") most recent samples. @tab-splits gives the size and
origin of each data split.

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
    reused, so the amount seen grows with the epochs trained (text).
  ],
) <tab-splits>

#assert(val("m_full_length_is_no"), message: "a family other than the offline neural operator now trains on whole shots")

Training uses Adam at a peak learning rate of #val("m_lr"), under cosine
annealing with warm restarts. Two cycle lengths appear: $T_0 = 30$ epochs,
capped at 100, for the noise and channel grids, and $T_0 = 90$, capped at 150,
for the feedback curve and the family comparison. A run stops when its
validation error has not improved by #val("m_min_rel_delta_pct")% in
#val("m_patience") epochs, but never before epoch #val("m_grace"); that grace
lets slow starters leave the zero-output plateau. Under $T_0 = 30$ the
#val("m_t30_cells") cells trained #val("m_t30_epochs_lo") to
#val("m_t30_epochs_hi") epochs, so each saw #val("m_t30_kshots_lo") k to
#val("m_t30_kshots_hi") k distinct shots. Under $T_0 = 90$ the
#val("m_t90_cells") cells trained #val("m_t90_epochs_lo") to
#val("m_t90_epochs_hi") epochs, #val("m_t90_kshots_lo") k to
#val("m_t90_kshots_hi") k shots. The #val("q3_a0_epochs")-epoch runs are
$alpha = 0$ controls, which cannot learn the sign, so their best checkpoint is
their first. Each cell is
one training seed unless stated. Three seeds of the noise-free base span
#val("q2_seed_range_pct")% of its mean error, and that range is the resolution
for ranking any two cells. Every cell ran on Apple #val("m_device") in
float#val("m_precision"), with Python 3.14 and torch 2.15.0.

Each error is the RMSE of a shot, averaged over the test shots. Velocity RMSE
is in $mu$m/ms. NRMSE divides each shot's RMSE by that shot's velocity std
before averaging, so it weights slow shots more. Displacement RMSE integrates
the predicted velocity over the shot and removes each start offset, as the
training code's displacement loss does; it is in $mu$m. The zero predictor,
always predicting zero velocity, scores #val("q1_zero") $mu$m/ms on the
noise-free three-channel test set at $C = 0.2$.

#fig(
  "spectral_mismatch",
  width: 75%,
  caption: [
    Velocity, not displacement, has its power where the fringes do. Hann-windowed
    power spectral densities, each divided by its peak, of the two targets and
    each channel's fringe frequency $2 |v| \/ lambda_k$, averaged over
    #val("spec_shots") shots at $C = 0.2$, $alpha = 3$: (a) the nominal
    point; (b) noise-free, with the same trajectories.
  ],
) <fig-spectral>

The networks predict velocity rather than displacement because velocity
matches the information in the signal (@fig-spectral). Each fringe frequency
follows $|v|$, while displacement weights the velocity spectrum by
$1 \/ f^2$ and puts #val("spec_x_low_pct")% of its power below 100 Hz, against
#val("spec_v_low_pct")% for velocity. Below the lowest resolved bin,
#val("spec_bin_hz") Hz, drive content enters a shot only as a trend.

== Likelihood versus fit <sec-likelihood>

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
most, so every fit below states its interval.

== The two measurements <sec-measurements>

Both run on shots from each acquisition at 16384 samples per shot, 488.28 kHz,
decoded on a GPU (`cuda:0`). They do *not* run on the same number of shots.
Method 1 is a closed-form least-squares solve, about a minute for 1000 shots
on a CPU, so it runs at N = 1000. Method 2's Viterbi decode costs about
5.85 s per shot, so it stays at N = 200. Every table below names its own N;
none of them can be read under the other's. Shot-level metrics are summarized
by their median unless stated otherwise.

=== Method 1: forward-model residual <sec-method1>

Displacement is known from the drive voltage, so the interference signal it
predicts is known up to an amplitude and phase. Both enter linearly once the
cosine is expanded into its cosine and sine components, so the fit is a
three-parameter least-squares solve -- cosine, sine, constant -- with a closed
form. No optimizer, no initial guess, no convergence to check, and a global
optimum by construction.

What the solve does not settle is *over what interval*. Fitting one amplitude,
phase and offset across the whole 33.6 ms record assumes those three
quantities are constant for 33.6 ms, and in this data they are not. The fit is therefore reported two ways:
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
artifact. Where a number is instead taken at a fixed $W$, the text says so.

A shorter window fits more of anything, so a windowed $R^2$ reported alone is
not a measurement. Every windowed number below therefore appears with a
*scrambled-theta control*: the same $W$, the same window count and the same
three parameters per window, but with the displacement taken from a different
shot. Matching $W$ between the fit and its control is what makes the pair a
comparison -- a fit with more fitting freedom than its control would be
scored on the freedom. The control is how much $R^2$ the fit buys from that
flexibility, and what the model earns is the gap between the two.

=== Method 2: multi-wavelength inverse <sec-method2>

The actual baseline. Given only the three photodiode signals, resolve fringe
ambiguity using the wavelength diversity across 635, 675 and 515 nm, decode
displacement on a 1601-point grid, then differentiate for velocity. Scored on
the same quantity the network predicts, in physical units: velocity in um/ms
and displacement in um, which is what the repository stores and what the
training loss is computed in.

The overall sign of the displacement is not scored, because under the
Michelson model it is unobservable (@sec-sign).

== How to read these numbers <sec-read>

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

== Predictions for the real data <sec-real-predictions>

// Published real-data numbers, defined once:
// the free-space causal TCN's test velocity RMSE (@sec-learned-table, mean
// over two seeds) and the free-space median true velocity RMS
// (@sec-r-method2), which is the zero predictor's per-shot error.
#let real-tcn-fs = decimal("0.1116")
#let real-v-rms-fs = decimal("0.373")
#let real-ratio = calc.round(real-tcn-fs / real-v-rms-fs, digits: 2)

#assert(
  float(val("pred_base_ratio")) < float(real-ratio)
    and float(real-ratio) < float(val("pred_c09_ratio")),
  message: "the real free-space ratio no longer lies between the noise-free base and every nominal-point variant",
)

The synthetic results are useful only if they say something checkable about
the rig. This section fixes what the synthetic program predicts for the real
acquisitions before they are read (@tab-real-predictions). Each prediction
names what it rests on and the observation that would refute it. Absolute
synthetic errors should not transfer, because the rig's $C$, $alpha$, SNR and
drift are unknown and its motion prior and training budget differ. The
predictions are therefore ratios and orderings, against the real bars of the
real-data results table (@sec-learned-table): the causal TCN at
#real-tcn-fs and 0.2540 $mu$m/ms and Barland at 0.0814 and 0.1643 $mu$m/ms,
in free space and behind multimode fiber. Those bars are two-seed means,
while the synthetic #val("q2_seed_range_pct")% seed range comes from one seed.
P6 therefore scores every falsifier with $s$, the seed range it measures on // math-ok: stage label
the real runs at the same two seeds.

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
    [`calibrate_sim.py` reads feedback on the free-space file],
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
    [No other offline family beats Barland's 0.0814 $mu$m/ms],
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
(@tab-q1-count). The reason is structural: with one unknown phase per
channel the Fisher matrix of $(v, phi_1, ..., phi_N)$ is block-arrowhead, so
the profiled velocity information is the sum of the per-channel profiled
values, each weighted by $lambda_k^(-2)$ as in @eq-fisher. The table is
`bounds.fisher_from_shots`, profiled over the phases, on the noise-free
detector at 30 dB SNR, $alpha = 3$, #val("ch_fisher_shots") shots, averaged
over #val("ch_count_seeds") seeds. Each seed draws one set of trajectories and
phases for all six colors and reuses it for every $N$ and both $C$, so each set
equals the sum of its single channels exactly and the feedback column is a
paired comparison. Adding 405 nm as the sixth channel raises the information
by #val("ch_info_jump_405_pct")%; adding 785 nm as the fourth, by
#val("ch_info_jump_785_pct")%. Feedback at $C = 0.2$ should scale every entry
by $Phi(0.2, 3) \/ Phi(0, 3)$ of @eq-phase-factor, a
#val("ch_fb_pred_pct")% rise. The paired ratio is
#val("ch_fb_meas_pct")% $plus.minus$ #val("ch_fb_se_pct")% over seeds and
entries, and a single entry scatters by #val("ch_fb_spread_pct")% from seed
to seed.

#assert(val("ch_additive_exact"), message: "the common-random-number channel sets no longer equal the sum of their singles")

#figure(
  table(
    columns: 5,
    align: (center, center, right, right, right),
    table.header([$N$], [added $lambda$ (nm)], [$J$, $C = 0.2$], [$J$, $C = 0$], [bound ratio]),
    ..val("ch_count_table").map(r => (
      [#r.n], [#r.lam], [#r.info02], [#r.info0], [#r.bound02],
    )).flatten(),
  ),
  caption: [
    Velocity information adds over channels. $J$: profiled per-sample
    velocity information, in $10^7 ("um/ms")^(-2)$, of the first $N$ colors of
    the list, mean over seeds. Bound ratio: $sqrt(J_1 \/ J_N)$ at $C = 0.2$, the CRB on velocity
    relative to 635 nm alone.
  ],
) <tab-q1-count>

=== Synthetic

_Status: the R1 channel-count grid is running; the text below predates it._

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
at the nominal point the gain shrinks to at most #val("q1_s4_gain_max_pct")%
(@fig-q1-subsets). The rig has three lasers, so the practical question is
what the second and third buy. @eq-fisher predicts that a channel goes blind at its fringe
extrema, and incommensurate wavelengths put the extrema of different channels
at different times. Extra channels should therefore help most near one
channel's extrema, including velocity turning points, where the fringes stall,
and elsewhere by the modest factor of @fig-bounds. To test this we train the
TCN and TCAN, which fuses per-channel TCN encodings by attention across
channels at the same 44k parameters, on every subset of the three channels. We score them
noise-free and, for the TCN, at the nominal point across SNR.

#fig(
  "q1_subsets",
  caption: [
    Noise-free at $C = 0.2$, a second wavelength cuts the error
    #val("q1_gain_tcn") times (TCN) and #val("q1_gain_tcan") times (TCAN)
    below the best single channel, 515 nm, and on Michelson one channel
    recovers nothing; at the nominal point the gain is at most
    #val("q1_s4_gain_max_pct")%. (a) Noise-free test velocity RMSE of each
    channel subset, one seed per cell, Michelson scored sign-invariant and
    $C = 0.2$ signed, $alpha = 3$; oracle: the classical
    inverse fit, given the true excursion. (b) The TCN at the nominal point
    with SNR varied: the lowest error among single channels, among pairs, and
    all three.
  ],
) <fig-q1-subsets>

The single-channel Michelson failure is unexplained. The global mirror of
@sec-sign cannot cause it, because the sign-invariant score forgives the
mirror. One untested explanation is that
for one channel the mirror also acts locally. Near a time $t_0$ the fringe is
$
  cos(psi_0 + (4 pi [d(t) - d(t_0)]) / lambda), quad
  psi_0 = (4 pi d(t_0)) / lambda + phi ,
$ <eq-local-mirror>
which is unchanged when the local offset $psi_0$ and the displacement
increment are both negated. The offset is uniform for any window, so a
windowed estimator cannot fix the sign locally. The whole-shot oracle also
fails, which no window argument explains, so this remains a gap.

#fig(
  "q1_binned",
  caption: [
    The multi-channel gain does not concentrate at the fringe extrema but
    grows with speed: noise-free, the 515 nm error over the three-channel
    error is #val("q1_bin_gain_near") near the 515 nm extremum against
    #val("q1_bin_gain_far") midway, and it vanishes in the slowest $|v|$
    decile (c). Pooled test RMSE of the TCN ($C = 0.2$, $alpha = 3$, one seed)
    in #val("q1_bin_width")-fringe bins of the distance from $phi_F$ to the
    nearest extremum, of the 515 nm channel (solid) and of the channel
    farthest from one (dotted): (a) noise-free, (b) the nominal point at
    30 dB. (c) The same in $|v|$ deciles.
  ],
) <fig-q1-binned>

The extremum prediction fails (@fig-q1-binned). The 30 dB panel, the proper
test of a noise-limited claim, rises in the predicted direction but far less
than a loss of all information would. One untested reading is that an
estimator seeing a window interpolates across an extremum, as the prior does
in the bounds of @sec-crb. The growth with speed runs against the
turning-point clause of the prediction.

At the nominal point the error is set by phase drift more than by channel
count. The gain left at
30 dB is several times the nominal seed range and comes from pairing 515 nm
with another channel (#ref(<fig-q1-subsets>)b). The classical oracle decodes under the Michelson model and scores worse than zero
at the nominal point, so it is not a reference there.

These runs are one seed each at the 30-epoch budget, ranked against a seed
spread measured on three channels, and the nominal point ran for the TCN only.

=== Real data

The Q1 gain should fall between its two synthetic limits (@sec-q1-count). // math-ok: question label
Three channels beat the best single channel by a factor
#val("pred_triple_free") noise-free and #val("pred_triple_30") at the
nominal 30 dB point, for the TCN (@fig-q1-subsets). The nominal
end is within the seed range of parity, so a ratio there cannot be told
from 1. Ratios between $1 - s$ and #val("pred_triple_30"), or between
#val("pred_triple_free") and #val("pred_triple_free") $(1 + s)$, neither
confirm nor falsify. Prediction 6 expects the real ratio nearer the
noise-free end the further prediction 3 places the rig from nominal. On
another rig, @matha2023multichannel found its best single channel ahead of
all three, so a ratio below 1 is a live outcome.

Predictions 5 and 7 rest on thinner evidence. 515 nm leads in the
$lambda_k^(-2)$ order of @eq-fisher and in every synthetic cell, but its
noise-free TCN lead is at the edge of the seed range. 515 nm alone recovers
the motion at $C = 0.2$, while on Michelson one channel scores at the zero
predictor (@fig-q1-subsets). Prediction 7 scores it with the
#val("q1_stall_tol_pct")% stall tolerance of the runner. A real 515 nm
channel at the zero predictor would mean feedback too weak for one channel to
read the sign.

== Q1.2: several colors against a one-color array <sec-q1-colors> // math-ok: question label

=== Analytic

A one-color array with quadrature offsets reads the sign without feedback;
distinct colors do not, and local precision barely separates the two
(@tab-q1-colors). We compare $N$ colors (independent phases) against $N$
copies of 635 nm sharing one unknown phase, with known offsets $0, pi \/ 2$
for $N = 2$ and $2 pi k \/ 3$ for $N = 3$, or all zero. Local Fisher
information follows $sum_k lambda_k^(-2)$: colors give
#val("ch_colors_vs_array_n3_pct")% for $N = 3$, where 515 nm enters, and
#val("ch_colors_vs_array_n2_pct")% for $N = 2$. The sign evidence of
@eq-sign-info separates them. At $C = 0$ it is zero for colors and for
zero offsets, as the Michelson symmetry of @sec-sign requires, while the
quadrature array carries
$#val("ch_sign_quad_n3_c0").mantissa times 10^(#val("ch_sign_quad_n3_c0").exponent)$
nats per shot at $N = 3$. That is #val("ch_sign_ratio_n3") times what
feedback alone gives the three colors at $C = 0.2$. This does not contradict
the $C^2$ leading order of @eq-sign-info, which holds for independent
channels, each fitting its own mirror phase. Sharing one phase with known
offsets, the mirror needs one $phi'$ with $phi' = -phi_0 - 2 o_k$ for every
$k$ at once, which distinct offsets forbid even at $C = 0$; this is quadrature
detection. The sign evidence is the fixed-trajectory upper bound of
`bounds.sign_information`, over #val("ch_sign_shots") shots. It assumes the
offsets known exactly; we have not computed how fast it degrades with an
offset error.

#figure(
  table(
    columns: 5,
    align: (center, center, left, right, right),
    table.header([$C$], [$N$], [configuration], [$J$], [sign (nats/shot)]),
    ..val("ch_colors_table").map(r => (
      [#r.c], [#r.n], [#r.arm.replace("_", " ")], [#r.info], [#r.sign],
    )).flatten(),
  ),
  caption: [
    Quadrature offsets, not colors, carry the sign at $C = 0$. Colors: the
    first $N$ of 635, 674.8, 515 nm with independent phases. Array: $N$ times
    635 nm sharing one phase, offsets quadrature-spread or zero. $J$ as in
    @tab-q1-count; same detector, 30 dB, $alpha = 3$.
  ],
) <tab-q1-colors>

=== Synthetic

_Status: the R2 colors-against-array grid is running._

=== Real data

_Status: not run; needs a same-color array on the rig._ The committed
prediction: at matched channel count and visibility, a 635 nm array with
offsets held near quadrature recovers the sign at the rig's weakest feedback,
where the three colors do not, and its local precision differs from the colors' by the
#val("ch_colors_vs_array_n3_pct")% of @tab-q1-colors. It is
falsified if the array's sign error at that feedback is no lower than the
colors', or if collapsing its offsets to zero leaves its sign error unchanged.

= Q2: which estimator <sec-q2> // math-ok: question label

== Q2.1: accuracy floor on noise-free data <sec-q2-floor> // math-ok: question label

=== Analytic

_Status: no per-model prediction; the bounds of @sec-crb are the floor._

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
above the classical oracle; with feedback TCAN is best on velocity, and the LSTMs and
the offline neural operator do not train under this protocol, on either
target (@fig-families). Every
result so far uses the TCN, so we ask whether the architecture is the limit.
Each family is matched to the TCN's 44k parameters and trained as the
feedback curve was, with one 90-epoch cycle on noise-free three-channel data:
the signed loss at $C = 0.2$, $alpha = 3$, and the sign-invariant loss on
Michelson. Causal families see only the past; offline ones see both sides. Context is matched at #val("q2_matched_lo") to
#val("q2_matched_hi") samples for the convolutional, transformer and causal
neural-operator families. The recurrent families (LSTM and the state-space
model Mamba) see unbounded context, the offline neural operator sees the whole
shot, and the MLP and Barland CNN average overlapping windows spanning
#val("q2_mlp_span") and #val("q2_barland_span") samples.

#fig(
  "model_families",
  width: 70%,
  caption: [
    Without feedback (Michelson, hollow) every family that trains, except
    the offline neural operator, lies within the #val("q2_seed_range_pct")%
    seed range of the TCN; with feedback TCAN is best on velocity,
    #val("q2_tcan_gain") times below the TCN in one seed, and both LSTMs and
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

Retargeting the stalled families to predict displacement directly does not
make them train usefully (#ref(<fig-families>)b). We retrain both LSTMs and
the offline neural operator on the displacement target, with a TCN control,
on the same sims, budget and test shots. With feedback all four stall, and without it the
LSTMs train but stay far behind the velocity-trained TCN. The TCN is not a
clean control for the stall. Its #val("q2_tcn_field")-sample field cannot
represent displacement, which depends on the whole history, so only the
LSTMs and the offline neural operator are informative; that the context
causes the stall is untested. Mamba was not retargeted: it already trains on
velocity (#val("s1d_mamba_velocity") $mu$m/ms at $C = 0.2$), and these runs
give no reason to expect displacement to help it.

#assert(val("disp_lowest_named"), message: "the three lowest families on displacement at C = 0.2 are no longer the two offline transformers and TCAN, tied with each other and below the TCN")

Displacement RMSE ranks the families differently from velocity RMSE
(@fig-displacement). Displacement is what the instrument reports, and it
weights the velocity error spectrum by $1 \/ f^2$ (@fig-spectral), so slow
error counts for more than in velocity RMSE. We rescore the checkpoints and
frozen test shots of @fig-families without retraining. Three seeds of the
noise-free base span #val("disp_seed_range_pct")% of its mean displacement
error, against #val("q2_seed_range_pct")% for velocity, and that wider range
decides which families are resolved. Why integration reorders the families
is untested.

#fig(
  "displacement_rmse",
  width: 80%,
  caption: [
    Displacement does not keep the families' velocity ranking. With
    feedback, TCAN, best on velocity, ties the offline PatchTransformer and
    UNet-Transformer at #val("disp_lowest_lo") to #val("disp_lowest_hi")
    #box[$mu$m], against the TCN's #val("disp_tcn") #box[$mu$m]. (a) The TCN
    against $C$, as in #ref(<fig-feedback>)a. (b) Displacement against velocity RMSE for every trained
    family of @fig-families, one seed each: hollow on Michelson,
    sign-invariant; filled at $C = 0.2$, signed.
  ],
) <fig-displacement>

Each cell is one seed at one budget, noise-free, and budgets differ:
#val("q2_early_n") trained cells stopped by patience before the restart at
epoch #val("q2_restart"), and #val("q2_second_n") overran into a second
cycle. The one gain well outside the seed range is TCAN over the TCN at
$C = 0.2$; more seeds of both would test it.

// Tabular cells, read straight from the runner's records. The document rule
// does not list synthetic_results.json, but synthetic_values.json depends on
// it, so a changed artifact still rebuilds this table.
#let syn-records = json("/results/synthetic_results.json").at("datasets").values().flatten()
#let tab-cell(prefix, sim) = syn-records.find(r => (
  r.at("sim_name", default: none) == sim and r.at("cell_key").starts-with(prefix + "__")
))
#let tab-metric(r, si) = r.at("metrics").at(
  if si { "velocity_nrmse_sign_invariant" } else { "velocity_nrmse" },
).at("mean")
// The best network on the same frozen test set, scored the same way.
#let tab-best-net(sim, si) = {
  let ds = tab-cell("ridge-w91", sim).at("dataset")
  let nets = syn-records.filter(r => (
    r.at("dataset", default: none) == ds
      and not r.at("cell_key").starts-with("ridge-")
      and not r.at("cell_key").starts-with("lightgbm-")
  ))
  calc.min(..nets.map(r => tab-metric(r, si)))
}
// (sim name, label, scored sign-invariant): Michelson cannot be scored
// signed, because the sign is unobservable there (@sec-sign).
#let tab-sims = (
  ("michelson", [Michelson, $C = 0$], true),
  ("smi-a3-feedback_c=0.05", [$C = 0.05$], false),
  ("smi-a3-feedback_c=0.1", [$C = 0.1$], false),
  ("smi-a3", [$C = 0.2$], false),
  ("smi-a3-feedback_c=0.4", [$C = 0.4$], false),
  ("smi-a3-feedback_c=0.7", [$C = 0.7$], false),
  ("smi-a3-feedback_c=0.9", [$C = 0.9$], false),
  ("smi-a5", [$C = 0.2$, $alpha = 5$], false),
)
#let tab-rows = tab-sims.map(((sim, label, si)) => (
  label: label,
  si: si,
  ridge: tab-metric(tab-cell("ridge-w91", sim), si),
  lgbm: tab-metric(tab-cell("lightgbm-w91", sim), si),
  net: tab-best-net(sim, si),
))
#let tab-field = tab-cell("ridge-w91", "smi-a3").at("receptive_field")
#let tab-feedback = tab-rows.slice(1, 7)
#let tab-ratios = tab-feedback.slice(2).map(r => r.lgbm / r.net)
#let f2(x) = {
  let (whole, ..frac) = str(calc.round(x, digits: 2)).split(".")
  whole + "." + (frac.at(0, default: "") + "00").slice(0, 2)
}
#assert(tab-rows.all(r => calc.abs(r.ridge - 1) < 0.01), message: "ridge now scores away from NRMSE 1 on some sim")
#assert(
  tab-feedback.windows(2).all(((a, b)) => b.lgbm < a.lgbm + 0.01),
  message: "LightGBM's NRMSE no longer falls with C",
)
#assert(tab-rows.all(r => r.lgbm > r.net), message: "LightGBM now matches the best network on some sim")

Ridge regression and LightGBM, the two tabular estimators of the real-data
comparison, do not approach the networks on any simulation
(@tab-tabular-synthetic). Ridge scores the zero predictor's NRMSE of 1 on
every sim. LightGBM improves with $C$, from #f2(tab-feedback.first().lgbm) at
$C = 0.05$ to #f2(tab-feedback.last().lgbm) at $C = 0.9$, flat above
$C = 0.7$, but stays
#f2(calc.min(..tab-ratios)) to #f2(calc.max(..tab-ratios)) times behind the
best network from $C = 0.2$ up. Both see three channels through
91-sample windows labelled at their centre, so each output depends on a
#[#tab-field]-sample two-sided span, and both are offline. Each fits
#tab-cell("ridge-w91", "smi-a3").at("config").at("n_train_shots") training
shots, seed-disjoint from the frozen test sets the networks score on.

#figure(
  {
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

=== Real data

The Q2 predictions claim orderings only, because the synthetic family ranks // math-ok: question label
already transferred imperfectly once. On the real data Barland beats SCNN by
15% (@sec-learned-table), where the synthetic cells tie them
(@fig-families). TCAN's #val("q2_tcan_gain")-fold gain over the TCN was
measured noise-free; under noise, the Q1 multi-channel gain shrank from // math-ok: question label
#val("q1_gain_tcn") to #val("q1_s4_30_gain_pct")%, so prediction 8 runs from
parity to that factor. A ratio between $1 - s$ and 1, or above
#val("q2_tcan_gain"), neither confirms nor falsifies it. The LSTMs stalled on
both simulations, with and without the grace period. Prediction 9 applies
only to the same protocol of default initialization, signed loss and
patience. Mamba has no prediction, because its $C = 0.2$ cell, which trains, ran only
after these predictions were fixed.

Prediction 10 covers SCNN, the MLP, both transformers, the offline neural
operator and the bidirectional LSTM. It rests on the published real ranking,
where Barland leads every configuration on both acquisitions
(@sec-learned-table), and not on the synthetic ties. That ranking includes
only SCNN and the MLP, so for the other four families it is an extrapolation.

=== Q2.1.1: does the self-mixing nonlinearity help? <sec-q2-nonlinearity> // math-ok: question label

==== Analytic

_Status: the prediction is @sec-sign and #ref(<sec-feedback>)._

==== Synthetic

#assert(val("q3_monotone"), message: "the 90-epoch C curve is no longer monotone")
#assert(val("q3_t30_c04_below"), message: "the 30-epoch C = 0.4 inversion is gone")

With feedback a signed estimator reads the direction of motion, and its
error falls steadily with $C$ up to 0.9, far below Michelson's error even
when Michelson is forgiven the sign (@fig-feedback). @sec-feedback predicts the
direction can be read but not how fast the error falls with $C$, or whether a
strongly tilted fringe is harder to decode. So we train the same causal TCN,
with the signed loss, on noise-free data at six values of $C$, and compare
each with Michelson scored sign-invariantly, the best a Michelson estimator
can do.

#fig(
  "feedback_curve",
  caption: [
    With feedback ($C > 0$, $alpha != 0$) a signed-loss network reads the
    direction of motion, and its error falls steadily with $C$, to
    #val("q3_gain_top") times below Michelson's sign-invariant RMSE at
    $C = #val("q3_c_top")$. (a) Test velocity RMSE in $mu$m/ms and (b) NRMSE,
    each shot's RMSE over its velocity std so the zero predictor scores 1,
    averaged over shots, of the causal TCN on three noise-free channels at
    $alpha = 3$, one seed; the 30-epoch curve dips at $C = 0.4$ because that
    run trained #val("q3_t30_c04_epochs") epochs, past its first restart. (c)
    Sign-invariant RMSE pooled in $|v|$ deciles for the 90-epoch Michelson,
    $C = 0.2$ and $C = 0.9$ cells.
  ],
) <fig-feedback>

RMSE and NRMSE disagree at weak feedback because they weight different shots. RMSE
pools samples, which fast motion dominates, and Michelson's error grows in
proportion to $|v|$ (#ref(<fig-feedback>)c): a wrong local direction costs
more the faster the mirror moves. NRMSE divides by each shot's velocity std,
which lets slow, low-amplitude shots dominate. There the estimator's own
floor in the slowest decile sets the score.

The sign is resolved on nearly every shot, since signed and sign-invariant
RMSE coincide (#ref(<fig-feedback>)a). The controls behave as @sec-feedback
predicts.

The curve holds for a fixed budget, and its values still move with it. One
#val("q3_t270_cycle")-epoch cycle at $C = 0.2$ reached #val("q3_c02_t270")
$mu$m/ms, below the 90-epoch value, so convergence is not demonstrated. Within
that caveat, the operating target is the strongest weak feedback sampled. On
the rig, $C$ is set by how much light returns into the laser: target
reflectivity or neutral-density attenuation in the return path, focusing and coupling
efficiency, and the external cavity length. Which of these is practical to
change is untested.

==== Real data

The published bars already show that the rig breaks the global sign symmetry
of @sec-sign. Those runs trained with a signed loss, the plain `MSELoss` of
`lit_module.py` at commit `556ab18`. On Michelson data that loss leaves the
TCN at the zero predictor (@fig-exit-check). The free-space TCN instead
scores #real-ratio of the median true velocity RMS of #real-v-rms-fs
$mu$m/ms (@sec-r-method2), which is the zero predictor's
per-shot error. That median comes from a different 200-shot subset, so the
ratio is approximate and likely a ceiling. Prediction 1 holds the sign
resolved on nearly every test shot, as in every 90-epoch cell of
@sec-q2-nonlinearity. Its interval widens the #val("q3_sign_gap_pct")% noise-free
gap to 10% for noise, which no cell measured.

Feedback with $C > 0$ and $alpha != 0$ is one way to break the symmetry
(@sec-feedback). Phase offsets that repeat from shot to shot, an amplitude
that depends on position, or drive crosstalk into a photodiode would also
break it, and the simulator models none of them. Prediction 2 therefore
requires the calibration to read the feedback the sign needs. Drive crosstalk
and photodiode nonlinearity also put power in the second fringe harmonic that
`calibrate_sim.py` reads, so prediction 2 can pass for the wrong reason. Its
threshold, $C = 0.05$, is the weakest feedback sampled in @fig-feedback, and
it already resolved the sign in RMSE.

The calibration returns nearest-grid point estimates with no uncertainty, so
prediction 2's bars carry its round-trip error (`tests/test_calibrate_sim.py`).
At $D = 58$ $"rad s"^(-1\/2)$ and $C = 0.2$ it under-reads $C$ by up to 32%,
so the falsifier sits at $C < 0.03$. Its bias at $C = 0.05$ is untested, and
a reading between 0.03 and 0.05 neither confirms nor falsifies. An $alpha$
below 0.5, a sixth of the nominal 3, counts as consistent with 0.

== Q2.2: robustness to noise <sec-q2-noise> // math-ok: question label

=== Analytic

_Status: no prediction beyond the bounds of #ref(<sec-crb>)._

=== Synthetic

#assert(val("noise_ranking"), message: "the noise ladder no longer ranks phase drift, then SNR, then the rest")
#assert(val("noise_e60_phase_monotone"), message: "at equal length the phase-drift error no longer rises monotonically with drift")
#assert(val("noise_bound_no_power"), message: "the noise-free base no longer sits over 5 times the 30 dB causal bound")
#assert(val("noise_electronics_flat"), message: "an ADC or PD-bandwidth row other than 8 bits or 3 kHz now leaves the base spread")
#assert(val("loo_electronics_max_pdbw"), message: "removing white noise or the ADC now moves the nominal error more than removing the PD bandwidth limit")
#assert(val("noise_dc_low_named"), message: "the DC-offset rows below the base are no longer the +/-0.3 and +/-1 rows")
#assert(val("loo_e60_outside"), message: "at equal length, removing phase drift no longer cuts the nominal error by more than the nominal seed spread")
#assert(val("loo_e60_same_ranking"), message: "at equal length, the base, no-phase-drift and nominal cells no longer rank in that order")

At the nominal operating point, per-laser phase drift dominates the error:
removing it cuts the error #val("loo_e60_cut")% at equal training length,
while no electronics knob moves it more than #val("loo_pdbw_cut")%
(@fig-loo). A real rig adds detector noise and drifts to the fringes, and a
hardware fix targets one at a time. Two designs ask which costs the most, all
at $C = 0.2$, $alpha = 3$, with the 30-epoch TCN. The ladder switches on one knob at a time from the noise-free
base (@fig-noise). The leave-one-out switches one knob off at a time from the
nominal point, where every knob is on (@fig-loo). The second design is the one
a hardware fix asks about, because the knobs need not add.

#fig(
  "noise_ladder",
  caption: [
    Alone, phase drift costs the most: #val("noise_phase_share")% of the
    summed one-knob costs, against #val("noise_snr_share")% for 30 dB white
    noise and at most #val("noise_minor_share_hi")% for each other knob at its
    nominal value (base seed spread: #val("noise_spread_share")%). Test
    velocity RMSE of the TCN with one knob on and the rest noise-free,
    $C = 0.2$, one seed per point; each x axis runs toward the worse signal.
  ],
) <fig-noise>

#fig(
  "leave_one_out",
  width: 62%,
  caption: [
    At the nominal point per-laser phase drift dominates: removing it cuts
    the error #val("loo_phase_cut")%, and #val("loo_e60_cut")% when both runs
    train exactly #val("loo_e60_epochs") epochs, while removing an electronics
    knob (white noise, ADC or photodiode bandwidth) cuts it at most
    #val("loo_pdbw_cut")%. Test velocity RMSE of the TCN, $C = 0.2$, with the
    named knob returned to noise-free and every other knob at nominal, one
    seed each.
  ],
) <fig-loo>

The simulation draws phase drift independently per laser, as frequency drift
of each laser would. A path-length drift shared by all channels would read as
displacement and is not simulated. So the lever these runs support is
per-laser frequency stability or a per-laser phase monitor, not common-path
stabilization. The verdict rests on nominal values that are assumptions. The
drift nominal, from the phase wander the method-1 fit leaves on real data, is a
lower bound on the true rate; the rest are guesses. Measuring the rig's
drifts and SNR would test the verdict.

The 30 dB causal bound of @sec-crb is no constraint here: the noise-free base
alone sits #val("noise_base_over_bound") times above it. Two DC-offset rows
score below the base, but both trained #val("noise_dc_low_epochs") epochs,
into a third cycle, and nothing here rests on them. The ladder and the
leave-one-out cover three channels at $C = 0.2$ only.

=== Real data

The same ratio puts the rig well below the synthetic nominal point. Over the
zero predictor, the 30-epoch ladder cells of @sec-q2-noise score
#val("pred_base_ratio") noise-free, #val("pred_nominal_ratio") at the
nominal point, #val("pred_no_drift_ratio") with phase drift removed and
#val("pred_c09_ratio") at $C = 0.9$. The real #real-ratio clears the nominal
point by a wide margin but sits within its own approximation of the $C =
0.9$ cell. It lies below the no-drift cell too, so drift alone cannot explain
the rig. The 60-epoch cells, which give the #val("loo_e60_cut")% drift cut,
improved the nominal point by only #val("loo_e60_nominal_gain")%, so
training length does not close the gap. Hence the nominal knobs or the
synthetic motion prior overstate the difficulty, and a narrower real motion
prior is an equal explanation. The drift-dominance verdict of @sec-q2-noise
may then not describe the rig, as its call to measure the drifts allowed.

Prediction 3 tests the knob half. Each bar clears nominal by the round
trip's acceptance bar, 20% or 1 dB for SNR, and a reading between nominal
and its bar neither confirms nor falsifies. A knob returned as None fails its
clause and drops out of the falsifier, and with all three None the
prediction is unscored. Drift is not a clause, because @sec-q2-noise takes
its nominal as a lower bound on the true rate. If prediction 3 fails, the
motion prior or a noise source the simulator lacks carries the gap.

Prediction 4 needs a per-laser phase monitor or a more stable acquisition,
neither of which exists, so it is directional only. The
#val("loo_e60_cut")% cut at equal length is its ceiling, and a rig less
drift-limited than nominal should gain less. The calibration reports drift as
$D$ in $"rad s"^(-1\/2)$, and `rad_per_sqrt_s_to_fringes` in `forward.py`
converts it to fringes per shot. Its nominal of #val("noise_phase_fringes")
fringes is $D = #val("noise_phase_rad")$ $"rad s"^(-1\/2)$. Prediction 4 is
falsified if new hardware reads $D$ below that and the TCN's error falls by
less than $s$.

== Q2.3: generalization across alignment <sec-q2-alignment> // math-ok: question label

=== Analytic

A wavelength shift common to all channels is exactly a velocity rescale. In
@eq-michelson the phase $4 pi d \/ (s lambda_k)$ at scaled wavelength
$s lambda_k$, $s = 1 + epsilon$, equals the phase of trajectory $d \/ s$ at
the trained wavelengths. No estimator can tell the two apart, so a model right
at nominal returns $v \/ s$ and scores
$
  "RMSE"(epsilon) approx sqrt("RMSE"_0^2 + (epsilon \/ (1 + epsilon))^2 v_"rms"^2) .
$ <eq-shift-scale>
With $v_"rms" = #val("shift_v_rms")$ $mu$m/ms from the motion prior, the
scale term overtakes the nominal error only at
$|epsilon| approx #val("shift_tcan_eps_cross")$. Feedback $C$ changes the
fringe shape of @eq-excess, not its rate, so we predict only a minimum at the
trained $C = 0.2$, steepest toward $C -> 0$, where the asymmetry that carries
the sign vanishes.

Training draws an independent uniform phase per channel and shot. Pinning the
channels to one shared phase with fixed offsets selects one slice of that
distribution, and the information bound barely depends on which slice. The
shared-phase profiled information of the three colors at $C = 0.2$
(`bounds.fisher_from_shots`, 30 dB) stays within
#val("shift_info_lo_pct")% to #val("shift_info_hi_pct")% of the
independent-phase value over every configuration scored below. Equal phases,
where @sec-fisher warns all channels could sit at their extrema
together, give #val("shift_info_zero_pct")%: with three wavelengths the
displacement sweeps the channels apart within a shot. The sign evidence does
vary, up to #val("shift_sign_max_ratio") times the independent-phase value
for spread offsets and #val("shift_sign_zero_ratio") times it at equal
phases, but it is large everywhere. We therefore predict that an efficient
estimator's error does not depend on the configuration beyond a percent or
two, and that the configuration-averaged error equals the nominal score.

=== Synthetic

We score the nominal-trained TCAN and UNet-Transformer, unchanged, on frozen
64-shot test sets shifted in one knob (@fig-shift). Each error is the mean
per-shot velocity RMSE. Its SE, IQR $\/ 1.349 \/ sqrt(64)$, is
#val("shift_tcan_se") $mu$m/ms for the TCAN (nominal
#val("shift_tcan_nominal")) and #val("shift_unet_se") for the UNet-Transformer
(#val("shift_unet_nominal")).

Feedback is the knob that matters. Both errors are lowest at
$C = #val("shift_tcan_best_c")$, #val("shift_tcan_best_c_se") SE from the
trained 0.2 for the TCAN, and rise #val("shift_tcan_c005_ratio") and
#val("shift_unet_c005_ratio") times at $C = 0.05$ against
#val("shift_tcan_c05_ratio") and #val("shift_unet_c05_ratio") times at
$C = 0.5$, as predicted.

Wavelength shifts up to $|epsilon| = 10^(-2)$, common or in the per-channel
pattern $(+1, -1, +0.5) epsilon$, stay within
#val("shift_unet_wave_max_se") SE of nominal, consistent with the
#val("shift_tcan_eps_pred_se") SE that @eq-shift-scale predicts at
$10^(-2)$.

The phase configuration moves the learned errors far more than the bound.
Over #val("shift_n_random") random shared-phase configurations, offsets
$(0, u_1, u_2)$ with $u$ uniform, the TCAN averages
#val("shift_tcan_rand_mean") $plus.minus$ #val("shift_tcan_rand_mean_se")
$mu$m/ms against its nominal #val("shift_tcan_nominal"), and the
UNet-Transformer #val("shift_unet_rand_mean") $plus.minus$
#val("shift_unet_rand_mean_se") against #val("shift_unet_nominal"): the
nominal score is the configuration average, as predicted. The spread across
configurations is #val("shift_tcan_rand_spread_se") and
#val("shift_unet_rand_spread_se") times the per-shot SE, against the percent
or two the bound allows, and it tracks neither the configuration's
information nor its sign evidence (Spearman $|rho| <= #val("shift_rho_max")$).
The configuration dependence is therefore the learned estimators' own. Equal
phases are the TCAN's worst case: #val("shift_tcan_zero1") $mu$m/ms,
#val("shift_tcan_zero1_ratio") times nominal, beyond all random
configurations (maximum #val("shift_tcan_rand_max")), and
#val("shift_tcan_zero2_ratio") times on a second test draw. The
UNet-Transformer is #val("shift_unet_zero1_ratio") times nominal there. At
the other end, the TCAN at a 2 to 3 rad step is
#val("shift_tcan_phase3_pct") to #val("shift_tcan_phase2_pct")% better than
nominal. We have no explanation for why equal phases in particular defeat the
TCAN.

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

_Status: blinded; no shifted real acquisition exists._ The committed
prediction: a frozen model degrades with a change of feedback strength
(realignment) by the synthetic factors above, and not with a diode
temperature change, which @eq-shift-scale bounds at about
$|epsilon| v_"rms"$. It is falsified if a temperature-only change of order
$Delta lambda \/ lambda = 10^(-3)$ raises the real error by more than
#val("shift_falsify_temp_pct")%, or if a realignment that halves $C$ changes
it by less than #val("shift_falsify_realign_pct")%, against the more than
twofold rise of the synthetic sweep at $C = 0.1$.

== Real data: the analytic baseline and learned models <sec-q2-real>

=== Results

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

==== Method 2: the baseline number <sec-r-method2>

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

===== $R^2$ and Pearson r are not interchangeable

Two statistics appear in this document and neither substitutes for the other.

/ $R^2$: the fraction of a signal's variance a fit explains, computed from a
  sum of squared residuals. It can go negative, and it is sensitive to *scale*:
  a fit at the wrong amplitude scores badly. Method 1 reports it.
/ Pearson r (disp.): the linear correlation between the decoded and the true
  displacement, the last column of the table above. It is scale-free, so it
  would score a decode that tracked the right shape at the wrong amplitude as
  highly as a perfect one.

Together they say more than either does alone. Pearson r of 0.53 and 0.42 puts
the decode well clear of noise: it follows the real motion partway before
losing it. The NRMSE just above 1 says that what it follows is not yet worth
more than a constant. Both readings are true of the same decode, and a decode
that recovered the shape at the wrong amplitude would look exactly like this.

===== The limits of this measurement

This is one inverse on these two acquisitions. The decode resolves fringe
ambiguity across the three wavelengths on a 1601-point displacement grid and
differentiates for velocity; a finer grid, a different fringe-ambiguity
strategy, or a smoother velocity estimate would each give a different number.
The RMSEs above are a floor a learned model must clear. They are not a bound
on classical processing. The sign caveat of @sec-method2 applies: the overall
sign of $d(t)$ is unobservable and is not scored.

The decoder itself is sound on fringes the Michelson model generated: run on
synthetic signals it recovers displacement to a median NRMSE below 0.06, with
at least 75% of shots under 0.10, roughly twenty times better than it manages
here. That bound is asserted by the test suite (`tests/test_baseline_fit.py`)
and so is rechecked on every run. Those synthetic fringes carry no measurement
noise and no within-shot phase drift, so the check establishes that the
decoder is correct, not that it is noise-robust, as @sec-read noted.

==== Method 1: does the Michelson model describe this data <sec-r-method1>

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
0.39 to 0.55 in $R^2$.

===== Why a control is mandatory

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

===== How the window length is chosen

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
`smi/estimators/phase_fit.py` beside the clip constants they justify, rather
than in the results artifacts.

===== Per-channel caveats

*Free-space 515 nm is the weak case.* It has the narrowest gap of the six,
0.21, and the lowest global $R^2$ of the six, 0.080. Averaging the three
free-space channels would report a gap near 0.31 and hide it, so every
table in this document reports the channels separately. 515 nm crosses the most
fringes per micron of the three wavelengths, which is the ordering to expect if
the residual is a phase effect; this document has no measurement that pins that
down.

*The mm-fiber 675 nm channel is the one to read carefully elsewhere.* Nothing
in the table above is affected, because both columns there use the known
displacement. But where a gain is fit per shot, that channel
picks a gain near twice the true one in 249 of 1000 shots, which re-aligns the
fringe pattern by a whole fringe order and scores well without describing the
motion. Its per-shot median $R^2$ of 0.336 is inflated by those aliases and is
not evidence that the channel is fit better than the table says.

==== What the fits look like on real shots <sec-r-shots>

Every number above is a median over a thousand shots. A median $R^2$ does not
say whether the fit follows individual fringes or merely happens to land close
to them on average, so @fig-traces tabulates the fit on three shots.

// Shot-trace figure (figures/plot_baseline_shot_traces.py) is out of the DAG
// until a committed raw-data extract restores it; TASKS.md tracks this.
#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header(
      [shot],
      [peak-to-peak (um)],
      [635 nm $R^2$ (control)],
      [675 nm $R^2$ (control)],
      [515 nm $R^2$ (control)],
      [joint $g$],
    ),
    [203], [0.29], [0.79 (0.11)], [0.91 (0.25)], [0.65 (0.26)], [1.028],
    [90], [1.51], [0.68 (0.29)], [0.72 (0.33)], [--], [1.095],
    [365], [3.74], [0.67 (0.38)], [--], [--], [1.036],
  ),
  caption: [
    Method 1 on three free-space shots, the ones nearest the 10th, 50th and
    95th percentile of displacement peak-to-peak over the N = 1000 shots of
    the table above: windowed $R^2$ at the fringe-matched $W$ with its
    scrambled-theta control, and the shared gain $g$ of the joint fit. Values
    are those quoted in the text; -- where it quotes none. The trace figure
    they come from needs the raw acquisition files and is not built here.
  ],
) <fig-traces>

These are the estimator of @sec-r-method1 applied shot by shot, so each shot
resolves its own $W$: on 515 nm, 4096 at low excursion down to 519 at high. A
loud shot gets a shorter window because it crosses more fringes in the same
time, which keeps one window's accumulated phase error comparable across
shots.

Every tabulated channel clears its control on all three shots, and the margin
narrows as excursion grows (@fig-traces).

The joint-fit gains of @fig-traces bracket the 1.10 to 1.13 that an
independent gain sweep finds by refitting each channel separately over a grid.
That
is a cross-check between two estimators of the same mechanical scale, one
using three wavelengths at once and one using each alone. The joint fit gains
little over the per-channel fits (0.70 against 0.67 on high-band 635 nm, and
within 0.01 elsewhere). With amplitude and phase free in every window there is
already enough freedom to absorb a gain error, so constraining the gain buys
little.

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
here. What these shots
settle is the narrower question the medians leave open: at low and medium
excursion, on 635 nm and 675 nm, the windowed fit follows individual fringes
rather than only their general shape.

===== Why the fringe-matched window and not a smooth envelope

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
  0.586 to 0.883 of the local amplitude.
]

=== What limits the fit <sec-limits>

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

==== The interval the fit holds its parameters over

The global curve falls from 0.453-0.730 in the smallest bin to
0.014-0.047 above 3 um, and does so in all six channel/acquisition
combinations. The windowed curve over the same shots stays at 0.510-0.867 in
that top bin, against controls of 0.365-0.400. Nothing about the model or the
displacement differs between those two curves; only the interval over which
amplitude and phase are held fixed.

The fringe phase and the fringe contrast both wander within a shot, and those
are the two quantities the global fit holds constant. Within-shot drift of 1
to 2.7 fringes has been measured on this data and is recorded in
`smi/estimators/phase_fit.py`; that magnitude lives there rather than in the
results artifacts, so this document cites it and does not tabulate it. For
scale, at the median true velocity RMS of 0.373 and 0.377 um/ms a
1024-sample window sweeps 2.3 to 3.1 fringes depending on wavelength, and the
whole 33.6 ms record sweeps 33 times as many. A single phase cannot describe a
record whose fringe phase wanders by a fringe or more, and a single amplitude
cannot describe one whose contrast varies, so a three-parameter global fit
leaves most of the variance unexplained. That also accounts for the excursion
dependence without any extra assumption: a longer sweep gives the drift more
time to accumulate.

==== A real gain error in the drive-to-displacement conversion, secondary <sec-limits-gain>

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
measure and are not comparable to the fringe-matched tables of @sec-r-method1;
what they establish is the *size* of the gain's effect, and that is the claim
being made. Second, the best gain is *not* consistent between the two
acquisitions -- 16.50 against 15.50 at 635 nm, 16.25 against 15.75 at 675 nm,
16.00 against 15.25 at 515 nm, a disagreement of 3.2 to 6.5%. Drive-to-
displacement gain is a property of the speaker, not of the optical path, so a
scalar `k` that lands in two different places on two acquisitions is absorbing
something other than a pure gain error. The gain step is 0.25, which is 1.7%
of the live value, so the disagreement is wider than the sweep's resolution.

For that reason the calibration constant in `smi/theory/calib_params.py`
has deliberately *not* been re-tuned here. Changing it would shift every
training target, and a constant the two acquisitions disagree about by up to
6.5% is not settled enough to shift them on.

The same sweep does refute the value the 2024 calibration implied,
`k = 33.42`, which is kept commented out in that file under the note
"Incorrect fringe counts in 2024 calibration?". At that gain the median $R^2$
is 0.012 to 0.029 across the six channel/acquisition combinations, which is
noise. The doubt in the comment was warranted.

==== A residual amplitude dependence that is not a scale error <sec-limits-amp>

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

==== What this section rules out

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

=== Prior art: two published networks <sec-prior>

Two papers from the same group have already put a convolutional network on
this problem, and both are the starting point for the learned models here.

@barland2021cnn trains a single-channel network on self-mixing fringes and
reports a Pearson correlation of 0.90 between the recovered and the true
displacement, at 57,153 parameters. @matha2023multichannel extends the
approach to three channels at 207,905 parameters, reporting a correlation of
0.96 and a velocity RMSE of 0.13 um/ms; its best single channel reaches 0.97
and 0.11 um/ms.

`smi/learning/models/barland_cnn.py` is a port of the 2021 architecture. At
one input channel it comes to 57,153 trainable parameters, an exact match
with the total the paper states, which is the check that establishes the port
is faithful rather than merely similar. At the three channels this data
supplies it comes to 57,377, and that is the configuration trained here.

*Those reported numbers are not comparable to this document's.* They were
measured on different apparatus, with training displacements drawn from eight
harmonic frequencies between 53 and 93 Hz, against targets calibrated on that
rig. The Barland row below is this repository's port of their architecture
on this repository's data. It is evidence about the architecture; it is not
their result, and the two should not be differenced.

=== Learned models against the analytic baseline <sec-learned>

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
    against the analytic decode of @sec-method2 drawn as the dashed line. Bars
    are the mean over two seeds and the whisker spans the seed-to-seed range,
    which two seeds give and a standard error does not. Row order is the
    free-space ranking and is held fixed in the right panel, so where the
    fiber ordering differs it reads as crossing rather than as a second
    unrelated chart. Hatched bars label the models that see samples after the
    timestamp they predict; only the TCN rows do not.
  ],
) <fig-models>

==== What every configuration scored <sec-learned-table>

Velocity RMSE in um/ms and displacement RMSE in um, each the mean over the
two seeds, from `report/results/learned_results.json`. Every row carries
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

FS is free space, MMF multimode fiber. For ridge the params column counts
coefficients and for LightGBM boosted trees (200 trees of 31 leaves, the 6,200
leaves the synthetic records count). Their field is the 91-sample window;
stride-1 reconstruction averages overlapping windows, so each output depends
on the 181-sample two-sided span that @tab-tabular-synthetic quotes. The analytic row is method 2 of
@sec-method2, the same decode the dashed line in @fig-models marks; the means
are quoted here because the learned rows are means, where the table of @sec-r-method2
quotes medians of the same decode.

Across all twenty runs per acquisition the mean velocity RMSE is 0.1712
(free space) and 0.2771 (mm fiber). The best single run is 0.0799 and 0.1632.
Those are different claims and neither substitutes for the other: the mean
says what a configuration drawn from this set achieves on average, and the
best says what the set's most fortunate seed of its best architecture
achieved once.

==== Every learned model clears the analytic baseline

On both acquisitions, all ten configurations score a lower velocity RMSE than
the analytic decode. The weakest, a ridge regression over a 91-sample window
with 274 coefficients, comes in at 0.3837 against the decode's 0.5611 in free
space and 0.3431 against 0.5647 in the fiber.

A linear fit over a 91-sample window is not a powerful estimator, so its
beating the multi-wavelength inverse says more about the inverse than about
the regression. @sec-r-method1 and @sec-limits already found that the Michelson model, given
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

==== Causality costs less than the architecture literature would suggest <sec-learned-causal>

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

==== The published architecture wins on both acquisitions

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
the next sweep should target: the open question is which part of the Barland stack
does the work -- the wider window, the pooling, or the fully connected head --
not whether the dilated stack can be tuned past it.

==== Measuring an architecture requires an output resolution finer than the score

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

==== The generalization gap is the finding <sec-learned-gap>

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

@sec-read found no free-space/fiber gap in the analytic baseline at all, and
these ratios say why that was not reassuring: the gap is invisible to an
estimator that is not resolving the structure the fiber destroys. It appears
as soon as one is.

==== Receptive field dominates parameter count <sec-learned-field>

Two TCN configurations run at a 29-sample receptive field, at 27,985 and
35,585 parameters. They score 0.2096 and 0.2021 in free space: 27% more
capacity bought 0.0075 um/ms, which is within the smaller model's own seed
spread of 0.0160. A TCN at 19,329 parameters and an 85-sample field scores
0.1401 -- better than either, with 46% fewer parameters than the larger
field-29 model.

So on this data, over this range, widening the receptive field buys what
adding parameters at a fixed field does not. That is the cheapest thing the
next session can act on.

==== What limits these numbers

Eight things bound what the section above establishes, each of them a threat
to a conclusion a reader would otherwise draw.

*Only the TCN is causal.* Eight of the ten configurations label each window
at its centre and so read samples after the timestamp they predict. Any
comparison that crosses the hatch boundary in @fig-models compares estimators
solving different problems.

*Neither published paper states whether its network is causal*, so the
causality claim is scoped to this repository's own TCN-versus-SCNN pair
(@sec-learned-causal).

*Cross-apparatus RMSE is indicative only.* The numbers of @sec-prior came
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

*Not every published figure in @sec-prior came off the page cleanly.* The two
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
`scripts/benchmark_loading.py` was measured on a production-size TCN at
batch 32, where training is GPU bound; it does not transfer to 19-44k
parameters, which is why the sweep had to be measured on its own.

=== Future directions <sec-future>

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

*A receptive-field sweep.* @sec-learned-field found 27% more parameters at a fixed
29-sample field buying nothing, while widening the field at fewer parameters
cut the error by 31%. Only two field lengths have been tried, 29 and 85, so
the axis that moved the numbers most is the one with two points on it.

*Feature engineering for the fiber gap.* Better motivated now than before:
@sec-learned-gap measures the degradation at 2.3x for the best causal model rather
than leaving it as a suspicion, and locates it specifically in the models
that resolve fine fringe structure. That is a target a feature can be
designed against.

=== Scope

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

@sec-learned establishes that the floor is clear of every learned model tried,
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
at this error level; @sec-learned-gap is where that gap becomes measurable,
and it is measured there only for the models trained under this round's budget.
It does not give a corrected value for `k`: the two acquisitions
disagree by 3.2 to 6.5%, so the sweep locates an error without resolving it.
And the residual amplitude dependence that survives a per-shot gain fit is
unexplained -- measured, bounded, and open.

@sec-future lists what to do on the learned side. On the classical side two
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

= Q3: fiber <sec-q3> // math-ok: question label

== Q3.1: a fiber-only monitor color <sec-q3-monitor> // math-ok: question label

_Status: not run in any tier._

== Q3.2: the orthogonal-polarization channel <sec-q3-polarization> // math-ok: question label

_Status: not run in any tier._

== Q3.3: flow-matching denoising <sec-q3-denoise> // math-ok: question label

_Status: not run in any tier._

== Q3.4: the shape of the fiber posterior <sec-q3-posterior> // math-ok: question label

_Status: not run in any tier._

== Real-data context: free space versus multimode fiber <sec-q3-real>

The fiber path is the deployment-relevant one and the expected-harder one. The
raw data does differ: the fiber acquisition carries less high-frequency
content than free space in the same channel.

// Spectrum figure (figures/plot_freespace_vs_fiber.py): out of the DAG until a
// committed raw-data extract restores it; its caption is in git history.

_Status: the 635 nm mean amplitude spectrum behind this comparison needs the
raw acquisition files and is not shown in this build._

That spectral difference does not translate into a baseline difference
(@sec-read). Both acquisitions are limited by the same things, and a
comparison at this level should not be cited as evidence either way about the
fiber gap.

The program makes no prediction about the fiber penalty. It has no model of
mode scrambling, which is Q3 and outside P0 to P3, so the published fiber // math-ok: question and stage labels
over free-space ratios of 2.28 for the TCN and 2.02 for Barland
(@sec-learned-gap) have no synthetic counterpart. Every prediction above
is for free space.

= Q4: Sagnac accelerometer <sec-q4> // math-ok: question label

Q4 asks whether a Sagnac loop read by self-mixing would give a common-path, // math-ok: question label
fiber-noise-insensitive accelerometer. Its readout is not insensitive to
reciprocal fiber phase: the loop returns the nonreciprocal phase in the
amplitude, and the reciprocal fiber phase comes back as $phi_0$. Whether that
drift matters depends on its bandwidth against the signal band, which we do
not model. Code: `smi/theory/sagnac.py`, tests `tests/test_sagnac.py`.

=== Analytic

A 2x2 coupler of power split $rho$ closed on a loop returns to the laser
$
  E_r = 2 i sqrt(rho (1 - rho)) cos(Delta phi \/ 2) e^(i Sigma \/ 2), quad
  Sigma = phi_"CW" + phi_"CCW", quad Delta phi = phi_"CW" - phi_"CCW"
$ <eq-loop-mirror>
for any $rho$ @mortimore1988loop, because each beam takes one bar and one
cross pass. The test sums the two beam fields and checks the transmitted port
for $rho in {0.3, 0.5, 0.9}$. The nonreciprocal phase sets only the
amplitude reflectivity $r = cos(Delta phi \/ 2)$. Self-mixing reads the round-trip
phase of the return @lang1980feedback @taimre2015tutorial, which contains
$Sigma \/ 2$ and the lead fiber in full. Reciprocal fiber noise therefore
enters $phi_0$ of @eq-excess. A conventional Sagnac readout at the second
coupler port is insensitive to it; the self-mixing readout is not.

The output encodes the velocity of an element placed off the loop midpoint,
not the acceleration of the loop. An element with phase $kappa x(t)$, crossed
by the two beams $tau$ apart, gives $Delta phi = kappa tau v$ at the midpoint
time, exactly for quadratic $x(t)$. Rotation is nonreciprocal @post1967sagnac.
Uniform linear acceleration of a rigid loop is not: the Sagnac delay is
proportional to $integral.cont g_(0 i) d x^i \/ g_(00)$
@rizzi2004rotating @tartaglia1998sagnac, and the Rindler metric has
$g_(0 i) = 0$. That step to a null result is our own, argued and not tested. A proof mass of
resonance $omega_0$ and quality $Q$ makes $Delta phi$ follow jerk below
resonance and acceleration, with gain $kappa tau Q \/ omega_0$, only near
it. Off-center strain of the coil itself acts as a distributed proof mass by
the same mechanism.

With feedback $C_0$ at full reflection, both feedback and modulation depth
scale with $r$:
$
  phi_F = phi_0 - C_0 r sin(phi_F + beta), quad y = m r cos phi_F + n .
$ <eq-sagnac-signal>
A bias $Delta phi_b = pi \/ 2$ is needed, since $partial r \/ partial Delta phi$
vanishes at 0. $C_0 > 1$ is allowed because $C_"eff" = C_0 r$, but
$C_0 cos(Delta phi \/ 2) < 1$ must hold over the whole swing, or the laser
turns multistable @lang1980feedback. At $C_0 = 1.3$ that leaves
$plus.minus #val("sagnac_c13_swing")$ rad. With $phi_0$ known and $g = partial Delta phi \/ partial a$, the
acceleration Fisher information has the form of @eq-fisher with prefactor
$m g sin(Delta phi_b \/ 2) \/ (2 sigma)$ and $F'$ replaced by
$partial (y \/ m) \/ partial r$, from implicit differentiation of
@eq-sagnac-signal. We call its ratio at $C_0$ to its value at $C_0 = 0$, at
the same bias and modulation depth, the feedback gain.

Feedback raises that information little unless $phi_0$ is held at its best
value (@fig-sagnac, @tab-sagnac). At the rig's $alpha = 3$ and $C_0 = 0.99$,
so at $C = C_"eff" = #val("sagnac_c099_ceff")$, the gain is
#val("sagnac_c099_avg") phase-averaged and #val("sagnac_c099_best") at the
best phase. The averaged gain is below the conventional phase readout's
$2 Phi = #val("sagnac_c099_smi")$ at the same $C$ (@eq-phase-factor), because
$Delta phi$ enters the amplitude, not the phase. The same laser in a
conventional SMI would see the full $C = 0.99$ and gain
$2 Phi = #val("sagnac_c099_smi_c0")$; the loop bias lowers its feedback to
$C_"eff"$ as well as moving $Delta phi$ into the amplitude. The best-phase value is a genie bound: holding $phi_0$ there is the
fiber-phase stabilization the premise hoped to avoid. Sign needs no feedback
asymmetry; the bias makes the response signed.

The averaged gain is larger at $alpha = 0$ than at $alpha = 3$, though the
best-phase gain at $alpha = 0$ is exactly 1. Over a uniform $phi_0$ the
feedback phase $phi_F$ dwells with density proportional to
$1 + C cos(phi_F + beta)$, longest near $phi_F = -beta$. At $alpha = 0$ that
is $phi_F = 0$, where the slope $partial (y \/ m) \/ partial r$ peaks; at
$alpha = 3$ it is $phi_F = -arctan 3$, where the slope is small. Weighting
$phi_F$ uniformly instead reverses the order and returns $2 Phi$ for both.

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
    Feedback gains, phase-averaged / best phase, bias $pi \/ 2$, against the
    conventional SMI phase readout at the same $C_"eff"$. Swing: for
    $C_0 > 1$, the largest $|Delta phi - Delta phi_b|$ keeping
    $C_0 r < 1$; blank rows are unbounded.
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

=== Real data

_Status: not built; needs a bench fiber Sagnac loop read by self-mixing._ The
committed prediction, from @eq-loop-mirror: with the loop's nonreciprocal phase
swept by a known bias, the self-mixing modulation amplitude follows
$|cos(Delta phi \/ 2)|$, and with a phase modulator in the lead fiber the
fringe phase follows the reciprocal phase in full. It is falsified if the
amplitude does not vanish at $Delta phi = pi$, or if a reciprocal lead-fiber
phase step leaves the self-mixing fringe phase unmoved.


#bibliography("refs.bib")
