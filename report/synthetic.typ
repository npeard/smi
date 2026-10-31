// Research report: what synthetic data and analysis say about velocity
// retrieval from multi-wavelength self-mixing fringes, before any real
// acquisition is unblinded.
//
// Build:  pixi run report-synthetic
//
// Every number in this document comes from a committed artifact or from a
// computation in a committed script:
// - smi/analysis/data/synthetic_results.json, the experiment runner's records,
//   read by the report/figures/plot_*.py scripts this document includes;
// - smi.analysis.bounds, evaluated by the figure scripts
//   report/figures/plot_{fisher_phase,velocity_bounds,sign_information}.py;
// - build/synthetic_values.json, written by report/figures/synthetic_values.py
//   from smi.analysis.bounds, learned_results.json, synthetic_results.json and
//   the experiment grids: every value quoted in the prose is read from it
//   below, so a library or artifact change rebuilds the prose rather than
//   leaving it stale.

// ASCII only, per the repo convention.

#import "lib/theme.typ": fig, report-theme, takeaway

#show: report-theme.with(
  title: "Velocity from self-mixing fringes: a synthetic study",
  subtitle: "What feedback, noise and extra channels do to velocity retrieval",
  author: "Nolan Peard",
  date: "Draft",
)

#set math.equation(numbering: "(1)")
// Floats go to the top or bottom of a page, so a figure that misses the space
// left on one page does not leave that space blank.
#show figure: set figure(placement: auto)

#let values = json("/build/synthetic_values.json")
#let val(key) = values.at(key)
// A result that awaits a rerun, marked so it cannot pass for a finding.
#let pending(body) = block(
  stroke: 1pt + rgb("#c0392b"),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
  text(fill: rgb("#c0392b"))[*Pending.* #body],
)

We want to measure how a mirror moves from the light it reflects into three
lasers at 635, 674.8 and 515 nm. On real acquisitions the true motion is
unknown, so this report works on simulated data, where it is known exactly.
The real acquisitions stay unread until the synthetic program's predictions
for them are fixed. The report answers three questions, Q3 in two parts, // math-ok: question label
and each answer has one figure.

#takeaway[
  + *Q3, feedback: weak optical feedback makes the direction of motion // math-ok: question label
    observable.* Without it, as in a plain Michelson interferometer, a network
    trained on signed velocity predicts zero (@fig-exit-check). With it the
    error falls steadily with feedback strength, to #val("q3_gain_top") times
    below the Michelson error up to sign (@fig-feedback).
  + *Q3, imperfections: at the nominal operating point, per-laser phase drift // math-ok: question label
    dominates.* Removing it cuts the error #val("loo_e60_cut")% at equal
    training length, while no electronics knob moves it more than
    #val("loo_pdbw_cut")%, at the assumed nominal values (@fig-noise,
    @fig-loo).
  + *Q1, channels: a second wavelength helps only without noise.* Noise-free
    it cuts the error #val("q1_gain_tcn") to #val("q1_gain_tcan") times below
    the best single channel, and on Michelson one channel alone recovers
    nothing. At the nominal point the gain is at most
    #val("q1_s4_gain_max_pct")% (@fig-q1-subsets).
  + *Q2, model families: only with feedback does the architecture matter.*
    There TCAN is best, #val("q2_tcan_gain") times below the TCN in one seed,
    while the LSTMs and the offline neural operator do not train
    (@fig-families).
]

= What the fringes can tell us <sec-bounds>

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
so each pair of runs differs in one factor: the loss or the fringe.

#fig(
  "exit_check",
  width: 62%,
  caption: [
    Without feedback a signed-loss network learns to predict zero; with
    feedback it recovers the motion and its sign. Michelson: a plain
    interferometer, feedback strength $C = 0$. Self-mixing (SMI): the light
    re-enters the laser, here with $C = 0.2$ and linewidth enhancement factor
    $alpha = 3$, which skews the fringe. Bars: test velocity NRMSE, the error
    of each shot (one simulated 33.6 ms record) over that shot's velocity std,
    so predicting zero scores 1 (dashed), for a causal temporal convolutional
    network (TCN) on three noise-free channels. Signed compares the prediction
    with the truth as is; sign-invariant takes the better of
    $plus.minus hat(v)$ per shot. The runs train at most
    #val("exit_max_epochs") epochs, so only the ordering is evidence.
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
    feedback, growing as $C^2$, and three channels ($K = 3$)
    carry #val("sign_k3_k1_lo") to #val("sign_k3_k1_hi") times the evidence of
    one ($K = 1$) at $C = 0.05$ to $0.7$. The bound is exactly zero at $C = 0$ and
    at $alpha = 0$. Upper bound on the sign evidence, the expected
    log-likelihood ratio in nats of the true signal over the closest
    sign-flipped one, for one 16384-sample shot
    at SNR 30 dB, visibility $V tilde U(0.5, 1)$, mean over 8 shots (seed 0);
    $K = 1$ is the 635 nm channel. Markers: computed; lines: the leading order,
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
    the blind spots at the extrema. The squared fringe slope $F'(theta)^2$
    (with prime: the derivative of the fringe function $F$), the
    phase-dependent factor of the Fisher information in @eq-fisher, over one
    fringe at $alpha = 3$. Dashed: its phase averages $Phi(C, alpha)$ of
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
    #val("causal_ratio_k3_k1"), and feedback lowers it less still; the offline
    bound sits below the causal one. Stationary RMS velocity bounds at SNR
    30 dB, $V tilde U(0.5, 1)$, $alpha = 3$, for $K$ channels: $K = 1$ is
    635 nm, $K = 2$ adds 674.8 nm and $K = 3$ adds 515 nm. Causal (past samples
    only): @eq-pcrb after convergence; offline (whole shot): the smoothing
    bound at mid-record, away from the shot's ends.
  ],
) <fig-bounds>

The causal bound scales close to the inverse fourth root of $macron(J)$
(@fig-bounds). It sits #val("real_gap") times below the causal TCN's error of
#val("real_tcn_rmse") $mu$m/ms on the real free-space test split of the
baseline report. Within its own model the bound is nearly attained: in
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

= Data, training and evaluation <sec-methods>

Every learned result below compares cells that differ in one factor: the
feedback, a noise knob, the channel subset or the model family. That
comparison is fair only because the data, the training budget and the scoring
are otherwise shared. This section states them.

The simulator draws each shot's trajectory as a sum of drive lines from 1 Hz
to 1 kHz with independent uniform phases and an amplitude drawn per shot from
$U(0, 1)$. It renders the three channels of @eq-signal over
#val("m_shot_length") samples at 488 kHz, a 33.6 ms shot. A noise-free
configuration turns every detector and drift knob off; the nominal
configuration sets each to a stated guess. Every model maps the three channel
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
    Velocity, not displacement, has its power where the fringes do. A
    channel's instantaneous fringe frequency, $2 |v| \/ lambda_k$, is the rate
    at which fringes pass, the quantity its signal encodes most directly, so a
    target whose spectrum shares its band is one the fringes inform. Power
    spectral densities, each divided by its peak, of the velocity target
    (solid), the displacement target (dashed) and each channel's fringe
    frequency (colored), averaged over #val("spec_shots") shots at $C = 0.2$,
    $alpha = 3$: (a) the nominal point, every detector and drift imperfection
    ("knob") at its assumed value, (b) noise-free with the same trajectories.
    Hann-windowed whole shots; shaded: the 1 Hz to 1 kHz band of the drive
    lines.
  ],
) <fig-spectral>

The networks predict velocity rather than displacement because velocity
matches the information in the signal (@fig-spectral). Each fringe frequency
follows $|v|$, while displacement weights the velocity spectrum by
$1 \/ f^2$ and puts #val("spec_x_low_pct")% of its power below 100 Hz, against
#val("spec_v_low_pct")% for velocity. Below the lowest resolved bin,
#val("spec_bin_hz") Hz, drive content enters a shot only as a trend.

= Q3: accuracy against feedback strength <sec-q3> // math-ok: question label

#assert(val("q3_monotone"), message: "the 90-epoch C curve is no longer monotone")
#assert(val("q3_t30_c04_below"), message: "the 30-epoch C = 0.4 inversion is gone")
#assert(
  val("q3_slow_falls_with_budget"),
  message: "the slowest-decile floor no longer falls from 30 to 90 epochs",
)

== The noise-free curve <sec-q3-curve>

With feedback a signed estimator reads the direction of motion, and its
error falls steadily with $C$ up to 0.9, far below Michelson's error even
when Michelson is forgiven the sign (@fig-feedback). Section 1 predicts the
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
    $C = #val("q3_c_top")$. Per shot (NRMSE) the gain there is
    #val("q3_gain_top_nrmse") times, and the weakest feedback does not beat
    Michelson. At $C = 0.2$ the $alpha = 0$ control, whose fringe stays even,
    ends within #val("q3_zero_gap_pct")% of the zero predictor's error, and
    $alpha = 5$ lies within #val("q3_a5_gap_pct")% of $alpha = 3$ (hollow).
    (a) Test velocity RMSE in $mu$m/ms, per shot and averaged over shots;
    (b) NRMSE, each shot's RMSE over its velocity std, so the zero predictor
    scores 1; signed for SMI; causal TCN, three noise-free channels,
    $alpha = 3$, one seed. Solid: one 90-epoch cosine cycle; dashed: a
    30-epoch cycle, whose inversion at $C = 0.4$ is a run that trained
    #val("q3_t30_c04_epochs") epochs, past its first restart. Dotted:
    Michelson, trained and scored sign-invariant, in its budget's color.
    (c) Sign-invariant RMSE of the test samples pooled in each $|v|$ decile,
    for three 90-epoch cells; grey: slope one.
  ],
) <fig-feedback>

RMSE and NRMSE disagree at weak feedback because they weight different shots. RMSE
pools samples, which fast motion dominates, and Michelson's error grows in
proportion to $|v|$ (#ref(<fig-feedback>)c): a wrong local direction costs
more the faster the mirror moves. NRMSE divides by each shot's velocity std,
which lets slow, low-amplitude shots dominate. There the estimator's own
floor, #val("q3_slow_lo") to #val("q3_slow_hi") $mu$m/ms in the slowest
decile for every $C$, sets the score, and it falls with training budget.

The sign is resolved on nearly every shot: signed and sign-invariant RMSE
agree within #val("q3_sign_gap_pct")% in every 90-epoch cell. The controls
behave as @sec-feedback predicts.

The curve holds for a fixed budget, and its values still move with it. One
#val("q3_t270_cycle")-epoch cycle at $C = 0.2$ reached #val("q3_c02_t270")
$mu$m/ms, below the 90-epoch value, so convergence is not demonstrated. Within
that caveat, the operating target is the strongest weak feedback sampled.

== Which imperfection costs the most <sec-q3-noise>

#assert(val("noise_ranking"), message: "the noise ladder no longer ranks phase drift, then SNR, then the rest")
#assert(val("noise_e60_phase_monotone"), message: "at equal length the phase-drift error no longer rises monotonically with drift")
#assert(val("noise_e60_half_keeps_most"), message: "at equal length the lowest swept drift no longer keeps over half the nominal drift's cost")
#assert(val("noise_bound_no_power"), message: "the noise-free base no longer sits over 5 times the 30 dB causal bound")
#assert(val("noise_electronics_flat"), message: "an ADC or PD-bandwidth row other than 8 bits or 3 kHz now leaves the base spread")
#assert(val("loo_electronics_max_pdbw"), message: "removing white noise or the ADC now moves the nominal error more than removing the PD bandwidth limit")
#assert(val("noise_drift_20_like_snr"), message: "the 20% drift rows' cost is no longer within a factor 1.5 of the 30 dB row's")
#assert(val("noise_snr20_doubles"), message: "the 20 dB row's cost is no longer about twice the 30 dB row's")
#assert(val("noise_dc_low_named"), message: "the DC-offset rows below the base are no longer the +/-0.3 and +/-1 rows")
#assert(val("loo_outside_named"), message: "the leave-one-out cells clearly below the nominal spread are no longer phase drift, then visibility and PD bandwidth")
#assert(val("loo_long_named"), message: "the leave-one-out cells that trained into a third cycle are no longer phase drift and PD bandwidth")
#assert(val("loo_e60_outside"), message: "at equal length, removing phase drift no longer cuts the nominal error by more than the nominal seed spread")
#assert(val("loo_e60_same_ranking"), message: "at equal length, the base, no-phase-drift and nominal cells no longer rank in that order")
#assert(val("loo_snr_subadditive"), message: "removing SNR from the nominal point now saves a third or more of its ladder cost")

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
    nominal value (base seed spread: #val("noise_spread_share")%). At equal
    length, cutting drift to #val("noise_e60_low_fringes") fringes per shot
    keeps #val("noise_e60_half_kept")% of its cost (squares). At four times its
    nominal value, amplitude or contrast drift costs about what 30 dB white
    noise does, and raising $C$ to 0.9 halves the all-nominal error (dotted,
    a factor #val("noise_c09_ratio"), one seed). Test velocity RMSE of the TCN
    with one knob on and the rest noise-free, $C = 0.2$, one seed per point.
    Ringed: the knob's nominal value. Grey and red bands, with a line at the
    mean: the noise-free base and every knob at nominal, over
    #val("noise_n_seeds") seeds. Squares and dashed line (phase drift only):
    the drift sweep and the base trained exactly #val("loo_e60_epochs")
    epochs. Each x axis runs toward the worse signal; the nominal drift of
    #val("noise_phase_fringes") fringes per shot is #val("noise_phase_rad")
    $"rad s"^(-1\/2)$.
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
    #val("loo_pdbw_cut")%. Next after drift come visibility
    (#val("loo_vis_cut")%) and photodiode bandwidth. The knobs do not add:
    30 dB white noise raises the noise-free base by #val("noise_snr_cost")
    $mu$m/ms (@fig-noise), but removing it here saves only
    #val("loo_snr_saving") $mu$m/ms. Test velocity RMSE of the TCN, $C = 0.2$,
    with the named knob returned to noise-free and every other knob at its
    assumed nominal value, one seed each. Red band: all knobs nominal over
    #val("noise_n_seeds") seeds; grey: the noise-free base. Purple: cells
    trained exactly #val("loo_e60_epochs") epochs. Hollow markers trained into
    a third cosine cycle, so part of their cut is training length.
  ],
) <fig-loo>

The simulation draws phase drift independently per laser, as frequency drift
of each laser would. A path-length drift shared by all channels would read as
displacement and is not simulated. So the lever these runs support is
per-laser frequency stability or a per-laser phase monitor, not common-path
stabilization. The verdict rests on nominal values that are assumptions. The
drift nominal, from the phase wander the baseline report's fit leaves, is a
lower bound on the true rate; the rest are guesses. At 20 dB the white-noise
cost doubles to #val("noise_snr20_cost") $mu$m/ms. Measuring the rig's drifts
and SNR would test the verdict.

The 30 dB causal bound of @sec-crb is no constraint here: the noise-free base
alone sits #val("noise_base_over_bound") times above it. Two DC-offset rows
score below the base, but both trained #val("noise_dc_low_epochs") epochs,
into a third cycle, and nothing here rests on them. The ladder and the
leave-one-out cover three channels at $C = 0.2$ only.

= Q1: do extra channels beat the best single channel? <sec-q1> // math-ok: question label

#assert(val("q1_multi_beats_single"), message: "a multi-channel set no longer beats the best single channel at C = 0.2 by more than the seed range")
#assert(val("q1_515_best"), message: "515 nm is no longer the best single channel at C = 0.2 for both families")
#assert(val("q1_tcn_multi_tied"), message: "the TCN's multi-channel sets at C = 0.2 now span more than the seed range")
#assert(val("q1_runner_up_tcn_at_edge"), message: "the TCN's 515 nm lead is no longer at the edge of the seed range")
#assert(val("q1_tcan_triple_at_edge"), message: "TCAN's three-channel gain over its best pair is no longer at the edge of the seed range")
#assert(val("q1_mich_single_at_zero"), message: "a Michelson single channel no longer scores at the zero predictor")
#assert(val("q1_mich_single_converged"), message: "the stall test now marks a Michelson single channel as not trained")
#assert(val("q1_mich_multi_far"), message: "a Michelson multi-channel set now comes near the zero predictor")
#assert(val("q1_oracle_single_above_zero"), message: "the single-channel Michelson oracle no longer scores above the zero predictor")
#assert(val("q1_bin_nf_flat"), message: "noise-free, an error curve now rises toward the 515 nm extremum or departs from flat")
#assert(val("q1_bin_nom_small_rise"), message: "at 30 dB the one- or three-channel error now departs from flat in extremum distance")
#assert(val("q1_bin_gain_not_near"), message: "the noise-free multi-channel gain is now larger near the 515 nm extremum than midway")
#assert(val("q1_bin_v_shape"), message: "the noise-free multi-channel gain by |v| no longer vanishes in the slowest bin and exceeds 1.5 higher up")
#assert(val("q1_bin_v_slow_above_zero"), message: "in the slowest |v| bin a noise-free model now scores at or below the zero predictor")
#assert(val("q1_s4_gain_shape"), message: "the nominal-point multi-channel gain no longer peaks at 10 to 30 dB, vanishes at 0 dB and reverses at 40 dB")
#assert(val("q1_s4_40_third_cycle"), message: "the 40 dB 515 nm cell is no longer the only one there trained into a third cycle")
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
TCN and TCAN, a temporal convolutional network with attention at the same
44k parameters, on every subset of the three channels. We score them
noise-free and, for the TCN, at the nominal point across SNR.

#fig(
  "q1_subsets",
  caption: [
    Noise-free at $C = 0.2$, a second wavelength cuts the error
    #val("q1_gain_tcn") times (TCN) and #val("q1_gain_tcan") times (TCAN)
    below the best single channel, 515 nm, and on Michelson one channel
    recovers nothing; at the nominal point the gain is at most
    #val("q1_s4_gain_max_pct")%. (a) A third channel adds nothing the TCN
    resolves over a pair, and #val("q1_tcan_triple_pct")% for TCAN, at the
    edge of the #val("q2_seed_range_pct")% seed range; 515 nm leads the next
    single channel by #val("q1_runner_up_tcn_pct")% (TCN) and
    #val("q1_runner_up_tcan_pct")% (TCAN). On Michelson every single channel,
    learned or oracle, scores within #val("q1_mich_single_gap_pct")% of the
    zero predictor. (b) The gain vanishes at 0 dB, is
    #val("q1_s4_30_gain_pct")% at 30 dB (three channels
    #val("pred_triple_30") times below the best single, against
    #val("pred_triple_free") noise-free) and reverses at 40 dB, where the
    single-channel cell trained into a third cycle. Panels: (a) noise-free
    test velocity RMSE in $mu$m/ms of each channel subset, one seed per cell,
    Michelson scored sign-invariant, $C = 0.2$ signed, $alpha = 3$; dashed:
    the zero predictor, #val("q1_zero") $mu$m/ms. The oracle is the baseline
    report's classical (not learned) inverse fit, given the true excursion.
    (b) The TCN at the nominal point with SNR
    varied: the lowest error among single channels, among pairs, and all
    three; error bar: three seeds of all three channels at 30 dB.
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
    The predicted concentration of the multi-channel gain at fringe extrema
    fails: noise-free the three-channel gain is #val("q1_bin_gain_near") times
    near the 515 nm extremum against #val("q1_bin_gain_far") midway, and it
    grows with speed to #val("q1_bin_v_gain_max") times, vanishing in the
    slowest $|v|$ decile. At 30 dB the error rises toward an extremum by only
    #val("q1_bin_nom_rise_one_pct")% for one channel and
    #val("q1_bin_nom_rise_three_pct")% for three. TCN on 515 nm alone, on 635 + 515 nm and on
    all three, $C = 0.2$, $alpha = 3$, one seed. Pooled test RMSE in
    #val("q1_bin_width")-fringe bins of the distance from $phi_F$ to the
    nearest extremum, of the 515 nm channel (solid) and of the channel farthest
    from one, so that every channel is near an extremum (dotted): (a)
    noise-free, (b) the nominal point at 30 dB. (c) In $|v|$ deciles,
    noise-free (solid) and at 30 dB (dashed), with the zero predictor (grey)
    and each subset's causal bound at 30 dB (dash-dotted).
  ],
) <fig-q1-binned>

The extremum prediction fails (@fig-q1-binned). The 30 dB panel, the proper
test of a noise-limited claim, rises in the predicted direction but far less
than a loss of all information would. One untested reading is that an
estimator seeing a window interpolates across an extremum, as the prior does
in the bounds of @sec-crb. The growth with speed runs against the
turning-point clause of the prediction.

At the nominal point phase drift dominates (@sec-q3-noise). The gain left at
30 dB is several times the nominal seed range and comes from pairing 515 nm
with another channel (#ref(<fig-q1-subsets>)b). The classical oracle decodes under the Michelson model and scores worse than zero
at the nominal point, so it is not a reference there.

These runs are one seed each at the 30-epoch budget, ranked against a seed
spread measured on three channels, and the nominal point ran for the TCN only.

= Q2: model families <sec-q2> // math-ok: question label

#assert(val("q2_mich_tied"), message: "the Michelson families other than the offline neural operator no longer lie within the seed range")
#assert(val("q2_oracle_beats_all"), message: "a learned family now comes within a factor 1.5 of the Michelson oracle")
#assert(val("q2_smi_vs_tcn"), message: "the families above and below the TCN at C = 0.2 have changed")
#assert(val("q2_tcan_best"), message: "TCAN is no longer the best family at C = 0.2 by more than the seed range")
#assert(val("q2_offline_span_close"), message: "the top offline families at C = 0.2 now span more than 1.5 times the seed range")
#assert(val("q2_scnn_pair"), message: "SCNN against TCN no longer ties on Michelson and wins at C = 0.2")
#assert(val("q2_learned_beat_oracle_smi"), message: "a trained family no longer beats the oracle at C = 0.2")
#assert(val("q2_stalled_named"), message: "the stalled cells are no longer the two LSTMs and the offline neural operator at C = 0.2")
#assert(val("q2_sliding_named"), message: "the sliding-window families are no longer Barland and the MLP")
#assert(val("q2_matched_named"), message: "the fixed-window families are no longer the eight named in the context paragraph")
#assert(val("q2_barland_ties_scnn"), message: "Barland and SCNN no longer tie at C = 0.2")
#assert(val("q2_oracle_worse_than_zero"), message: "the oracle's signed C = 0.2 cell no longer scores worse than the zero predictor")
#assert(val("q2_mamba_trained"), message: "Mamba no longer trains on both sims")

Without feedback no family separates from the TCN, and all sit far above the
classical oracle; with feedback TCAN is best, and the LSTMs and the offline
neural operator do not train under this protocol (@fig-families). Every
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
    Without feedback (Michelson, hollow) every family but the offline neural
    operator lies
within the #val("q2_seed_range_pct")% seed range of the TCN
    (#val("q2_mich_lo") to #val("q2_mich_hi") $mu$m/ms), all
    #val("q2_oracle_gap") times or more above the oracle; with feedback TCAN is
    best, #val("q2_tcan_gain") times below the TCN and
    #val("q2_tcan_margin_pct")% below the next family, from one seed. Both
    LSTMs, and the offline neural operator at $C = 0.2$, never leave the
    zero-output plateau (crosses: best validation error within
    #val("q2_stall_tol_pct")% of the zero predictor's), even with a 30-epoch
    grace; Mamba trains
    on both. At $C = 0.2$ four offline families beat the TCN within
    #val("q2_offline_span_pct")% of each other, so one seed does not order
    them, and SCNN, the TCN with symmetric padding, ties it on Michelson and
    beats it by #val("q2_scnn_gain_pct")%. Noise-free test velocity RMSE of
    each family at 44k parameters, one seed per cell. Bands: the TCN's value
    $plus.minus$ the seed range, so a family outside a band is resolved from
    it. The oracle, a classical inverse fit given the true excursion, decodes
    under the Michelson model, so its signed $C = 0.2$ cell is misspecified
    and scores worse than the zero predictor (#val("q2_oracle_smi_signed")
    against #val("q2_zero_smi") $mu$m/ms).
  ],
) <fig-families>

At default initialization the LSTM's sensitivity to past input decays within
a few samples, while velocity must be read over a fringe period of hundreds;
chrono gate initialization and a larger learning rate did not move it. The
offline neural operator mixes a few global Fourier modes and has no local path
at the fringe frequency.

#assert(val("disp_lowest_named"), message: "the three lowest families on displacement at C = 0.2 are no longer the two offline transformers and TCAN, tied with each other and below the TCN")
#assert(val("disp_mamba_last"), message: "Mamba is no longer last on displacement at C = 0.2, tied with the TCN on both scores")
#assert(val("disp_mich_spread"), message: "the Michelson families tied on velocity no longer spread beyond the displacement seed range")
#assert(val("disp_curve_monotone"), message: "the 90-epoch displacement curve is no longer monotone in C")

Displacement RMSE ranks the families differently from velocity RMSE
(@fig-displacement). Displacement is what the instrument reports, and it
weights the velocity error spectrum by $1 \/ f^2$ (@fig-spectral), so slow
error counts for more than in velocity RMSE. We rescore the checkpoints and
frozen test shots of @fig-families without retraining. Three seeds of the
noise-free base span #val("disp_seed_range_pct")% of its mean displacement
error, against #val("q2_seed_range_pct")% for velocity, and that wider range
decides which families are resolved. Why integration reorders the families
is untested, and a grid that retrains the stalled families and a TCN control
on the displacement target is still running.

#fig(
  "displacement_rmse",
  width: 80%,
  caption: [
    Displacement does not keep the families' velocity ranking. With feedback,
    TCAN, best on velocity, ties the offline PatchTransformer and
    UNet-Transformer at #val("disp_lowest_lo") to #val("disp_lowest_hi")
    #box[$mu$m], against the TCN's #val("disp_tcn") #box[$mu$m]. Mamba, tied
    with the TCN on velocity, is last at #val("disp_mamba") #box[$mu$m] but
    within the displacement seed range of the TCN. Without feedback, the
    families tied on velocity span #val("disp_mich_span") times in
    displacement. (a) The TCN against $C$, as in #ref(<fig-feedback>)a,
    falling to #val("disp_gain_top") times below Michelson (dotted,
    sign-invariant) at $C = #val("disp_c_top")$. (b) Displacement against
    velocity RMSE for every trained family of @fig-families, one seed each:
    hollow on Michelson, scored under each shot's better global sign; filled
    at $C = 0.2$, signed.
  ],
) <fig-displacement>

Each cell is one seed at one budget, noise-free, and budgets differ:
#val("q2_early_n") trained cells stopped by patience before the restart at
epoch #val("q2_restart"), and #val("q2_second_n") overran into a second
cycle. The one gain well outside the seed range is TCAN over the TCN at
$C = 0.2$; more seeds of both would test it.

= Predictions for the real data <sec-real-predictions>

// Published real-data numbers, defined once (baseline report, report.typ):
// the free-space causal TCN's test velocity RMSE (Section 9.1 results table,
// mean over two seeds) and the free-space median true velocity RMS (Section
// 4.1), which is the zero predictor's per-shot error.
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
baseline report's results table (its Section 9.1): the causal TCN at
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
    Errors are test velocity RMSE under the baseline report's protocol;
    ratios are the single channel's or the TCN's error over the comparator's.
    $s$ is the seed range P6 measures on the real runs at two seeds. // math-ok: stage label
  ],
) <tab-real-predictions>

The published bars already show that the rig breaks the global sign symmetry
of @sec-sign. Those runs trained with a signed loss, the plain `MSELoss` of
`lit_module.py` at commit `556ab18`. On Michelson data that loss leaves the
TCN at the zero predictor (@fig-exit-check). The free-space TCN instead
scores #real-ratio of the median true velocity RMS of #real-v-rms-fs
$mu$m/ms (baseline report, Section 4.1), which is the zero predictor's
per-shot error. That median comes from a different 200-shot subset, so the
ratio is approximate and likely a ceiling. Prediction 1 holds the sign
resolved on nearly every test shot, as in every 90-epoch cell of
@sec-q3-curve. Its interval widens the #val("q3_sign_gap_pct")% noise-free
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

The same ratio puts the rig well below the synthetic nominal point. Over the
zero predictor, the 30-epoch ladder cells of @sec-q3-noise score
#val("pred_base_ratio") noise-free, #val("pred_nominal_ratio") at the
nominal point, #val("pred_no_drift_ratio") with phase drift removed and
#val("pred_c09_ratio") at $C = 0.9$. The real #real-ratio clears the nominal
point by a wide margin but sits within its own approximation of the $C =
0.9$ cell. It lies below the no-drift cell too, so drift alone cannot explain
the rig. The 60-epoch cells, which give the #val("loo_e60_cut")% drift cut,
improved the nominal point by only #val("loo_e60_nominal_gain")%, so
training length does not close the gap. Hence the nominal knobs or the
synthetic motion prior overstate the difficulty, and a narrower real motion
prior is an equal explanation. The drift-dominance verdict of @sec-q3-noise
may then not describe the rig, as its call to measure the drifts allowed.

Prediction 3 tests the knob half. Each bar clears nominal by the round
trip's acceptance bar, 20% or 1 dB for SNR, and a reading between nominal
and its bar neither confirms nor falsifies. A knob returned as None fails its
clause and drops out of the falsifier, and with all three None the
prediction is unscored. Drift is not a clause, because @sec-q3-noise takes
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

The Q1 gain should fall between its two synthetic limits (@sec-q1). // math-ok: question label
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

The Q2 predictions claim orderings only, because the synthetic family ranks // math-ok: question label
already transferred imperfectly once. On the real data Barland beats SCNN by
15% (baseline report, Section 9.1), where the synthetic cells tie them
(@fig-families). TCAN's #val("q2_tcan_gain")-fold gain over the TCN was
measured noise-free; under noise, the Q1 multi-channel gain shrank from // math-ok: question label
#val("q1_gain_tcn") to #val("q1_s4_30_gain_pct")%, so prediction 8 runs from
parity to that factor. A ratio between $1 - s$ and 1, or above
#val("q2_tcan_gain"), neither confirms nor falsifies it. The LSTMs stalled on
both simulations, with and without the grace period. Prediction 9 applies
only to the same protocol of default initialization, signed loss and
patience. Mamba has no prediction, because its $C = 0.2$ cell had not run at
this commit.

Prediction 10 covers SCNN, the MLP, both transformers, the offline neural
operator and the bidirectional LSTM. It rests on the published real ranking,
where Barland leads every configuration on both acquisitions (baseline
report, Section 9.1), and not on the synthetic ties. That ranking includes
only SCNN and the MLP, so for the other four families it is an extrapolation.

The program makes no prediction about the fiber penalty. It has no model of
mode scrambling, which is Q4 and outside P0 to P3, so the published fiber // math-ok: question and stage labels
over free-space ratios of 2.28 for the TCN and 2.02 for Barland (baseline
report, Section 9.6) have no synthetic counterpart. Every prediction above
is for free space.

#bibliography("refs.bib")
