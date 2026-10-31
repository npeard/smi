// Research report: what synthetic data and analysis say about velocity
// retrieval from multi-wavelength self-mixing fringes, before any real
// acquisition is unblinded.
//
// Build:  pixi run report-synthetic
//
// Every number in this document comes from a committed artifact or from a
// computation in a committed script:
// - smi/analysis/data/synthetic_results.json, the experiment runner's records,
//   read by report/figures/plot_{exit_check,feedback_curve}.py;
// - smi.analysis.bounds, evaluated by the figure scripts
//   report/figures/plot_{fisher_phase,velocity_bounds,sign_information}.py;
// - build/synthetic_values.json, written by report/figures/synthetic_values.py
//   from smi.analysis.bounds, learned_results.json and synthetic_results.json:
//   every value quoted in the prose and in the table is read from it below, so
//   a library or artifact change rebuilds the prose rather than leaving it
//   stale.

// ASCII only, per the repo convention.

#import "lib/theme.typ": fig, report-theme, takeaway

#show: report-theme.with(
  title: "Velocity from self-mixing fringes: a synthetic study",
  subtitle: "Identifiability, information bounds, and what feedback and extra channels buy",
  author: "Nolan Peard",
  date: "Draft",
)

#set math.equation(numbering: "(1)")

#let values = json("/build/synthetic_values.json")
#let val(key) = values.at(key)
#let sci(cell) = $#cell.mantissa times 10^#str(cell.exponent)$

This report asks, on synthetic data where the true motion is known exactly,
what limits velocity retrieval from three self-mixing photodiode channels at
635, 674.8 and 515 nm. Every error is the velocity RMSE of each shot, in
$mu$m/ms, averaged over the shots of a frozen, seeded synthetic test set;
NRMSE divides each shot's RMSE by that shot's velocity std before averaging.
Every bound is an RMS pooled over trajectories. The real acquisitions stay
unread until the synthetic verdicts are fixed; real-data numbers appear only
as context, quoted from the baseline report.

= Identifiability, information and bounds <sec-bounds>

Each channel $k$ of the simulator reads
$
  y_k (t) = a_k (t) [1 + V_k (t) F(theta_k (t))] - 1 + n_k (t),
  quad theta_k = (4 pi d) / lambda_k + phi_k ,
$ <eq-signal>
where $d$ is the mirror displacement, $lambda_k$ the wavelength, $phi_k$ an
unknown phase offset drawn uniformly on $[0, 2 pi)$, $a_k$ and $V_k$ the
amplitude and visibility, and $n_k$ the detector noise. For a Michelson
interferometer $F = cos$. The training target is the velocity $v = dot(d)$.

== The Michelson global sign is unobservable <sec-sign>

In the Michelson model no number of channels can tell $v(t)$ from $-v(t)$.
The map $d -> -d$, $phi_k -> -phi_k$ sends every $theta_k$ to $-theta_k$, and
$cos$ is even, so it leaves every channel of @eq-signal unchanged. The
uniform law of $phi_k$ is invariant under $phi_k -> -phi_k$ modulo $2 pi$, so
integrating the offsets out gives
$
  p(y | d) = integral p(y | d, phi) p(phi) dif phi
  = integral p(y | -d, -phi) p(-phi) dif phi = p(y | -d) .
$ <eq-sign-likelihood>
Nothing else in @eq-signal depends on the sign of $d$.

The minimum-MSE signed velocity estimate is therefore zero. The drive is a
sum of lines with independent uniform phases, so the trajectory prior is also
symmetric, $p(d) = p(-d)$. With @eq-sign-likelihood the posterior is
symmetric, $E[v | y] = 0$, and the best achievable signed MSE is the prior
variance of $v$. A network trained with signed MSE on Michelson data should
collapse toward predicting zero. A network trained to recover the motion must
instead be scored per shot up to the global sign, with the sign-invariant
loss $min_(epsilon in {plus.minus 1}) "MSE"(epsilon hat(v), v)$.

@fig-exit-check confirms both predictions. The causal TCN (44k parameters, 85-sample lookback) trained with signed MSE
on Michelson data scores at the zero-predictor level on both metrics. The
same network trained with the sign-invariant loss recovers the motion up to
sign, and its signed error, with a random sign per shot, exceeds the zero
predictor's. Under weak feedback with signed MSE the two
metrics agree, so the sign is resolved on nearly every test shot. The signed
Michelson run stopped by patience after #val("exit_signed_epochs") epochs at
the zero-predictor level, and the other two ran to the
#val("exit_max_epochs")-epoch cap, so only the ordering, not the absolute
values, is evidence.

#fig(
  "exit_check",
  width: 62%,
  caption: [
    The Michelson sign collapse, and its absence under weak feedback.
    Velocity NRMSE on a frozen noise-free test set of #val("test_shots")
    shots (seed #val("test_seed")), three channels, for the causal TCN trained
    for at most #val("exit_max_epochs") epochs of #val("exit_steps") steps.
    Signed NRMSE compares the prediction with the truth as is; sign-invariant
    NRMSE takes the better of $plus.minus hat(v)$ per shot. Michelson runs use
    $C = 0$; the SMI run uses $C = 0.2$, $alpha = 3$. The dashed line,
    NRMSE $= 1$, is the error of always predicting zero. RMSE, in the same
    records, orders the runs alike.
  ],
) <fig-exit-check>

== Feedback breaks the symmetry; one channel only through the prior <sec-feedback>

Under weak optical feedback the fringe is $F(theta) = cos phi_F$, where the
feedback phase $phi_F$ solves the excess-phase equation
$
  phi_F = theta - C sin(phi_F + beta), quad beta = arctan alpha ,
$ <eq-excess>
with feedback strength $0 <= C < 1$ and linewidth enhancement factor
$alpha$. The map $phi_F |-> phi_F + C sin(phi_F + beta)$ has derivative at
least $1 - C > 0$, so @eq-excess has exactly one root for each $theta$.

The symmetry of @sec-sign survives feedback when $alpha = 0$. With
$beta = 0$, negating @eq-excess shows that $-phi_F$ solves it at $-theta$; by
uniqueness $phi_F (-theta) = -phi_F (theta)$, so $F$ is still even and
@eq-sign-likelihood holds for every $C$. For $C > 0$ and $alpha != 0$ the
fringe tilts toward a sawtooth whose slow and steep edges swap with the
direction of motion.

With one channel the symmetry is broken only through the motion prior. The
warped phase $theta'' = phi_F + C sin(phi_F - beta)$ satisfies
@eq-excess at $-theta''$ with root $-phi_F$, so
$
  F(-theta'') = F(theta) quad "exactly", quad
  theta'' - theta = -2 C sin beta cos phi_F .
$ <eq-warp>
A sign-flipped trajectory, displaced by a warp of depth
$C sin beta thin lambda \/ (2 pi)$ that follows the fringe, therefore
reproduces a single channel noise-free. Only the motion prior's penalty on
that warp separates the two hypotheses. With two or more channels the warps
of @eq-warp differ, because each depends on its own $phi_F$ and $lambda_k$,
so this mirror does not carry over.

We bound the sign evidence from above by holding the trajectory fixed. For each
shot and channel, the sign information is the Kullback-Leibler divergence
between the noise law of the true signal and that of the closest
sign-flipped signal $m + b F(-4 pi d \/ lambda_k + phi')$, with offset $m$,
gain $b >= 0$ and phase $phi'$ free. The gain is kept non-negative because
photodiode polarity is fixed and known. Freeing the trajectory can only lower
the value, to the prior-limited level for one channel. To leading order in
$C$, for $L$ samples per shot, noise std $sigma$ and uniform phase coverage,
$
  I_"sign" = sum_k (L C^2 E[V_k^2] alpha^2) / (4 sigma^2 (1 + alpha^2)) .
$ <eq-sign-info>
The derivation expands $F = cos theta + C sin theta sin(theta + beta) +
O(C^2)$. The unwarped mirror ($b = 1$, $phi' = -phi$) then leaves the
residual $V C sin beta sin 2 theta$, which is orthogonal to the directions
$(m, b, phi')$ can move.

#fig(
  "sign_information",
  width: 62%,
  caption: [
    An upper bound on the sign evidence grows as $C^2$ at weak feedback, and
    three channels carry about three times one channel's. The bound is for
    one 16384-sample shot at SNR 30 dB and $V tilde U(0.5, 1)$, mean over 8
    shots (seed 0), for one
    channel (635 nm) and all three. Markers are the computed value; lines are
    the leading-order form of @eq-sign-info, which is the same for both
    $alpha$ to within the marker size. The value is exactly zero at $C = 0$
    and at $alpha = 0$.
  ],
) <fig-sign>

Because the bound is far above one nat per shot over the whole weak-feedback
range (@fig-sign), no mirror on the true trajectory reproduces the signals;
that is numerical, not proven. The bound does not quantify the evidence once the trajectory is free. For two or more
channels, whether some other trajectory mirrors every channel, for example
one that switches fringe branch at the extrema, is an exposed gap.

#figure(
  align(center, table(
    columns: 5,
    align: (left, right, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header(
      [$C$],
      [$alpha = 3$, $K = 1$],
      [$alpha = 3$, $K = 3$],
      [$alpha = 5$, $K = 1$],
      [$alpha = 5$, $K = 3$],
    ),
    ..values.sign_table.map(row => ([#row.c], ..row.cells.map(sci))).flatten(),
  )),
  caption: [
    Upper bound on the sign evidence (nats per 16384-sample shot) at the
    settings of @fig-sign. $K = 1$ is the 635 nm channel.
  ],
) <tab-sign>

== Fisher information per sample <sec-fisher>

Away from the global sign, how precisely a sample fixes $d$ is set by its
Fisher information. For channel $k$ at unit amplitude, with white Gaussian
noise of std $sigma$,
$
  J_k (theta) = ((4 pi V_k) / (lambda_k sigma))^2 F'(theta)^2, quad
  F'(theta) = - (sin phi_F) / (1 + C cos(phi_F + beta)) ,
$ <eq-fisher>
where $F'$ follows from implicit differentiation of @eq-excess. The factor
$F'^2$ vanishes at the fringe extrema, $sin phi_F = 0$, where a channel is
blind to motion (@fig-fisher). The channel noises are independent, so the
information of $K$ channels is the sum of the $J_k$. Because $J_k$ scales as
$lambda_k^(-2)$, the 515 nm channel carries $(0.635 \/ 0.515)^2 = 1.52$ times
the information of the 635 nm channel at equal visibility and noise.

#fig(
  "fisher_phase",
  width: 62%,
  caption: [
    Feedback concentrates the information on the steep fringe edge and leaves
    the blind spots. $F'(theta)^2$, the phase-dependent factor of
    @eq-fisher, over one fringe at $alpha = 3$. Dashed lines are its phase
    averages $Phi(C, alpha)$, which the averaged bounds use.
  ],
) <fig-fisher>

Averaged over a uniform fringe phase, the factor has a closed form,
$
  Phi(C, alpha) = chevron.l F'^2 chevron.r_theta
  = 1 / (2 s) [1 - cos(2 beta) (1 - s) / (1 + s)], quad s = sqrt(1 - C^2) .
$ <eq-phase-factor>
Substituting $theta -> phi_F$ in the average, which is monotone for $C < 1$,
reduces it to two standard integrals over a uniform angle,
$chevron.l 1 \/ (1 + C cos u) chevron.r = 1 \/ s$ and
$chevron.l cos 2u \/ (1 + C cos u) chevron.r = (1 - s) \/ (s (1 + s))$.
The Michelson limit is $Phi = 1 \/ 2$. For $|alpha| > 1$, $cos 2 beta < 0$,
so feedback adds magnitude information: at $alpha = 3$,
$Phi = #val("phase_factor_c02")$ at $C = 0.2$ and #val("phase_factor_c07") at
$C = 0.7$.

== Causal and offline Bayesian bounds <sec-crb>

The time-series bounds combine @eq-fisher with a model of how the motion
evolves. We fit a Gauss-Markov surrogate prior to the simulator's motion: the
state $x = (d, v, dot(v))$ is driven by white noise through one real pole and
one resonance, discretized exactly at the sample rate. The fit matches the
in-band velocity line powers of 256 seeded trajectories to
#val("prior_psd_rms_db") dB rms, and it reproduces their velocity and
displacement variances. Its pooled velocity std, #val("prior_velocity_std")
$mu$m/ms, exceeds a typical shot's own std because the drive amplitude is
drawn per shot.

The causal bound is the posterior Cramer-Rao bound of @tichavsky1998pcrb. For
linear-Gaussian dynamics $x_(n+1) = A x_n + w_n$, $w_n tilde cal(N)(0, Q)$,
its information recursion reduces to
$
  J_(n+1) = (Q + A J_n^(-1) A^T)^(-1) + macron(J) c_d c_d^T ,
$ <eq-pcrb>
where $c_d$ selects $d$ from the state and $macron(J)$ is the per-sample
information summed over channels. That is the covariance recursion of a
Kalman filter that measures $d$ linearly. The offline bound is the Bayesian
(Van Trees) bound @vantrees1968detection, whose information matrix is the
block-tridiagonal prior information plus $macron(J) c_d c_d^T$ on each
block; its inverse diagonal blocks are the surrogate's smoother covariances.
Since $theta_k mod 2 pi$ is uniform whatever $d$ is, $macron(J) = sum_k
E[V_k^2] Phi(C, alpha) (4 pi \/ (lambda_k sigma))^2$ exactly, with no Monte
Carlo. Averaging the information over the phase before inverting gives
$(E J)^(-1) <= E[J^(-1)]$ by Jensen's inequality, so the averaged bound is
valid and looser than one conditioned on the phases.

#fig(
  "velocity_bounds",
  width: 62%,
  caption: [
    Extra channels lower the velocity bound more than feedback does.
    Stationary RMS velocity bounds at SNR 30 dB, $V tilde U(0.5, 1)$ and
    $alpha = 3$. Causal is the posterior CRB of @eq-pcrb after convergence;
    offline is the smoothing bound at mid-record. $K = 1$ is 635 nm, $K = 2$
    adds 674.8 nm and $K = 3$ adds 515 nm.
  ],
) <fig-bounds>

The bound depends only weakly on both the channel count and $C$
(@fig-bounds). At the nominal point ($K = 3$, $C = 0.2$) the causal bound is
#val("causal_k3") $mu$m/ms and the offline bound #val("offline_k3") $mu$m/ms.
Going from one channel to three multiplies $macron(J)$ by
#val("info_ratio_k3_k1") but lowers the causal bound only by a factor
#val("causal_ratio_k3_k1"), close to $macron(J)^(-1 \/ 4)$.

The causal bound is #val("real_gap") times below the causal TCN's error of
#val("real_tcn_rmse") $mu$m/ms on the real free-space test split in the
baseline report. That gap measures the bound's omissions as much as any
estimator's shortfall, since within its own model the bound is nearly
attained. In `tests/test_bounds.py` an extended Kalman filter given the
genie phases, on three-channel $C = 0.2$ trajectories from the surrogate
prior, must score below twice the causal bound and above it within three
standard errors.

The bounds are exact for a simpler model than the simulator, so they are not
a floor for it. Four simplifications separate the two.
+ They are exact for the fitted Gauss-Markov prior, not
  for the simulator's motion. That motion is a finite sum of drive lines with
  a per-shot amplitude drawn from $U(0, 1)$. An estimator that knows the line
  structure could exploit it beyond the surrogate.
+ The offsets $phi_k$ and visibilities $V_k$ are known
  to the bound, and the amplitude is 1.
+ Amplitude, contrast and phase drift, DC offset, the
  photodiode low-pass and the ADC are absent; drift in particular raises
  $E[a^2 V^2]$ above the value used here.
+ They are local and sign-resolved: they see neither the global sign nor the
  fringe order. At $C = 0$ or $alpha = 0$ they stay finite while the
  signed-MSE floor is the prior variance (@sec-sign), so there they compare
  only with a sign-invariant RMSE.

== Predictions for Q3 and Q1 <sec-predictions> // math-ok: question labels

*Q3, accuracy against feedback.* With three channels, signed error should // math-ok: question label
fall from the zero-predictor level at $C = 0$ to the sign-invariant level at
$C > 0$, as the SMI run of @fig-exit-check already does at $C = 0.2$, $alpha = 3$.
Whether one channel reads the sign is open. Its evidence comes only through
the motion prior (@eq-warp), but the warp that mirrors it has velocity
amplitude about $2 C |v| sin beta$ at the fringe rate $2|v| \/ lambda$, a
fast component the prior may penalize heavily. Because $Phi$ rises with $C$
at these $alpha$ (@eq-phase-factor), an optimum of the sign-invariant error
below $C = 1$ would come from the estimator's difficulty with sawtooth
fringes, not from lost local Fisher information. For an efficient causal
estimator, 30 dB white noise costs at least the bound of @fig-bounds. The
noise ladder measures what it costs the TCN, by training the same
architecture and budget noise-free and at 30 dB.

*Q1, multiple channels.* // math-ok: question label
@eq-fisher predicts that a single channel loses all
information at its fringe extrema, and incommensurate wavelengths put the
extrema of different channels at different times. A three-channel estimator
should therefore gain over the best single channel mainly near fringe
extrema, and elsewhere by the modest factor of @fig-bounds. This includes
velocity turning points, where the fringes stall and one channel can dwell at
an extremum while the others do not. The test is error binned by distance to
the nearest fringe extremum: it should rise toward the extremum for one
channel and stay flat for three.

= Q3: noise-free accuracy against feedback $C$ <sec-q3> // math-ok: question label

#assert(val("q3_monotone"), message: "the 90-epoch C curve is no longer monotone")
#assert(val("q3_t30_c04_below"), message: "the 30-epoch C = 0.4 inversion is gone")
#assert(
  val("q3_slow_falls_with_budget"),
  message: "the slowest-decile floor no longer falls from 30 to 90 epochs",
)

== The fixed-budget curve <sec-q3-curve>

In RMSE at the 90-epoch budget, the causal TCN's signed error falls
monotonically from $C = 0.05$ to $C = 0.9$, and every $alpha = 3$ cell lies
below the Michelson sign-invariant error (#ref(<fig-feedback>)a). At $C = #val("q3_c_top")$ the RMSE is
#val("q3_gain_top") times below Michelson's. Each cell trains the TCN of
@sec-sign with the signed loss on noise-free three-channel data at
$alpha = 3$, under cosine annealing with warm restarts, a 90-epoch period and
no warmup. Patience (no 1% gain in validation error over 10 epochs) ends
every run on the curve at epoch #val("q3_t90_epochs_lo") to
#val("q3_t90_epochs_hi"), just after the first restart, so each run is one
cosine cycle, scored at its best checkpoint.

#fig(
  "feedback_curve",
  caption: [
    Feedback brings the causal TCN's noise-free RMSE below Michelson's
    sign-invariant level at every $C$ by removing the part of Michelson's
    error that grows with $|v|$, but per shot the weakest $C$ does not beat
    Michelson. (a) Velocity RMSE and (b) NRMSE, each the mean over the
    #val("test_shots") frozen test shots (seed #val("test_seed")), signed for
    SMI. TCN-44k, three channels, $alpha = 3$, one training seed. Solid: one
    90-epoch cosine cycle; dashed: a 30-epoch cycle. Dotted: Michelson,
    trained and scored sign-invariant, in its budget's color. Hollow markers:
    the $alpha = 5$ and $alpha = 0$ controls at $C = 0.2$. (c) Sign-invariant
    RMSE of the test samples pooled in each $|v|$ decile, at the decile's
    midpoint speed, for three 90-epoch cells; grey line: slope one.
  ],
) <fig-feedback>

Per shot the gain is smaller, and the weakest feedback does not pay. The
NRMSE exceeds Michelson's at $C <= #val("q3_nrmse_worse_t90")$ for the 90-epoch
cells and at $C <= #val("q3_nrmse_worse_t30")$ for the 30-epoch cells
(#ref(<fig-feedback>)b). At
$C = #val("q3_c_top")$ the NRMSE gain is #val("q3_gain_top_nrmse") times by
mean and #val("q3_gain_top_nrmse_median") by median. The case for more
feedback at low $C$ therefore rests on pooled error, not on every shot.

The two metrics disagree because RMSE pools samples, which fast motion
dominates, while NRMSE divides each shot by its velocity std, which lets
low-amplitude shots dominate. Michelson's
sign-invariant error grows roughly in proportion to $|v|$
(#ref(<fig-feedback>)c), the cost of a wrong local direction between turning
points, and feedback removes that part. Near $|v| = 0$ a direction error
costs little, so the slowest decile, #val("q3_slow_lo") to
#val("q3_slow_hi") $mu$m/ms for Michelson and every $C$, shows the
estimator's own floor. Divided by the small std of low-amplitude shots, that
floor sets the mean NRMSE. On noise-free data it belongs to the fixed-budget
estimator, not to the information of @eq-phase-factor, and it is lower at 90
epochs than at 30 in every cell.

The sign is resolved on all but a few low-amplitude shots. Signed and
sign-invariant RMSE agree within #val("q3_sign_gap_pct")% in every 90-epoch
cell, but in NRMSE, which weights those shots, signed exceeds sign-invariant
by up to #val("q3_n_sign_gap_max_pct")%, at $C = #val("q3_n_sign_gap_max_c")$.
The worst shot's signed NRMSE lies between #val("q3_n_worst_lo") and
#val("q3_n_worst_hi") across the curve. From $C = #val("q3_flip_c")$ up it
reaches 2, the score of a zero-mean shot predicted with the wrong sign.

The $alpha$ controls at $C = 0.2$ behave as @sec-feedback predicts. With
$alpha = 0$ the fringe stays even and the signed-MSE optimum is zero, and both
$alpha = 0$ runs match the zero predictor's test RMSE within
#val("q3_zero_gap_pct")%. They coincide because their best checkpoint is the
first validation, after one epoch at the peak learning rate both schedules
share; patience then stops them at epoch #val("q3_a0_epochs"). With
$alpha = 5$ the error is within #val("q3_a5_gap_pct")% of $alpha = 3$ at both
budgets, consistent with @eq-sign-info's factor $alpha^2 \/ (1 + alpha^2)$,
which noise-free runs that resolve the sign cannot test.

Of the Q3 predictions in @sec-predictions, the sign prediction holds in RMSE, // math-ok: question label
with the low-amplitude exceptions above. No optimum appears below $C = 1$ up
to $C = 0.9$ at this budget, so sawtooth fringes did not defeat the
estimator. Whether one channel reads the sign, and what noise costs, this
curve does not test.

If the advantage survives noise, the operating target for this estimator is
the top of the weak-feedback range sampled, a reading from one seed at a
fixed budget that per shot needs $C$ above #val("q3_nrmse_worse_t90"). Three levers set $C$: the attenuation of the
return path, including the target's reflectivity; the coupling and focus of
the returned light into the laser cavity; and the external cavity length.
Which of them is practical on the rig is untested, and the moderate regime
$C > 1$, with its hysteresis, is out of scope.

Every value here holds for a fixed budget and still moves with it, as the two
curves of #ref(<fig-feedback>)a show. One #val("q3_t270_cycle")-epoch cycle
at $C = 0.2$ stopped by patience at epoch #val("q3_t270_epochs"), early in its
decay, at #val("q3_c02_t270") $mu$m/ms, so convergence is not demonstrated.
The 30-epoch curve inverts once because its $C = 0.4$ run kept improving
after its first restart and trained #val("q3_t30_c04_epochs") epochs, against
#val("q3_t30_other_epochs_lo") to #val("q3_t30_other_epochs_hi") for the
other cells. Each cell is one training seed, and $alpha = 5$ is sampled only
at $C = 0.2$.

== Noise ladder <sec-q3-noise>

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

Phase drift dominates the error at the nominal operating point
(@fig-noise). Each row trains the 30-epoch TCN of @fig-feedback at
$C = 0.2$ with one detector or drift knob switched on at a time. Measured
against the mean of the noise-free base, phase drift alone at its nominal
#val("noise_phase_fringes") fringes per shot gives
#val("noise_phase_share")% of the error that every knob at nominal adds, and
30 dB white noise gives #val("noise_snr_share")%. No other knob gives more
than #val("noise_minor_share_hi")%, against a base seed spread of
#val("noise_spread_share")% on the same scale. How the knobs combine the
ladder cannot show; @tab-loo removes each from the nominal point instead.

#fig(
  "noise_ladder",
  caption: [
    Test velocity RMSE, mean over the
    #val("test_shots") frozen shots (seed #val("test_seed")), of TCN-44k on
    three channels at $C = 0.2$, $alpha = 3$, seed 0, with one knob on and the
    rest noise-free. Ringed: the knob's nominal value. Grey and red bands, each
    with a line at its mean: the noise-free base and every knob at nominal,
    over #val("noise_n_seeds") seeds; dotted: every knob at nominal at
    $C = 0.9$, one seed. Squares and dashed line (phase drift only): the drift
    sweep and the base trained exactly #val("loo_e60_epochs") epochs. Each x axis runs toward the worse signal; the nominal
    phase drift is #val("noise_phase_rad") rad$\/sqrt("s")$.
  ],
) <fig-noise>

A partial cut in phase drift recovers only part of its cost (@fig-noise,
squares). The ladder's drift rows stopped at #val("noise_phase_epochs_lo")
to #val("noise_phase_epochs_hi") epochs, so the sweep was rerun at exactly
#val("loo_e60_epochs"), where the error rises monotonically with drift.
Cutting drift from the nominal #val("noise_phase_fringes") to
#val("noise_e60_low_fringes") fringes per shot keeps
#val("noise_e60_half_kept")% of its cost over the equal-length base. No run
sampled drift between 0 and #val("noise_e60_low_fringes") fringes.

Removing phase drift from the nominal point cuts its error by
#val("loo_phase_cut")%, the only large single removal (@tab-loo). Each cell
is the nominal point with one knob returned to its noise-free value, at the
ladder's budget and one seed. Of the other seven, only full visibility, at
#val("loo_vis_cut")%, and the photodiode bandwidth, at
#val("loo_pdbw_cut")%, fall below the nominal seeds by more than their own
spread.

#figure(
  align(center, table(
    columns: 3,
    align: (left, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header([Knob removed], [RMSE], [Epochs]),
    [none, #val("noise_n_seeds") seeds], [#val("loo_nominal_lo")--#val("loo_nominal_hi")],
    [#val("loo_nominal_epochs_lo")--#val("loo_nominal_epochs_hi")],
    ..val("loo_rows").map(row => ([#row.label], [#row.rmse], [#row.epochs])).flatten(),
  )),
  caption: [
    Test velocity RMSE ($mu$m/ms) on the frozen shots of
    @fig-noise, TCN-44k on three channels at $C = 0.2$, $alpha = 3$, seed 0,
    with the named knob noise-free and the rest at nominal. Epochs: when
    patience stopped the run, under the 30-epoch cycle.
  ],
) <tab-loo>

Training length does not explain the cut. With plateau stopping disabled, the
nominal point, the nominal point without phase drift and the noise-free base
each trained exactly #val("loo_e60_epochs") epochs, one seed each. At equal
length, removing phase drift cuts the nominal error by #val("loo_e60_cut")%,
far more than the nominal seed spread. The same runs leave the ranking unchanged, though the
extra epochs help the base (#val("loo_e60_base_gain")%) far more than the
nominal point (#val("loo_e60_nominal_gain")%), against the ladder's seeds.

White noise and the other knobs do not add. Alone at 30 dB it raises the
base error by #val("noise_snr_cost") $mu$m/ms, but removing it from the
nominal point saves #val("loo_snr_saving") $mu$m/ms. These grids do not
show which other knob its cost depends on.

With every other knob on, removing any one electronic knob (white noise,
the ADC or the bandwidth limit) moves the error by at most
#val("loo_pdbw_cut")%, and that largest move came with the longer training.
No run removed all three together, and the knobs do not add. In the ladder,
the ADC and bandwidth rows rise above the base only at 8 bits or 3 kHz.
White noise alone gives #val("noise_snr_share")% of the nominal excess, so
it is the likely next limit once drift is fixed.

In the simulation, phase drift is an independent random walk per channel,
as per-laser frequency drift would be. A path-length drift shared by all
channels would read as displacement and was not simulated. The measured 1
to 2.7 fringes is the phase wander the baseline report's global fit leaves,
with no source attributed. The lever this supports is per-laser frequency
stability or a per-laser phase monitor, not a common-path stabilization.
Full visibility is an alignment and speckle lever, and raising $C$ to 0.9,
in one seed, cuts the nominal-point error by a factor
#val("noise_c09_ratio"). No grid here tests any of these levers on the rig.

Every nominal value is itself an assumption. The nominal drive gain and phase
drift are anchored to numbers the baseline report published, as context
rather than a calibration, and the phase-drift nominal is a lower bound on
the true rate. SNR, visibility, DC offset, amplitude and contrast drift
and the photodiode bandwidth are stated guesses with wide sweeps. At 20%,
four times their assumed value, amplitude and contrast drift each cost
comparably to 30 dB white noise (@fig-noise). The verdict also rests on the
guessed 30 dB: at 20 dB the white-noise cost doubles, to
#val("noise_snr20_cost") $mu$m/ms. Measuring these drifts and the rig's SNR
would therefore test the verdict above.

The 30 dB row is consistent with the causal floor of @sec-predictions, but
the comparison has no power here. The bound floors the total error of an
efficient estimator, within the simplifications of @sec-crb, not the
increase a knob adds. The noise-free base alone sits
#val("noise_base_over_bound") times above the bound of
#val("noise_snr_bound") $mu$m/ms at the 30 dB row's settings.

Two DC-offset rows, $plus.minus 0.3$ and $plus.minus 1$, score below every
seed of the noise-free base. Both trained #val("noise_dc_low_epochs")
epochs, into a third cosine cycle, against #val("noise_base_epochs_lo") to
#val("noise_base_epochs_hi") for the base seeds. Whether the longer run or
the offset causes the drop is untested, and nothing here rests on it.

Each ladder row and leave-one-out cell is one training seed at one budget,
read against the seed spread of @fig-noise. Budgets also differ by row: #val("noise_n_long") of
the #val("noise_n_rows") rows, the nominal phase-drift row among them, trained #val("noise_long_epochs_lo")
to #val("noise_long_epochs_hi") epochs, against
#val("noise_short_epochs_lo") to #val("noise_short_epochs_hi") for the rest.
The ladder covers three channels at $C = 0.2$, $alpha = 3$ only. Every run
used Python 3.14 and torch 2.15.0 on Apple MPS in float32.

= Q1: does multi-channel beat any single channel? <sec-q1> // math-ok: question label

#fig(
  "spectral_mismatch",
  caption: [
    Shot-averaged power spectral densities, each divided by its own peak, of
    the velocity target (solid), the displacement target (dashed) and each
    channel's instantaneous fringe frequency (colored), over
    #val("spec_shots") simulated shots at $C = 0.2$, $alpha = 3$. (a) Every
    noise knob at nominal; (b) noise-free, with the same trajectories. The
    fringe frequency is the time derivative of the analytic signal's phase
    over $2 pi$; each periodogram is Hann-windowed over the whole
    16384-sample shot. Shaded: the drive band.
  ],
) <fig-spectral>

The displacement target puts #val("spec_x_low_pct")% of its power below
100 Hz, against #val("spec_v_low_pct")% for the velocity target
(@fig-spectral). The lowest resolved bin is #val("spec_bin_hz") Hz, so
drive content below it enters each shot only as a trend. Each channel's
fringe frequency, which follows $|v|$, peaks at twice the velocity target's
peak frequency and keeps power near its peak down to the lowest bin. Above
1 kHz it sits on a broadband floor that the nominal noise raises
(#ref(<fig-spectral>)a against b).

_The channel-subset (S2) results are to be written // math-ok: stage label
from `smi/analysis/data/synthetic_results.json` when the
Q1 grid // math-ok: question label
 finishes._

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

Without feedback no family separates from the TCN, and none comes near the
classical oracle. With feedback, at $C = 0.2$, TCAN beats the TCN by a factor
#val("q2_tcan_gain") and is the best family, causal or offline
(@tab-families). Each family is matched to the TCN's 44k
parameters and trained as in @sec-q3-curve: one seed, a 90-epoch cosine
cycle that #val("q2_second_n") cells overran into a second, noise-free
three-channel data, the signed loss at $C = 0.2$, $alpha = 3$ and the
sign-invariant loss on Michelson. All cells are scored on the frozen test
shots of @fig-feedback. The recurrent families are the LSTM (causal and
bidirectional) and the state-space model Mamba. The neural operator is our
own code, not the `neuraloperator` library.#footnote[The original cells ran
in Python 3.12. The six cells that stalled in that run were rerun with a
grace period that withholds early stopping for the first 30 epochs; they,
and Mamba, ran in Python 3.14 with torch 2.15.0. Every learned cell ran on
Apple MPS in float32.]

#figure(
  align(center, table(
    columns: 4,
    align: (left, right, right, right),
    stroke: 0.5pt + rgb("#d0d0d0"),
    table.header(
      [Family], [Context], [Michelson, sign-invariant], [$C = 0.2$, signed],
    ),
    table.cell(colspan: 4)[_Causal_],
    ..val("q2_causal_rows").map(row => ([#row.label], [#row.context], ..row.cells.map(c => [#c]))).flatten(),
    table.cell(colspan: 4)[_Offline_],
    ..val("q2_offline_rows").map(row => ([#row.label], [#row.context], ..row.cells.map(c => [#c]))).flatten(),
    [Inverse-fit oracle], [whole shot], ..val("q2_oracle_cells").map(c => [#c]),
  )),
  caption: [
    Test velocity RMSE ($mu$m/ms) of each model family at a matched 44k
    parameters, one training seed per cell. Context is the span of input
    samples one output depends on: past samples only for causal rows, both
    sides for offline rows. Stalled: the run never left the zero-output
    plateau, so the cell measures the protocol, not the estimator. Pending:
    Mamba's stage has not yet run, so its context is also unknown (#"--"). The
    oracle is given the true excursion and decodes under the Michelson model,
    so its $C = 0.2$ cell is misspecified and scores worse than the zero
    predictor (#val("q2_zero_smi")).
  ],
) <tab-families>

On Michelson the families do not separate. Every trained family but the
offline neural operator lies between #val("q2_mich_lo") and
#val("q2_mich_hi") $mu$m/ms, a span of #val("q2_mich_span_pct")%, inside the
#val("q2_seed_range_pct")% that three seeds of the noise-free base span
(@sec-q3-noise). That range was measured on the TCN at $C = 0.2$, signed, at
$T_0 = 30$, and serves here as a rough resolution for both columns, so the
TCN ties every other family here.
The offline neural operator, at #val("q2_no_offline_mich") $mu$m/ms, falls
behind the rest. The oracle is the classical inverse fit of the baseline
report, decoding each whole shot under the Michelson model and told the true
excursion. It reaches #val("q2_oracle_mich") $mu$m/ms, #val("q2_oracle_gap")
times below the best learned family. The learned models therefore fall
short of an oracle given the true excursion, and no architecture tried
closes that gap at this budget.

At $C = 0.2$ the families separate. In the causal column TCAN is
#val("q2_tcan_gain") times below the TCN and the causal PatchTransformer
#val("q2_patch_gain") times below it, while the causal UNet-Transformer and
the causal neural operator lie above it by more than the seed range. TCAN
also beats the best offline family by #val("q2_tcan_margin_pct")%, so no
offline family matches it here. Four offline families,
SCNN, Barland, PatchTransformer and UNet-Transformer, beat the TCN, but they
span #val("q2_offline_span_pct")% among themselves, close to the seed range,
and one seed does not settle their order. SCNN is the TCN with symmetric
padding, the one exactly matched causal and offline pair. The two tie on
Michelson, and at $C = 0.2$ SCNN is #val("q2_scnn_gain_pct")% lower than the
TCN, at one seed. The oracle stops being a ceiling here.
Its Michelson model has no feedback term, so it does not read the sign
(signed RMSE #val("q2_oracle_smi_signed") $mu$m/ms). Even scored
sign-invariant it reaches only #val("q2_oracle_smi_si") $mu$m/ms,
#val("q2_oracle_smi_gap") times above the weakest trained family.

Context is matched at #val("q2_matched_lo") to #val("q2_matched_hi")
samples for the TCN, TCAN, SCNN, both PatchTransformers, both
UNet-Transformers and the causal neural operator. The recurrent families see
unbounded context, and the offline neural operator sees the whole shot and
trains on full shots, about four times the samples per step of its siblings'
4096-sample crops. The MLP and Barland records give a field of
#val("q2_window_field") samples, which counts one window. Each output,
however, averages every window that covers it, so a gradient probe finds
they depend on #val("q2_mlp_span") and #val("q2_barland_span") samples. The
wider context bought nothing visible: at $C = 0.2$ Barland ties SCNN
(#val("q2_barland_smi") against #val("q2_scnn_smi") $mu$m/ms), and the MLP
trails the TCN.

Five cells did not train under this protocol, and they are kept out of every
ranking above: the LSTM and the bidirectional LSTM on both sims, and the
offline neural operator at $C = 0.2$. In each, the best validation error
stayed within #val("q2_stall_tol_pct")% of the zero predictor's. Each also stalled in the first run,
without the grace, and stalled again with it, stopping at epoch
#val("q2_stall_epochs"). The diagnosis differs by family. At default
initialization the LSTM's sensitivity to past input decays within a few
samples, while velocity has to be read over a fringe period of hundreds.
In short diagnostic runs at $C = 0.2$, chrono gate initialization and a
larger learning rate did not move it, nor did a 40-epoch run without early
stopping. For the offline neural operator, three diagnostic seeds all
stalled at $C = 0.2$. The likely cause is that it mixes a few global Fourier
modes and has no local path at the fringe frequency, though its training on
full 16384-sample shots is a confound. On Michelson, where it trained, it is
the weakest family. The offline UNet-Transformer on Michelson is the
opposite case: its stall was a slow start. Without early stopping, seed 0
left the plateau at epoch 15, which the 30-epoch grace now covers, and seeds
1 and 2 escaped in their first epoch. #if val("q2_mamba_pending") [Mamba runs as its own stage, on a
parallel scan that matches the sequential loop to float32 rounding; its row
is pending.] else [Mamba ran as its own stage, on a parallel scan that
matches the sequential loop to float32 rounding; its row arrived after this
text was written, and the per-family comparisons above leave it out.]

Each cell is one seed, at one budget, noise-free, with three channels and
the two feedback settings. Budgets also differ by cell. Of the trained cells,
#val("q2_early_n") stopped by patience at epoch #val("q2_early_lo") to
#val("q2_early_hi"), before the restart at #val("q2_restart"), and the rest
ran #val("q2_late_lo") to #val("q2_late_hi"). The one gain well outside the
seed range is TCAN over the TCN at $C = 0.2$ (a factor #val("q2_tcan_gain")); more
seeds of both would test it. The order among families within the seed range
of each other is not resolved.

#bibliography("refs.bib")
