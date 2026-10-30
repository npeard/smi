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

== The fixed-budget curve <sec-q3-curve>

Within the weak-feedback regime, more feedback is better for the causal TCN.
At the 90-epoch budget its signed error falls monotonically from $C = 0.05$
to $C = 0.9$, and every SMI cell lies below the Michelson sign-invariant
error (@fig-feedback a). At $C = #val("q3_c_top")$ the RMSE is
#val("q3_gain_top") times below Michelson's. Each cell trains the TCN of
@sec-sign with the signed loss on noise-free three-channel data at
$alpha = 3$, under one 90-epoch cosine cycle. A run ends at a 150-epoch cap
or when its validation error has not improved by 1% over 10 checks, and is
scored at its best checkpoint.

#fig(
  "feedback_curve",
  caption: [
    Within weak feedback, the causal TCN's noise-free error falls
    monotonically with $C$ and stays below the Michelson sign-invariant level.
    (a) Velocity RMSE and (b) NRMSE, each the mean over the
    #val("test_shots") frozen test shots (seed #val("test_seed")), signed for
    SMI. TCN-44k, three channels, $alpha = 3$, one training seed. Solid: one
    90-epoch cosine cycle; dashed: a 30-epoch cycle. Dotted lines: Michelson,
    trained and scored sign-invariant, at the matching budget. Hollow markers:
    the $alpha = 5$ and $alpha = 0$ controls at $C = 0.2$; the two budgets'
    $alpha = 0$ runs coincide.
  ],
) <fig-feedback>

The sign is resolved at every $C$ sampled, so the fall with $C$ is in the
magnitude error. Signed and sign-invariant RMSE agree within
#val("q3_sign_gap_pct")% in every cell of the 90-epoch curve. @sec-predictions
allowed an optimum below $C = 1$ if sawtooth fringes were harder for the
estimator to read, and none appears up to $C = 0.9$ at this budget. Without
noise there is no information limit to approach, so this curve measures how
well the estimator reads the tilted fringe, not the information of
@eq-phase-factor.

Normalized per shot, the gain over Michelson shrinks to
#val("q3_gain_top_nrmse") times at $C = #val("q3_c_top")$
(@fig-feedback b), because the two errors scale differently with speed. In
the records' per-sample errors binned by $|v|$ decile, Michelson's
sign-invariant error rises from #val("q3_mich_lo") $mu$m/ms in the slowest
decile to #val("q3_mich_hi") in the fastest. At $C = #val("q3_c_top")$ it
goes only from #val("q3_top_lo") to #val("q3_top_hi") $mu$m/ms. Feedback
therefore gains most on fast motion, which dominates the mean RMSE, and
nothing in the slowest decile, which weighs heavily in the mean NRMSE.

The $alpha$ controls at $C = 0.2$ behave as @sec-feedback predicts. With
$alpha = 0$ the fringe stays even and the signed-MSE optimum is zero. Both
$alpha = 0$ runs stop by patience after #val("q3_a0_epochs") epochs at the
zero-predictor level, NRMSE #val("q3_a0_nrmse"). With $alpha = 5$ the error
is within #val("q3_a5_gap_pct")% of $alpha = 3$ at both budgets.
@eq-sign-info depends on $alpha$ only through $alpha^2 \/ (1 + alpha^2)$,
which changes by a factor #val("sign_info_a5_a3") between the two.

If the advantage survives noise, the operating target is the top of the
weak-feedback range, near the largest $C$ sampled here. Three levers set
$C$: the attenuation of the return path, including the target's
reflectivity; the coupling and focus of the returned light into the laser
cavity; and the external cavity length. Which of them is practical on the
rig is untested, and the moderate regime $C > 1$, with its hysteresis, is out
of scope.

Every value here holds for a fixed budget and still moves with it. Tripling the cosine
cycle from 30 to 90 epochs lowers the $C = 0.2$ error from
#val("q3_c02_t30") to #val("q3_c02_t90") $mu$m/ms and Michelson's from
#val("q3_mich_t30") to #val("q3_mich_t90"). One #val("q3_t270_cycle")-epoch
cycle at $C = 0.2$ stopped by patience at epoch #val("q3_t270_epochs"),
early in its decay, at #val("q3_c02_t270") $mu$m/ms, so convergence is not
demonstrated. At 30 epochs the curve inverts once: the $C = 0.4$ run lasted
#val("q3_t30_c04_epochs") epochs against #val("q3_t30_other_epochs_lo") to
#val("q3_t30_other_epochs_hi") for the other cells, and lies below
$C = 0.7$ and $0.9$. Each cell is one training seed, and $alpha = 5$ is
sampled only at $C = 0.2$.

== Noise ladder <sec-q3-noise>

_To be written from `smi/analysis/data/synthetic_results.json` when the
noise-ladder grid finishes: which noise term moves the error most at the
nominal operating point, and the case it makes for or against new
electronics._

= Q1: does multi-channel beat any single channel? <sec-q1> // math-ok: question label

_To be written from `smi/analysis/data/synthetic_results.json` when the
Q1 grid // math-ok: question label
 finishes._

#bibliography("refs.bib")
