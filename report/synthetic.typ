// Research report: what synthetic data and analysis say about velocity
// retrieval from multi-wavelength self-mixing fringes, before any real
// acquisition is unblinded.
//
// Build:  pixi run report-synthetic
//
// Every number in this document comes from a committed artifact or from a
// computation in a committed script:
// - smi/analysis/data/exit_check_results.json (the three exit-check runs of
//   section 1.1, copied from the Task 10 diagnostic with its provenance);
// - smi.analysis.bounds, evaluated by the figure scripts
//   report/figures/plot_{fisher_phase,velocity_bounds,sign_information}.py;
// - build/synthetic_values.json, written by report/figures/synthetic_values.py
//   from smi.analysis.bounds and learned_results.json: every value quoted in
//   section 1's prose and in its table is read from it below, so a library
//   change rebuilds the prose rather than leaving it stale.
// Sections 2 and 3 are written from smi/analysis/data/synthetic_results.json
// once the experiment grids finish.
//
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
every bound is an RMS pooled over trajectories. The real acquisitions stay
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
metrics agree, so the sign is resolved on nearly every test shot. None of the
three runs had converged, so only the ordering, not the absolute values, is
evidence.

#fig(
  "exit_check",
  width: 62%,
  caption: [
    The Michelson sign collapse, and its absence under weak feedback.
    Velocity RMSE on a frozen noise-free test set of 64 shots (seed 1234),
    three channels, for the causal TCN trained for 20 epochs of 200 steps.
    Signed RMSE compares the prediction with the truth as is; sign-invariant
    RMSE takes the better of $plus.minus hat(v)$ per shot. Michelson runs use
    $C = 0$; the SMI run uses $C = 0.2$, $alpha = 3$. The dashed line is the
    mean per-shot std of the true velocity, about the error of always
    predicting zero. NRMSE, in the same artifact, orders the runs alike.
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
$mu$m/ms, exceeds the mean per-shot std of @fig-exit-check because the drive
amplitude is drawn per shot.

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

_To be written from `smi/analysis/data/synthetic_results.json` when the
Q3 grid // math-ok: question label
 finishes._

= Q1: does multi-channel beat any single channel? <sec-q1> // math-ok: question label

_To be written from `smi/analysis/data/synthetic_results.json` when the
Q1 grid // math-ok: question label
 finishes._

#bibliography("refs.bib")
