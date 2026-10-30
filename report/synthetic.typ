// Research report: what synthetic data and analysis say about velocity
// retrieval from multi-wavelength self-mixing fringes, before any real
// acquisition is unblinded.
//
// Build:  pixi run report  (or, from report/: snakemake build/synthetic.pdf)
//
// Every number in this document comes from a committed artifact or from a
// computation in a committed script:
// - smi/analysis/data/exit_check_results.json (the three exit-check runs of
//   section 1.1, copied from the Task 10 diagnostic with its provenance);
// - smi.analysis.bounds, evaluated by the figure scripts
//   report/figures/plot_{fisher_phase,velocity_bounds,sign_information}.py and
//   by the same calls for the values quoted in the text and table;
// - report/report.typ and smi/analysis/data/learned_results.json, for the
//   real-data numbers quoted as context only (the program is blinded).
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

This report asks, on synthetic data where the true motion is known exactly,
what limits velocity retrieval from three self-mixing photodiode channels at
635, 674.8 and 515 nm. Every error is a velocity RMSE in um/ms on a frozen,
seeded synthetic test set unless stated otherwise. The real acquisitions stay
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
loss $min_(s in {plus.minus 1}) "MSE"(s hat(v), v)$.

@fig-exit-check confirms both predictions on noise-free three-channel data.
The causal TCN (44k parameters, 85-sample lookback) trained with signed MSE
on Michelson data scores at the zero-predictor level on both metrics. The
same network trained with the sign-invariant loss recovers the motion up to
sign, and its signed error, with a random sign per shot, exceeds the zero
predictor's. Under weak feedback with signed MSE the two
metrics agree, so the sign is resolved on nearly every test shot. None of the
three runs had converged at 20 epochs, so only the ordering, not the absolute
values, is evidence.

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
    std of the true velocity, the RMSE of always predicting zero.
  ],
) <fig-exit-check>

== Weak feedback breaks the symmetry, but one channel cannot use it <sec-feedback>

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
so no single trajectory mirrors every channel and the sign evidence is real.

We bound that evidence from above by holding the trajectory fixed. For each
shot and channel, the sign information is the Kullback-Leibler divergence
between the noise law of the true signal and that of the closest
sign-flipped signal $a + b F(-4 pi d \/ lambda_k + phi')$, with offset $a$,
gain $b >= 0$ and phase $phi'$ free. The gain is kept non-negative because
photodiode polarity is fixed and known. Freeing the trajectory can only lower
the value, to the prior-limited level for one channel. To leading order in
$C$, for $L$ samples per shot, noise std $sigma$ and uniform phase coverage,
$
  I_"sign" = sum_k (L C^2 E[V_k^2] alpha^2) / (4 sigma^2 (1 + alpha^2)) .
$ <eq-sign-info>
The derivation expands $F = cos theta + C sin theta sin(theta + beta) +
O(C^2)$. The exact mirror then leaves the residual $V C sin beta sin 2 theta$,
which is orthogonal to the directions $(a, b, phi')$ can move.

#fig(
  "sign_information",
  width: 62%,
  caption: [
    Sign evidence grows as $C^2$ at weak feedback, and three channels carry about three times
    one channel's. Upper bound on the sign evidence of one 16384-sample shot
    at SNR 30 dB and $V tilde U(0.5, 1)$, mean over 8 shots (seed 0), for one
    channel (635 nm) and all three. Markers are the computed value; lines are
    the leading-order form of @eq-sign-info, which is the same for both
    $alpha$ to within the marker size. The value is exactly zero at $C = 0$
    and at $alpha = 0$.
  ],
) <fig-sign>

The tilt breaks the symmetry: at the nominal SNR the bound is far above one
nat per shot over the whole weak-feedback range (@fig-sign). This is a
numerical result, not a proof; no analytic argument here excludes a phase
shift $s$ with $F(-theta + s) = F(theta)$. It is an upper bound, and for one channel
the true evidence is set by the motion prior, so @fig-sign does not show that
a single-channel estimator can read the sign. Its trend in $alpha$ is weak,
as $sin^2 beta$ in @eq-sign-info predicts. The values in @tab-sign anchor the
figure.

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
    [0.05], [$1.10 times 10^4$], [$3.04 times 10^4$], [$1.18 times 10^4$], [$3.25 times 10^4$],
    [0.2], [$1.72 times 10^5$], [$4.75 times 10^5$], [$1.83 times 10^5$], [$5.06 times 10^5$],
    [0.7], [$1.63 times 10^6$], [$4.38 times 10^6$], [$1.70 times 10^6$], [$4.64 times 10^6$],
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
    the blind spots in place. $F'(theta)^2$, the phase-dependent factor of
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
reduces it to two elementary integrals over a uniform angle,
$chevron.l 1 \/ (1 + C cos u) chevron.r = 1 \/ s$ and
$chevron.l cos 2u \/ (1 + C cos u) chevron.r = (1 - s) \/ (s (1 + s))$.
The Michelson limit is $Phi = 1 \/ 2$. For $|alpha| > 1$, $cos 2 beta < 0$,
so feedback adds magnitude information: at $alpha = 3$, $Phi = 0.514$ at
$C = 0.2$ and $0.794$ at $C = 0.7$.

== Causal and offline Bayesian bounds <sec-crb>

The time-series bounds combine @eq-fisher with a model of how the motion
evolves. We fit a Gauss-Markov surrogate prior to the simulator's motion: the
state $x = (d, v, dot(v))$ is driven by white noise through one real pole and
one resonance, discretized exactly at the sample rate. The fit matches the
in-band velocity line powers of 256 seeded trajectories to 0.73 dB rms, and
it reproduces their velocity and displacement variances, a velocity std of
0.506 um/ms. The coil resonance survives the drive pre-compensation, so the
velocity spectrum is not flat.

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
valid and slightly looser than one conditioned on the phases.

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
(@fig-bounds).
At the nominal point ($K = 3$, $C = 0.2$) the causal bound is 0.0077 um/ms
and the offline bound 0.0019 um/ms. Going from one channel to three
multiplies $macron(J)$ by 3.4 but lowers the causal bound only by a factor
0.74, close to $macron(J)^(-1 \/ 4)$. The causal bound is 14 times below the
causal TCN's real free-space error of 0.1116 um/ms in the baseline report.
That gap measures the bound's omissions as much as any estimator's
shortfall. Within its own model the bound is nearly attained.
`tests/test_bounds.py` runs an extended Kalman filter, given the genie
phases, on three-channel $C = 0.2$ trajectories drawn from the surrogate
prior. Its MSE must lie above the causal bound, within three standard
errors, and below twice it.

The bounds are exact for a simpler model than the simulator, so they are not
a floor for it. Four simplifications separate the two.
+ They are exact for the fitted Gauss-Markov prior, not
  for the simulator's motion. That motion is a finite sum of drive lines with
  a per-shot power drawn from $U(0, 1)$. An estimator that knows the line
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

// math-ok: Q3 and Q1 below are question labels from the program spec.
*Q3, accuracy against feedback.* @sec-sign and @sec-feedback predict that
signed error with three channels falls from the zero-predictor level at
$C = 0$ to the sign-invariant level at the smallest swept $C > 0$, for
$alpha in {3, 5}$, because the sign evidence of @fig-sign is large there.
With one channel the sign is prior-limited (@eq-warp), so its signed error
may stay above its sign-invariant error at small $C$. Because $Phi$ rises
with $C$ at these $alpha$ (@eq-phase-factor), an optimum of the
sign-invariant error below $C = 1$ would come from the estimator's
difficulty with sawtooth fringes, not from lost information. The noise-free SMI error of
@fig-exit-check, at 20 epochs, is already an order of magnitude above the
nominal-SNR causal bound of @fig-bounds. The noise ladder should therefore
find additive white noise at the nominal SNR a minor term; the bounds say
nothing about the terms they omit.

*Q1, multiple channels.* // math-ok: question label
@eq-fisher predicts that a single channel loses all
information at its fringe extrema, and incommensurate wavelengths put the
extrema of different channels at different times. A three-channel estimator
should therefore gain over the best single channel mainly near fringe
extrema and at velocity turning points, and elsewhere by the modest factor
of @fig-bounds. The test is error binned by distance to the nearest fringe
extremum: it should rise toward the extremum for one channel and stay flat
for three. A single channel also carries only prior-limited sign evidence,
so its signed error should exceed its sign-invariant error where three
channels' do not.

= Q3: noise-free accuracy against feedback $C$ <sec-q3> // math-ok: question label

_To be written from `smi/analysis/data/synthetic_results.json` when the
Q3 grid // math-ok: question label
 finishes._

= Q1: does multi-channel beat any single channel? <sec-q1> // math-ok: question label

_To be written from `smi/analysis/data/synthetic_results.json` when the
Q1 grid // math-ok: question label
 finishes._

#bibliography("refs.bib")
