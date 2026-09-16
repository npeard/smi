#!/usr/bin/env python
"""Three real shots on all three channels, with the method-1 fits drawn.

Every other number in the baseline report is a summary statistic: a median
$R^2$ per channel, a trend across excursion bins. None of them lets a reader
see whether an $R^2$ of 0.49 means the fit tracks fringes or is coincidentally
correlated. This figure shows the waveforms so that can be judged by eye.

Layout is channels down the rows and excursion bands across the columns, with
the two fit modes overlaid on the measured trace. Displacement (um) gets its
own top row rather than a second y-axis on a photodiode panel: two units on
one axes is the chart mistake that makes both unreadable.

Two fit modes are drawn, and the difference between them is the physics:

- *individual*, each channel fit against the drive-implied displacement as
  given, amplitude and phase free per window. What the report's tables score.
- *joint*, one displacement scale factor ``g`` shared by all three channels
  with amplitude and phase still free per channel. The mirror moves once and
  three lasers observe that one motion, so the scale is a shared mechanical
  quantity while the amplitude and phase are per-channel optics and detector.
  Three fringe rates differing by 31% pin a ``g`` that one channel alone would
  trade almost freely against a phase offset.

The window length is fringe-matched rather than fixed (see
``fringe_matched_window``): ``W`` is scaled by each shot's own fringe rate so
every window spans about two fringes. At fixed ``W = 1024`` the fitted
amplitude runs systematically under the data's local amplitude at high
excursion -- median ratio 0.586, below 1 in 94% of windows -- because a
least-squares amplitude attenuates toward zero when the phase regressor is
wrong. Fringe-matching is what recovers it, to 0.883.

Shot selection is a rule, not a choice. Over the same first N = 1000 shots the
committed artifact scored, each shot's displacement peak-to-peak is computed
and the shots nearest the 10th, 50th and 95th percentiles of that distribution
are drawn. Hand-picking three shots out of a thousand would make this figure
worthless as evidence, so the percentiles are named constants below and the
resolved indices are stated in the caption.

Unlike the other baseline figures, this one reads the HDF5 directly rather
than `baseline_results.json`: it has to show waveforms, and no summary
artifact contains any. Cost is 1000 drive rows plus 9 photodiode rows, not a
whole channel (0.66 GB each). The fits are closed-form least squares, so
recomputing them here is seconds and gives the same numbers the artifact
reports.

Note on the flat-topped fringe peaks: they are not detector clipping. Only 1
of 400 shots reaches the channel maximum, the fraction of samples near each
shot's extremes does not grow with excursion, and the topmost distinct values
are spaced at the 0.000122 V ADC quantum with a single sample at the maximum.
A clipped channel would pile hundreds of samples on one rail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _figcommon import (
    CHANNEL_COLORS,
    FREE_SPACE_H5,
    PD_CHANNELS,
    apply_style,
    parse_args,
    read_shot,
    save,
    time_axis,
)

from smi.analysis.baseline_fit import (
    fit_phase_amplitude,
    fringe_matched_window,
    joint_gain_fit,
    scrambled_theta_control,
    windowed_forward_model_residual,
)
from smi.synthetic.coil_driver import CoilDriver

# Drive voltage channel. Named here rather than imported so this script does
# not depend on which module happens to re-export it.
DRIVE_KEY = 'RP1_CH1'

# Matches the artifact's method-1 shot count, so the percentiles below are
# taken over exactly the population the report's medians describe.
N_SHOTS = 1000

# The selection rule. One band per percentile of the displacement
# peak-to-peak distribution; the shot nearest each percentile is drawn.
# Percentiles rather than fixed indices so the rule survives a different
# dataset, and so no shot is chosen by how its trace looks.
BANDS: tuple[tuple[str, float], ...] = (('low', 10.0), ('medium', 50.0), ('high', 95.0))

# Partners the scrambled-theta control is averaged over. The artifact pairs
# each shot with the next one only, which is the right rule for a median over
# 1000 shots but too noisy to annotate a single panel with: a single-partner
# control ranges 0.418-0.798 on the low band and correlates *negatively* with
# the partner's own excursion (r = -0.63 to -0.75), because a low-excursion
# partner supplies slowly-varying theta the fit absorbs easily. One draw would
# understate the gap this figure is about. The median over the next 16 shots
# is stable to within 0.01-0.02 from 8 partners to 32.
N_CONTROL_PARTNERS = 16

# Fringes the zoom window should span, rather than a fixed duration.
#
# A fixed width cannot serve all nine panels. The window is placed at each
# shot's fastest motion, which is the most fringes per millisecond, and the
# three channels see different fringe counts over that same stretch: 515 nm
# crosses 1.31x as many per micron as 675 nm. At a fixed 1.5 ms the 515 nm
# high-excursion panel held so many fringes that neither the data nor the fit
# was resolvable, which made the weakest channel also the least legible -- a
# reader cannot judge a fit from a window where the fringes are a blur.
#
# Scaling the width by the panel's own fringe rate shows every panel about the
# same number of cycles, so the three channels are compared on what the fit
# does per fringe rather than on how many fringes got crammed in.
ZOOM_FRINGES = 6.0

# Bounds on the derived width (ms). The lower bound keeps enough samples for
# the trace to read as a waveform; the upper stops a nearly-static shot from
# zooming out to the whole record.
ZOOM_MS_MIN = 0.35
ZOOM_MS_MAX = 4.0

# Ink, not theme. The measured trace and the two fits need to be
# distinguishable from each other and from the per-channel colors, and these
# three are the same in any matplotlib style.
MEASURED_COLOR = 'black'
JOINT_COLOR = 'tab:gray'


def _displacements(n_shots: int) -> tuple[np.ndarray, float]:
    """Drive-implied displacement for the first ``n_shots`` shots (um).

    Reads only the drive channel, which is what the selection rule needs; the
    photodiode rows are read one at a time later.

    Args:
        n_shots: Number of leading shots to read.

    Returns:
        Tuple of (per-shot zero-mean displacement of shape
        ``(n_shots, n_time)``, sample rate in Hz).
    """
    with h5py.File(FREE_SPACE_H5, 'r') as handle:
        sample_rate = float(handle.attrs['sample_rate'])
        n_shots = min(n_shots, handle[DRIVE_KEY].shape[0])
        drive = np.asarray(handle[DRIVE_KEY][:n_shots], dtype=np.float64)

    coil = CoilDriver()
    displacement = np.empty_like(drive)
    for i in range(drive.shape[0]):
        _, displacement[i] = coil.get_velocity_and_displacement(drive[i], sample_rate)
    return displacement - displacement.mean(axis=1, keepdims=True), sample_rate


def _select_shots(displacement: np.ndarray) -> list[tuple[str, float, int, float]]:
    """Apply the selection rule to a displacement array.

    Args:
        displacement: Per-shot displacement, shape ``(n_shots, n_time)``.

    Returns:
        One ``(band, percentile, shot_index, ptp_um)`` per entry of ``BANDS``.
    """
    ptp = np.ptp(displacement, axis=1)
    chosen = []
    for band, percentile in BANDS:
        target = float(np.percentile(ptp, percentile))
        index = int(np.argmin(np.abs(ptp - target)))
        chosen.append((band, percentile, index, float(ptp[index])))
    return chosen


def _reconstruct(
    signal: np.ndarray, theta: np.ndarray, window_samples: int
) -> np.ndarray:
    """Fitted waveform ``A cos(theta + phi) + c``, refit per window.

    This draws what the R^2 columns score. The fit is the same closed-form
    least squares the scoring functions use; only the reconstruction is done
    here, because those functions return statistics rather than a waveform.

    Args:
        signal: Zero-mean photodiode samples (V).
        theta: Model phase, ``4*pi/lambda * displacement`` (radians).
        window_samples: Window length in samples.

    Returns:
        The fitted waveform, same shape as ``signal``.
    """
    out = np.empty_like(signal)
    for begin in range(0, signal.size, window_samples):
        end = min(begin + window_samples, signal.size)
        block, block_theta = signal[begin:end], theta[begin:end]
        # Each window is fit about its own mean and that mean added back, so
        # the drawn curve is continuous with the trace it sits on.
        mean = float(block.mean())
        fit = fit_phase_amplitude(block - mean, block_theta)
        out[begin:end] = (
            fit.amplitude * np.cos(block_theta + fit.phase) + fit.offset + mean
        )
    return out


def _control_r2(
    signal: np.ndarray,
    displacement: np.ndarray,
    index: int,
    wavelength_um: float,
    window_samples: int,
) -> float:
    """Median scrambled-theta control over ``N_CONTROL_PARTNERS`` partners.

    Same fit, same window length and same parameter count as the real one,
    with the displacement taken from other shots, so what it explains is the
    fit's own capacity rather than agreement with the data. The window length
    is passed in rather than defaulted: the fringe-matched fit has more
    freedom than the fixed-W one, and a control run at a different window
    length would not be a control at all.

    Args:
        signal: Zero-mean photodiode samples for this shot (V).
        displacement: Per-shot displacement, shape ``(n_shots, n_time)``.
        index: Index of the shot being scored.
        wavelength_um: Laser wavelength for this channel (microns).
        window_samples: Window length, matching the real fit exactly.

    Returns:
        The median control R^2.
    """
    n_shots = displacement.shape[0]
    scores = [
        scrambled_theta_control(
            signal,
            displacement[(index + 1 + offset) % n_shots],
            wavelength_um,
            window_samples=window_samples,
        ).r_squared
        for offset in range(N_CONTROL_PARTNERS)
    ]
    return float(np.median(scores))


def _zoom_center(displacement: np.ndarray, n_probe: int) -> int:
    """Sample index of the shot's fastest motion.

    Fastest motion is the most fringes per millisecond, which is where a phase
    error shows up as a visible slip rather than as a slow baseline wander.
    Deterministic: an argmax over a boxcar-smoothed speed, with no randomness
    and no manual nudging. The center is a property of the shot, so all three
    channels are drawn over the same stretch of time even though each gets its
    own width.

    Args:
        displacement: Displacement for one shot (um).
        n_probe: Boxcar width in samples used to smooth the speed.

    Returns:
        Index of the window center.
    """
    speed = np.abs(np.gradient(displacement))
    smoothed = np.convolve(speed, np.ones(n_probe) / n_probe, mode='same')
    return int(np.argmax(smoothed))


def _zoom_slice(
    displacement: np.ndarray, center: int, wavelength_um: float, sample_rate: float
) -> slice:
    """Window of about ``ZOOM_FRINGES`` cycles, centered on ``center``.

    The fringe rate is ``2 * speed / wavelength`` (a lambda/2 displacement is
    one fringe), so the width that holds a fixed number of fringes depends on
    the channel as well as the shot. Measuring the speed locally rather than
    over the whole record is what makes the width match the stretch actually
    drawn.

    Args:
        displacement: Displacement for one shot (um).
        center: Window center in samples, from :func:`_zoom_center`.
        wavelength_um: Channel wavelength (microns).
        sample_rate: Sample rate (Hz).

    Returns:
        The slice to draw.
    """
    # Local speed in um/s around the center, over a short fixed probe so the
    # estimate does not depend on the width being solved for.
    probe = max(1, int(0.2e-3 * sample_rate))
    lo = max(0, center - probe)
    hi = min(displacement.size, center + probe)
    speed = float(np.abs(np.gradient(displacement[lo:hi])).mean()) * sample_rate

    fringe_rate = 2.0 * speed / wavelength_um
    width_ms = ZOOM_MS_MAX if fringe_rate <= 0 else ZOOM_FRINGES / fringe_rate * 1e3
    width_ms = float(np.clip(width_ms, ZOOM_MS_MIN, ZOOM_MS_MAX))

    n_zoom = max(2, int(width_ms * 1e-3 * sample_rate))
    n_zoom = min(n_zoom, displacement.size)
    begin = max(0, min(center - n_zoom // 2, displacement.size - n_zoom))
    return slice(begin, begin + n_zoom)


def _annotate_scores(
    ax: plt.Axes, individual: float, joint: float, control: float, window: int
) -> None:
    """Title the panel with its three R^2 values, naming the quantity.

    A number on a panel has to say what it is without the reader reaching for
    the caption, so the quantity is spelled out rather than left as three bare
    decimals. Above the axes rather than inside it: a box in the corner sits
    on top of the waveform this figure exists to show, and at high excursion
    the fringes fill the panel with nowhere for it to go.

    Args:
        ax: Axes to annotate.
        individual: R^2 of the per-channel fit.
        joint: R^2 at the joint fit's shared gain.
        control: Median scrambled-theta control R^2.
        window: Window length the three were computed at.
    """
    ax.set_title(
        f'W = {window} | $R^2$ indiv. {individual:.2f} '
        f'/ joint {joint:.2f} / ctrl {control:.2f}',
        fontsize=6.5,
    )


def main() -> None:
    args = parse_args(__doc__ or '')
    apply_style()

    displacement, sample_rate = _displacements(N_SHOTS)
    selected = _select_shots(displacement)
    wavelengths = tuple(wavelength for _, wavelength in PD_CHANNELS)
    # Boxcar for locating the fastest motion. Fixed, and independent of the
    # per-channel widths derived from it, so all three channels center on the
    # same stretch of the record.
    n_probe = int(1.5e-3 * sample_rate)

    # One row of displacement panels on top, then one row per channel.
    fig, axes = plt.subplots(
        len(PD_CHANNELS) + 1, len(BANDS), figsize=(7.2, 8.0), constrained_layout=True
    )

    for column, (band, percentile, index, ptp) in enumerate(selected):
        shot_disp = displacement[index]
        times = time_axis(shot_disp.size, sample_rate)
        center = _zoom_center(shot_disp, n_probe)
        # Widths differ per channel, so the shaded span on the displacement
        # panel shows the widest of the three: it is the envelope of what the
        # column below it draws, not any single panel's window.
        windows = [
            _zoom_slice(shot_disp, center, wavelength, sample_rate)
            for wavelength in wavelengths
        ]
        widest = max(windows, key=lambda s: s.stop - s.start)

        # All three channels of this shot, read once, so the joint fit sees
        # the same samples the panels draw.
        signals = np.stack(
            [read_shot(FREE_SPACE_H5, channel, index)[0] for channel, _ in PD_CHANNELS]
        )
        signals = signals - signals.mean(axis=1, keepdims=True)

        # The shared gain is a whole-shot, whole-instrument quantity, so it is
        # fit once per shot against all three channels at once. Global rather
        # than windowed: with amplitude and phase refit per window there is
        # enough freedom to absorb a gain error, and the search wanders to the
        # bracket edge instead of finding the mechanical scale.
        joint = joint_gain_fit(signals, shot_disp, wavelengths)

        top = axes[0, column]
        top.plot(times, shot_disp, color=MEASURED_COLOR, linewidth=0.7)
        # The zoom is a fraction of a 33.6 ms record, so a shaded span alone is
        # a sliver that reads as a plotting artifact. Edge lines make it legible
        # at the width this panel gets in the document.
        top.axvspan(
            times[widest.start],
            times[widest.stop - 1],
            color='tab:red',
            alpha=0.25,
            linewidth=0,
        )
        for edge in (widest.start, widest.stop - 1):
            top.axvline(times[edge], color='tab:red', linewidth=0.6)
        top.set_title(
            f'{band}: shot {index} (p{percentile:.0f})\n'
            f'{ptp:.2f} um ptp, joint g = {joint.gain:.3f}',
            fontsize=8,
        )
        if column == 0:
            top.set_ylabel('displacement (um)')

        for row, ((_channel, wavelength_um), color) in enumerate(
            zip(PD_CHANNELS, CHANNEL_COLORS, strict=True), start=1
        ):
            signal = signals[row - 1]
            # This channel's own zoom: same center, width set by its fringe
            # rate (see ZOOM_FRINGES). 515 nm crosses 1.31x as many fringes
            # per micron as 675 nm, so a shared width left the shortest
            # wavelength the least readable panel in the figure.
            window = windows[row - 1]
            # Fringe-matched: W scaled so each window spans about two
            # fringes, which is what keeps the fitted amplitude from
            # attenuating at high excursion. Both fit modes and the control
            # use the same W, so the three numbers are comparable.
            window_samples = fringe_matched_window(shot_disp, wavelength_um)

            theta = 4.0 * np.pi / wavelength_um * shot_disp
            joint_theta = 4.0 * np.pi / wavelength_um * joint.gain * shot_disp
            individual_fit = _reconstruct(signal, theta, window_samples)
            joint_fit = _reconstruct(signal, joint_theta, window_samples)

            individual_r2 = windowed_forward_model_residual(
                signal, shot_disp, wavelength_um, window_samples=window_samples
            ).r_squared
            joint_r2 = windowed_forward_model_residual(
                signal,
                joint.gain * shot_disp,
                wavelength_um,
                window_samples=window_samples,
            ).r_squared
            control_r2 = _control_r2(
                signal, displacement, index, wavelength_um, window_samples
            )

            ax = axes[row, column]
            ax.plot(
                times[window],
                signal[window],
                color=MEASURED_COLOR,
                linewidth=0.9,
                label='measured',
            )
            ax.plot(
                times[window],
                individual_fit[window],
                color=color,
                linewidth=0.9,
                label='individual fit',
            )
            ax.plot(
                times[window],
                joint_fit[window],
                color=JOINT_COLOR,
                linewidth=0.9,
                linestyle='--',
                label='joint fit, shared g',
            )
            _annotate_scores(ax, individual_r2, joint_r2, control_r2, window_samples)
            if column == 0:
                ax.set_ylabel(f'{wavelength_um * 1e3:.0f} nm\nPD signal (V)')

    for ax in axes[-1, :]:
        ax.set_xlabel('time (ms)')

    # The panel titles have to abbreviate to fit the column width, so the
    # three quantities are named in the legend's own title. A reader must be
    # able to learn what a number on a panel is without leaving the figure,
    # and hanging the key off the legend is what keeps the layout engine
    # aware of it -- a bare fig.text lands under the legend instead.
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='outside lower center',
        ncol=3,
        fontsize=8,
        title=(
            'panel titles: W = fringe-matched window (samples); whole-shot $R^2$ '
            'for the per-channel\nfit (indiv.), the shared-gain joint fit (joint), '
            f'and the scrambled-theta control (ctrl),\na median over '
            f'{N_CONTROL_PARTNERS} partner shots at the same W.'
        ),
        title_fontsize=6.5,
    )

    save(fig, args.output)


if __name__ == '__main__':
    main()
