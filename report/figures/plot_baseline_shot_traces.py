#!/usr/bin/env python
"""Three real shots, with the method-1 fits drawn on the waveforms.

Every other number in the baseline report is a summary statistic: a median
$R^2$ per channel, a trend across excursion bins. None of them lets a reader
see whether an $R^2$ of 0.49 means the fit tracks fringes or is coincidentally
correlated. This figure shows the waveforms so that can be judged by eye.

Left column is the displacement the drive voltage implies for that shot, with
the zoom window shaded. Right column is the photodiode trace inside that
window with the global fit and the windowed refit over it. Displacement (um)
and photodiode signal (V) never share an axis -- two units on one axes is the
chart mistake that makes both unreadable, so they get separate panels.

Shot selection is a rule, not a choice. Over the same first N = 1000 shots the
committed artifact scored, each shot's displacement peak-to-peak is computed
and the shots nearest the 10th, 50th and 95th percentiles of that distribution
are drawn. Hand-picking three shots out of a thousand would make this figure
worthless as evidence, so the percentiles are named constants below and the
resolved indices are stated in the caption.

Unlike the other baseline figures, this one reads the HDF5 directly rather
than `baseline_results.json`: it has to show waveforms, and no summary
artifact contains any. Cost is 1000 drive rows plus 3 photodiode rows, not a
whole channel (0.66 GB each). The fits are closed-form least squares, so
recomputing them here is milliseconds and gives the same numbers the artifact
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
    COLORS,
    FREE_SPACE_H5,
    apply_style,
    parse_args,
    read_shot,
    save,
    time_axis,
)

from smi.analysis.baseline_fit import (
    DEFAULT_WINDOW_SAMPLES,
    DRIVE_KEY,
    PD_WAVELENGTHS_UM,
    fit_phase_amplitude,
    forward_model_residual,
    measure_phase_drift,
    scrambled_theta_control,
    windowed_forward_model_residual,
)
from smi.synthetic.coil_driver import CoilDriver

# Matches the artifact's method-1 shot count, so the percentiles below are
# taken over exactly the population the report's medians describe.
N_SHOTS = 1000

# The selection rule. One band per percentile of the displacement
# peak-to-peak distribution; the shot nearest each percentile is drawn.
# Percentiles rather than fixed indices so the rule survives a different
# dataset, and so no shot is chosen by how its trace looks.
BANDS: tuple[tuple[str, float], ...] = (('low', 10.0), ('medium', 50.0), ('high', 95.0))

# The channel shown. 635 nm free space is the report's lead channel and the
# one whose global/windowed/control numbers appear first in its table.
CHANNEL = 'RP1_CH2'

# Partners the scrambled-theta control is averaged over. The artifact pairs
# each shot with the next one only, which is the right rule for a median over
# 1000 shots but too noisy to annotate a single panel with: for shot 203 that
# one partner happens to have almost the same peak-to-peak (0.28 against
# 0.29 um) and scores 0.79, against 0.55 for the median over 16 partners and a
# 0.45-0.71 spread across 200 of them. A single draw that high would
# understate the gap this figure is about. The median over the next 16 shots
# is stable to within 0.01-0.02 from 8 partners to 32.
N_CONTROL_PARTNERS = 16

# Width of the fringe zoom, in milliseconds. Wide enough to hold several
# fringes at the low-excursion shot, narrow enough that individual fringes are
# resolved at the high one.
ZOOM_MS = 1.5


def _displacements(n_shots: int) -> tuple[np.ndarray, float]:
    """Drive-implied displacement for the first ``n_shots`` shots (um).

    Reads only the drive channel, which is what the selection rule needs; the
    photodiode rows are read one at a time later.

    Returns:
        Tuple of (displacement array of shape ``(n_shots, n_time)``, mean
        removed per shot, and the sample rate in Hz).
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
    signal: np.ndarray, theta: np.ndarray, window_samples: int | None
) -> np.ndarray:
    """Fitted waveform ``A cos(theta + phi) + c``, globally or per window.

    This draws what the R^2 columns score. The fit is the same closed-form
    least squares the scoring functions use; only the reconstruction is done
    here, because those functions return statistics rather than a waveform.

    Args:
        signal: Zero-mean photodiode samples (V).
        theta: Model phase, ``4*pi/lambda * displacement`` (radians).
        window_samples: Window length, or ``None`` for one global fit.

    Returns:
        The fitted waveform, same shape as ``signal``.
    """
    if window_samples is None:
        fit = fit_phase_amplitude(signal, theta)
        return fit.amplitude * np.cos(theta + fit.phase) + fit.offset

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
    signal: np.ndarray, displacement: np.ndarray, index: int, wavelength_um: float
) -> float:
    """Median scrambled-theta control over ``N_CONTROL_PARTNERS`` partners.

    Same windowed fit and same parameter count as the real one, with the
    displacement taken from other shots, so what it explains is the
    windowing's own fitting capacity. Extending the artifact's single-partner
    pairing to a median over the next several shots, because one partner is
    too noisy to annotate a panel with -- see ``N_CONTROL_PARTNERS``.

    Args:
        signal: Zero-mean photodiode samples for this shot (V).
        displacement: Per-shot displacement, shape ``(n_shots, n_time)``.
        index: Index of the shot being scored.
        wavelength_um: Laser wavelength for this channel (microns).

    Returns:
        The median control R^2.
    """
    n_shots = displacement.shape[0]
    scores = [
        scrambled_theta_control(
            signal, displacement[(index + 1 + offset) % n_shots], wavelength_um
        ).r_squared
        for offset in range(N_CONTROL_PARTNERS)
    ]
    return float(np.median(scores))


def _zoom_slice(displacement: np.ndarray, n_zoom: int) -> slice:
    """Pick the zoom window at the shot's fastest motion.

    Fastest motion is the most fringes per millisecond, which is where a phase
    error shows up as a visible slip rather than as a slow baseline wander.
    Deterministic: it is an argmax over a boxcar-smoothed speed, with no
    randomness and no manual nudging.
    """
    speed = np.abs(np.gradient(displacement))
    smoothed = np.convolve(speed, np.ones(n_zoom) / n_zoom, mode='same')
    center = int(np.argmax(smoothed))
    begin = max(0, min(center - n_zoom // 2, displacement.size - n_zoom))
    return slice(begin, begin + n_zoom)


def main() -> None:
    args = parse_args(__doc__ or '')
    apply_style()

    wavelength_um = PD_WAVELENGTHS_UM[CHANNEL]
    displacement, sample_rate = _displacements(N_SHOTS)
    selected = _select_shots(displacement)
    n_zoom = int(ZOOM_MS * 1e-3 * sample_rate)

    fig, axes = plt.subplots(
        len(BANDS),
        2,
        figsize=(7.0, 6.0),
        gridspec_kw={'width_ratios': [1.0, 1.7]},
        constrained_layout=True,
    )

    for row, (band, percentile, index, ptp) in enumerate(selected):
        signal, _ = read_shot(FREE_SPACE_H5, CHANNEL, index)
        signal = signal - signal.mean()
        shot_disp = displacement[index]
        theta = 4.0 * np.pi / wavelength_um * shot_disp

        # Whole-shot statistics, from the same functions that wrote the
        # artifact's table, so the annotation and the report agree by
        # construction rather than by transcription.
        global_r2 = forward_model_residual(signal, shot_disp, wavelength_um).r_squared
        windowed_r2 = windowed_forward_model_residual(
            signal, shot_disp, wavelength_um
        ).r_squared
        control_r2 = _control_r2(signal, displacement, index, wavelength_um)
        drift = measure_phase_drift(signal, shot_disp, wavelength_um).drift_fringes

        global_fit = _reconstruct(signal, theta, None)
        windowed_fit = _reconstruct(signal, theta, DEFAULT_WINDOW_SAMPLES)
        times = time_axis(signal.size, sample_rate)
        window = _zoom_slice(shot_disp, n_zoom)

        left = axes[row, 0]
        left.plot(times, shot_disp, color=COLORS['drive'], linewidth=0.7)
        left.axvspan(
            times[window.start],
            times[window.stop - 1],
            color=COLORS['accent'],
            alpha=0.15,
            linewidth=0,
        )
        left.set_ylabel(f'{band}\ndisplacement (um)')
        left.set_title(
            f'shot {index} (p{percentile:.0f}): {ptp:.2f} um ptp,\n'
            f'drift {drift:.2f} fringes',
            fontsize=8,
        )

        right = axes[row, 1]
        right.plot(
            times[window],
            signal[window],
            color=COLORS['drive'],
            linewidth=1.0,
            label='measured',
        )
        right.plot(
            times[window],
            global_fit[window],
            color=COLORS['ch635'],
            linewidth=1.0,
            linestyle=':',
            label='global fit',
        )
        right.plot(
            times[window],
            windowed_fit[window],
            color=COLORS['accent'],
            linewidth=1.0,
            label='windowed fit, W = 1024',
        )
        right.set_ylabel('PD signal (V)')
        # Whole-shot statistics annotated inside the panel rather than as a
        # title: as one line of title text this overflows the axes width, and
        # these numbers belong beside the trace they describe anyway. Ink
        # color, not series colors, so it is not misread as a fourth series.
        right.set_title('whole-shot $R^2$', fontsize=8)
        right.annotate(
            f'global {global_r2:.2f}\n'
            f'windowed {windowed_r2:.2f}\n'
            f'control {control_r2:.2f}',
            xy=(0.015, 0.03),
            xycoords='axes fraction',
            ha='left',
            va='bottom',
            fontsize=7.5,
            color='#333333',
            bbox={
                'boxstyle': 'round,pad=0.25',
                'facecolor': 'white',
                'edgecolor': '#d0d0d0',
                'linewidth': 0.5,
                'alpha': 0.9,
            },
        )

    axes[-1, 0].set_xlabel('time (ms)')
    axes[-1, 1].set_xlabel('time (ms)')

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3)

    save(fig, args.output)


if __name__ == '__main__':
    main()
