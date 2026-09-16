#!/usr/bin/env python
"""Method-1 fit quality against how far the speaker actually moved.

The forward-model residual is computed with the displacement already known,
so what it fails to explain is a property of the estimator and the model, not
of any inverse. The two estimators drawn here differ only in the interval
over which amplitude and phase are held constant. The global fit forces a
single amplitude, phase and offset across all 16384 samples, and that is what
collapses with excursion: the longer the sweep, the more the fringe phase and
contrast wander away from any one setting. The windowed refit holds them
constant only over a window spanning about two fringes -- W is resolved per
shot per channel from that shot's own fringe rate, so the measure means the
same thing in every excursion bin -- and pools the variances into one R^2.

Three series per panel, because a windowed R^2 on its own is not evidence: a
short window can fit anything, so the scrambled-theta control -- the same W,
the same window count and the same three parameters, with the displacement
taken from a different shot -- is what turns the windowed curve into a
measurement. Matching W between fit and control is not a detail: comparing a
fit against a control with less fitting freedom measures the freedom. The
*gap* between windowed and control is the claim; the shading draws it.

Per channel and per acquisition, never averaged: free-space 515 nm has a gap
narrow enough that averaging it with the other two would report agreement
this figure does not support.

Reads the committed results artifact. It recomputes nothing -- a document
build should not be refitting 1000 shots x 3 channels x 2 acquisitions, and
reading a fixed artifact is what keeps the figure deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    BASELINE_DATASETS,
    COLORS,
    FIGSIZE_TALL,
    PD_CHANNELS,
    RESULTS_JSON,
    parse_args,
    save,
)

# Bin keys as written by baseline_fit, mapped to an axis label. Kept explicit
# rather than taken from the artifact so the axis reads in microns and the
# script fails loudly if the binning ever changes underneath it.
BIN_LABELS: dict[str, str] = {
    '0-0.3': '<0.3',
    '0.3-1': '0.3-1',
    '1-2': '1-2',
    '2-3': '2-3',
    '3+': '>3',
}

# The three artifact keys this figure draws, with how each is styled. Order is
# the reading order of the legend: the single-window contrast, the windowed
# measure, and the floor that measure has to clear.
#
# The windowed label is filled in from the artifact's own config rather than
# written here, because a label naming a window the artifact was not
# generated with is exactly the drift this figure cannot afford.
SERIES: tuple[tuple[str, str, str, str], ...] = (
    ('r_squared_by_drive_ptp', 'global fit', COLORS['muted'], ':'),
    ('r_squared_windowed_by_drive_ptp', '{windowed}', COLORS['accent'], '-'),
    (
        'r_squared_windowed_scrambled_by_drive_ptp',
        'scrambled control, same W',
        COLORS['ch635'],
        '--',
    ),
)


def _windowed_label(config: dict) -> str:
    """Legend label for the windowed series, from the artifact's config."""
    if config.get('windowing') == 'fringe-matched':
        fringes = config['fringes_per_window']
        return f'windowed, W fringe-matched ({fringes:g} fringes)'
    return f'windowed, W = {config["window_samples"]}'


def _medians(channel_stats: dict, key: str, bin_keys: list[str]) -> np.ndarray:
    """Median per excursion bin, NaN where the bin holds no shots.

    An empty bin carries no 'median' key (see binned_by_amplitude): the bin
    edges are absolute, so regenerating with a smaller --n-shots, or on a
    quieter dataset, can leave one unpopulated. NaN draws as a gap, which is
    the honest rendering of "no shots here"; raising KeyError would fail the
    document build over it.
    """
    bins = {b['ptp_um']: b for b in channel_stats[key]}
    return np.array([bins[b].get('median', float('nan')) for b in bin_keys])


def main() -> None:
    args = parse_args(__doc__ or '')
    results = json.loads(RESULTS_JSON.read_text(encoding='utf-8'))
    n_shots = results['config']['n_shots']
    windowed_label = _windowed_label(results['config'])

    bin_keys = list(BIN_LABELS)
    x = np.arange(len(bin_keys), dtype=float)

    fig, axes = plt.subplots(
        len(BASELINE_DATASETS),
        len(PD_CHANNELS),
        figsize=FIGSIZE_TALL,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for row, (key, title) in enumerate(BASELINE_DATASETS):
        method1 = results['datasets'][key]['method1']
        for col, (channel, wavelength_um) in enumerate(PD_CHANNELS):
            ax = axes[row, col]
            stats = method1[channel]
            curves = {
                artifact_key: _medians(stats, artifact_key, bin_keys)
                for artifact_key, _, _, _ in SERIES
            }

            # Shade windowed-minus-control first, so the lines sit on top of
            # it. This band is the finding; the absolute windowed height is
            # not, since the control shows how much of it is fitting capacity.
            ax.fill_between(
                x,
                curves['r_squared_windowed_scrambled_by_drive_ptp'],
                curves['r_squared_windowed_by_drive_ptp'],
                color=COLORS['accent'],
                alpha=0.16,
                linewidth=0,
                label='gap over control',
            )
            for artifact_key, label, color, style in SERIES:
                ax.plot(
                    x,
                    curves[artifact_key],
                    linestyle=style,
                    color=color,
                    marker='o',
                    markersize=3,
                    label=label.format(windowed=windowed_label),
                )

            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            if row == 0:
                ax.set_title(f'{wavelength_um * 1e3:.0f} nm')
            if col == 0:
                ax.set_ylabel(f'{title}\nmedian $R^2$')
            # A fringe-matched W differs per shot, so the panel has to say
            # what window its own numbers came from. The median and the
            # interquartile range are what the artifact records.
            resolved = stats['window_samples_resolved']
            ax.text(
                0.97,
                0.95,
                f'W {resolved["median"]:.0f}\n({resolved["p25"]:.0f}-'
                f'{resolved["p75"]:.0f})',
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=5.5,
                color=COLORS['muted'],
            )
            if row == len(BASELINE_DATASETS) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([BIN_LABELS[b] for b in bin_keys])
                ax.set_xlabel('drive peak-to-peak (um)')

    # Bin populations, once rather than per panel: they are identical across
    # channels of one acquisition (the same shots are fit in all three) and
    # differ between acquisitions only by a few shots.
    counts = {
        label: _binned_counts(results['datasets'][key]['method1'], bin_keys)
        for key, label in BASELINE_DATASETS
    }
    population = '; '.join(
        f'{label} n = ' + '/'.join(str(n) for n in ns) for label, ns in counts.items()
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4)
    fig.supxlabel(
        f'N = {n_shots} shots per acquisition (method 1). Shots per bin: {population}',
        fontsize=7,
    )

    save(fig, args.output)


def _binned_counts(method1: dict, bin_keys: list[str]) -> list[int]:
    """Shot count per excursion bin, read off the first channel."""
    first = method1[PD_CHANNELS[0][0]]['r_squared_by_drive_ptp']
    bins = {b['ptp_um']: b for b in first}
    return [bins[b]['n'] for b in bin_keys]


if __name__ == '__main__':
    main()
