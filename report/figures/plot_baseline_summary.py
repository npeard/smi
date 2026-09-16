#!/usr/bin/env python
"""The two baseline measurements side by side, on the scales that read them.

Left is the honest floor: how much of each photodiode channel the Michelson
model explains when it is handed the true displacement. Middle is the baseline
proper -- the inverse decode's displacement error, drawn against the RMS of
the true displacement, because an error bar means nothing without the signal
it is being compared to. Right puts the same error in the dimensionless form
the network will be scored in, with the unity line marked: NRMSE 1 is the
error of predicting a flat zero.

Each panel is titled rather than lettered. The titles carry the distinction
that matters (method 1 versus method 2) and a letter in the corner would sit
on top of the legends.

Medians throughout, with the interquartile range as the error bar. The means
are in the artifact and are not used here: near-motionless shots drive the
NRMSE denominator toward zero, which pulls the mean to several times the
median without saying anything about a typical shot.

Reads the committed results artifact; recomputes nothing.
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
    CHANNEL_COLORS,
    COLORS,
    PD_CHANNELS,
    RESULTS_JSON,
    parse_args,
    save,
)


def _iqr_bar(stats: dict[str, float]) -> list[list[float]]:
    """Asymmetric matplotlib error bar spanning p25 to p75 about the median."""
    return [[stats['median'] - stats['p25']], [stats['p75'] - stats['median']]]


def main() -> None:
    args = parse_args(__doc__ or '')
    results = json.loads(RESULTS_JSON.read_text(encoding='utf-8'))
    datasets = results['datasets']
    # The two methods run at different shot counts (method 2's Viterbi decode
    # is ~5.85 s per shot), so one N cannot label all three panels.
    n_shots = results['config']['n_shots']
    n_shots_method2 = results['config'].get('n_shots_method2', n_shots)

    # Wider than FIGSIZE_WIDE. At matplotlib's default font sizes the two
    # category labels ("free space", "mm fiber") need more horizontal room than
    # a 7.0 in figure gives across three panels: at that width they ran
    # together into "free spacemm fiber" and the y-axis label was clipped. The
    # house rcParams used to shrink the fonts enough to hide it; with those
    # gone the figure has to be the right size instead.
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.6), constrained_layout=True)

    x = np.arange(len(BASELINE_DATASETS), dtype=float)
    labels = [label for _, label in BASELINE_DATASETS]

    # (a) Method 1: per-channel explained variance, the model's own ceiling.
    #
    # Three numbers per channel, because the windowed R^2 alone overstates the
    # fit and the global one understates it. Bars are the windowed refit; the
    # tick below each bar is the scrambled-theta control it has to clear, and
    # the hollow marker is the global single-window fit -- the measurement an
    # earlier draft of the report quoted on its own. The bar above its control
    # tick is the claim.
    ax = axes[0]
    width = 0.26
    for offset, ((channel, wavelength_um), color) in enumerate(
        zip(PD_CHANNELS, CHANNEL_COLORS, strict=True)
    ):
        positions = x + (offset - 1) * width

        def _medians(stat: str, channel: str = channel) -> list[float]:
            return [
                datasets[key]['method1'][channel][stat]['median']
                for key, _ in BASELINE_DATASETS
            ]

        ax.bar(
            positions,
            _medians('r_squared_windowed'),
            width=width,
            color=color,
            label=f'{wavelength_um * 1e3:.0f} nm',
        )
        ax.scatter(
            positions,
            _medians('r_squared_windowed_scrambled'),
            marker='_',
            s=90,
            linewidths=1.4,
            color=COLORS['drive'],
            zorder=3,
            label='control' if offset == 0 else None,
        )
        ax.scatter(
            positions,
            _medians('r_squared'),
            marker='o',
            s=12,
            facecolors='none',
            edgecolors=COLORS['drive'],
            linewidths=1.0,
            zorder=3,
            label='global fit' if offset == 0 else None,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # Headroom above 1.0 so the legend does not sit on the topmost tick label.
    # The axis still runs to 1.0 in ticks, which is what makes the bars
    # readable as "fraction of variance explained".
    # The legend now sits below the axes, so the old headroom above 1.0 is
    # dead space that shrinks the bars for nothing.
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('median $R^2$, displacement known')
    ax.set_title('Method 1: does the model fit?')
    # Below the axes rather than inside it. Five entries at default font size
    # do not fit in the headroom above the bars -- they landed on top of the
    # bars and the y-axis label. constrained_layout reserves space for a
    # legend anchored outside, which it cannot do for one placed inside.
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        columnspacing=0.7,
        frameon=False,
    )

    # (b) Method 2 displacement error against the signal it has to resolve.
    ax = axes[1]
    width = 0.32
    errors = [
        datasets[key]['method2']['displacement_rmse_um'] for key, _ in BASELINE_DATASETS
    ]
    scales = [
        datasets[key]['signal_scale']['true_displacement_rms_um']
        for key, _ in BASELINE_DATASETS
    ]
    ax.bar(
        x - width / 2,
        [s['median'] for s in scales],
        width=width,
        color=COLORS['muted'],
        label='true displacement RMS',
    )
    ax.bar(
        x + width / 2,
        [e['median'] for e in errors],
        width=width,
        color=COLORS['accent'],
        yerr=np.hstack([_iqr_bar(e) for e in errors]),
        capsize=3,
        ecolor=COLORS['drive'],
        label='decode RMSE',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('displacement (um)')
    ax.set_title('Method 2: error vs signal')
    ax.set_ylim(0, 0.92)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

    # (c) The same error, normalized, against the do-nothing predictor.
    ax = axes[2]
    nrmse = [
        datasets[key]['method2']['displacement_nrmse'] for key, _ in BASELINE_DATASETS
    ]
    ax.bar(
        x,
        [s['median'] for s in nrmse],
        width=0.45,
        color=COLORS['highlight'],
        yerr=np.hstack([_iqr_bar(s) for s in nrmse]),
        capsize=3,
        ecolor=COLORS['drive'],
    )
    ax.axhline(1.0, color=COLORS['ch635'], linewidth=1.2, linestyle='--')
    # Annotate in the gap left of the first bar, which is the only region of
    # this panel that neither a bar nor a whisker reaches. Sat directly on the
    # dashed line at 1.06 once the default font size grew; lifted clear of it.
    ax.text(
        -0.55,
        1.20,
        'predicting zero',
        color=COLORS['ch635'],
        fontsize=7.5,
        ha='left',
        va='bottom',
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 2.4)
    ax.set_ylabel('median displacement NRMSE')
    ax.set_title('Method 2: normalized')

    # Two lines. As one line this overran the figure width at default font
    # sizes; the break is placed at the semicolon so each line is a complete
    # clause rather than wherever the width happened to run out.
    fig.suptitle(
        f'Bars are medians per acquisition -- N = {n_shots} shots for method 1, '
        f'N = {n_shots_method2} for method 2\n'
        'whiskers span the interquartile range',
        fontsize=8,
        color=COLORS['muted'],
    )

    save(fig, args.output)


if __name__ == '__main__':
    main()
