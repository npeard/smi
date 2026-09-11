#!/usr/bin/env python
"""Method-1 fit quality against how far the speaker actually moved.

The forward-model residual is computed with the displacement already known, so
whatever it fails to explain is the model's fault rather than an estimator's.
Binning those per-shot R^2 values by the drive's peak-to-peak excursion shows
the failure is not uniform: every channel in both acquisitions explains
small-excursion shots several times better than large-excursion ones, and all
six collapse above 2 um. From the 0.3-1 um bin onward every channel decreases
at every step; three of the six rise into that bin from the smallest one,
which is also the thinnest. The shared direction is the most actionable thing
in the baseline, so it gets its own figure.

Reads the committed results artifact. It does not recompute anything -- the
fits took ~20 minutes per acquisition on a GPU and are not something a
document build should be doing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    CHANNEL_COLORS,
    FIGSIZE_WIDE,
    PD_CHANNELS,
    RESULTS_JSON,
    panel_label,
    parse_args,
    save,
)

# Datasets in the order they are plotted, with the label to show.
DATASETS: tuple[tuple[str, str], ...] = (
    ('free-space', 'free space'),
    ('mmfiber', 'mm fiber'),
)

# Bin keys as written by baseline_fit, mapped to an axis label. Kept explicit
# rather than taken from the artifact so the axis reads in microns and the
# script fails loudly if the binning ever changes underneath it.
BIN_LABELS: dict[str, str] = {
    '0-0.3': '< 0.3',
    '0.3-1': '0.3 - 1',
    '1-2': '1 - 2',
    '2+': '> 2',
}


def main() -> None:
    args = parse_args(__doc__ or '')
    results = json.loads(RESULTS_JSON.read_text(encoding='utf-8'))

    fig, axes = plt.subplots(
        1, len(DATASETS), figsize=FIGSIZE_WIDE, sharey=True, constrained_layout=True
    )

    bin_keys = list(BIN_LABELS)
    x = np.arange(len(bin_keys), dtype=float)
    width = 0.26

    for panel, (ax, (key, title)) in enumerate(zip(axes, DATASETS, strict=True)):
        method1 = results['datasets'][key]['method1']
        counts: list[int] = []

        for offset, ((channel, wavelength_um), color) in enumerate(
            zip(PD_CHANNELS, CHANNEL_COLORS, strict=True)
        ):
            bins = {b['ptp_um']: b for b in method1[channel]['r_squared_by_drive_ptp']}
            # An empty bin carries no 'median' key (see _binned_by_amplitude):
            # the bin edges are absolute, so regenerating with a smaller
            # --n-shots, or on a quieter dataset, can leave one unpopulated.
            # NaN draws as a gap, which is the honest rendering of "no shots
            # here"; raising KeyError would fail the document build over it.
            medians = [bins[b].get('median', float('nan')) for b in bin_keys]
            if not counts:
                counts = [bins[b]['n'] for b in bin_keys]
            ax.bar(
                x + (offset - 1) * width,
                medians,
                width=width,
                color=color,
                label=f'{wavelength_um * 1e3:.0f} nm',
            )

        # Shot count per bin, so a reader can see the small-excursion bins are
        # the thinly populated ones without having to look it up.
        for xi, n in zip(x, counts, strict=True):
            ax.text(xi, 0.02, f'n = {n}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([BIN_LABELS[b] for b in bin_keys])
        ax.set_xlabel('drive peak-to-peak displacement (um)')
        ax.set_title(title)
        ax.set_ylim(0, 0.95)
        panel_label(ax, f'({"ab"[panel]})')

    axes[0].set_ylabel('median $R^2$ of the forward-model fit')
    axes[0].legend(loc='upper right', ncol=3, columnspacing=1.0)

    save(fig, args.output)


if __name__ == '__main__':
    main()
