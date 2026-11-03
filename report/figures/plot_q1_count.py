#!/usr/bin/env python
"""Q1.1 and Q1.2: test velocity RMSE against channel count.

One panel per sim (noise-free, nominal), one line per family over the first N
of q1_r1_count's colors. Small dots are the q1_r1_seeds reruns at K = 1, 3;
open markers at K = 2 the other two-color pairs (q1_r1_pairs); stars the
one-color quadrature arrays of q1_r2_colors(_n3). The dotted line is the
profiled velocity CRB's ratio sqrt(J_1 / J_N) (channel_analytics.json) scaled
to TCAN's K = 1 seed mean, so a line parallel to it gains what the bound gains.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    R1_COLORS_UM,
    R1_SEEDS,
    R1_SIMS,
    R_FAMILIES,
    R_FAMILY_LABELS,
    RESULTS_DIR,
    mean_metric,
    panel_label,
    parse_args,
    r_cell,
    save,
    synthetic_records,
)

from smi.experiments.runner import cell_key

STYLE = {
    R_FAMILIES[0]: (COLORS['accent'], '-', 'o'),
    R_FAMILIES[1]: (COLORS['ch675'], '--', 's'),
}
OTHER_PAIRS = ((0.635, 0.515), (0.6748, 0.515))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    data = json.loads((RESULTS_DIR / 'channel_analytics.json').read_text())
    seeds = [data['q1_count'][str(k)]['0.2'] for k in data['count_seeds']]
    info = np.array([np.mean([sd[i]['info'] for sd in seeds]) for i in range(6)])
    bound = (info / info[0]) ** -0.5
    ns = np.arange(1, 7)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, (sim, title), label in zip(
        axes, R1_SIMS.items(), ['(a)', '(b)'], strict=True
    ):
        for fam in R_FAMILIES:
            color, ls, marker = STYLE[fam]
            runs = [
                [
                    mean_metric(records[k], 'velocity_rmse')
                    for s in R1_SEEDS
                    if (k := cell_key(fam, R1_COLORS_UM[:n], sim, s)) in records
                ]
                for n in ns
            ]
            means = [np.mean(r) for r in runs]
            ax.plot(
                ns, means, color=color, ls=ls, marker=marker, label=R_FAMILY_LABELS[fam]
            )
            for n, r in zip(ns, runs, strict=True):
                if len(r) > 1:
                    ax.plot([n] * len(r), r, '.', color=color, ms=4, alpha=0.7)
            pairs = [
                mean_metric(r_cell(records, fam, p, sim), 'velocity_rmse')
                for p in OTHER_PAIRS
            ]
            ax.plot(
                [2.15] * 2, pairs, marker=marker, mfc='white', color=color, ls='none'
            )
            arrays = [
                mean_metric(
                    r_cell(records, fam, (0.635,) * n, f'{sim}-1color'), 'velocity_rmse'
                )
                for n in (2, 3)
            ]
            ax.plot([2, 3], arrays, marker='*', ms=9, color=color, ls=':', lw=0.8)
            if fam == R_FAMILIES[0]:
                ax.plot(ns, bound * means[0], color=COLORS['muted'], ls=':', lw=1.4)
        ax.set_yscale('log')
        ax.set_ylim(2.5e-3, 0.6)
        ax.set_xticks(
            ns, [f'{n}\n{R1_COLORS_UM[n - 1] * 1e3:.0f}' for n in ns], fontsize=8
        )
        ax.set_xlabel('channels K (added color, nm)')
        ax.set_title(title, fontsize=9)
        panel_label(ax, label)
    axes[0].set_ylabel(r'test velocity RMSE ($\mu$m/ms)')
    handles = [
        plt.Line2D([], [], color=COLORS['muted'], ls=':', lw=1.4),
        plt.Line2D([], [], color='k', marker='o', mfc='white', ls='none'),
        plt.Line2D([], [], color='k', marker='*', ms=9, ls='none'),
        plt.Line2D([], [], color='k', marker='.', ls='none'),
    ]
    extra = [
        'CRB ratio (scaled)',
        'other 2-color pairs',
        '635 nm array, quadrature',
        'seeds 0-2',
    ]
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(
        h + handles,
        lab + extra,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=7.5,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
