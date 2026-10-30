#!/usr/bin/env python
"""Noise-free velocity error of the causal TCN against SMI feedback strength C.

Two panels share one layout: (a) velocity RMSE, (b) velocity NRMSE, each the
mean over the frozen test shots of the per-shot value. SMI cells are scored
signed; the Michelson reference is scored sign-invariant, the best any
estimator can do there. One curve per cosine schedule, so the reader sees how
much of each value is training budget.

Reads the runner's records in synthetic_results.json and recomputes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIGSIZE_WIDE,
    Q3_BUDGETS,
    Q3_FEEDBACK_C,
    mean_metric,
    panel_label,
    parse_args,
    save,
    synthetic_cell,
    synthetic_records,
)

# Per schedule: color and line style, so the two curves differ without color.
STYLE = {90: (COLORS['accent'], '-', 'o'), 30: (COLORS['muted'], '--', 's')}
# The alpha controls, all at C = 0.2.
CONTROLS = (('smi-a5', '^', r'$\alpha=5$'), ('smi-a0', 'X', r'$\alpha=0$'))
# (metric, axis label, labelled y ticks): log axes spanning under a decade
# get no major tick labels by default.
PANELS = (
    ('velocity_rmse', r'velocity RMSE ($\mu$m/ms)', (0.02, 0.05, 0.1, 0.2, 0.5)),
    ('velocity_nrmse', 'velocity NRMSE', (0.1, 0.2, 0.5, 1.0)),
)


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharex=True)
    for ax, (metric, ylabel, yticks), tag in zip(
        axes, PANELS, ['(a)', '(b)'], strict=True
    ):
        for family, epochs in Q3_BUDGETS:
            color, linestyle, marker = STYLE[epochs]
            curve = [
                mean_metric(
                    synthetic_cell(records, family, f'smi-a3-feedback_c={c}'), metric
                )
                for c in Q3_FEEDBACK_C
            ]
            ax.plot(
                Q3_FEEDBACK_C,
                curve,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
            )
            michelson = synthetic_cell(records, family, 'michelson')
            ax.axhline(
                mean_metric(michelson, f'{metric}_sign_invariant'),
                color=color,
                linestyle=':',
                linewidth=1.5,
            )
            for sim, control_marker, _ in CONTROLS:
                ax.plot(
                    0.2,
                    mean_metric(synthetic_cell(records, family, sim), metric),
                    marker=control_marker,
                    markersize=8,
                    markerfacecolor='none',
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    linestyle='none',
                )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_off()
        ax.set_xticks(Q3_FEEDBACK_C, [f'{c:g}' for c in Q3_FEEDBACK_C], fontsize=8)
        ax.set_yticks(yticks, [f'{y:g}' for y in yticks], fontsize=8)
        ax.set_xlabel('feedback strength $C$')
        ax.set_ylabel(ylabel)
        panel_label(ax, tag)

    handles = [
        Line2D(
            [],
            [],
            color=STYLE[90][0],
            linestyle='-',
            marker='o',
            label='90-epoch cycle',
        ),
        Line2D(
            [],
            [],
            color=STYLE[30][0],
            linestyle='--',
            marker='s',
            label='30-epoch cycle',
        ),
        Line2D([], [], color='k', linestyle=':', label='Michelson, sign-inv.'),
        *(
            Line2D(
                [],
                [],
                color='k',
                marker=m,
                markerfacecolor='none',
                linestyle='none',
                label=lab,
            )
            for _, m, lab in CONTROLS
        ),
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
