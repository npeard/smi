#!/usr/bin/env python
"""Displacement RMSE beside velocity RMSE, from the same checkpoints and test shots.

(a) Noise-free displacement RMSE of the causal TCN against feedback strength
C, per cosine schedule, with Michelson (sign-invariant) as a dotted line, the
layout of the feedback-curve figure's velocity panel. (b) Displacement against
velocity RMSE for every trained S1 family at both sims, so a family whose
velocity ranking does not carry over to displacement leaves the trend.

Displacement is the predicted velocity integrated over the shot, each start
offset removed (``DISPLACEMENT_METRICS``). Until the records carry it, both
panels say PENDING rather than plot nothing. Reads synthetic_results.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    DISPLACEMENT_METRICS,
    FIGSIZE_WIDE,
    Q2_SIMS,
    Q2_STEMS,
    Q3_BUDGETS,
    Q3_FEEDBACK_C,
    c_sim,
    displacement_pending,
    mean_metric,
    panel_label,
    parse_args,
    pending_panel,
    q2_record,
    save,
    synthetic_cell,
    synthetic_records,
)

from smi.analysis.results_artifact import RunRecord

SIM_STYLE = {
    'michelson': (COLORS['muted'], 'Michelson, sign-inv.'),
    'smi-a3': (COLORS['accent'], '$C=0.2$, signed'),
}


# The C = 0.2 families the caption names, as (label, offset in points,
# alignment); offsets keep the clustered lowest three apart.
LABELED = {
    'patchtransformer-offline-44k': ('PatchTr. (off.)', (5, -4), 'left'),
    'tcan-44k': ('TCAN', (0, -8), 'center'),
    'unettransformer-offline-44k': ('UNet-Tr. (off.)', (5, 4), 'left'),
    'tcn-44k': ('TCN', (5, -5), 'left'),
    'mamba-44k': ('Mamba', (-5, 3), 'right'),
}


def plain_log(axis: Axis) -> None:
    """Log ticks at 1-2-5 per decade, printed plainly rather than as powers."""
    axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))


def curve_panel(ax: plt.Axes, records: dict[str, RunRecord]) -> None:
    signed = DISPLACEMENT_METRICS['velocity_rmse']
    si = DISPLACEMENT_METRICS['velocity_rmse_sign_invariant']
    for budget in Q3_BUDGETS:
        ax.plot(
            Q3_FEEDBACK_C,
            [
                mean_metric(synthetic_cell(records, budget.family, c_sim(c)), signed)
                for c in Q3_FEEDBACK_C
            ],
            color=budget.color,
            linestyle=budget.linestyle,
            marker=budget.marker,
            markersize=5,
            label=f'{budget.epochs}-epoch cycle',
        )
        ax.axhline(
            mean_metric(synthetic_cell(records, budget.family, 'michelson'), si),
            color=budget.color,
            linestyle=':',
            linewidth=1.5,
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Headroom so the Michelson lines clear the panel label.
    ax.set_ylim(top=1.6 * ax.get_ylim()[1])
    plain_log(ax.yaxis)
    ax.set_xticks(Q3_FEEDBACK_C, [f'{c:g}' for c in Q3_FEEDBACK_C], fontsize=8)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('feedback strength $C$')
    ax.set_ylabel(r'displacement RMSE ($\mu$m)')
    ax.legend(fontsize=7, title='dotted: Michelson', title_fontsize=7)


def families_panel(ax: plt.Axes, cells: list[tuple[str, str, RunRecord]]) -> None:
    for stem, sim, record in cells:
        if not record.converged:
            continue
        metric = Q2_SIMS[sim]
        color, _ = SIM_STYLE[sim]
        xy = (
            mean_metric(record, metric),
            mean_metric(record, DISPLACEMENT_METRICS[metric]),
        )
        ax.plot(
            *xy,
            marker='o',
            color=color,
            markerfacecolor=color if sim == 'smi-a3' else 'white',
            markeredgewidth=1.3,
            linestyle='none',
        )
        if sim == 'smi-a3' and stem in LABELED:
            text, offset, ha = LABELED[stem]
            ax.annotate(
                text,
                xy,
                xytext=offset,
                textcoords='offset points',
                ha=ha,
                va='center',
                fontsize=7,
                color=color,
            )
    for color, label in SIM_STYLE.values():
        ax.plot([], [], marker='o', color=color, ls='none', label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Room for the labels of the lowest points inside the axes.
    ax.margins(0.12)
    plain_log(ax.xaxis)
    plain_log(ax.yaxis)
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'velocity RMSE ($\mu$m/ms)')
    ax.set_ylabel(r'displacement RMSE ($\mu$m)')
    ax.legend(fontsize=7)


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    curve = [
        synthetic_cell(records, b.family, sim)
        for b in Q3_BUDGETS
        for sim in [*map(c_sim, Q3_FEEDBACK_C), 'michelson']
    ]
    families = [
        (stem, sim, r)
        for stem in Q2_STEMS
        for sim in Q2_SIMS
        if (r := q2_record(records, stem, sim)) is not None
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FIGSIZE_WIDE[0], 2.9))
    if displacement_pending([*curve, *(r for _, _, r in families)]):
        pending_panel(a, 'displacement RMSE against $C$\n(rerun plan item a)')
        pending_panel(b, 'displacement against velocity RMSE\n(rerun plan item a)')
    else:
        curve_panel(a, records)
        families_panel(b, families)
    panel_label(a, '(a)')
    panel_label(b, '(b)')
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
