#!/usr/bin/env python
"""Noise-free velocity error of the causal TCN against SMI feedback strength C.

(a) velocity RMSE and (b) velocity NRMSE, each the mean over the frozen test
shots of the per-shot value. SMI cells are scored signed; the Michelson
reference is scored sign-invariant, the best any estimator can do there. One
curve per cosine schedule, so the reader sees how much of each value is
training budget. (c) bins the 90-epoch cells' per-sample error by |v| decile,
which is why (a) and (b) disagree: RMSE pools samples, NRMSE weights shots.

Reads the runner's records in synthetic_results.json and recomputes nothing.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    FIGSIZE_WIDE,
    Q3_BUDGETS,
    Q3_FEEDBACK_C,
    c_sim,
    mean_metric,
    panel_label,
    parse_args,
    save,
    synthetic_cell,
    synthetic_records,
)

from smi.analysis.results_artifact import RunRecord

# The alpha controls, all at C = 0.2.
CONTROLS = (('smi-a5', '^', r'$\alpha=5$'), ('smi-a0', 'X', r'$\alpha=0$'))
# (metric, axis label, labelled y ticks): log axes spanning under a decade
# get no major tick labels by default.
PANELS = (
    ('velocity_rmse', r'velocity RMSE ($\mu$m/ms)', (0.02, 0.05, 0.1, 0.2, 0.5)),
    ('velocity_nrmse', 'velocity NRMSE', (0.1, 0.2, 0.5, 1.0)),
)
# Panel (c): (sim, line style, marker, direct label), all 90-epoch cells.
BINNED = (
    ('michelson', ':', 's', 'Michelson'),
    ('smi-a3-feedback_c=0.2', '--', 'o', '$C=0.2$'),
    ('smi-a3-feedback_c=0.9', '-', 'o', '$C=0.9$'),
)


def log_axes(ax: plt.Axes, xticks: Sequence[float], yticks: Sequence[float]) -> None:
    """Log both axes, labelling exactly ``xticks`` and ``yticks``."""
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_off()
    ax.set_xticks(xticks, [f'{x:g}' for x in xticks], fontsize=8)
    ax.set_yticks(yticks, [f'{y:g}' for y in yticks], fontsize=8)


def sign_agreement(
    ax: plt.Axes, records: dict[str, RunRecord], signed: list[float]
) -> None:
    """Overlay the 90-epoch curve's sign-invariant RMSE and the largest gap.

    The two coinciding is the evidence that the sign is resolved on nearly
    every shot, so the gap is printed on the panel rather than left to prose.
    """
    family = Q3_BUDGETS[0].family
    si = [
        mean_metric(
            synthetic_cell(records, family, c_sim(c)), 'velocity_rmse_sign_invariant'
        )
        for c in Q3_FEEDBACK_C
    ]
    ax.plot(Q3_FEEDBACK_C, si, ls='none', marker='+', ms=8, mew=1.2, color='k')
    gap = max(s / i - 1 for s, i in zip(signed, si, strict=True))
    ax.text(
        0.04,
        0.04,
        f'+: sign-inv., within {100 * gap:.0f}%\nof signed (90 epochs)',
        transform=ax.transAxes,
        fontsize=7,
    )


def slow_floor(ax: plt.Axes, records: dict[str, RunRecord]) -> None:
    """Print the slowest decile's error range over every C, per budget.

    The floor is a training-budget property (section 2), so the panel states
    its range at both budgets over Michelson and all six C, not only the
    three plotted cells.
    """
    sims = ('michelson', *map(c_sim, Q3_FEEDBACK_C))
    lines = []
    for budget in Q3_BUDGETS:
        slow = []
        for sim in sims:
            binned = synthetic_cell(records, budget.family, sim).binned
            assert binned is not None, sim
            slow.append(binned['abs_velocity'][0]['rmse_sign_invariant'])
        lines.append(f'{budget.epochs} ep. {min(slow):.3f}-{max(slow):.3f}')
    ax.text(
        0.04,
        0.88,
        'slowest decile,\nall $C$:\n' + '\n'.join(lines),
        transform=ax.transAxes,
        fontsize=6.5,
        va='top',
    )


def binned_panel(ax: plt.Axes, records: dict[str, RunRecord]) -> None:
    """Sign-invariant RMSE by |v| decile, at each decile's midpoint speed."""
    budget = Q3_BUDGETS[0]
    for sim, linestyle, marker, label in BINNED:
        binned = synthetic_cell(records, budget.family, sim).binned
        assert binned is not None, sim
        rows = binned['abs_velocity']
        speed = [(r['lo'] + r['hi']) / 2 for r in rows]
        error = [r['rmse_sign_invariant'] for r in rows]
        ax.plot(
            speed,
            error,
            color=budget.color,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=1.5,
        )
        ax.annotate(
            label,
            (speed[-1], error[-1]),
            xytext=(4, 0),
            textcoords='offset points',
            va='center',
            fontsize=8,
        )
        if sim == 'michelson':
            # Slope-one guide, through half of Michelson's fastest bin, so
            # its proportionality to |v| reads off the log axes.
            k = 0.5 * error[-1] / speed[-1]
            guide = (0.1, speed[-1])
            ax.plot(guide, [k * x for x in guide], color='0.35', lw=1.0, ls='-.')
            ax.annotate(
                r'$\propto |v|$',
                (guide[1], k * guide[1]),
                xytext=(3, -4),
                textcoords='offset points',
                fontsize=8,
                color='0.35',
            )
    slow_floor(ax, records)
    log_axes(ax, (0.01, 0.1, 1.0), (0.02, 0.05, 0.1, 0.2, 0.5))
    ax.set_xlim(right=8)
    # Headroom above Michelson's slow bins for the floor's range.
    ax.set_ylim(top=2.5)
    ax.set_xlabel(r'$|v|$ decile midpoint ($\mu$m/ms)')
    ax.set_ylabel(r'sign-inv. RMSE ($\mu$m/ms)')
    panel_label(ax, '(c)')


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    for ax, (metric, ylabel, yticks), tag in zip(
        axes[:2], PANELS, ['(a)', '(b)'], strict=True
    ):
        for budget in Q3_BUDGETS:
            curve = [
                mean_metric(synthetic_cell(records, budget.family, c_sim(c)), metric)
                for c in Q3_FEEDBACK_C
            ]
            ax.plot(
                Q3_FEEDBACK_C,
                curve,
                color=budget.color,
                linestyle=budget.linestyle,
                marker=budget.marker,
                markersize=5,
            )
            if metric == 'velocity_rmse' and budget is Q3_BUDGETS[0]:
                sign_agreement(ax, records, curve)
            michelson = synthetic_cell(records, budget.family, 'michelson')
            ax.axhline(
                mean_metric(michelson, f'{metric}_sign_invariant'),
                color=budget.color,
                linestyle=':',
                linewidth=1.5,
            )
            for sim, control_marker, _ in CONTROLS:
                ax.plot(
                    0.2,
                    mean_metric(synthetic_cell(records, budget.family, sim), metric),
                    marker=control_marker,
                    markersize=8,
                    markerfacecolor='none',
                    markeredgecolor=budget.color,
                    markeredgewidth=1.2,
                    linestyle='none',
                )
        log_axes(ax, Q3_FEEDBACK_C, yticks)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xlabel('feedback strength $C$')
        ax.set_ylabel(ylabel)
        panel_label(ax, tag)

    binned_panel(axes[2], records)

    # Each Michelson line and control marker is drawn once per budget, in that
    # budget's color, so the legend shows both colors side by side.
    def per_budget(**style: object) -> tuple[Line2D, ...]:
        return tuple(Line2D([], [], color=b.color, **style) for b in Q3_BUDGETS)

    handles = [
        *(
            Line2D([], [], color=b.color, linestyle=b.linestyle, marker=b.marker)
            for b in Q3_BUDGETS
        ),
        per_budget(linestyle=':', linewidth=1.5),
        *(
            per_budget(
                marker=m,
                markersize=8,
                markerfacecolor='none',
                markeredgewidth=1.2,
                linestyle='none',
            )
            for _, m, _ in CONTROLS
        ),
    ]
    labels = [
        *(f'{b.epochs}-epoch cycle' for b in Q3_BUDGETS),
        'Michelson, sign-inv.',
        *(lab for _, _, lab in CONTROLS),
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='lower center',
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
