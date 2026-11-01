#!/usr/bin/env python
"""Model families at a matched 44k parameters: noise-free test RMSE per family.

(a) One row per family, causal families above the offline ones and the classical
oracle last. Hollow grey markers are Michelson, scored sign-invariant; filled
blue markers are C = 0.2, scored signed. Each sim's shaded band is the TCN's
value widened by the seed range, so a family outside the band is resolved from
the TCN. A cell that never left the zero-output plateau is drawn as a red cross
at its score. (b) The S1d retargeting test: displacement RMSE of each family
the S1d grid retrained on displacement, beside its velocity-target cell on the
same test shots, in the same sim styles. Reads the runner's records in
synthetic_results.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    DISPLACEMENT_METRICS,
    Q2_CAUSAL,
    Q2_OFFLINE,
    Q2_SIMS,
    S1D_FAMILIES,
    STALL_TOLERANCE,
    SYNTHETIC_WAVELENGTHS_UM,
    mean_metric,
    noise_seeds,
    oracle_cell,
    panel_label,
    parse_args,
    q2_record,
    s1d_record,
    save,
    synthetic_records,
)

from smi.experiments.results_artifact import RunRecord
from smi.experiments.runner import cell_key

SIM_STYLE = {'michelson': COLORS['muted'], 'smi-a3': COLORS['accent']}
STALLED = 'C3'


def stalled(record: RunRecord) -> bool:
    """Whether the run never left the zero-output plateau (the runner's test)."""
    if record.converged:
        return False
    assert record.best_monitor is not None
    assert record.zero_predictor_monitor is not None
    assert record.best_monitor >= (1 - STALL_TOLERANCE) * record.zero_predictor_monitor
    return True


def seed_resolution(records: dict[str, RunRecord]) -> float:
    """Seed range over mean of the noise-free base, as synthetic_values.py reads it."""
    base = [
        mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, 'ladder-base')
    ]
    return (max(base) - min(base)) / (sum(base) / len(base))


def sim_marker(ax: plt.Axes, x: float, y: float, sim: str, marker: str) -> None:
    """One cell in its sim's style: hollow grey Michelson, filled blue C = 0.2."""
    ax.plot(
        x,
        y,
        marker=marker,
        markersize=6,
        color=SIM_STYLE[sim],
        markerfacecolor=SIM_STYLE[sim] if sim == 'smi-a3' else 'white',
        markeredgewidth=1.4,
        linestyle='none',
    )


def retarget_panel(ax: plt.Axes, records: dict[str, RunRecord]) -> None:
    """Displacement RMSE of each S1d family, velocity target against displacement.

    The two sims of a family share its row, offset apart; a thin line joins a
    family's velocity-target cell (circle) to its displacement-target cell
    (square), and a stalled cell carries the red cross of panel (a).
    """
    for y, (stem, _) in enumerate(S1D_FAMILIES):
        for sim, metric in Q2_SIMS.items():
            row = y + (-0.17 if sim == 'michelson' else 0.17)
            score = DISPLACEMENT_METRICS[metric]
            velocity = q2_record(records, stem, sim)
            assert velocity is not None
            cells = ((velocity, 'o'), (s1d_record(records, stem, sim), 's'))
            xs = [mean_metric(r, score) for r, _ in cells]
            ax.plot(xs, [row, row], color=SIM_STYLE[sim], lw=0.8, alpha=0.6)
            for x, (record, marker) in zip(xs, cells, strict=True):
                sim_marker(ax, x, row, sim, marker)
                if stalled(record):
                    ax.plot(x, row, marker='x', color=STALLED, markersize=7, mew=1.6)
    ax.set_yticks(
        range(len(S1D_FAMILIES)), [lab for _, lab in S1D_FAMILIES], fontsize=8
    )
    ax.set_ylim(len(S1D_FAMILIES) - 0.5, -0.5)
    ax.set_xscale('log')
    ticks = (0.05, 0.1, 0.2, 0.5, 1.0)
    ax.minorticks_off()
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks], fontsize=8)
    ax.set_xlim(0.04, 1.0)
    ax.set_xlabel(r'noise-free test displacement RMSE ($\mu$m)')
    ax.legend(
        handles=[
            Line2D([], [], marker='o', color='k', mfc='white', ls='none'),
            Line2D([], [], marker='s', color='k', mfc='white', ls='none'),
        ],
        labels=['trained on velocity', 'trained on displacement'],
        loc='lower left',
        fontsize=7.5,
    )
    panel_label(ax, '(b)')


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    rows = [*Q2_CAUSAL, *Q2_OFFLINE]
    resolution = seed_resolution(records)

    fig, (ax, ax_b) = plt.subplots(
        2, 1, figsize=(5.6, 6.2), height_ratios=(15, 5.2), layout='constrained'
    )
    for sim, metric in Q2_SIMS.items():
        tcn = mean_metric(
            records[cell_key('tcn-44k-T90', SYNTHETIC_WAVELENGTHS_UM, sim, 0)], metric
        )
        ax.axvspan(
            tcn / (1 + resolution),
            tcn * (1 + resolution),
            color=SIM_STYLE[sim],
            alpha=0.18,
            lw=0,
        )
        for y, (stem, _) in enumerate(rows):
            record = q2_record(records, stem, sim)
            if record is None:
                continue
            x = mean_metric(record, metric)
            if stalled(record):
                ax.plot(x, y, marker='x', color=STALLED, markersize=7, mew=1.6)
                continue
            sim_marker(ax, x, y, sim, 'o')
        oracle = mean_metric(oracle_cell(records, sim), metric)
        ax.plot(
            oracle,
            len(rows),
            marker='D',
            markersize=5.5,
            color=SIM_STYLE[sim],
            markerfacecolor=SIM_STYLE[sim] if sim == 'smi-a3' else 'white',
            markeredgewidth=1.4,
        )
    zero = mean_metric(
        records[cell_key('tcn-44k-T90', SYNTHETIC_WAVELENGTHS_UM, 'smi-a0', 0)],
        'velocity_rmse',
    )
    ax.axvline(zero, color='k', ls='--', lw=1.0)
    ax.axhline(len(Q2_CAUSAL) - 0.5, color=COLORS['muted'], lw=0.6)
    ax.axhline(len(rows) - 0.5, color=COLORS['muted'], lw=0.6)
    labels = [label for _, label in rows] + ['inverse-fit oracle']
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for y, text in ((0, 'causal'), (len(Q2_CAUSAL), 'offline')):
        ax.annotate(
            text,
            (0.99, y),
            xycoords=('axes fraction', 'data'),
            ha='right',
            va='center',
            fontsize=7.5,
            style='italic',
            color=COLORS['muted'],
        )
    ax.invert_yaxis()
    ax.set_xscale('log')
    ticks = (0.02, 0.05, 0.1, 0.2, 0.5)
    ax.minorticks_off()
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks], fontsize=8)
    ax.set_xlabel(r'noise-free test velocity RMSE ($\mu$m/ms)')
    panel_label(ax, '(a)')
    retarget_panel(ax_b, records)
    handles = [
        Line2D(
            [],
            [],
            marker='o',
            color=SIM_STYLE['michelson'],
            markerfacecolor='white',
            mew=1.4,
            ls='none',
        ),
        Line2D([], [], marker='o', color=SIM_STYLE['smi-a3'], ls='none'),
        Line2D([], [], marker='x', color=STALLED, mew=1.6, ls='none'),
        Patch(color=SIM_STYLE['michelson'], alpha=0.18, lw=0),
        Patch(color=SIM_STYLE['smi-a3'], alpha=0.18, lw=0),
        Line2D([], [], color='k', ls='--', lw=1.0),
    ]
    labels = [
        'Michelson, sign-invariant',
        '$C=0.2$, signed',
        'stalled at the zero-output plateau',
        'TCN $\\pm$ seed range, Michelson',
        'TCN $\\pm$ seed range, $C=0.2$',
        'zero predictor',
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=7.5,
    )
    save(fig, args.output)


if __name__ == '__main__':
    main()
