#!/usr/bin/env python
"""Leave-one-out at the nominal point: test RMSE with one knob returned to zero.

One row per knob, ranked by error. The red band is every knob at nominal over
its seeds and the grey band the noise-free base, so a row's distance below the
red band is what removing that knob alone buys. Hollow markers are runs that
trained past their second warm restart, whose gain is partly training length;
purple squares are the cells rerun at exactly the matched-length grid's epoch
count. Reads the runner's records in synthetic_results.json.
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
    LEAVE_ONE_OUT,
    NOISE_KNOBS,
    matched_length_cells,
    mean_metric,
    noise_seeds,
    parse_args,
    save,
    synthetic_records,
)

from smi.analysis.results_artifact import RunRecord

EQUAL_LENGTH = COLORS['highlight']


def rmse(record: RunRecord) -> float:
    return mean_metric(record, 'velocity_rmse')


def band(ax: plt.Axes, runs: list[RunRecord], color: str) -> float:
    """Seed range of ``runs`` as a vertical band, with a line at its mean."""
    xs = [rmse(r) for r in runs]
    mean = sum(xs) / len(xs)
    ax.axvspan(min(xs), max(xs), color=color, alpha=0.25, lw=0)
    ax.axvline(mean, color=color, lw=1.2)
    return mean


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    rows = []
    for field, stem in LEAVE_ONE_OUT:
        (record,) = noise_seeds(records, f'nominal-no-{stem}')
        rows.append((NOISE_KNOBS[field], stem, record))
    # Ranked so the largest single cut sits at the top.
    rows.sort(key=lambda row: -rmse(row[2]))
    equal, _ = matched_length_cells(records)
    (equal_epochs,) = {r.epochs_run for r in equal.values()}

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    nominal = band(ax, noise_seeds(records, 'nominal'), 'C3')
    base = band(ax, noise_seeds(records, 'ladder-base'), COLORS['muted'])
    # How much the longer, equal-length training alone buys each end: much
    # at the base, little at the nominal point.
    for name, label, mean, ls, side, y in (
        ('ladder-base', 'noise-free', base, '--', 1, len(rows) - 3),
        ('nominal', 'all nominal', nominal, ':', -1, 1.5),
    ):
        x = rmse(equal[name])
        ax.axvline(x, color=EQUAL_LENGTH, ls=ls, lw=1.0 if ls == '--' else 1.2)
        ax.annotate(
            f'{label}, {equal_epochs} epochs:\n'
            f'{100 * (1 - x / mean):.0f}% below its seed mean',
            (x, y),
            xytext=(3 * side, 0),
            textcoords='offset points',
            ha='left' if side > 0 else 'right',
            va='center',
            fontsize=7,
            color=EQUAL_LENGTH,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1},
        )
    for y, (_, stem, record) in enumerate(rows):
        third = (record.epochs_run or 0) > 2 * (record.T_0 or 0)
        ax.plot(
            rmse(record),
            y,
            marker='o',
            markersize=7,
            color=COLORS['accent'],
            markerfacecolor='white' if third else COLORS['accent'],
            markeredgewidth=1.4,
            linestyle='none',
        )
        if (rerun := equal.get(f'nominal-no-{stem}')) is not None:
            # Offset below its row so it cannot hide the circle.
            ax.plot(
                rmse(rerun), y - 0.3, marker='s', color=EQUAL_LENGTH, linestyle='none'
            )
    ax.set_yticks(range(len(rows)), [label for label, _, _ in rows], fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.4)
    ax.set_xlim(0, None)
    ax.set_xlabel(r'test velocity RMSE ($\mu$m/ms), that knob removed')
    ax.tick_params(axis='x', labelsize=8)
    handles = [
        Line2D([], [], marker='o', color=COLORS['accent'], linestyle='none'),
        Line2D(
            [],
            [],
            marker='o',
            color=COLORS['accent'],
            markerfacecolor='white',
            linestyle='none',
        ),
        Patch(color='C3', alpha=0.4),
        Patch(color=COLORS['muted'], alpha=0.4),
        Line2D([], [], marker='s', color=EQUAL_LENGTH, linestyle='none'),
        Line2D([], [], color=EQUAL_LENGTH, ls=':'),
        Line2D([], [], color=EQUAL_LENGTH, ls='--'),
    ]
    labels = [
        'stopped by patience',
        'trained into a third cycle',
        'all knobs nominal (seeds)',
        'noise-free base (seeds)',
        f'knob removed, {equal_epochs} epochs exactly',
        f'all nominal, {equal_epochs} epochs exactly',
        f'noise-free, {equal_epochs} epochs exactly',
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=7.5,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
