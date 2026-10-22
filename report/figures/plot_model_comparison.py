#!/usr/bin/env python
"""Every trained configuration against the analytic baseline it has to clear.

One panel per acquisition, configurations sorted by their free-space score so
both panels keep the same row order and any reordering between them is visible
as crossing rather than as two unrelated rankings. What the fixed order shows
is that the ranking barely reorders while the bars stretch: every learned
model degrades behind fiber, and the models that win biggest in free space
degrade the most.

Hatching marks the models that see samples after the timestamp they predict.
Only the TCN does not, so a reader comparing bar lengths across the hatch
boundary is comparing a causal estimator against ones given lookahead -- which
is the report's argument and not a footnote to it.

The dashed line is the analytic Michelson decode (method 2) on the same
acquisition, read from the committed baseline artifact.

Means over the two seeds, with the seed-to-seed spread as the error bar. Two
seeds give a range, not a standard error, so the bar is drawn as the full
spread and labelled as such.

Reads the two committed artifacts and recomputes nothing.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    BASELINE_DATASETS,
    COLORS,
    DATA_DIR,
    RESULTS_JSON,
    parse_args,
    save,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.analysis.results_artifact import RunRecord, read_results

LEARNED_JSON = DATA_DIR / 'learned_results.json'

METRIC = 'velocity_rmse_um_per_ms'


def _label(record: RunRecord) -> str:
    """Architecture, parameter count and receptive field, as one row label.

    All three are needed to read the ranking: two TCN rows differ only in
    receptive field, and two others only in parameter count.
    """
    # Ridge and LightGBM have hundreds of parameters, not tens of thousands,
    # so a fixed "k" suffix would round both to "0k" and erase the very
    # contrast the row is there to make.
    params = (
        f'{record.total_params / 1e3:.0f}k'
        if record.total_params >= 1000
        else str(record.total_params)
    )
    return f'{record.model_type} {params} (field {record.receptive_field})'


@dataclass
class Config:
    """One configuration's score on each dataset, collapsed over seeds.

    Attributes:
        sees_future: Whether the model reads samples after the predicted one.
        mean: Dataset key -> mean velocity RMSE over the seeds.
        spread: Dataset key -> full seed-to-seed range of that RMSE.
    """

    sees_future: bool
    mean: dict[str, float] = field(default_factory=dict)
    spread: dict[str, float] = field(default_factory=dict)


def _grouped(records: list[RunRecord]) -> dict[str, Config]:
    """Collapse the per-seed records into one entry per configuration."""
    by_key: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        by_key[(_label(record), record.dataset)].append(record)

    grouped: dict[str, Config] = {}
    for (label, dataset), runs in by_key.items():
        values = [run.metrics[METRIC]['mean'] for run in runs]
        config = grouped.setdefault(label, Config(sees_future=runs[0].sees_future))
        config.mean[dataset] = float(np.mean(values))
        config.spread[dataset] = float(np.ptp(values))
    return grouped


def main() -> None:
    args = parse_args(__doc__ or '')

    grouped = _grouped(read_results(LEARNED_JSON))
    analytic = json.loads(RESULTS_JSON.read_text(encoding='utf-8'))['datasets']

    # Sort by free space descending. barh puts y=0 at the bottom, so this
    # renders the best model on the top row of both panels, and any
    # fiber-panel reordering reads as crossing against a fixed row order.
    order = sorted(
        grouped, key=lambda label: grouped[label].mean['free-space'], reverse=True
    )
    y = np.arange(len(order), dtype=float)

    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 4.2), sharey=True, constrained_layout=True
    )

    for ax, (key, label) in zip(axes, BASELINE_DATASETS, strict=True):
        means = [grouped[name].mean[key] for name in order]
        spreads = [grouped[name].spread[key] for name in order]
        causal = [not grouped[name].sees_future for name in order]

        ax.barh(
            y,
            means,
            height=0.66,
            xerr=np.array(spreads) / 2.0,
            capsize=2.5,
            ecolor=COLORS['muted'],
            color=[COLORS['accent'] if c else COLORS['highlight'] for c in causal],
            hatch=['' if c else '//' for c in causal],
            edgecolor='white',
            linewidth=0.0,
        )

        baseline = analytic[key]['method2']['velocity_rmse_um_per_ms']['mean']
        ax.axvline(baseline, color=COLORS['ch635'], linestyle='--', linewidth=1.2)
        # Left of the line: right of it there is only the axis edge, and the
        # label ran past it at this x-limit.
        ax.text(
            baseline - 0.012,
            len(order) - 0.4,
            f'analytic decode {baseline:.3f}',
            color=COLORS['ch635'],
            fontsize=7.5,
            ha='right',
            va='center',
        )

        ax.set_xlabel('velocity RMSE (um/ms)')
        ax.set_title(label)
        ax.set_xlim(0, 0.66)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=8)
    axes[0].set_ylim(-0.7, len(order) - 0.05)

    # Below the axes. The bottom rows are the longest bars, so a legend
    # anchored inside "lower right" sat on top of them.
    fig.legend(
        handles=[
            Patch(facecolor=COLORS['accent'], label='causal (no lookahead)'),
            Patch(
                facecolor=COLORS['highlight'],
                hatch='//',
                edgecolor='white',
                label='sees future samples',
            ),
        ],
        loc='outside lower center',
        ncol=2,
        fontsize=8,
        frameon=False,
    )

    fig.suptitle(
        'Mean over 2 seeds at a 20-epoch budget; whiskers span the seed-to-seed '
        'range\nrow order is the free-space ranking, held fixed in both panels',
        fontsize=8,
        color=COLORS['muted'],
    )

    save(fig, args.output)


if __name__ == '__main__':
    main()
