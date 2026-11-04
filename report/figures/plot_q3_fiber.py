#!/usr/bin/env python
"""Q3.1 and Q3.2: test velocity RMSE of each q3_fiber arm, per family.

Error bars are the per-cell shot SE (``shot_se``); hollow markers did not
converge. The grey line is the zero predictor, which every arm shares because
the arms draw the same motion shot for shot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIBER_ARMS,
    R_FAMILIES,
    R_FAMILY_LABELS,
    mean_metric,
    parse_args,
    r_cell,
    save,
    shot_se,
    synthetic_records,
)

from smi.data.frozen import DEFAULT_ROOT, load_or_create
from smi.experiments.diagnostics import shot_speed
from smi.experiments.grids import GRIDS
from smi.experiments.runner import expand_grid

LABELS = {
    'nominal': 'free space',
    'fiber': 'fiber',
    'fiber-4color': 'fiber + 785 nm to target',
    'fiber-monitor': 'fiber + 785 nm monitor',
    'fiber-pbs': r'fiber + $P_\perp$',
}
STYLE = {
    R_FAMILIES[0]: (COLORS['accent'], 'o', -0.12),
    R_FAMILIES[1]: (COLORS['ch675'], 's', 0.12),
}


def zero_rmse(records: dict, arm: str) -> float:
    """Zero predictor on ``arm``'s frozen set: mean over shots of the velocity RMS."""
    record = r_cell(records, R_FAMILIES[0], FIBER_ARMS[arm], arm)
    (cell,) = [c for c in expand_grid(GRIDS['q3_fiber']) if c.key == record.cell_key]
    n = int(record.dataset.rsplit('_n', 1)[1])
    path = load_or_create(cell.sim, record.test_seed, n, DEFAULT_ROOT)
    return float(shot_speed(path).mean())


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    arms = list(FIBER_ARMS)
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.axvline(
        zero_rmse(records, 'fiber'),
        color=COLORS['muted'],
        lw=1.2,
        label='zero predictor',
    )
    for fam in R_FAMILIES:
        color, marker, dy = STYLE[fam]
        for y, arm in enumerate(arms):
            r = r_cell(records, fam, FIBER_ARMS[arm], arm)
            ax.errorbar(
                mean_metric(r, 'velocity_rmse'),
                y + dy,
                xerr=shot_se(r),
                marker=marker,
                color=color,
                mfc=color if r.converged else 'white',
                capsize=2,
                label=R_FAMILY_LABELS[fam] if y == 0 else None,
            )
    ax.set_yticks(range(len(arms)), [LABELS[a] for a in arms], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.set_xlabel(r'test velocity RMSE ($\mu$m/ms)')
    ax.legend(fontsize=7.5, loc='lower left')
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
