#!/usr/bin/env python
"""Q3.4: velocity RMSE and 90% band coverage of the encoder flow and the regressors.

Left, test velocity RMSE per estimator in free space and through the R7 fiber;
hollow markers rescore the flow's own estimators on the fresh seed-2000 set.
Right, coverage of each estimator's 90% band, where it has one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import COLORS, flow_rows, parse_args, save, synthetic_records
from plot_q3_fiber import zero_rmse

ARMS = {
    'fs': ('free space', COLORS['accent'], 'o', -0.12),
    'fiber': ('fiber', COLORS['ch675'], 's', 0.12),
}
BAND_LEVEL = 0.9


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    rows = flow_rows(records)
    fig, (ax, ax_c) = plt.subplots(
        1,
        2,
        figsize=(6.4, 2.9),
        sharey=True,
        width_ratios=(2.2, 1),
        layout='constrained',
    )
    ax.axvline(zero_rmse(records, 'fiber'), color=COLORS['muted'], lw=1.2, label='zero')
    for y, row in enumerate(rows.values()):
        for arm, (label, color, marker, dy) in ARMS.items():
            test, fresh = getattr(row, arm), getattr(row, f'{arm}_fresh')
            if test is not None:
                ax.plot(
                    test, y + dy, marker, color=color, label=label if y == 0 else None
                )
            if fresh is not None:
                ax.plot(
                    fresh,
                    y + dy,
                    marker,
                    color=color,
                    mfc='white',
                    label=f'{label}, fresh set' if y == len(rows) - 1 else None,
                )
            cov = getattr(row, f'coverage_{arm}')
            if cov is not None:
                ax_c.plot(cov, y + dy, marker, color=color)
    ax_c.axvline(BAND_LEVEL, color=COLORS['muted'], lw=1.0, ls='--')
    ax.set_yticks(range(len(rows)), [r.label for r in rows.values()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.set_xlabel(r'test velocity RMSE ($\mu$m/ms)')
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel('90% band coverage')
    fig.legend(fontsize=7, loc='outside lower center', ncol=5)
    save(fig, args.output)


if __name__ == '__main__':
    main()
