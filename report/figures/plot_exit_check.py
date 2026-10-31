#!/usr/bin/env python
"""Signed and sign-invariant velocity NRMSE of the three exit-check runs.

NRMSE divides each shot's RMSE by that shot's velocity std, so a predictor
that always outputs zero scores exactly 1: the level the Michelson model
trained with signed MSE is predicted to collapse to. RMSE would need the test
set's velocity std for that reference, which the records do not carry.

Reads the runner's records in synthetic_results.json and recomputes nothing.
The signed Michelson run lacks zero_predictor_monitor and was stopped by
patience at epoch 11, inside the min_epochs grace; the report quotes that
collapse, so it is exempt from the grace check (SIGN_COLLAPSE_CONTROLS).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIGSIZE_HALF,
    SIGN_COLLAPSE_CONTROLS,
    mean_metric,
    parse_args,
    save,
    synthetic_cell,
    synthetic_records,
)

FAMILY = 'tcn-44k-E20'
# Plot order and two-line labels: interferometer, then training loss.
RUNS = (
    ('michelson-signed', 'Michelson\nsigned MSE'),
    ('michelson-signinv', 'Michelson\nsign-inv. MSE'),
    ('smi-c0.2-signed', 'SMI $C=0.2$\nsigned MSE'),
)


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records(SIGN_COLLAPSE_CONTROLS)

    x = np.arange(len(RUNS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(FIGSIZE_HALF[0] * 1.3, FIGSIZE_HALF[1]))
    for offset, metric, label, color in (
        (-width / 2, 'velocity_nrmse', 'signed', COLORS['accent']),
        (
            width / 2,
            'velocity_nrmse_sign_invariant',
            'sign-invariant',
            COLORS['highlight'],
        ),
    ):
        values = [
            mean_metric(synthetic_cell(records, FAMILY, sim), metric) for sim, _ in RUNS
        ]
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.axhline(1.0, color=COLORS['muted'], linestyle='--', label='zero predictor')
    ax.set_xticks(x, [label for _, label in RUNS])
    ax.set_ylabel('velocity NRMSE')
    # Headroom above the reference line so the legend clears the tallest bar.
    ax.set_ylim(0, 1.4)
    ax.legend(loc='upper center', fontsize=8, ncol=3, columnspacing=1.0)
    save(fig, args.output)


if __name__ == '__main__':
    main()
