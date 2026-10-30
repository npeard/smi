#!/usr/bin/env python
"""Signed and sign-invariant velocity RMSE of the three exit-check runs.

The horizontal line is the std of the true velocity on the test set: the RMSE
of a predictor that always outputs zero, which is where the Michelson model
trained with signed MSE is predicted to land.

Reads the committed exit-check artifact and recomputes nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import COLORS, DATA_DIR, FIGSIZE_HALF, parse_args, save

EXIT_CHECK_JSON = DATA_DIR / 'exit_check_results.json'

# Plot order and two-line labels: interferometer, then training loss.
RUNS = (
    ('C_michelson_signed', 'Michelson\nsigned MSE'),
    ('B_michelson_sign_inv', 'Michelson\nsign-inv. MSE'),
    ('A_smi_signed', 'SMI $C=0.2$\nsigned MSE'),
)


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    results = json.loads(EXIT_CHECK_JSON.read_text(encoding='utf-8'))

    x = np.arange(len(RUNS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(FIGSIZE_HALF[0] * 1.3, FIGSIZE_HALF[1]))
    for offset, key, label, color in (
        (-width / 2, 'velocity_rmse', 'signed', COLORS['accent']),
        (
            width / 2,
            'velocity_rmse_sign_invariant',
            'sign-invariant',
            COLORS['highlight'],
        ),
    ):
        values = [results['runs'][name][key] for name, _ in RUNS]
        ax.bar(x + offset, values, width, label=label, color=color)
    ax.axhline(
        results['std_v_um_per_ms'],
        color=COLORS['muted'],
        linestyle='--',
        label=r'mean per-shot std of $v$',
    )
    ax.set_xticks(x, [label for _, label in RUNS])
    ax.set_ylabel(r'velocity RMSE ($\mu$m/ms)')
    # Headroom above the std line so the legend clears the tallest bar.
    ax.set_ylim(0, 0.52)
    ax.legend(loc='upper center', fontsize=8, ncol=3, columnspacing=1.0)
    save(fig, args.output)


if __name__ == '__main__':
    main()
