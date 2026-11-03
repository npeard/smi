#!/usr/bin/env python
"""Feedback gain of the Sagnac self-mixing acceleration Fisher against C_eff (Q4).

Gain is the Fisher information at feedback C_0 over that at C_0 = 0, at the
pi/2 loop bias and fixed modulation depth: phase-averaged (dashed) and at the
best phase phi_0 (solid, a genie bound). Reads sagnac_results.json, written
by scripts/sagnac_table.py from smi.theory.sagnac.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import COLORS, FIGSIZE_HALF, RESULTS_DIR, parse_args, save

# alpha = 5 is in the JSON and the table but not drawn: within a few percent
# of alpha = 3 on this log axis, it only overlaps it.
ALPHA_COLORS = {'0': COLORS['muted'], '3': COLORS['accent']}


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    curves = json.loads((RESULTS_DIR / 'sagnac_results.json').read_text())['curves']
    fig, ax = plt.subplots(figsize=(FIGSIZE_HALF[0] * 1.3, FIGSIZE_HALF[1]))
    for alpha, color in ALPHA_COLORS.items():
        c = curves[alpha]
        ax.plot(c['c_eff'], c['best'], color=color, label=rf'$\alpha={alpha}$')
        ax.plot(c['c_eff'], c['avg'], color=color, ls='--')
    ax.plot([], [], 'k-', label='best phase')
    ax.plot([], [], 'k--', label='phase-averaged')
    ax.set_yscale('log')
    ax.set_xlim(0, 0.95)
    ax.set_xlabel(r'$C_\mathrm{eff} = C_0 \cos(\Delta\phi_b / 2)$')
    ax.set_ylabel('acceleration Fisher gain')
    ax.legend(fontsize=7, loc='upper left')
    save(fig, args.output)


if __name__ == '__main__':
    main()
