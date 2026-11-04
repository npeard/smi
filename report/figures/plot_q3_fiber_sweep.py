#!/usr/bin/env python
"""Q3: TCAN velocity RMSE as each fiber impairment turns on alone.

One panel per knob of q3_fiber_sweep (mode count, speckle drift, path drift),
each from the one-mode static fiber with the other knobs off. Filled: signed
RMSE, bars the per-cell shot SE; hollow: sign-invariant RMSE. Reference lines:
free space and the zero predictor, which every point shares (paired motion).
Reads ``fiber_sweep_mechanism.json`` and the per-shot errors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import COLORS, parse_args, save

from smi.core.paths import RESULTS_DIR
from smi.experiments.grids import GRIDS
from smi.experiments.grids.q3_fiber_sweep import (
    MODES,
    PATH_UM_PER_SQRT_S,
    SPECKLE_RAD_PER_SQRT_S,
)
from smi.experiments.runner import expand_grid

PANELS = (
    ('mode count $M$', [f'modes-{n}' for n in MODES], MODES),
    (
        r'speckle drift (rad/s$^{1/2}$)',
        [f'speckle-{r:g}' for r in SPECKLE_RAD_PER_SQRT_S],
        SPECKLE_RAD_PER_SQRT_S,
    ),
    (
        r'path drift ($\mu$m/s$^{1/2}$)',
        [f'path-{r:g}' for r in PATH_UM_PER_SQRT_S],
        PATH_UM_PER_SQRT_S,
    ),
)


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    mech = json.loads((RESULTS_DIR / 'fiber_sweep_mechanism.json').read_text())
    per_shot = json.loads((RESULTS_DIR / 'per_shot_rmse.json').read_text())
    sweep, zero = mech['sweep'], mech['zero_predictor']

    keys = {c.sim_name: c.key for c in expand_grid(GRIDS['q3_fiber_sweep'])}

    def se(name: str) -> float:
        x = np.array(per_shot[keys[name]]['velocity_rmse'])
        return float(x.std(ddof=1) / np.sqrt(x.size))

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, (label, names, xs) in zip(axes, PANELS, strict=True):
        ax.axhline(zero, color=COLORS['muted'], lw=1.2, label='zero predictor')
        ax.axhline(
            sweep['nominal']['rmse'],
            color=COLORS['muted'],
            lw=1.2,
            ls='--',
            label='free space',
        )
        ax.errorbar(
            xs,
            [sweep[n]['rmse'] for n in names],
            yerr=[se(n) for n in names],
            marker='o',
            color=COLORS['accent'],
            capsize=2,
            label='signed',
        )
        ax.plot(
            xs,
            [sweep[n]['rmse_sign_invariant'] for n in names],
            ls='none',
            marker='o',
            ms=8,
            mfc='none',
            color=COLORS['ch675'],
            label='sign-invariant',
        )
        ax.set_xscale('log', base=2 if names[0].startswith('modes') else 10)
        ax.set_xticks(xs, [f'{x:g}' for x in xs])
        ax.minorticks_off()
        ax.set_xlabel(label)
    axes[0].set_ylabel(r'test velocity RMSE ($\mu$m/ms)')
    axes[0].set_ylim(0, None)
    axes[0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
