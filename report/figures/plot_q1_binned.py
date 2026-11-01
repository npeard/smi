#!/usr/bin/env python
"""Q1's binned test: TCN error by fringe-extremum distance and by |v|.

Reads smi/analysis/data/q1_binned_errors.json (written by
``python -m smi.analysis.q1_binned`` from gitignored checkpoints): the TCN on
515 nm alone, on 635 + 515 nm and on all three channels, at C = 0.2, alpha =
3, noise-free and at the nominal point at 30 dB. (a, b) pooled RMSE against
distance to the nearest extremum of the 515 nm fringe (solid) and of every
channel at once (dotted). (c) pooled RMSE by |v| bin, with the zero
predictor's and the 30 dB causal bounds of section 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIGSIZE_WIDE,
    Q1_BINNED_JSON,
    SYNTHETIC_WAVELENGTHS_UM,
    panel_label,
    parse_args,
    save,
    velocity_bounds,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.core.acquisition import STEMLAB_125_14
from smi.theory import bounds
from smi.theory.forward import NOMINAL_ALPHA, NOMINAL_FEEDBACK_C, SimConfig

# Subsets as smi.analysis.q1_binned evaluates them; the pair and the triple
# take fig-bounds' K = 2 and K = 3 colors.
SUBSETS = (
    ((0.515,), '515 nm', COLORS['ch515']),
    ((0.635, 0.515), '635 + 515 nm', COLORS['accent']),
    ((0.635, 0.6748, 0.515), 'all three', COLORS['highlight']),
)
CONDITIONS = (
    ('noise_free', 'noise-free', '-'),
    ('nominal_30db', 'nominal, 30 dB', '--'),
)


def midpoints(edges: list[float]) -> np.ndarray:
    e = np.asarray(edges)
    return (e[:-1] + e[1:]) / 2


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    data = json.loads(Q1_BINNED_JSON.read_text('utf-8'))
    cells = {
        (c['condition'], tuple(c['wavelengths_um'])): c for c in data['cells'].values()
    }
    dist = midpoints(data['fringe_distance_edges_fringes'])
    speed = midpoints(data['abs_velocity_edges_um_per_ms'])

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)
    for ax, (condition, title, _), tag in zip(
        axes[:2], CONDITIONS, ('(a)', '(b)'), strict=True
    ):
        for wavelengths, label, color in SUBSETS:
            table = cells[condition, wavelengths]['fringe_distance']
            ax.plot(
                dist, table['515']['rmse'], color=color, marker='o', ms=3, label=label
            )
            if 'all' in table:
                ax.plot(dist, table['all']['rmse'], color=color, linestyle=':')
        ax.set_title(title, fontsize=9, loc='right')
        ax.set_xlabel('distance to extremum (fringes)')
        ax.set_xlim(0, 0.25)
        # From zero, so a flat curve reads as flat.
        ax.set_ylim(0, None)
        panel_label(ax, tag)
    axes[0].set_ylabel(r'pooled RMSE ($\mu$m/ms)')
    axes[0].plot([], [], color='k', linestyle=':', label='channel farthest from one')

    ax = axes[2]
    prior = bounds.motion_prior(
        SimConfig.nominal(STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, 0.0, NOMINAL_ALPHA)
    )
    for wavelengths, _, color in SUBSETS:
        for condition, _, linestyle in CONDITIONS:
            rmse = cells[condition, wavelengths]['abs_velocity']['rmse']
            ax.plot(speed, rmse, color=color, linestyle=linestyle, marker='o', ms=2.5)
        cfg = SimConfig.nominal(
            STEMLAB_125_14, wavelengths, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
        )
        causal, _ = velocity_bounds(prior, cfg)
        ax.axhline(causal, color=color, linestyle='-.', lw=0.8)
    zero = next(iter(cells.values()))['abs_velocity']['zero']
    ax.plot(speed, zero, color=COLORS['muted'], lw=1, label='zero predictor')
    ax.plot([], [], color='k', linestyle='-', label='noise-free')
    ax.plot([], [], color='k', linestyle='--', label='nominal, 30 dB')
    ax.plot([], [], color='k', linestyle='-.', lw=0.8, label='causal bound, 30 dB')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$|v|$ bin midpoint ($\mu$m/ms)')
    panel_label(ax, '(c)')
    handles, labels = axes[0].get_legend_handles_labels()
    more = ax.get_legend_handles_labels()
    fig.legend(
        handles + more[0],
        labels + more[1],
        loc='lower center',
        ncol=4,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
