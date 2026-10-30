#!/usr/bin/env python
"""Per-sample displacement information against fringe phase, for three C.

Plots the dimensionless factor F'(theta)^2, which the per-sample information
multiplies by (V 4 pi / (lambda sigma))^2; its zeros are the fringe extrema,
where a channel is blind. Dashed lines are the phase averages
(``bounds.phase_factor``), which is what the averaged bounds use.

Computed from smi.analysis.bounds, deterministic with no randomness.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import COLORS, FIGSIZE_HALF, parse_args, save

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.analysis import bounds

ALPHA = 3.0
CURVES = ((0.0, COLORS['muted']), (0.2, COLORS['accent']), (0.7, COLORS['highlight']))


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    theta = torch.linspace(0, 2 * math.pi, 1001, dtype=torch.float64)

    fig, ax = plt.subplots(figsize=(FIGSIZE_HALF[0] * 1.3, FIGSIZE_HALF[1]))
    for feedback_c, color in CURVES:
        slope_sq = bounds.fringe_slope(theta, feedback_c, ALPHA) ** 2
        ax.plot(theta.numpy(), slope_sq.numpy(), color=color, label=f'$C={feedback_c}$')
        ax.axhline(
            bounds.phase_factor(feedback_c, ALPHA),
            color=color,
            linestyle='--',
            linewidth=0.8,
        )
    ax.set_xticks(
        [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi],
        ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'],
    )
    ax.set_xlim(0, 2 * math.pi)
    ax.set_xlabel(r'fringe phase $\theta$')
    ax.set_ylabel(r"$F'(\theta)^2$")
    ax.legend(loc='upper right', fontsize=8, title=rf'$\alpha={ALPHA:g}$')
    save(fig, args.output)


if __name__ == '__main__':
    main()
