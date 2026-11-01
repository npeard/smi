#!/usr/bin/env python
"""Upper bound on the sign evidence of one shot against C, for K = 1 and 3.

Markers are ``bounds.sign_information`` (8 shots, seed 0, nominal SNR 30 dB);
thin lines are its leading-order form sum_k L C^2 E[V^2] alpha^2 / (4 sigma^2
(1 + alpha^2)). The value is zero at C = 0 and at alpha = 0, which a log axis
cannot show.

Computed from smi.analysis.bounds; seeded, so deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import COLORS, FIGSIZE_HALF, SYNTHETIC_WAVELENGTHS_UM, parse_args, save

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.analysis import bounds
from smi.core.acquisition import STEMLAB_125_14
from smi.synthetic.forward import SimConfig, noise_std

FEEDBACK_C = (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9)
ALPHAS = ((3.0, 'o'), (5.0, 's'))
CHANNELS = ((1, COLORS['ch635']), (3, COLORS['accent']))
N_SHOTS = 8
SEED = 0


def leading_order(cfg: SimConfig, n_samples: int) -> float:
    """The O(C^2) sign evidence, as stated in ``bounds.sign_information``."""
    lo, hi = cfg.visibility_range
    mean_v2 = (lo * lo + lo * hi + hi * hi) / 3
    per_channel = (
        n_samples
        * cfg.feedback_c**2
        * mean_v2
        * cfg.alpha**2
        / (4 * noise_std(cfg.snr_db) ** 2 * (1 + cfg.alpha**2))
    )
    return len(cfg.wavelengths_um) * per_channel


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    some_cfg = SimConfig.nominal(STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, 0.2, 3.0)
    n_samples = bounds.ideal_shots(some_cfg, 1, SEED).velocity.shape[-1]
    c_line = np.geomspace(FEEDBACK_C[0], FEEDBACK_C[-1], 50)

    fig, ax = plt.subplots(figsize=(FIGSIZE_HALF[0] * 1.3, FIGSIZE_HALF[1]))
    for n_channels, color in CHANNELS:
        waves = SYNTHETIC_WAVELENGTHS_UM[:n_channels]
        for alpha, marker in ALPHAS:
            values = [
                bounds.sign_information(
                    SimConfig.nominal(STEMLAB_125_14, waves, c, alpha), N_SHOTS, SEED
                )
                for c in FEEDBACK_C
            ]
            ax.plot(
                FEEDBACK_C,
                values,
                marker=marker,
                linestyle='none',
                color=color,
                markerfacecolor='none' if alpha == 5.0 else color,
                label=rf'$K={n_channels}$, $\alpha={alpha:g}$',
            )
            line = [
                leading_order(
                    SimConfig.nominal(STEMLAB_125_14, waves, float(c), alpha), n_samples
                )
                for c in c_line
            ]
            ax.plot(c_line, line, color=color, linewidth=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('feedback $C$')
    ax.set_ylabel('sign evidence per shot (nats),\nupper bound')
    ax.legend(loc='upper left', fontsize=7)
    save(fig, args.output)


if __name__ == '__main__':
    main()
