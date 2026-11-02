#!/usr/bin/env python
"""Causal and offline velocity bounds against C, for one, two and three channels.

Stationary RMS bounds at the nominal rig point (SNR 30 dB, V ~ U(0.5, 1),
alpha = 3): the causal posterior CRB after it has converged, and the offline
smoothing bound at mid-record. Channel subsets add lasers in the order
635, 674.8, 515 nm.

Computed from smi.theory.bounds, deterministic: the motion prior is fit to
a fixed-seed trajectory sample inside ``bounds.motion_prior``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import COLORS, FIGSIZE_HALF, SYNTHETIC_WAVELENGTHS_UM, parse_args, save

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.core.acquisition import STEMLAB_125_14
from smi.theory import bounds
from smi.theory.forward import NOMINAL_ALPHA, SimConfig

FEEDBACK_C = np.linspace(0.0, 0.9, 19)
# The causal bound converges within about 100 samples from the prior, and the
# smoother's mid-record value is its stationary value once the record is many
# correlation times long, so a 2000-sample record suffices for both.
N_STEPS = 2000
CHANNEL_COLORS = (COLORS['ch635'], COLORS['accent'], COLORS['highlight'])


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    prior = bounds.motion_prior(
        SimConfig.nominal(STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, 0.0, NOMINAL_ALPHA)
    )

    fig, ax = plt.subplots(figsize=(FIGSIZE_HALF[0] * 1.3, FIGSIZE_HALF[1]))
    for n_channels, color in zip((1, 2, 3), CHANNEL_COLORS, strict=True):
        causal, offline = [], []
        for feedback_c in FEEDBACK_C:
            cfg = SimConfig.nominal(
                STEMLAB_125_14,
                SYNTHETIC_WAVELENGTHS_UM[:n_channels],
                float(feedback_c),
                NOMINAL_ALPHA,
            )
            causal.append(float(bounds.pcrb_filter(prior, cfg, N_STEPS)[-1]) ** 0.5)
            offline.append(
                float(bounds.bcrb_smoother(prior, cfg, N_STEPS)[N_STEPS // 2]) ** 0.5
            )
        ax.plot(FEEDBACK_C, causal, color=color, label=f'$K={n_channels}$')
        ax.plot(FEEDBACK_C, offline, color=color, linestyle='--')
    ax.plot([], [], color='k', label='causal')
    ax.plot([], [], color='k', linestyle='--', label='offline')
    ax.set_yscale('log')
    ax.set_xlabel('feedback $C$')
    ax.set_ylabel(r'RMS velocity bound ($\mu$m/ms)')
    ax.legend(loc='center right', fontsize=8, ncol=2)
    save(fig, args.output)


if __name__ == '__main__':
    main()
