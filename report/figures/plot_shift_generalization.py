#!/usr/bin/env python
"""Nominal-trained checkpoints scored on test sets shifted in one knob (Q2.3).

(a) feedback C; (b) wavelength shift, common scale and per-channel
independent, against the common-scale prediction; (c) shared-phase offset,
on two test draws (the second nudged right so the markers do not overlap),
against the nominal score of each draw, with the spread of the random
shared-phase configurations shaded. The pi/3 step sits on the 1 rad one and is
left to the text. Error bars are the
per-shot SE of ``_figcommon.shift_scores``. Reads shift_results.json.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIGSIZE_WIDE,
    NOMINAL_ALPHA,
    SHIFT_DEFAULT_TEST_SEED,
    SHIFT_FAMILIES,
    STEMLAB_125_14,
    SYNTHETIC_WAVELENGTHS_UM,
    parse_args,
    save,
    shift_scores,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from smi.theory import bounds
from smi.theory.forward import SimConfig

LABELS = {'tcan-44k-T90': 'TCAN', 'unettransformer-offline-44k-T90': 'UNet-Tr.'}
FAMILY_COLORS = {
    'tcan-44k-T90': COLORS['accent'],
    'unettransformer-offline-44k-T90': COLORS['highlight'],
}
SECOND_TEST_SEED = 2000
TRAINED_C = 0.2


def velocity_rms() -> float:
    """RMS velocity of the motion prior's fitted trajectories (um/ms)."""
    cfg = SimConfig.nominal(
        STEMLAB_125_14, SYNTHETIC_WAVELENGTHS_UM, TRAINED_C, NOMINAL_ALPHA
    )
    return math.sqrt(bounds.motion_prior(cfg).empirical_velocity_var)


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    scores = shift_scores()
    v_rms = velocity_rms()
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE_WIDE)

    def series(
        family: str, knob: str, seed: int = SHIFT_DEFAULT_TEST_SEED
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keys = sorted(k for k in scores if k[:2] == (family, knob) and k[3] == seed)
        return (
            np.array([k[2] for k in keys]),
            np.array([scores[k].rmse for k in keys]),
            np.array([scores[k].se for k in keys]),
        )

    for family in SHIFT_FAMILIES:
        color, label = FAMILY_COLORS[family], LABELS[family]
        x, y, se = series(family, 'feedback_c')
        axes[0].errorbar(x, y, se, color=color, marker='o', ms=3, label=label)

        nominal = scores[(family, 'feedback_c', TRAINED_C, SHIFT_DEFAULT_TEST_SEED)]
        x, y, se = series(family, 'wavelength_scale')
        eps = x - 1
        axes[1].errorbar(eps, y, se, color=color, marker='o', ms=3, ls='none')
        x, y, se = series(family, 'wavelength_independent')
        axes[1].errorbar(x, y, se, color=color, marker='s', ms=3, mfc='none', ls='none')
        grid = np.linspace(-0.01, 0.01, 201)
        axes[1].plot(
            grid,
            np.sqrt(nominal.rmse**2 + (grid / (1 + grid)) ** 2 * v_rms**2),
            color=color,
            lw=0.8,
            ls='--',
        )

        x, y, _ = series(family, 'phase_random')
        axes[2].axhspan(
            y.mean() - y.std(ddof=1),
            y.mean() + y.std(ddof=1),
            color=color,
            alpha=0.15,
            lw=0,
        )
        for seed, marker, dx in (
            (SHIFT_DEFAULT_TEST_SEED, 'o', 0.0),
            (SECOND_TEST_SEED, 's', 0.12),
        ):
            x, y, se = series(family, 'phase_offset', seed)
            keep = ~np.isclose(x, math.pi / 3, atol=1e-3)
            x, y, se = x[keep], y[keep], se[keep]
            axes[2].errorbar(
                x + dx,
                y,
                se,
                color=color,
                marker=marker,
                ms=3,
                ls='none',
                mfc=None if seed == SHIFT_DEFAULT_TEST_SEED else 'none',
            )
            base = scores[(family, 'wavelength_independent', 0.0, seed)].rmse
            axes[2].axhline(
                base,
                color=color,
                lw=0.8,
                ls='-' if seed == SHIFT_DEFAULT_TEST_SEED else ':',
            )

    axes[0].axvline(TRAINED_C, color=COLORS['muted'], lw=0.8, ls=':')
    axes[0].set_yscale('log')
    axes[0].set_yticks([0.03, 0.1, 0.3], ['0.03', '0.1', '0.3'])
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel('test feedback $C$')
    axes[0].set_ylabel(r'velocity RMSE ($\mu$m/ms)')
    axes[0].legend(fontsize=7, loc='upper right')
    axes[1].set_xlabel(r'relative wavelength shift $\epsilon$')
    axes[1].plot([], [], 'ko', ms=3, ls='none', label='common')
    axes[1].plot([], [], 'ks', ms=3, mfc='none', ls='none', label='independent')
    axes[1].plot([], [], 'k--', lw=0.8, label='common, predicted')
    axes[1].legend(fontsize=6, loc='center right')
    axes[2].set_xlabel('phase-offset step (rad)')
    axes[2].plot([], [], 'ko', ms=3, ls='none', label='test seed 1000')
    axes[2].plot([], [], 'ks', ms=3, mfc='none', ls='none', label='test seed 2000')
    axes[2].plot([], [], 'k-', lw=0.8, label='nominal, per seed')
    axes[2].fill_between(
        [], [], color='k', alpha=0.15, lw=0, label='random configs, +/-1 sd'
    )
    axes[2].legend(fontsize=6, loc='upper right')
    for ax, tag in zip(axes, 'abc', strict=True):
        ax.set_title(f'({tag})', loc='left', fontsize=9)
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
