#!/usr/bin/env python
"""Channel subsets: what extra channels buy, noise-free and at nominal noise.

(a) Noise-free test RMSE of every channel subset, for the TCN and TCAN, on
Michelson (sign-invariant) and at C = 0.2 (signed), with the classical oracle on
Michelson and the zero predictor as a reference line. (b) The nominal point
(every knob nominal) against SNR, for the TCN: the best single channel, the best
pair and all three, so the multi-channel gain reads as the gap between lines.
Reads the runner's records in synthetic_results.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIGSIZE_WIDE,
    Q1_FAMILIES,
    Q1_PAIRS,
    Q1_S2_SIMS,
    Q1_SINGLES,
    Q1_SNR_DB,
    Q1_SUBSETS,
    SYNTHETIC_WAVELENGTHS_UM,
    mean_metric,
    noise_seeds,
    panel_label,
    parse_args,
    save,
    synthetic_records,
)

from smi.analysis.experiments import ORACLE_FAMILY, cell_key
from smi.analysis.results_artifact import RunRecord

# (family, marker, label) for panel (a).
FAMILY_MARKERS = {'tcn-44k': ('o', 'TCN'), 'tcan-44k': ('s', 'TCAN')}
SIM_STYLE = {
    's2-michelson': (COLORS['muted'], 'Michelson, sign-inv.'),
    's2-smi-a3': (COLORS['accent'], '$C=0.2$, signed'),
}


def zero_rmse(records: dict[str, RunRecord]) -> float:
    """The zero predictor's test RMSE: the alpha = 0 control scores at it.

    synthetic_values.py computes it from the frozen set and asserts the
    control matches; reading the control keeps this figure free of the HDF5.
    """
    return mean_metric(
        records[cell_key('tcn-44k-T90', SYNTHETIC_WAVELENGTHS_UM, 'smi-a0', 0)],
        'velocity_rmse',
    )


def subsets_panel(ax: plt.Axes, records: dict[str, RunRecord]) -> None:
    ys = range(len(Q1_SUBSETS))
    for sim, metric in Q1_S2_SIMS.items():
        color, sim_label = SIM_STYLE[sim]
        for k, family in enumerate(Q1_FAMILIES):
            marker, family_label = FAMILY_MARKERS[family]
            xs = [
                mean_metric(records[cell_key(family, w, sim, 0)], metric)
                for w, _ in Q1_SUBSETS
            ]
            ax.plot(
                xs,
                [y + (k - 0.5) * 0.28 for y in ys],
                marker=marker,
                markersize=5.5,
                linestyle='none',
                color=color,
                markerfacecolor=color if family == 'tcn-44k' else 'white',
                markeredgewidth=1.3,
                label=f'{family_label}, {sim_label}',
            )
    oracle = [
        mean_metric(
            records[cell_key(ORACLE_FAMILY, w, 's2-michelson', None)],
            'velocity_rmse_sign_invariant',
        )
        for w, _ in Q1_SUBSETS
    ]
    ax.plot(
        oracle, ys, marker='x', color='k', linestyle='none', label='oracle, Michelson'
    )
    ax.axvline(zero_rmse(records), color='k', ls='--', lw=1.0)
    ax.axhline(2.5, color=COLORS['muted'], lw=0.6)
    ax.axhline(5.5, color=COLORS['muted'], lw=0.6)
    ax.set_xscale('log')
    ticks = (0.05, 0.1, 0.2, 0.5)
    ax.minorticks_off()
    ax.set_xticks(ticks, [f'{t:g}' for t in ticks], fontsize=8)
    ax.set_yticks(
        list(ys),
        [f'{name} nm' if len(w) == 1 else name for w, name in Q1_SUBSETS],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel(r'noise-free test RMSE ($\mu$m/ms)')
    panel_label(ax, '(a)')


def snr_panel(ax: plt.Axes, records: dict[str, RunRecord]) -> None:
    def rmse(w: tuple[float, ...], snr: float) -> float:
        key = cell_key('tcn-44k', w, f'q1-nominal-snr_db={snr:g}', 0)
        return mean_metric(records[key], 'velocity_rmse')

    curves = (
        ('best single channel', COLORS['ch515'], Q1_SINGLES, 'o', '-'),
        ('best pair', COLORS['ch675'], Q1_PAIRS, 's', '--'),
        ('all three', COLORS['accent'], [SYNTHETIC_WAVELENGTHS_UM], 'D', '-'),
    )
    for label, color, among, marker, ls in curves:
        ax.plot(
            Q1_SNR_DB,
            [min(rmse(w, snr) for w in among) for snr in Q1_SNR_DB],
            color=color,
            marker=marker,
            linestyle=ls,
            markersize=4.5,
            label=label,
        )
    seeds = [mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, 'nominal')]
    ax.errorbar(
        [31.5],
        [sum(seeds) / len(seeds)],
        yerr=[
            [sum(seeds) / len(seeds) - min(seeds)],
            [max(seeds) - sum(seeds) / len(seeds)],
        ],
        color='k',
        capsize=3,
        lw=1,
        label=f'all three, {len(seeds)}-seed range',
    )
    ax.axhline(zero_rmse(records), color='k', ls='--', lw=1.0)
    ax.set_ylim(top=1.08 * zero_rmse(records))
    ax.set_xticks(Q1_SNR_DB, [f'{s:g}' for s in Q1_SNR_DB], fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel('SNR (dB), every other knob nominal')
    ax.set_ylabel(r'test RMSE ($\mu$m/ms)')
    ax.legend(fontsize=7, loc='center right')
    panel_label(ax, '(b)')


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    fig, (a, b) = plt.subplots(
        1, 2, figsize=FIGSIZE_WIDE, gridspec_kw={'width_ratios': [1.25, 1]}
    )
    subsets_panel(a, records)
    snr_panel(b, records)
    handles, labels = a.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color='k', ls='--', lw=1.0))
    labels.append('zero predictor')
    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=7.5,
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
