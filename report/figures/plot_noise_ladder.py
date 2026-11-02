#!/usr/bin/env python
"""Noise ladder: test RMSE of the causal TCN with one noise knob on at a time.

One panel per knob, each swept alone from the noise-free base, with every x
axis ordered so the knob degrades the signal to the right. Shaded bands are
the seed spread of the noise-free base and of every knob at nominal, each
with a line at its seed mean; the dotted line is every knob at nominal with
C = 0.9. The phase-drift panel adds the same sweep and base rerun at exactly
60 epochs, since the ladder's drift rows stopped at different lengths. Reads
the runner's records in synthetic_results.json and recomputes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIGSIZE_TALL,
    NOISE_KNOBS,
    matched_length_cells,
    mean_metric,
    noise_ladder,
    noise_seeds,
    parse_args,
    save,
    synthetic_records,
)

from smi.theory.forward import rad_per_sqrt_s_to_fringes

# Per knob: (x label, log x, invert x). Inverted axes
# are those where a smaller value is the worse signal.
PANELS: dict[str, tuple[str, bool, bool]] = {
    'phase_drift_rad_per_sqrt_s': ('fringes per shot', True, False),
    'snr_db': ('SNR (dB)', False, True),
    'visibility_range': (r'lowest $V$', False, True),
    'amplitude_drift_rel': ('relative std at shot end', True, False),
    'contrast_drift_rel': ('relative std at shot end', True, False),
    'dc_offset_range': ('half-range (fringe units)', True, False),
    'pd_bandwidth_hz': ('kHz', True, True),
    'adc_bits': ('bits', False, True),
}


def knob_x(field: str, value: object) -> float:
    """A knob's value on its panel's x axis."""
    if field in ('visibility_range', 'dc_offset_range'):
        lo, hi = cast(tuple[float, float], value)
        return lo if field == 'visibility_range' else hi
    x = float(cast(float, value))
    if field == 'phase_drift_rad_per_sqrt_s':
        return rad_per_sqrt_s_to_fringes(x)
    if field == 'pd_bandwidth_hz':
        return x / 1e3
    return x


LINE = COLORS['accent']
BASE = COLORS['muted']
NOMINAL = COLORS['ch635']
EQUAL_LENGTH = COLORS['highlight']
PHASE = 'phase_drift_rad_per_sqrt_s'


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    ladder = noise_ladder(records)

    def rmse_span(sim: str) -> tuple[float, float, float, int]:
        values = [mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, sim)]
        return min(values), max(values), sum(values) / len(values), len(values)

    *base, n_base = rmse_span('ladder-base')
    *nominal, n_nominal = rmse_span('nominal')
    (c09,) = noise_seeds(records, 'nominal-c0.9')
    # The 60-epoch reruns repeat the ladder's drift sweep, so a row's
    # (field, value) is its rerun's sweep point.
    cells, e60 = matched_length_cells(records)
    (epochs,) = {r.epochs_run for r in cells.values()}
    e60_base = mean_metric(cells['ladder-base'], 'velocity_rmse')

    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE_TALL, sharey=True)
    for ax, (field, title) in zip(axes.flat, NOISE_KNOBS.items(), strict=True):
        xlabel, log_x, invert = PANELS[field]
        rows = ladder[field]
        x = [knob_x(field, row.value) for row in rows]
        y = [mean_metric(row.record, 'velocity_rmse') for row in rows]
        for (lo, hi, mean), color in ((base, BASE), (nominal, NOMINAL)):
            ax.axhspan(lo, hi, color=color, alpha=0.25, linewidth=0)
            ax.axhline(mean, color=color, lw=1.2)
        ax.axhline(mean_metric(c09, 'velocity_rmse'), color=NOMINAL, ls=':', lw=1.2)
        ax.plot(x, y, color=LINE, marker='o', markersize=4, linewidth=1.5)
        (nom,) = [
            (xi, yi) for xi, yi, row in zip(x, y, rows, strict=True) if row.nominal
        ]
        ax.plot(*nom, marker='o', markersize=9, markerfacecolor='none', color=NOMINAL)
        if field == PHASE:
            ax.axhline(e60_base, color=EQUAL_LENGTH, ls='--', lw=1.2)
            ax.plot(
                x,
                [mean_metric(e60[field, row.value], 'velocity_rmse') for row in rows],
                color=EQUAL_LENGTH,
                marker='s',
                markersize=4,
                linewidth=1.5,
            )
        if log_x:
            ax.set_xscale('log')
            ax.minorticks_off()
        ax.set_xticks(x, [f'{xi:.3g}' for xi in x], fontsize=7, rotation=45)
        if invert:
            ax.invert_xaxis()
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r'velocity RMSE ($\mu$m/ms)', fontsize=8)
    axes[0, 0].set_ylim(bottom=0)

    handles = [
        Line2D([], [], color=LINE, marker='o', markersize=4),
        Line2D(
            [],
            [],
            marker='o',
            markersize=9,
            markerfacecolor='none',
            color=NOMINAL,
            ls='',
        ),
        (Patch(color=BASE, alpha=0.25, linewidth=0), Line2D([], [], color=BASE)),
        (Patch(color=NOMINAL, alpha=0.25, linewidth=0), Line2D([], [], color=NOMINAL)),
        Line2D([], [], color=NOMINAL, ls=':', lw=1.2),
        Line2D([], [], color=EQUAL_LENGTH, marker='s', markersize=4),
        Line2D([], [], color=EQUAL_LENGTH, ls='--', lw=1.2),
    ]
    labels = [
        'one knob on',
        'its nominal value',
        f'noise-free, {n_base} seeds',
        f'all knobs nominal, {n_nominal} seeds',
        r'all nominal, $C=0.9$',
        f'phase drift, {epochs} epochs',
        f'noise-free, {epochs} epochs',
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
