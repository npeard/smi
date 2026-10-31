#!/usr/bin/env python
"""Noise ladder: test RMSE of the causal TCN with one noise knob on at a time.

One panel per knob, each swept alone from the noise-free base, with every x
axis ordered so the knob degrades the signal to the right. Shaded bands are
the seed spread of the noise-free base and of every knob at nominal; the
dotted line is every knob at nominal with C = 0.9. Reads the runner's records
in synthetic_results.json and recomputes nothing.
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
    NOMINAL_PHASE_DRIFT_FRINGES,
    mean_metric,
    noise_ladder,
    noise_seeds,
    parse_args,
    save,
    synthetic_records,
)

from smi.synthetic.forward import NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S

# (field, panel title, x label, log x, invert x). Inverted axes are those
# where a smaller value is the worse signal.
PANELS: tuple[tuple[str, str, str, bool, bool], ...] = (
    ('phase_drift_rad_per_sqrt_s', 'phase drift', 'fringes per shot', True, False),
    ('snr_db', 'white noise', 'SNR (dB)', False, True),
    ('visibility_range', 'visibility', r'lowest $V$', False, True),
    ('amplitude_drift_rel', 'amplitude drift', 'relative', True, False),
    ('contrast_drift_rel', 'contrast drift', 'relative', True, False),
    ('dc_offset_range', 'DC offset', 'half-range (fringe units)', True, False),
    ('pd_bandwidth_hz', 'PD bandwidth', 'kHz', True, True),
    ('adc_bits', 'ADC', 'bits', False, True),
)


def knob_x(field: str, value: object) -> float:
    """A knob's value on its panel's x axis."""
    if field in ('visibility_range', 'dc_offset_range'):
        lo, hi = cast(tuple[float, float], value)
        return lo if field == 'visibility_range' else hi
    x = float(cast(float, value))
    if field == 'phase_drift_rad_per_sqrt_s':
        return x / NOMINAL_PHASE_DRIFT_RAD_PER_SQRT_S * NOMINAL_PHASE_DRIFT_FRINGES
    if field == 'pd_bandwidth_hz':
        return x / 1e3
    return x


LINE = COLORS['accent']
BASE = COLORS['muted']
NOMINAL = COLORS['ch635']


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    records = synthetic_records()
    ladder = noise_ladder(records)

    def rmse_span(sim: str) -> tuple[float, float]:
        values = [mean_metric(r, 'velocity_rmse') for r in noise_seeds(records, sim)]
        return min(values), max(values)

    base = rmse_span('ladder-base')
    nominal = rmse_span('nominal')
    (c09,) = noise_seeds(records, 'nominal-c0.9')

    fig, axes = plt.subplots(2, 4, figsize=FIGSIZE_TALL, sharey=True)
    for ax, (field, title, xlabel, log_x, invert) in zip(
        axes.flat, PANELS, strict=True
    ):
        rows = ladder[field]
        x = [knob_x(field, row.value) for row in rows]
        y = [mean_metric(row.record, 'velocity_rmse') for row in rows]
        ax.axhspan(*base, color=BASE, alpha=0.35, linewidth=0)
        ax.axhspan(*nominal, color=NOMINAL, alpha=0.35, linewidth=0)
        ax.axhline(mean_metric(c09, 'velocity_rmse'), color=NOMINAL, ls=':', lw=1.2)
        ax.plot(x, y, color=LINE, marker='o', markersize=4, linewidth=1.5)
        (nom,) = [
            (xi, yi) for xi, yi, row in zip(x, y, rows, strict=True) if row.nominal
        ]
        ax.plot(*nom, marker='o', markersize=9, markerfacecolor='none', color=NOMINAL)
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
        Patch(color=BASE, alpha=0.35, linewidth=0),
        Patch(color=NOMINAL, alpha=0.35, linewidth=0),
        Line2D([], [], color=NOMINAL, ls=':', lw=1.2),
    ]
    labels = [
        'one knob on',
        'its nominal value',
        'noise-free, 3 seeds',
        'all knobs nominal, 3 seeds',
        r'all nominal, $C=0.9$',
    ]
    fig.legend(
        handles=handles,
        labels=labels,
        loc='lower center',
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
