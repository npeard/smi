"""Shared layout of the appendix example-shot figures.

Reads ``report/results/example_shots.{npz,json}`` (``scripts/example_shots.py``);
never runs a model. One column per arm, rows sharing time: the photodiode
inputs, velocity, displacement, then a full-rate zoom of the inputs and of the
velocity where the fringes are densest. The fiber-sweep figure instead puts
every shot type in one figure, a zoom of the inputs over the whole-shot
velocity for each, one column per sweep arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _figcommon import CHANNEL_COLORS, RESULTS_DIR, pending_panel, save

NPZ = RESULTS_DIR / 'example_shots.npz'
INDEX = RESULTS_DIR / 'example_shots.json'

# One color per model across every example-shot figure; the target is black.
MODEL_STYLE = {
    'tcan-44k': ('TCAN (causal)', 'C0'),
    'unettransformer-offline-44k': ('UNetTransformer (offline)', 'C4'),
    'head': ('encoder head (whole-shot regressor)', 'C5'),
    'flow': ('flow posterior mean, 90% band', 'C8'),
}
SHOT_KINDS = ('low', 'mid', 'high')
ARM_TITLE = {
    'nominal': 'free space',
    'fiber': 'multimode fiber',
    'real': 'real free space',
    'modes-1': 'fiber, 1 mode, static',
    'modes-8': 'fiber, 8 modes, static',
}


def index() -> dict:
    return json.loads(INDEX.read_text())


def _column(
    axes: np.ndarray,
    arrays: np.lib.npyio.NpzFile,
    tag: str,
    entry: dict,
    wavelengths_um: list[float],
    sample_rate: float,
    decimate: int,
) -> None:
    ms = 1e3 / sample_rate
    sig = arrays[f'{tag}/signals']
    t = np.arange(sig.shape[-1]) * decimate * ms
    start = entry['zoom_start']
    zsig = arrays[f'{tag}/signals_zoom']
    tz = (start + np.arange(zsig.shape[-1])) * ms
    ax_in, ax_v, ax_d, ax_zin, ax_zv = axes
    for i, w in enumerate(wavelengths_um):
        color = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
        label = f'{w * 1e3:.0f} nm'
        ax_in.plot(t, sig[i], color=color, lw=0.4, label=label)
        ax_zin.plot(tz, zsig[i], color=color, lw=0.7, label=label)
    for ax, key, tt, suffix in (
        (ax_v, 'velocity', t, ''),
        (ax_d, 'displacement', t, ''),
        (ax_zv, 'velocity', tz, '_zoom'),
    ):
        ax.plot(
            tt,
            arrays[f'{tag}/{key}{suffix}'],
            color='k',
            lw=1.1,
            label='target',
            zorder=3,
        )
        if key == 'velocity' and entry['band']:
            ax.fill_between(
                tt,
                arrays[f'{tag}/flow/velocity_lo{suffix}'],
                arrays[f'{tag}/flow/velocity_hi{suffix}'],
                color=MODEL_STYLE['flow'][1],
                alpha=0.2,
                lw=0,
            )
        for model, (label, color) in MODEL_STYLE.items():
            if model in entry['models']:
                ax.plot(
                    tt,
                    arrays[f'{tag}/{model}/{key}{suffix}'],
                    color=color,
                    lw=0.7,
                    label=label,
                )
    for ax in (ax_in, ax_v, ax_d):
        ax.axvspan(tz[0], tz[-1], color='0.85', zorder=0)
    ax_in.set_ylabel('photodiode (V)')
    ax_v.set_ylabel(r'velocity ($\mu$m/ms)')
    ax_d.set_ylabel(r'displacement ($\mu$m)')
    ax_zin.set_ylabel('photodiode (V)')
    ax_zv.set_ylabel(r'velocity ($\mu$m/ms)')
    ax_d.set_xlabel('time (ms)')
    ax_zv.set_xlabel('time (ms), zoom at full rate')
    for ax in axes:
        ax.margins(x=0)


def _layout(n_cols: int) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(7.0, 8.4), layout='constrained')
    sub = fig.subfigures(2, 1, height_ratios=(3, 2))
    top = sub[0].subplots(3, n_cols, sharex='col', squeeze=False)
    bottom = sub[1].subplots(2, n_cols, sharex='col', squeeze=False)
    return fig, np.concatenate([top, bottom])


def _legend(fig: plt.Figure, axes: np.ndarray) -> None:
    handles, labels = [], []
    for ax in (axes[0, 0], axes[1, 0]):
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=10)


def plot_synthetic(kind: str, output: Path) -> None:
    """Free space and fiber columns for shot type ``kind`` (low, mid, high)."""
    idx = index()
    syn = idx['synthetic']
    arrays = np.load(NPZ)
    arms = list(syn['arms'])
    fig, axes = _layout(len(arms))
    for col, arm in enumerate(arms):
        info = syn['arms'][arm]
        entry = info['shots'][kind]
        _column(
            axes[:, col],
            arrays,
            f'{arm}/{kind}',
            entry,
            info['wavelengths_um'],
            syn['sample_rate_hz'],
            idx['decimate'],
        )
        axes[0, col].set_title(f'{ARM_TITLE[arm]}, test shot {entry["shot"]}')
    for row in axes:
        row[1].sharey(row[0])
        row[1].set_ylabel('')
    _legend(fig, axes)
    save(fig, output)


def plot_real(output: Path) -> None:
    """The three real shots' columns, or a placeholder while the data is blinded."""
    idx = index()
    real = idx['real']
    if real['status'] != 'present':
        fig, ax = plt.subplots(figsize=(7.0, 1.6))
        pending_panel(
            ax, f'real data blinded: builds when {real["dataset"]} is present'
        )
        save(fig, output)
        return
    arrays = np.load(NPZ)
    kinds = list(real['shots'])
    fig, axes = _layout(len(kinds))
    for col, kind in enumerate(kinds):
        entry = real['shots'][kind]
        _column(
            axes[:, col],
            arrays,
            f'real/{kind}',
            entry,
            real['wavelengths_um'],
            real['sample_rate_hz'],
            idx['decimate'],
        )
        axes[0, col].set_title(f'{kind} excursion, shot {entry["shot"]}')
    _legend(fig, axes)
    save(fig, output)


def plot_sweep(output: Path) -> None:
    """Each shot type at the low- and high-fading fiber-sweep arms."""
    idx = index()
    syn = idx['synthetic']
    arrays = np.load(NPZ)
    arms = list(syn['sweep_arms'])
    kinds = list(SHOT_KINDS)
    ms = 1e3 / syn['sample_rate_hz']
    fig, axes = plt.subplots(
        2 * len(kinds), len(arms), figsize=(7.0, 8.4), layout='constrained'
    )
    for col, arm in enumerate(arms):
        info = syn['sweep_arms'][arm]
        for row, kind in enumerate(kinds):
            entry = info['shots'][kind]
            tag = f'{arm}/{kind}'
            ax_in, ax_v = axes[2 * row, col], axes[2 * row + 1, col]
            zsig = arrays[f'{tag}/signals_zoom']
            tz = (entry['zoom_start'] + np.arange(zsig.shape[-1])) * ms
            for i, w in enumerate(info['wavelengths_um']):
                ax_in.plot(
                    tz,
                    zsig[i],
                    color=CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                    lw=0.7,
                    label=f'{w * 1e3:.0f} nm',
                )
            v = arrays[f'{tag}/velocity']
            t = np.arange(v.size) * idx['decimate'] * ms
            ax_v.plot(t, v, color='k', lw=1.1, label='target', zorder=3)
            label, color = MODEL_STYLE['tcan-44k']
            ax_v.plot(
                t, arrays[f'{tag}/tcan-44k/velocity'], color=color, lw=0.7, label=label
            )
            ax_v.axvspan(tz[0], tz[-1], color='0.85', zorder=0)
            ax_in.set_title(f'{kind} excursion, {ARM_TITLE[arm]}', fontsize=8)
            if col == 0:
                ax_in.set_ylabel('photodiode (V)')
                ax_v.set_ylabel(r'velocity ($\mu$m/ms)')
            else:
                ax_v.sharey(axes[2 * row + 1, 0])
                ax_in.sharey(axes[2 * row, 0])
            for ax in (ax_in, ax_v):
                ax.margins(x=0)
    axes[-1, 0].set_xlabel('time (ms)')
    axes[-1, 1].set_xlabel('time (ms)')
    _legend(fig, axes)
    save(fig, output)
