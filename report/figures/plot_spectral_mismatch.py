#!/usr/bin/env python
"""Spectra of the velocity and displacement targets against the traces' fringe rate.

Shot-averaged PSDs, each scaled to unit peak, of the velocity target, the
displacement target and each channel's instantaneous fringe frequency, at the
nominal operating point (C = 0.2, alpha = 3) with (a) every noise knob at
nominal and (b) none. Simulated on CPU from a fixed seed; see
``_figcommon.mismatch_spectra`` for the estimator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    CHANNEL_COLORS,
    COLORS,
    FIGSIZE_WIDE,
    SYNTHETIC_WAVELENGTHS_UM,
    mismatch_spectra,
    panel_label,
    parse_args,
    save,
)

from smi.synthetic.forward import NOMINAL_DRIVE_BAND_HZ

# The velocity PSD falls to float round-off above a few kHz; clip there so
# the fringe-frequency floor sets the scale. The top leaves room for the panel label.
Y_MIN = 1e-7


def main() -> None:
    args = parse_args(__doc__.splitlines()[0])
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, sharey=True)
    for ax, noise_free, tag, title in zip(
        axes,
        (False, True),
        ('(a)', '(b)'),
        ('nominal noise', 'noise-free'),
        strict=True,
    ):
        spectra = mismatch_spectra(noise_free)
        f = spectra.freq_hz
        ax.axvspan(*NOMINAL_DRIVE_BAND_HZ, color=COLORS['drive'], alpha=0.12, lw=0)
        for psd, color, wavelength_um in zip(
            spectra.fringe_rate, CHANNEL_COLORS, SYNTHETIC_WAVELENGTHS_UM, strict=True
        ):
            ax.plot(
                f,
                psd,
                color=color,
                lw=1,
                label=f'fringe frequency, {wavelength_um * 1e3:.4g} nm',
            )
        ax.plot(f, spectra.velocity, color='k', lw=1.5, label='velocity target')
        ax.plot(
            f,
            spectra.displacement,
            color='k',
            lw=1.5,
            ls='--',
            label='displacement target',
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(f[0], f[-1])
        ax.set_ylim(Y_MIN, 10)
        ax.set_xlabel('frequency (Hz)')
        ax.set_title(title, fontsize=9)
        panel_label(ax, tag)
    axes[0].set_ylabel('PSD / peak')
    axes[0].text(
        30, 2 * Y_MIN, 'drive band', fontsize=8, color=COLORS['drive'], va='bottom'
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
