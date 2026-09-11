#!/usr/bin/env python
"""The distribution shift between the free-space and multimode-fiber datasets.

The two acquisitions drive the same speaker with the same waveform family, so
any difference in the photodiode statistics is the fiber's contribution --
mode scrambling and drift on top of the interference term. Left: one shot of
the same channel from each file. Right: the amplitude spectrum of that channel
averaged over a fixed sample of shots, which is the summary a model actually
sees the difference in.

Reads a bounded sample of shots (never a whole channel: one channel is
0.66 GB).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
from _figcommon import (
    COLORS,
    FIGSIZE_WIDE,
    FREE_SPACE_H5,
    MMFIBER_H5,
    panel_label,
    parse_args,
    read_shot,
    save,
    time_axis,
)

CHANNEL = 'RP1_CH2'  # 635 nm photodiode
SHOT_INDEX = 7
WINDOW = 1500
N_SHOTS = 64  # shots averaged for the spectrum; keeps the read ~4 MB per file


def mean_spectrum(
    h5_path: Path, channel: str, n_shots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum of ``channel`` averaged over the first ``n_shots``."""
    with h5py.File(h5_path, 'r') as f:
        block = np.asarray(f[channel][:n_shots, :], dtype=np.float64)
        sample_rate = float(f.attrs['sample_rate'])
    block = block - block.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(block, axis=1)).mean(axis=0)
    freq = np.fft.rfftfreq(block.shape[1], d=1.0 / sample_rate)
    return freq, spectrum


def main() -> None:
    args = parse_args(__doc__ or '')

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, constrained_layout=True)

    sources = (
        ('free space', FREE_SPACE_H5, COLORS['accent']),
        ('mm fiber', MMFIBER_H5, COLORS['ch635']),
    )

    ax = axes[0]
    for label, path, color in sources:
        trace, sample_rate = read_shot(path, CHANNEL, SHOT_INDEX)
        trace = trace[:WINDOW]
        ax.plot(
            time_axis(trace.size, sample_rate),
            trace - trace.mean(),
            color=color,
            label=label,
            alpha=0.85,
        )
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('635 nm signal (V, DC removed)')
    ax.set_title(f'Same channel, shot {SHOT_INDEX}')
    ax.legend(loc='upper right')
    panel_label(ax, '(a)')

    ax = axes[1]
    for label, path, color in sources:
        freq, spectrum = mean_spectrum(path, CHANNEL, N_SHOTS)
        keep = (freq > 0) & (freq <= 20e3)
        ax.loglog(freq[keep], spectrum[keep], color=color, label=label, alpha=0.85)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('mean |FFT| (arb.)')
    ax.set_title(f'Mean amplitude spectrum, N = {N_SHOTS} shots')
    ax.legend(loc='lower left')
    panel_label(ax, '(b)')

    save(fig, args.output)


if __name__ == '__main__':
    main()
