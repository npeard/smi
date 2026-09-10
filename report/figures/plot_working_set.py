#!/usr/bin/env python
"""Resident working set of one acquisition file against available GPU memory.

The on-disk file is small because it is gzip-compressed; what matters for a
GPU-resident data pipeline is the decompressed float32 footprint. This figure
computes that footprint from the real dataset shapes and dtypes, and compares
it against the two cards on the training rig, so the "can the whole dataset
live on the GPU?" question is answered with numbers rather than intuition.
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
    PD_CHANNELS,
    VOLTAGE_CHANNEL,
    parse_args,
    save,
)

GIB = 1024**3

# Cards on the training rig, nominal VRAM in GiB.
GPUS: tuple[tuple[str, float], ...] = (('Quadro RTX 4000', 8.0), ('RTX 3090 Ti', 24.0))


def channel_bytes(h5_path: Path, channel: str) -> float:
    """Decompressed size of one channel, from its shape and dtype."""
    with h5py.File(h5_path, 'r') as f:
        dset = f[channel]
        return float(np.prod(dset.shape)) * dset.dtype.itemsize


def main() -> None:
    args = parse_args(__doc__ or '')

    one_channel = channel_bytes(FREE_SPACE_H5, VOLTAGE_CHANNEL) / GIB
    n_pd = len(PD_CHANNELS)
    on_disk = FREE_SPACE_H5.stat().st_size / GIB

    # Cumulative build-up of what a fully resident pipeline must hold. Targets
    # (velocity, displacement) are float32 arrays of the same shape as one
    # channel, derived from the drive voltage.
    stages = [
        ('gzip file\non disk', on_disk, COLORS['muted']),
        ('3 PD channels\n(model input)', n_pd * one_channel, COLORS['accent']),
        ('+ velocity\ntarget', (n_pd + 1) * one_channel, COLORS['ch675']),
        ('+ displacement\ntarget', (n_pd + 2) * one_channel, COLORS['highlight']),
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, constrained_layout=True)

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    positions = np.arange(len(stages))

    ax.bar(positions, values, color=colors, width=0.62)
    for pos, value in zip(positions, values, strict=True):
        ax.text(
            pos, value + 0.25, f'{value:.2f} GiB', ha='center', va='bottom', fontsize=8
        )

    for name, capacity in GPUS:
        ax.axhline(capacity, color=COLORS['drive'], linestyle='--', linewidth=0.9)
        ax.text(
            len(stages) - 0.45,
            capacity + 0.2,
            f'{name} ({capacity:.0f} GiB)',
            ha='right',
            va='bottom',
            fontsize=8,
            color=COLORS['drive'],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel('resident size (GiB)')
    ax.set_ylim(0, max(GPUS[-1][1], *values) * 1.12)
    ax.set_title(
        f'{FREE_SPACE_H5.name}: compressed on disk vs float32 in memory '
        f'(10000 shots x 16384 samples)'
    )

    save(fig, args.output)


if __name__ == '__main__':
    main()
