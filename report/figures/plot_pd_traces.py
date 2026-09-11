#!/usr/bin/env python
"""Acquired photodiode traces beside the speaker drive voltage.

Reads one real shot out of the free-space acquisition file. This is the raw
model input: three interferometer channels at 635, 675 and 515 nm, plus the
drive voltage the ground-truth velocity is derived from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    CHANNEL_COLORS,
    COLORS,
    FIGSIZE_TALL,
    FREE_SPACE_H5,
    PD_CHANNELS,
    VOLTAGE_CHANNEL,
    panel_label,
    parse_args,
    read_shot,
    save,
    time_axis,
)

# Which shot to show. Fixed so the figure is reproducible.
SHOT_INDEX = 7

# Plot a 2048-sample window: enough to show several fringes without the trace
# turning into a solid band at slide size.
WINDOW = 2048


def main() -> None:
    args = parse_args(__doc__ or '')

    fig, axes = plt.subplots(
        4, 1, figsize=FIGSIZE_TALL, sharex=True, constrained_layout=True
    )

    drive, sample_rate = read_shot(FREE_SPACE_H5, VOLTAGE_CHANNEL, SHOT_INDEX)
    drive = drive[:WINDOW]
    t_ms = time_axis(drive.size, sample_rate)

    axes[0].plot(t_ms, drive, color=COLORS['drive'])
    axes[0].set_ylabel('drive (V)')
    panel_label(axes[0], '(a)')

    for ax, (channel, wavelength_um), color, label in zip(
        axes[1:], PD_CHANNELS, CHANNEL_COLORS, 'bcd', strict=True
    ):
        trace, _ = read_shot(FREE_SPACE_H5, channel, SHOT_INDEX)
        ax.plot(t_ms, trace[:WINDOW], color=color)
        ax.set_ylabel(f'{wavelength_um * 1e3:.0f} nm (V)')
        panel_label(ax, f'({label})')

    axes[-1].set_xlabel('time (ms)')
    axes[0].set_title(
        f'Free-space acquisition, shot {SHOT_INDEX}: drive voltage and three '
        f'photodiode channels'
    )

    save(fig, args.output)


if __name__ == '__main__':
    main()
