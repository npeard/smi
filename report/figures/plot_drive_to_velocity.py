#!/usr/bin/env python
"""How the speaker drive voltage becomes the supervision target.

The coil driver is a damped resonator: its complex transfer function maps
drive voltage to mirror displacement, and multiplying by j*2*pi*f gives
velocity. Panel (a) is that transfer function; panels (b) and (c) push a real
acquired drive waveform through it, which is exactly what the dataset does to
produce the (velocity, displacement) targets the network is trained against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _figcommon import (
    COLORS,
    FIGSIZE_TALL,
    FREE_SPACE_H5,
    VOLTAGE_CHANNEL,
    panel_label,
    parse_args,
    read_shot,
    save,
    time_axis,
)

from smi.synthetic.coil_driver import CoilDriver

SHOT_INDEX = 7
WINDOW = 4096

# The drive waveform is band-limited to 1 kHz (see the `end_freq` file
# attribute), so cut the transfer above that when converting.
MAX_FREQ = 1000.0


def main() -> None:
    args = parse_args(__doc__ or '')
    driver = CoilDriver()

    fig, axes = plt.subplots(
        3, 1, figsize=FIGSIZE_TALL, sharex=False, constrained_layout=True
    )

    # (a) magnitude and phase of the displacement transfer function
    freq = np.linspace(1.0, 1500.0, 2000)
    transfer = driver.get_transfer_function(freq)
    ax = axes[0]
    ax.plot(freq, np.abs(transfer), color=COLORS['accent'])
    ax.set_ylabel('gain (um/V)')
    ax.set_xlabel('frequency (Hz)')
    ax.axvline(driver.params.f0, color=COLORS['muted'], linestyle='--', linewidth=0.9)
    ax.annotate(
        f'f0 = {driver.params.f0:.0f} Hz, Q = {driver.params.Q:.1f}',
        xy=(driver.params.f0, np.abs(transfer).max()),
        xytext=(6, -4),
        textcoords='offset points',
        fontsize=8,
        color=COLORS['muted'],
    )
    ax.set_title('Coil-driver response and the derived training targets')
    panel_label(ax, '(a)')

    # (b), (c): a real acquired drive waveform mapped to displacement/velocity
    drive, sample_rate = read_shot(FREE_SPACE_H5, VOLTAGE_CHANNEL, SHOT_INDEX)
    displacement, _, _ = driver.get_displacement(drive, sample_rate, max_freq=MAX_FREQ)
    velocity, _, _ = driver.get_velocity(drive, sample_rate, max_freq=MAX_FREQ)

    t_ms = time_axis(WINDOW, sample_rate)

    ax = axes[1]
    ax.plot(t_ms, drive[:WINDOW], color=COLORS['drive'], label='drive (V)')
    ax.set_ylabel('drive (V)')
    ax.set_xlabel('time (ms)')
    twin = ax.twinx()
    twin.plot(
        t_ms, displacement[:WINDOW], color=COLORS['highlight'], label='displacement'
    )
    twin.set_ylabel('displacement (um)', color=COLORS['highlight'])
    twin.tick_params(axis='y', labelcolor=COLORS['highlight'])
    twin.grid(visible=False)
    panel_label(ax, '(b)')

    ax = axes[2]
    ax.plot(t_ms, velocity[:WINDOW], color=COLORS['accent'])
    ax.set_ylabel('velocity (um/s)')
    ax.set_xlabel('time (ms)')
    panel_label(ax, '(c)')

    save(fig, args.output)


if __name__ == '__main__':
    main()
