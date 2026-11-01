"""A scope-style view of whatever channels the manager is configured to acquire."""

import matplotlib.pyplot as plt
import numpy as np

from .manager import Acquisition, RedPitayaManager


class LiveView:
    """Repeatedly arm, acquire and redraw the raw channel traces.

    Args:
        manager: Already configured for acquisition (and generation, if any).
        histogram: Also show a histogram of each channel's samples.
        timeout_s: Per-frame trigger/fill timeout passed to ``acquire``.
    """

    def __init__(self, manager: RedPitayaManager, histogram: bool, timeout_s: float):
        self.manager = manager
        self.histogram = histogram
        self.timeout_s = timeout_s
        self.fig = None
        self._axes: dict[tuple[int, int], np.ndarray] = {}

    def run(self, n_frames: int | None) -> Acquisition:
        """Show ``n_frames`` acquisitions (forever if None, until the window closes).

        Returns:
            The last acquisition shown.
        """
        frame = 0
        while True:
            self.manager.start_acquisition()
            acquisition = self.manager.acquire(self.timeout_s)
            self.update(acquisition)
            plt.pause(0.001)
            frame += 1
            closed = self.fig is not None and not plt.fignum_exists(self.fig.number)
            if closed or (n_frames is not None and frame >= n_frames):
                return acquisition

    def update(self, acquisition: Acquisition) -> None:
        """Redraw every panel from one acquisition."""
        keys = sorted(acquisition.data)
        if self.fig is None:
            ncols = 2 if self.histogram else 1
            self.fig, axes = plt.subplots(
                len(keys), ncols, squeeze=False, figsize=(10, 2.5 * len(keys))
            )
            self._axes = dict(zip(keys, axes, strict=True))
        hosts = acquisition.metadata.board_ids
        for (board, chan), trace in acquisition.data.items():
            row = self._axes[board, chan]
            rate = acquisition.metadata.sample_rate_hz[board]
            t = np.arange(trace.size) / rate
            row[0].cla()
            row[0].plot(t, trace, lw=0.8)
            row[0].set(title=f'{hosts[board]} CH{chan}', xlabel='t (s)', ylabel='V')
            if self.histogram:
                row[1].cla()
                row[1].hist(trace, bins=64)
                row[1].set(xlabel='V', ylabel='count')
        self.fig.tight_layout()
