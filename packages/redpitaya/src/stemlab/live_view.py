"""A scope-style view of whatever channels the manager is configured to acquire."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch

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
        self._lines: dict[tuple[int, int], Line2D] = {}
        self._hists: dict[tuple[int, int], StepPatch] = {}

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
            frame += 1
            closed = self.fig is not None and not plt.fignum_exists(self.fig.number)
            if closed or (n_frames is not None and frame >= n_frames):
                return acquisition

    def update(self, acquisition: Acquisition) -> None:
        """Redraw every panel from one acquisition.

        Artists are created on the first frame and updated in place afterwards,
        so a frame costs a data swap and one idle redraw rather than rebuilding
        every axis.
        """
        keys = sorted(acquisition.data)
        first = self.fig is None
        if first:
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
            if first:
                self._lines[board, chan] = row[0].plot(t, trace, lw=0.8)[0]
                row[0].set(title=f'{hosts[board]} CH{chan}', xlabel='t (s)', ylabel='V')
            else:
                self._lines[board, chan].set_data(t, trace)
            self._rescale(row[0])
            if self.histogram:
                counts, edges = np.histogram(trace, bins=64)
                if first:
                    self._hists[board, chan] = row[1].stairs(counts, edges, fill=True)
                    row[1].set(xlabel='V', ylabel='count')
                else:
                    self._hists[board, chan].set_data(counts, edges)
                self._rescale(row[1])
        if first:
            self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    @staticmethod
    def _rescale(ax: Axes) -> None:
        ax.relim()
        ax.autoscale_view()
