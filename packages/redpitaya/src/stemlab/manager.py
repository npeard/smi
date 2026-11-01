"""Drive and read one or more (optionally daisy-chained) Red Pitaya boards.

The manager knows nothing about what the signals mean: generation takes an
arbitrary float array, acquisition returns raw per-channel arrays in volts.
"""

import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import numpy as np

from .board import BoardSpec

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 0.1
# The board needs a moment after ACQ:START before it accepts a trigger source;
# arming sooner can trigger on stale buffer contents.
_ARM_SETTLE_S = 0.2


class Transport(Protocol):
    """The subset of ``stemlab.scpi.SCPI`` the manager uses."""

    def close(self) -> None: ...

    def tx_txt(self, msg: str) -> None: ...

    def txrx_txt(self, msg: str) -> str | None: ...

    def sour_set(
        self,
        chan: int,
        func: str,
        volt: float,
        freq: float,
        offset: float,
        *,
        data: np.ndarray | None,
        trig: str,
    ) -> None: ...

    def acq_set(self, dec: int, trig_lvl: float, trig_delay: int) -> None: ...

    def acq_data(
        self, chan: int, *, num_samples: int | None, old: bool, convert: bool
    ) -> list | bytes | bool | str | None: ...


@dataclass(frozen=True)
class AcquisitionTrigger:
    """Acquisition trigger settings.

    Attributes:
        source: SCPI trigger source, e.g. ``NOW``, ``CH1_PE``, ``EXT_NE``.
        delay_samples: Samples recorded after the trigger before the buffer
            is reported full.
        level_v: Trigger level for channel sources.
    """

    source: str
    delay_samples: int
    level_v: float


@dataclass(frozen=True)
class _AcquisitionSettings:
    decimation: int
    trigger: AcquisitionTrigger
    n_samples: int
    channels: tuple[int, ...]


@dataclass(frozen=True)
class AcquisitionMetadata:
    """What produced an ``Acquisition``.

    Attributes:
        timestamp: UTC time the buffers were read.
        board_ids: Host of each board, indexed by board number.
        decimation: Decimation per acquired board.
        sample_rate_hz: Sample rate per acquired board.
        trigger: Trigger settings per acquired board.
    """

    timestamp: datetime
    board_ids: tuple[str, ...]
    decimation: dict[int, int]
    sample_rate_hz: dict[int, float]
    trigger: dict[int, AcquisitionTrigger]


@dataclass(frozen=True)
class Acquisition:
    """Raw acquired buffers.

    Attributes:
        data: Volts, keyed by ``(board, channel)``; each has shape ``(n_samples,)``.
        metadata: Settings and time of the acquisition.
    """

    data: dict[tuple[int, int], np.ndarray]
    metadata: AcquisitionMetadata


class RedPitayaManager:
    """Generation and acquisition across boards, addressed by index into ``hosts``.

    Args:
        hosts: Board host names or IPs; index 0 is the primary until
            ``configure_daisy_chain`` says otherwise.
        transport_factory: Opens a connection to one host, normally
            ``stemlab.SCPI``; tests pass ``stemlab.testing.FakeTransport``.
        board: Constants of the board model.
    """

    def __init__(
        self,
        hosts: Sequence[str],
        transport_factory: Callable[[str], Transport],
        board: BoardSpec,
    ):
        self.hosts = tuple(hosts)
        self.board = board
        self.devices: list[Transport] = [transport_factory(h) for h in self.hosts]
        self.primary = 0
        self._acquisition: dict[int, _AcquisitionSettings] = {}

    def close(self) -> None:
        """Close every connection."""
        for device in self.devices:
            with contextlib.suppress(OSError):
                device.close()

    def reset(self) -> None:
        """Reset generation and acquisition on every board."""
        for device in self.devices:
            device.tx_txt('GEN:RST')
            device.tx_txt('ACQ:RST')
        self._acquisition.clear()

    def configure_daisy_chain(self, primary: int) -> None:
        """Share clock and trigger from ``primary`` to every other board.

        Secondaries should then be configured to acquire on ``EXT_NE``, the
        trigger the primary forwards.
        """
        self._device(primary)
        self.primary = primary
        for device in self.devices:
            device.tx_txt('DAISY:SYNC:TRig ON')
            device.tx_txt('DAISY:SYNC:CLK ON')
            device.tx_txt('DAISY:TRIG_O:SOUR ADC')
            device.tx_txt('DIG:PIN LED5,1')

    def configure_generation(
        self,
        device: int,
        channel: int,
        samples: np.ndarray,
        sample_rate_hz: float,
        amplitude_v: float,
        offset_v: float,
        trigger: str,
    ) -> None:
        """Load an arbitrary waveform onto one output channel.

        Args:
            device: Board index.
            channel: Output channel, 1 or 2.
            samples: One period of the waveform in [-1, 1], at most
                ``board.buffer_size`` long; scaled by ``amplitude_v``.
            sample_rate_hz: Rate at which ``samples`` are played out.
            amplitude_v: Output volts for a sample of 1.
            offset_v: DC offset in volts.
            trigger: Generation trigger source (``INT``, ``EXT_PE``,
                ``EXT_NE``, ``GATED``).
        """
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1 or not 0 < samples.size <= self.board.buffer_size:
            raise ValueError(
                f'samples must be 1-D with 1..{self.board.buffer_size} entries'
            )
        if np.max(np.abs(samples)) > 1.0:
            raise ValueError('samples must lie in [-1, 1]; scale with amplitude_v')
        transport = self._device(device)
        transport.tx_txt('GEN:RST')
        # The board stretches the array over its buffer and repeats it at FREQ,
        # so one array period lasts len(samples) / sample_rate_hz.
        transport.sour_set(
            channel,
            'ARBITRARY',
            amplitude_v,
            sample_rate_hz / samples.size,
            offset_v,
            data=samples,
            trig=trigger,
        )

    def enable_output(self, device: int, channel: int, enable: bool) -> None:
        """Switch an output channel on or off."""
        state = 'ON' if enable else 'OFF'
        self._device(device).tx_txt(f'OUTPUT{channel}:STATE {state}')

    def trigger_generation(self, device: int, channel: int) -> None:
        """Fire an internal generation trigger on one channel."""
        self._device(device).tx_txt(f'SOUR{channel}:TRig:INT')

    def configure_acquisition(
        self,
        device: int,
        decimation: int,
        trigger: AcquisitionTrigger,
        n_samples: int,
        channels: Sequence[int],
    ) -> None:
        """Set up one board's acquisition; only configured boards are read."""
        if not 0 < n_samples <= self.board.buffer_size:
            raise ValueError(f'n_samples must be in 1..{self.board.buffer_size}')
        transport = self._device(device)
        transport.tx_txt('ACQ:RST')
        transport.acq_set(decimation, trigger.level_v, trigger.delay_samples)
        self._acquisition[device] = _AcquisitionSettings(
            decimation, trigger, n_samples, tuple(channels)
        )

    def start_acquisition(self) -> None:
        """Arm every configured board, the primary last.

        Secondaries wait on the primary's forwarded trigger, so they must be
        armed before the primary can fire it.
        """
        order = sorted(self._acquisition, key=lambda i: i == self.primary)
        for idx in order:
            transport = self.devices[idx]
            transport.tx_txt('ACQ:START')
            time.sleep(_ARM_SETTLE_S)
            transport.tx_txt(f'ACQ:TRig {self._acquisition[idx].trigger.source}')

    def acquire(self, timeout_s: float) -> Acquisition:
        """Wait for every armed board to trigger and fill, then read its buffers.

        Raises:
            TimeoutError: A board did not trigger or fill within ``timeout_s``.
        """
        if not self._acquisition:
            raise RuntimeError('configure_acquisition first')
        for idx in self._acquisition:
            self._wait(idx, 'ACQ:TRig:STAT?', 'TD', timeout_s)
            self._wait(idx, 'ACQ:TRig:FILL?', '1', timeout_s)
        data = {}
        for idx, settings in self._acquisition.items():
            for chan in settings.channels:
                data[idx, chan] = self._read(idx, chan, settings.n_samples)
        rates = {
            i: self.board.base_sample_rate_hz / s.decimation
            for i, s in self._acquisition.items()
        }
        metadata = AcquisitionMetadata(
            timestamp=datetime.now(UTC),
            board_ids=self.hosts,
            decimation={i: s.decimation for i, s in self._acquisition.items()},
            sample_rate_hz=rates,
            trigger={i: s.trigger for i, s in self._acquisition.items()},
        )
        return Acquisition(data=data, metadata=metadata)

    def blink_led(self, device: int, led: int, n_blinks: int, period_s: float) -> None:
        """Blink an LED, to identify a board or confirm the SCPI server answers."""
        transport = self._device(device)
        for _ in range(n_blinks):
            transport.tx_txt(f'DIG:PIN LED{led},1')
            time.sleep(period_s / 2)
            transport.tx_txt(f'DIG:PIN LED{led},0')
            time.sleep(period_s / 2)

    def _device(self, idx: int) -> Transport:
        if not 0 <= idx < len(self.devices):
            raise ValueError(f'Device index {idx} out of range')
        return self.devices[idx]

    def _wait(self, idx: int, query: str, done: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while self.devices[idx].txrx_txt(query) != done:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{self.hosts[idx]}: {query} never returned {done}')
            time.sleep(_POLL_INTERVAL_S)

    def _read(self, idx: int, chan: int, n_samples: int) -> np.ndarray:
        transport = self.devices[idx]
        # A full read uses the plain DATA? query, the path proven on hardware;
        # shorter reads take the oldest n samples, i.e. those nearest the trigger.
        if n_samples == self.board.buffer_size:
            raw = transport.acq_data(chan, num_samples=None, old=False, convert=True)
        else:
            raw = transport.acq_data(
                chan, num_samples=n_samples, old=True, convert=True
            )
        return np.asarray(raw, dtype=float)
