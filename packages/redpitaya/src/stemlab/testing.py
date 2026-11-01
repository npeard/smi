"""A hardware-free transport for tests of code built on ``stemlab``.

``FakeTransport`` subclasses ``SCPI`` and replaces only the socket, so the real
``sour_set``/``acq_set``/``acq_data`` formatting runs and the exact command
strings a board would receive are recorded in ``sent``.
"""

import numpy as np

from .scpi import SCPI


class FakeTransport(SCPI):
    """Records sent commands and answers queries with canned replies.

    Args:
        host: Recorded only; no connection is made.
        buffer: Samples returned for every channel data query.
    """

    def __init__(self, host: str, buffer: np.ndarray):
        self.host = host
        self._socket = None
        self.buffer = buffer
        self.sent: list[str] = []
        self.closed = False
        self._reply: str | None = None

    def close(self) -> None:
        self.closed = True

    def tx_txt(self, msg: str) -> None:
        self.sent.append(msg)
        self._reply = self._answer(msg)

    def rx_txt(self, chunksize: int = 4096) -> str | None:
        del chunksize  # replies are whole messages; kept for SCPI's signature
        return self._reply

    def _answer(self, msg: str) -> str | None:
        query = msg.upper()
        if query == 'ACQ:TRIG:STAT?':
            return 'TD'
        if query == 'ACQ:TRIG:FILL?':
            return '1'
        if query == 'ACQ:DATA:UNITS?':
            return 'VOLTS'
        if query.startswith('ACQ:SOUR') and '?' in query:
            n = int(msg.rsplit(' ', 1)[-1]) if ':N?' in query else self.buffer.size
            return '{' + ','.join(f'{v:.6f}' for v in self.buffer[:n]) + '}'
        return None
