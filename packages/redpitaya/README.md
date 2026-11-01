# stemlab

Arbitrary waveform generation and multi-channel acquisition on Red Pitaya
STEMlab 125-14 boards, over the board's SCPI server. The library is
signal-agnostic: you hand it sample arrays and get raw channel arrays (volts)
back. It knows nothing about what the signals mean.

The import name is `stemlab` because the vendor ships a module called
`redpitaya` on the board itself.

## Install

```bash
pip install -e packages/redpitaya    # or as a path dependency
```

Depends on numpy and matplotlib only. Importing opens no connection.

## API

```python
from stemlab import (
    SCPI,
    STEMLAB_125_14_SPEC,
    BoardSpec,
    RedPitayaManager,
    AcquisitionTrigger,
    Acquisition,
    AcquisitionMetadata,
    Transport,
)
from stemlab.live_view import LiveView
from stemlab.testing import FakeTransport
```

- `BoardSpec(base_sample_rate_hz, buffer_size, adc_bits, input_full_scale_v)`,
  frozen; `STEMLAB_125_14_SPEC` is 125 MHz, 16384 samples, 14 bits, +/-1 V.
- `RedPitayaManager(hosts, transport_factory, board)`: one connection per
  host, made by `transport_factory(host)` (normally `SCPI`). Boards are
  addressed by their index in `hosts`.
  - `configure_daisy_chain(primary)` - share clock and trigger from `primary`.
    Configure the secondaries to acquire on `EXT_NE`.
  - `configure_generation(device, channel, samples, sample_rate_hz, amplitude_v, offset_v, trigger)` -
    load one period of an arbitrary waveform (`samples` in [-1, 1], at most
    `buffer_size` long, sent at 5 decimal places) played at `sample_rate_hz`.
    `trigger` is `INT`, `EXT_PE`, `EXT_NE` or `GATED`. It does not reset the
    generator (GEN:RST clears both channels); call `reset()` once first.
  - `enable_output(device, channel, enable)`, `trigger_generation(device, channel)`.
  - `configure_acquisition(device, decimation, trigger, n_samples, channels)` -
    `trigger` is `AcquisitionTrigger(source, delay_samples, level_v)`;
    only configured boards are armed and read.
  - `start_acquisition()` - arm every configured board, primary last.
  - `acquire(timeout_s) -> Acquisition` - wait for trigger and buffer fill
    (raises `TimeoutError`; `timeout_s` bounds each wait on each board, so
    the worst case is `2 * timeout_s * n_boards`), then read. `Acquisition.data` maps
    `(board, channel)` to a `(n_samples,)` float array;
    `Acquisition.metadata` holds `timestamp` (UTC), `board_ids`, and per-board
    `decimation`, `sample_rate_hz` and `trigger`.
  - `reset()`, `blink_led(device, led, n_blinks, period_s)`, `close()`.
- `LiveView(manager, histogram, timeout_s).run(n_frames)` - a scope-style
  matplotlib view of the configured channels, optionally with a histogram
  per channel; `n_frames=None` runs until the window closes.
- `stemlab.testing.FakeTransport(host, buffer)` - a socket-free `SCPI` that
  records every command in `.sent` and answers data queries from `buffer`,
  for testing code built on this library without hardware.

## Example

```python
import numpy as np
from stemlab import SCPI, STEMLAB_125_14_SPEC, AcquisitionTrigger, RedPitayaManager

rp = RedPitayaManager(['rp-f0c04a.local', 'rp-f0c026.local'], SCPI, STEMLAB_125_14_SPEC)
rp.reset()
rp.configure_daisy_chain(primary=0)
wave = np.sin(2 * np.pi * np.arange(16384) / 16384)
rp.configure_generation(
    0,
    1,
    wave,
    sample_rate_hz=125e6 / 8192,
    amplitude_v=0.5,
    offset_v=0.0,
    trigger='INT',
)
rp.enable_output(0, 1, enable=True)
rp.configure_acquisition(
    0, 256, AcquisitionTrigger('CH1_PE', 16384, 0.0), 16384, [1, 2]
)
rp.configure_acquisition(
    1, 256, AcquisitionTrigger('EXT_NE', 16384, 0.0), 16384, [1, 2]
)
rp.start_acquisition()
shot = rp.acquire(timeout_s=5.0)
```

## Tests

```bash
pip install -e 'packages/redpitaya[test]'
pytest packages/redpitaya/tests
```
