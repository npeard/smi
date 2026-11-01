"""Play a random coil drive and record every board, one shot at a time."""

import logging
import time
from datetime import datetime
from pathlib import Path

import h5py
from stemlab import Acquisition, AcquisitionTrigger, RedPitayaManager
from stemlab.live_view import LiveView

from smi.acquisition.rig import PRIMARY, TRIGGER_DELAY_SAMPLES
from smi.data.schema import SAMPLE_RATE_ATTR, write_shot
from smi.theory.coil_driver import CoilDriver
from smi.theory.waveform import Waveform

logger = logging.getLogger(__name__)

DRIVE_CHANNEL = 1
ACQUIRED_CHANNELS = (1, 2)
# A rig that times out this many shots in a row is disconnected or mis-wired;
# stop rather than loop forever.
MAX_CONSECUTIVE_TIMEOUTS = 3


def channel_key(board: int, channel: int) -> str:
    """HDF5 dataset name of one acquired channel, e.g. ``RP1_CH2``."""
    return f'RP{board + 1}_CH{channel}'


def configure_shot(
    manager: RedPitayaManager, waveform: Waveform, coil: CoilDriver
) -> None:
    """Load a fresh drive onto the primary, arm every board and start playing.

    The primary triggers on the drive's own positive edge and forwards that
    trigger to the others, which wait on ``EXT_NE``. ``manager.reset()`` must
    have been called once beforehand.
    """
    # Pre-compensate the coil's gain so the displacement spectrum comes out flat.
    _, drive, _ = coil.sample(waveform, normalize_gain=True)
    manager.configure_generation(
        PRIMARY,
        DRIVE_CHANNEL,
        drive,
        waveform.gen_sample_rate,
        amplitude_v=1.0,
        offset_v=0.0,
        trigger='INT',
    )
    for board in range(len(manager.hosts)):
        source = 'CH1_PE' if board == PRIMARY else 'EXT_NE'
        manager.configure_acquisition(
            board,
            waveform.acq_dec,
            AcquisitionTrigger(source, TRIGGER_DELAY_SAMPLES, level_v=0.0),
            waveform.acquisition.buffer_size,
            ACQUIRED_CHANNELS,
        )
    manager.start_acquisition()
    manager.enable_output(PRIMARY, DRIVE_CHANNEL, True)
    manager.trigger_generation(PRIMARY, DRIVE_CHANNEL)


def _write(path: Path, acquisition: Acquisition, waveform: Waveform) -> None:
    shot = {channel_key(b, c): v for (b, c), v in acquisition.data.items()}
    attrs: dict[str, object] = {
        SAMPLE_RATE_ATTR: acquisition.metadata.sample_rate_hz[PRIMARY],
        'decimation': waveform.acq_dec,
        'buffer_size': waveform.acquisition.buffer_size,
        'gen_dec': waveform.gen_dec,
        'acq_dec': waveform.acq_dec,
        'start_freq': waveform.start_freq,
        'end_freq': waveform.end_freq,
        'trigger_delay': TRIGGER_DELAY_SAMPLES,
    }
    if not path.exists():
        attrs['creation_time'] = datetime.now().isoformat()
    with h5py.File(path, 'a') as f:
        write_shot(f, shot, attrs)


def run_shots(
    manager: RedPitayaManager,
    waveform: Waveform,
    coil: CoilDriver,
    n_shots: int,
    path: Path,
    timeout_s: float,
    delay_s: float,
) -> Path:
    """Acquire ``n_shots`` shots, appending each to ``path`` as soon as it is read.

    Writing per shot means a lost connection keeps every finished shot. A shot
    whose trigger times out is logged and skipped, never filled in, so the file
    may hold fewer than ``n_shots``. Velocity and displacement are not stored:
    they are recomputed from the drive channel by the readers.

    Args:
        manager: Connected boards, already daisy-chained if there are several.
        waveform: Random drive generator; sets decimations and frequency band.
        coil: Coil model used to pre-compensate the drive.
        n_shots: Shots to attempt.
        path: HDF5 file, created or appended to.
        timeout_s: Wait per trigger and per buffer fill, on each board.
        delay_s: Pause between shots.

    Returns:
        ``path``.

    Raises:
        RuntimeError: ``MAX_CONSECUTIVE_TIMEOUTS`` shots timed out in a row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    timeouts = 0
    manager.reset()
    try:
        for i in range(n_shots):
            logger.info(f'Shot {i + 1}/{n_shots}')
            configure_shot(manager, waveform, coil)
            try:
                acquisition = manager.acquire(timeout_s)
            except TimeoutError:
                timeouts += 1
                logger.warning(f'Shot {i + 1} timed out; skipped')
                if timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    raise RuntimeError(
                        f'{timeouts} consecutive shots timed out; check the rig'
                    ) from None
            else:
                timeouts = 0
                _write(path, acquisition, waveform)
                written += 1
            # The rig needs the settle time after a skipped shot as well.
            if i < n_shots - 1:
                time.sleep(delay_s)
    finally:
        manager.reset()  # stops the coil drive
        logger.info(f'Wrote {written}/{n_shots} shots to {path}')
    return path


def live_view(
    manager: RedPitayaManager, waveform: Waveform, coil: CoilDriver, timeout_s: float
) -> None:
    """Play the drive and show every channel live until the window is closed."""
    manager.reset()
    try:
        configure_shot(manager, waveform, coil)
        LiveView(manager, histogram=True, timeout_s=timeout_s).run(None)
    finally:
        manager.reset()
