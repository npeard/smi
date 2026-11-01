"""The real-data HDF5 schema: key names, attributes, and the shot read/write pair.

Every file the pipeline reads (acquisitions and frozen synthetic sets) holds one
``(n_shots, n_samples)`` float32 dataset per channel plus file-level attributes.
Readers and writers go through this module instead of spelling keys themselves.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from smi.core.acquisition import STEMLAB_125_14

logger = logging.getLogger(__name__)

# Photodiode channel -> laser wavelength in microns. RP1_CH1 is the speaker
# drive voltage and is not a photodiode. The 675 nm laser is spelled 0.675
# here and 0.6748 in the simulator configs (smi.data.frozen matches the two to
# RIG_WAVELENGTH_TOL_UM).
PD_WAVELENGTHS_UM: dict[str, float] = {
    'RP1_CH2': 0.635,
    'RP2_CH1': 0.675,
    'RP2_CH2': 0.515,
}
DRIVE_KEY = 'RP1_CH1'
PD_KEYS: tuple[str, ...] = tuple(PD_WAVELENGTHS_UM)
CHANNEL_KEYS: tuple[str, ...] = (DRIVE_KEY, *PD_KEYS)

# Ground-truth datasets that only frozen synthetic sets carry.
VELOCITY_KEY = 'velocity'
DISPLACEMENT_KEY = 'displacement'
OPTICAL_PHASE_KEY = 'optical_phase'

SAMPLE_RATE_ATTR = 'sample_rate'
NUM_SAMPLES_ATTR = 'num_samples'

# Files written before the ``sample_rate`` attribute existed were all acquired
# on the STEMlab 125-14 at decimation 256. This is the one place that fallback
# is spelled; files that carry the attribute never reach it.
LEGACY_SAMPLE_RATE_HZ: float = STEMLAB_125_14.sample_rate_dec1_hz / 256


def read_sample_rate(attrs: Mapping[str, Any]) -> float:
    """Sample rate (Hz) from a file's attributes, else ``LEGACY_SAMPLE_RATE_HZ``."""
    if SAMPLE_RATE_ATTR in attrs:
        return float(attrs[SAMPLE_RATE_ATTR])
    logger.warning('No sample_rate attr; using legacy %.2f Hz', LEGACY_SAMPLE_RATE_HZ)
    return LEGACY_SAMPLE_RATE_HZ


def write_shot(
    h5file: h5py.File, shot: Mapping[str, np.ndarray], attrs: Mapping[str, Any]
) -> None:
    """Append one shot to an open file, creating its channel datasets on first use.

    Args:
        h5file: File opened writable ('w' or 'a').
        shot: Channel key -> 1-D signal of this shot.
        attrs: File-level attributes, set (overwriting) on every call.

    The datasets are float32, gzip-4 and chunked one shot per chunk, so a shot
    reads back without touching its neighbours.
    """
    for name, value in attrs.items():
        h5file.attrs[name] = value
    for key, signal in shot.items():
        if key not in h5file:
            h5file.create_dataset(
                key,
                shape=(0, *signal.shape),
                maxshape=(None, *signal.shape),
                dtype='float32',
                chunks=(1, *signal.shape),
                compression='gzip',
                compression_opts=4,
            )
        dataset = h5file[key]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = signal
    if shot:
        h5file.attrs[NUM_SAMPLES_ATTR] = h5file[next(iter(shot))].shape[0]


def read_shots(
    path: str | Path,
    keys: Sequence[str] | None = None,
    start: int = 0,
    stop: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read shots ``start:stop`` of the given channels, plus the file attributes.

    Args:
        path: HDF5 file.
        keys: Channels to read; every dataset in the file when None.
        start: First shot.
        stop: One past the last shot; the end of the file when None.

    Returns:
        ``(arrays, attrs)``: channel key -> ``(shots, n_samples)`` array in the
        stored dtype, and the file-level attributes.
    """
    with h5py.File(path, 'r') as f:
        names = list(f) if keys is None else list(keys)
        arrays = {k: f[k][start:stop] for k in names}
        return arrays, dict(f.attrs)
