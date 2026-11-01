"""Seeded frozen synthetic test sets, written in the real-data HDF5 schema.

One file per ``(SimConfig, seed, n_shots)``, so every estimator is scored on
the same shots and ``smi.data.datasets`` reads it like an acquisition.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import torch

from smi.data.schema import (
    DISPLACEMENT_KEY,
    DRIVE_KEY,
    NUM_SAMPLES_ATTR,
    OPTICAL_PHASE_KEY,
    PD_WAVELENGTHS_UM,
    SAMPLE_RATE_ATTR,
    VELOCITY_KEY,
)
from smi.theory.forward import SIMULATOR_VERSION, SimConfig, simulate_batch

# Shots per simulate_batch call. Part of the file identity: random draws are
# made per batch (all trajectories, then all phases, ...), so the stream a shot
# sees depends on the batch size. Every call simulates a FULL chunk and the
# last one is truncated, so the first k shots of a set depend only on
# (cfg, seed), never on n_shots. Cost: even a 3-shot set simulates a full
# chunk. Stored as the `chunk_shots` attr and checked by load_or_create.
CHUNK_SHOTS = 256

# The rig's three lasers as the simulator configs spell them, in channel order.
RIG_WAVELENGTHS_UM = (0.635, 0.6748, 0.515)
# A config's rig wavelengths are matched to this tolerance: the simulator
# configs spell the 675 nm laser 0.6748 while smi.data.schema records 0.675.
RIG_WAVELENGTH_TOL_UM = 1e-3

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'analysis' / 'data' / 'frozen'


def frozen_set_path(cfg: SimConfig, seed: int, n_shots: int, root: Path) -> Path:
    """Path of the frozen set for ``(cfg, seed, n_shots)`` under ``root``."""
    return Path(root) / f'{cfg.hash()}_seed{seed}_n{n_shots}.h5'


def pd_keys(wavelengths_um: tuple[float, ...]) -> list[str]:
    """Photodiode dataset keys: the rig's names for the rig's three lasers.

    The existing loader finds channels by these names, so any other channel
    count or wavelength set gets generic ``PD_i`` keys.
    """
    rig = tuple(PD_WAVELENGTHS_UM.values())
    if len(wavelengths_um) == len(rig) and all(
        abs(w - r) <= RIG_WAVELENGTH_TOL_UM
        for w, r in zip(wavelengths_um, rig, strict=True)
    ):
        return list(PD_WAVELENGTHS_UM)
    return [f'PD_{i}' for i in range(len(wavelengths_um))]


def write_frozen_set(cfg: SimConfig, seed: int, n_shots: int, path: Path) -> Path:
    """Simulate ``n_shots`` on CPU with one seeded generator and write them.

    Datasets are float32, ``[n_shots, L]``, chunked per shot with gzip level 4,
    as ``RedPitayaManager.save_data`` writes acquisitions. The file is built
    under a temporary name and renamed, so a crash never leaves a truncated
    file that ``load_or_create`` would trust.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # pid-suffixed so concurrent creators never share a temp file.
    tmp = path.with_name(f'{path.name}.{os.getpid()}.part')
    keys = pd_keys(cfg.wavelengths_um)
    generator = torch.Generator().manual_seed(seed)
    cpu = torch.device('cpu')

    with h5py.File(tmp, 'w') as f:
        f.attrs[SAMPLE_RATE_ATTR] = cfg.sample_rate
        f.attrs['pd_wavelengths_um'] = np.asarray(cfg.wavelengths_um)
        f.attrs['sim_config'] = json.dumps(asdict(cfg), sort_keys=True)
        f.attrs['seed'] = seed
        f.attrs['chunk_shots'] = CHUNK_SHOTS
        f.attrs['simulator_version'] = SIMULATOR_VERSION
        f.attrs[NUM_SAMPLES_ATTR] = n_shots
        datasets: dict[str, h5py.Dataset] = {}
        for start in range(0, n_shots, CHUNK_SHOTS):
            batch = simulate_batch(cfg, CHUNK_SHOTS, cpu, generator)
            stop = min(start + CHUNK_SHOTS, n_shots)
            arrays = {
                DRIVE_KEY: batch.voltage,
                VELOCITY_KEY: batch.velocity,
                DISPLACEMENT_KEY: batch.displacement,
                OPTICAL_PHASE_KEY: batch.optical_phase,
            }
            arrays.update({k: batch.signals[:, i] for i, k in enumerate(keys)})
            for name, value in arrays.items():
                if name not in datasets:
                    shape = (n_shots, *value.shape[1:])
                    datasets[name] = f.create_dataset(
                        name,
                        shape=shape,
                        maxshape=(None, *shape[1:]),
                        dtype='float32',
                        chunks=(1, *shape[1:]),
                        compression='gzip',
                        compression_opts=4,
                        track_times=False,
                    )
                datasets[name][start:stop] = value[: stop - start].numpy()
    tmp.replace(path)
    return path


def load_or_create(cfg: SimConfig, seed: int, n_shots: int, root: Path) -> Path:
    """Path of the frozen set, writing it first if it does not exist.

    A file written with a different ``CHUNK_SHOTS`` holds different random
    streams under the same name, and one written by another
    ``SIMULATOR_VERSION`` holds another simulator's shots, so either is
    regenerated rather than reused.
    """
    path = frozen_set_path(cfg, seed, n_shots, root)
    if path.exists():
        with h5py.File(path, 'r') as f:
            if (
                f.attrs.get('chunk_shots') == CHUNK_SHOTS
                and f.attrs.get('simulator_version') == SIMULATOR_VERSION
            ):
                return path
    write_frozen_set(cfg, seed, n_shots, path)
    return path
