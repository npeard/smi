"""Write the example-shot arrays behind the report's appendix figures.

Picks a low-, mid- and high-excursion shot (displacement peak-to-peak) from
the nominal arm's frozen test set, and the same shots in the q3_fiber fiber
arm, whose motion is drawn identically (paired). For each it stores the
photodiode inputs, the target velocity and displacement, and the predictions
of TCAN-44k, UNetTransformer-offline-44k and, where its checkpoint loads, the
R8 conditioning-encoder posterior flow (mean and 90% band) and that flow's
encoder head alone as a deterministic regressor. The same shots also run
through two q3_fiber_sweep arms, a low-impairment and a high-fading static
fiber, scored by their TCAN-44k, so fiber degradation shows by eye. The
figures read only these arrays: running models is not work a document build
should repeat.

Traces are stored decimated by ``DECIMATE`` (plain subsampling: these are for
the eye, and the fringes are what the zoom window keeps at full rate).
Predicted displacement is the integrated predicted velocity started at zero,
the target likewise, as ``smi.experiments.evaluate`` scores it.

When the real acquisitions are present (``smi.core.paths.DATASETS``), the
nominal checkpoints also run on three real shots, chosen the same way; when
not, the index records the real set as blinded and the figure says so.

Run: pixi run python scripts/example_shots.py --device mps
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
from torch import Tensor

from smi.core.cli import add_verbosity, setup_logging
from smi.core.kinematics import integrate_velocity
from smi.core.paths import DATA_DIR, DATASETS, FROZEN_DIR, GRID_RUNS_DIR, RESULTS_DIR
from smi.data.frozen import PERP_SUFFIX, load_or_create, pd_keys
from smi.data.schema import (
    DISPLACEMENT_KEY,
    DRIVE_KEY,
    PD_WAVELENGTHS_UM,
    VELOCITY_KEY,
    read_sample_rate,
)
from smi.experiments.evaluate import CheckpointEstimator
from smi.experiments.flow_eval import EncoderEstimator, PosteriorEstimator
from smi.experiments.flow_runner import POSTERIOR_SAMPLING, POSTERIORS_ENC, checkpoint
from smi.experiments.grids import GRIDS
from smi.experiments.runner import Cell, expand_grid, trained_checkpoint
from smi.learning.flow_lit_module import FlowLitModule
from smi.theory.coil_driver import CoilDriver

logger = logging.getLogger(__name__)

GRID = 'q3_fiber'
ARMS = ('nominal', 'fiber')
FAMILIES = ('tcan-44k', 'unettransformer-offline-44k')
#: q3_fiber_sweep arms at the two ends of coupling fading, TCAN-44k only.
SWEEP_GRID = 'q3_fiber_sweep'
SWEEP_ARMS = ('modes-1', 'modes-8')
#: Shot types, by quantile of displacement peak-to-peak over the test set.
SHOT_QUANTILES = {'low': 0.1, 'mid': 0.5, 'high': 0.9}
DECIMATE = 16
#: Full-rate zoom window, centred on the shot's peak |velocity|, where the
#: fringes are densest.
ZOOM = 400
#: Real shots considered for selection, the first of each file.
REAL_SHOTS = 256
BAND = (0.05, 0.95)
NPZ = RESULTS_DIR / 'example_shots.npz'
INDEX = RESULTS_DIR / 'example_shots.json'


def select_shots(p2p: np.ndarray, quantiles: dict[str, float]) -> dict[str, int]:
    """The shot nearest each quantile of ``p2p``, distinct across types.

    Raises:
        ValueError: If two quantiles land on the same shot.
    """
    picks = {
        name: int(np.argmin(np.abs(p2p - np.quantile(p2p, q))))
        for name, q in quantiles.items()
    }
    if len(set(picks.values())) != len(picks):
        raise ValueError(f'shot types collide: {picks}')
    return picks


def zoom_slice(velocity: np.ndarray, width: int) -> slice:
    """``width`` samples centred on the peak |velocity|, clipped to the shot."""
    centre = int(np.argmax(np.abs(velocity)))
    start = min(max(centre - width // 2, 0), velocity.size - width)
    return slice(start, start + width)


def _store(out: dict[str, np.ndarray], prefix: str, trace: np.ndarray, z: slice):
    out[f'{prefix}'] = trace[..., ::DECIMATE].astype(np.float32)
    out[f'{prefix}_zoom'] = trace[..., z].astype(np.float32)


def _displacement(velocity: Tensor, sample_rate: float) -> np.ndarray:
    return cast(Tensor, integrate_velocity(velocity.double(), sample_rate)).numpy()


def _shot_entry(
    out: dict[str, np.ndarray],
    tag: str,
    signals: Tensor,
    velocity: Tensor,
    sample_rate: float,
    predictions: dict[str, Tensor],
    band: tuple[Tensor, Tensor] | None,
) -> dict[str, object]:
    """Store one shot's arrays under ``tag/...``; returns its index entry."""
    v = velocity.numpy()
    z = zoom_slice(v, ZOOM)
    _store(out, f'{tag}/signals', signals.numpy(), z)
    _store(out, f'{tag}/velocity', v, z)
    _store(out, f'{tag}/displacement', _displacement(velocity[None], sample_rate)[0], z)
    for name, pred in predictions.items():
        _store(out, f'{tag}/{name}/velocity', pred.numpy(), z)
        disp = _displacement(pred[None], sample_rate)[0]
        _store(out, f'{tag}/{name}/displacement', disp, z)
    if band is not None:
        for edge, trace in zip(('lo', 'hi'), band, strict=True):
            _store(out, f'{tag}/flow/velocity_{edge}', trace.numpy(), z)
    return {
        'zoom_start': z.start,
        'models': sorted(predictions),
        'band': band is not None,
    }


def _flow(arm: str, device: str) -> FlowLitModule | None:
    path = checkpoint(POSTERIORS_ENC[arm])
    try:
        module = FlowLitModule.load_from_checkpoint(
            path, map_location=device, fit_norm_stats=False
        ).eval()
    except (FileNotFoundError, RuntimeError, KeyError, TypeError) as err:
        logger.warning(f'{arm}: flow {path} did not load ({err}); no band')
        return None
    return module


def _read(path: Path, keys: list[str], idx: list[int]) -> tuple[Tensor, Tensor]:
    with h5py.File(path, 'r') as f:
        signals = np.stack([np.asarray(f[k])[idx] for k in keys], axis=1)
        velocity = np.asarray(f[VELOCITY_KEY])[idx]
    return torch.from_numpy(signals.astype(np.float32)), torch.from_numpy(velocity)


def _arm(
    out: dict[str, np.ndarray],
    arm: str,
    cells: dict[str, Cell],
    grid: str,
    picks: dict[str, int],
    p2p: np.ndarray,
    sample_rate: float,
    device: str,
    with_flow: bool,
) -> dict[str, object]:
    """Run ``cells`` (family -> cell) of one arm on the picked shots; store them."""
    sim = next(iter(cells.values())).sim
    test = GRIDS[grid].test_set
    keys = pd_keys(sim.wavelengths_um)
    if sim.n_signal_channels > len(keys):
        keys += [k + PERP_SUFFIX for k in keys]
    dataset = load_or_create(sim, test.seed, test.n_shots, FROZEN_DIR)
    signals, velocity = _read(dataset, keys, list(picks.values()))
    preds = {}
    for family, cell in cells.items():
        ckpt = trained_checkpoint(cell, grid, GRID_RUNS_DIR)
        est = CheckpointEstimator(ckpt, device)
        preds[family] = est.predict_velocity(signals, sim.wavelengths_um)
    band = None
    module = _flow(arm, device) if with_flow else None
    if module is not None:
        flow = PosteriorEstimator(module, POSTERIOR_SAMPLING, f'flow:{arm}')
        flow.predict_velocity(signals, sim.wavelengths_um)
        draws = flow.draws[0]
        preds['flow'] = draws.mean(0)
        preds['head'] = EncoderEstimator(module, f'head:{arm}').predict_velocity(
            signals, sim.wavelengths_um
        )
        band = (
            torch.quantile(draws, BAND[0], dim=0),
            torch.quantile(draws, BAND[1], dim=0),
        )
    shots = {
        kind: {
            'shot': shot,
            'displacement_p2p_um': float(p2p[shot]),
            **_shot_entry(
                out,
                f'{arm}/{kind}',
                signals[i],
                velocity[i],
                sample_rate,
                {k: v[i] for k, v in preds.items()},
                None if band is None else (band[0][i], band[1][i]),
            ),
        }
        for i, (kind, shot) in enumerate(picks.items())
    }
    return {
        'dataset': dataset.name,
        'wavelengths_um': list(sim.wavelengths_um),
        'channels': keys,
        'shots': shots,
    }


def synthetic(device: str, out: dict[str, np.ndarray]) -> dict[str, object]:
    cells = {(c.sim_name, c.family): c for c in expand_grid(GRIDS[GRID])}
    nominal = cells['nominal', FAMILIES[0]].sim
    test = GRIDS[GRID].test_set
    path = load_or_create(nominal, test.seed, test.n_shots, FROZEN_DIR)
    with h5py.File(path, 'r') as f:
        p2p = np.ptp(np.asarray(f[DISPLACEMENT_KEY]), axis=1)
        sample_rate = read_sample_rate(f.attrs)
    picks = select_shots(p2p, SHOT_QUANTILES)
    arms = {
        arm: _arm(
            out,
            arm,
            {fam: cells[arm, fam] for fam in FAMILIES},
            GRID,
            picks,
            p2p,
            sample_rate,
            device,
            with_flow=True,
        )
        for arm in ARMS
    }
    # The sweep's frozen sets draw the same motion as q3_fiber's (paired), so
    # the picked shot indices are the same shots.
    sweep = {(c.sim_name, c.family): c for c in expand_grid(GRIDS[SWEEP_GRID])}
    sweep_arms = {
        arm: _arm(
            out,
            arm,
            {FAMILIES[0]: sweep[arm, FAMILIES[0]]},
            SWEEP_GRID,
            picks,
            p2p,
            sample_rate,
            device,
            with_flow=False,
        )
        for arm in SWEEP_ARMS
    }
    return {'sample_rate_hz': sample_rate, 'arms': arms, 'sweep_arms': sweep_arms}


def real(device: str, out: dict[str, np.ndarray]) -> dict[str, object]:
    """The nominal checkpoints on real free-space shots, or the blinded status."""
    path = DATA_DIR / DATASETS['free-space']
    if not path.is_file():
        return {'status': 'blinded', 'dataset': path.name}
    keys = list(PD_WAVELENGTHS_UM)
    with h5py.File(path, 'r') as f:
        sample_rate = read_sample_rate(f.attrs)
        n = min(REAL_SHOTS, f[DRIVE_KEY].shape[0])
        volts = torch.from_numpy(np.asarray(f[DRIVE_KEY][:n], dtype=np.float32))
        velocity, disp = CoilDriver().get_velocity_and_displacement_torch(
            volts, sample_rate
        )
        picks = select_shots(np.ptp(disp.numpy(), axis=1), SHOT_QUANTILES)
        idx = list(picks.values())
        signals = torch.from_numpy(
            np.stack([np.asarray(f[k])[idx] for k in keys], axis=1).astype(np.float32)
        )
    cells = {(c.sim_name, c.family): c for c in expand_grid(GRIDS[GRID])}
    preds = {}
    for family in FAMILIES:
        cell = cells['nominal', family]
        est = CheckpointEstimator(trained_checkpoint(cell, GRID, GRID_RUNS_DIR), device)
        preds[family] = est.predict_velocity(signals, cell.wavelengths_um)
    shots = {
        kind: {
            'shot': shot,
            **_shot_entry(
                out,
                f'real/{kind}',
                signals[i],
                velocity[shot].float(),
                sample_rate,
                {k: v[i] for k, v in preds.items()},
                None,
            ),
        }
        for i, (kind, shot) in enumerate(picks.items())
    }
    return {
        'status': 'present',
        'dataset': path.name,
        'sample_rate_hz': sample_rate,
        'wavelengths_um': list(PD_WAVELENGTHS_UM.values()),
        'channels': keys,
        'shots': shots,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', required=True, help='e.g. cpu, mps, cuda:0')
    add_verbosity(parser)
    args = parser.parse_args()
    setup_logging(args.verbosity)
    torch.manual_seed(0)
    out: dict[str, np.ndarray] = {}
    index = {
        'grid': GRID,
        'families': list(FAMILIES),
        'shot_quantiles': SHOT_QUANTILES,
        'decimate': DECIMATE,
        'zoom_samples': ZOOM,
        'band': list(BAND),
        'synthetic': synthetic(args.device, out),
        'real': real(args.device, out),
    }
    np.savez_compressed(NPZ, **out)
    INDEX.write_text(json.dumps(index, indent=2) + '\n')
    logger.info(f'wrote {NPZ} ({NPZ.stat().st_size / 1e3:.0f} kB) and {INDEX}')


if __name__ == '__main__':
    main()
