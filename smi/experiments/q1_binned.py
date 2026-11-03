"""Q1's binned test: velocity error against |v| and fringe-extremum distance.

The analytic foundation of report.typ predicts that one channel loses its information at
its fringe extrema and three channels do not, so the error should rise toward
an extremum for one channel and stay flat for three. The runner's records bin
only by the pre-feedback phase of the shortest channel, and the checkpoints
are gitignored, so this evaluates the chosen checkpoints once on their cells'
frozen test sets and writes a small committed JSON the report reads.

A fringe extremum is a zero of sin(phi_F) mod pi, with phi_F the feedback
phase of eq-excess, because the simulated fringe is cos(phi_F). Distance is
in fractions of a fringe, so it runs from 0 (at an extremum) to 1/4 (midway
between two).

Usage: pixi run python scripts/q1_binned.py [--out PATH]
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from smi.core.metrics import best_sign, per_shot_rmse
from smi.core.paths import GRID_RUNS_DIR, RESULTS_DIR
from smi.data.datasets import select_pd_keys
from smi.data.frozen import DEFAULT_ROOT, load_or_create
from smi.experiments.evaluate import EVAL_CHUNK_SHOTS, N_BINS, CheckpointEstimator
from smi.experiments.grids import grid as shipped_grid
from smi.experiments.results_artifact import read_results
from smi.experiments.runner import Cell, cell_key, expand_grid
from smi.theory.forward import smi_excess_phase

RESULTS = RESULTS_DIR / 'synthetic_results.json'

FAMILY = 'tcn-44k'
# 515 nm is the best single channel and 635 + 515 the best pair of the TCN in
# both grids (lowest test RMSE; at C = 0.2 noise-free the 674.8 + 515 pair
# ties it). Comparing the best single channel with the best pair and with all
# three is the Q1 contrast; the noise-free cell shows it with nothing masking
# it, and the nominal point at 30 dB shows what is left under drift and noise.
SUBSETS = ((0.515,), (0.635, 0.515), (0.635, 0.6748, 0.515))
CONDITIONS = (
    ('noise_free', 'q1_s2_subsets', 's2-smi-a3'),
    ('nominal_30db', 'q1_s4_snr', 'q1-nominal-snr_db=30'),
)
# Shot-pooled per-shot RMSE must match the runner's record to this: the record
# was scored on MPS, this on CPU, both float32.
RECORD_RTOL = 1e-3


def extremum_distance(phase_f: np.ndarray) -> np.ndarray:
    """Distance from ``phase_f`` to the nearest multiple of pi, in fringes."""
    return np.abs(phase_f - np.pi * np.round(phase_f / np.pi)) / (2 * np.pi)


def fringe_phase(theta: np.ndarray, feedback_c: float, alpha: float) -> np.ndarray:
    """The feedback phase phi_F of eq-excess, whose cosine is the fringe."""
    t = torch.from_numpy(np.asarray(theta, dtype=np.float64))
    delta, _ = smi_excess_phase(t, feedback_c, alpha)
    return (t + delta).numpy()


def nearest_extremum_distances(
    phase_f: np.ndarray, wavelengths_um: tuple[float, ...]
) -> dict[str, np.ndarray]:
    """Extremum distance per channel, and of the nearest and farthest channel.

    Args:
        phase_f: ``[B, C, L]`` feedback phase, channels in ``wavelengths_um``
            order.
        wavelengths_um: The channels.

    Returns:
        ``[B, L]`` distances keyed by wavelength in nm and, for two or more
        channels, ``any`` (the nearest extremum of any channel: small when one
        channel is blind) and ``all`` (the farthest channel's: small only when
        every channel is near an extremum at once).
    """
    d = extremum_distance(phase_f)
    out = {f'{round(w * 1000)}': d[:, i] for i, w in enumerate(wavelengths_um)}
    if len(wavelengths_um) > 1:
        out['any'] = d.min(axis=1)
        out['all'] = d.max(axis=1)
    return out


def binned_rmse(
    sq_err: np.ndarray, values: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled RMSE and sample count of ``sq_err`` in bins of ``values``.

    Values on the top edge land in the last bin, as in evaluate's binning.
    """
    idx = np.digitize(values.ravel(), edges[1:-1])
    n_bins = len(edges) - 1
    n = np.bincount(idx, minlength=n_bins)
    total = np.bincount(idx, weights=sq_err.ravel(), minlength=n_bins)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(total / n), n


def _table(
    sq_err: np.ndarray, v: np.ndarray, values: np.ndarray, edges: np.ndarray
) -> dict[str, list]:
    """Per-bin RMSE, the zero predictor's (the bin's RMS velocity), and counts."""
    rmse, n = binned_rmse(sq_err, values, edges)
    zero, _ = binned_rmse(v**2, values, edges)
    return {'rmse': rmse.tolist(), 'zero': zero.tolist(), 'n': n.tolist()}


def grid_checkpoint(grid_name: str, key: str) -> Path:
    """The runner's checkpoint of cell ``key`` of the grid ``grid_name``."""
    return GRID_RUNS_DIR / grid_name / f'{key}.ckpt'


def evaluate_cell(
    grid_name: str, wavelengths_um: tuple[float, ...], sim_name: str
) -> tuple[Cell, float, np.ndarray, np.ndarray, np.ndarray]:
    """Squared error, truth and feedback phase of one cell on its frozen test set.

    Raises:
        ValueError: If the record does not match the cell or its test set, or
            the rescored RMSE does not reproduce the record's.

    Returns:
        ``(cell, mean per-shot RMSE, squared error [B, L], velocity [B, L],
        phase_f [B, C, L])``.
    """
    grid = shipped_grid(grid_name)
    cells = expand_grid(grid)
    key = cell_key(FAMILY, wavelengths_um, sim_name, 0)
    (cell,) = [c for c in cells if c.key == key]
    record = {r.cell_key: r for r in read_results(RESULTS)}[key]
    if record.sim_config_hash != cell.sim.hash() or record.test_seed is None:
        raise ValueError(f'{key}: record does not match its grid cell')
    path = load_or_create(
        cell.sim, record.test_seed, grid.test_set.n_shots, DEFAULT_ROOT
    )
    if path.stem != record.dataset:
        raise ValueError(f'{key}: scored on {record.dataset}, not {path.stem}')
    est = CheckpointEstimator(grid_checkpoint(grid_name, key), 'cpu')
    with h5py.File(path, 'r') as f:
        keys = select_pd_keys(f, cell.wavelengths_um)
        signals = torch.from_numpy(np.stack([f[k][:] for k in keys], axis=1))
        velocity = torch.from_numpy(f['velocity'][:])
        theta = f['optical_phase'][:]
    pred = torch.cat(
        [
            est.predict_velocity(signals[i : i + EVAL_CHUNK_SHOTS], cell.wavelengths_um)
            for i in range(0, len(signals), EVAL_CHUNK_SHOTS)
        ]
    )
    if cell.sign_invariant:
        pred = best_sign(pred, velocity) * pred
    rmse = float(per_shot_rmse(pred, velocity).mean())
    metric = 'velocity_rmse_sign_invariant' if cell.sign_invariant else 'velocity_rmse'
    if not math.isclose(rmse, record.metrics[metric]['mean'], rel_tol=RECORD_RTOL):
        raise ValueError(f'{key}: rescored {rmse}, record {record.metrics[metric]}')
    phase_f = fringe_phase(theta, cell.sim.feedback_c, cell.sim.alpha)
    sq_err = ((pred - velocity) ** 2).double().numpy()
    return cell, rmse, sq_err, velocity.double().numpy(), phase_f


def run_q1_binned(out: Path) -> None:
    """Evaluate every Q1 cell and write the binned-error JSON to ``out``."""
    dist_edges = np.linspace(0.0, 0.25, N_BINS + 1)
    v_edges = None
    reference_velocity = None
    cells = {}
    for condition, grid, sim_name in CONDITIONS:
        for wavelengths in SUBSETS:
            cell, rmse, sq_err, v, phase_f = evaluate_cell(grid, wavelengths, sim_name)
            if reference_velocity is None:
                reference_velocity = v
                v_edges = np.quantile(np.abs(v), np.linspace(0.0, 1.0, N_BINS + 1))
            # One |v| binning for every cell, valid because the trajectories
            # are drawn first from the shared test seed, before any per-channel
            # or noise draw (smi.theory.forward.simulate_batch).
            if not np.array_equal(v, reference_velocity):
                raise ValueError(f'{cell.key}: test trajectories differ')
            assert v_edges is not None
            cells[cell.key] = {
                'condition': condition,
                'wavelengths_um': list(cell.wavelengths_um),
                'rmse': rmse,
                'abs_velocity': _table(sq_err, v, np.abs(v), v_edges),
                'fringe_distance': {
                    name: _table(sq_err, v, d, dist_edges)
                    for name, d in nearest_extremum_distances(
                        phase_f, cell.wavelengths_um
                    ).items()
                },
            }
    assert v_edges is not None
    payload = {
        'abs_velocity_edges_um_per_ms': v_edges.tolist(),
        'fringe_distance_edges_fringes': dist_edges.tolist(),
        'cells': cells,
    }
    out.write_text(json.dumps(payload, indent=1) + '\n', encoding='utf-8')
