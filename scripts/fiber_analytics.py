"""Q3.1 and Q3.2 analytic tier: fiber phase as a Fisher nuisance, and P_perp.

Q3.1: profiled velocity information and condition number of the local Fisher
matrix (``bounds.fisher_matrix`` with ``SimConfig.fiber``) for free space, a
fiber with no monitor, a fourth color through the fiber to the target, and
that color returned by a dichroic at the distal face, against the dispersion
of the fiber-phase drift. Q3.2: corr(P_perp, |kappa|^2) against mode count
over static fully mixed fibers, with corr * sqrt(N) (the 1/sqrt(N) law) and
the Haar round-trip (COE) mean E|kappa|^2 = 2 / (2N + 1), which includes the
factor-2 coherent-backscatter enhancement of a reciprocal fiber, plus the
within-shot decorrelation of kappa against drift rate (to pick Task 3's grid
point). CPU, float64 analytics; seeded.

Usage::

    pixi run python scripts/fiber_analytics.py --help
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch

from smi.core.acquisition import STEMLAB_125_14
from smi.core.paths import RESULTS_DIR
from smi.data.frozen import RIG_WAVELENGTHS_UM
from smi.theory import bounds
from smi.theory.fiber import FiberConfig, simulate_fiber
from smi.theory.forward import NOMINAL_ALPHA, NOMINAL_FEEDBACK_C, SimConfig

DEFAULT_OUTPUT = RESULTS_DIR / 'fiber_results.json'
# Fourth color for the monitor arms: a common diode line clear of the rig's.
MONITOR_UM = 0.785
MODES = (1, 2, 4, 8, 16, 32, 64)
DISPERSIONS_PER_UM = (0.0, 1e-3, 1e-2, 1e-1)
DRIFTS_RAD_PER_SQRT_S = (1.0, 3.0, 10.0, 30.0, 100.0)
CPU = torch.device('cpu')


def _fiber(
    n_modes: int, drift: float, dispersion: float, monitor: float | None
) -> FiberConfig:
    return FiberConfig(
        n_modes=n_modes,
        drift_rad_per_sqrt_s=drift,
        path_drift_um_per_sqrt_s=0.0,
        dispersion_per_um=dispersion,
        spectral_corr_width_um=0.01,
        monitor_wavelength_um=monitor,
        pbs_channel=False,
    )


def _profiled(j: np.ndarray) -> float:
    """Velocity information with every nuisance profiled out (Schur)."""
    return float(j[0, 0] - j[0, 1:] @ np.linalg.solve(j[1:, 1:], j[1:, 0]))


def _cond(j: np.ndarray) -> float:
    """Condition number of the unit-diagonal (correlation) form: unit-free."""
    d = 1 / np.sqrt(np.diag(j))
    return float(np.linalg.cond(j * d[:, None] * d[None, :]))


def q31(n_shots: int, seed: int) -> dict:
    """Fisher table for the four arms against dispersion."""
    rig = SimConfig.nominal(
        STEMLAB_125_14, RIG_WAVELENGTHS_UM, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
    )
    four = (*RIG_WAVELENGTHS_UM, MONITOR_UM)
    ref = _profiled(bounds.fisher_matrix(rig, n_shots, seed))
    rows = [{'arm': 'free_space', 'dispersion_per_um': None, 'info': ref}]
    rows[0].update(cond=_cond(bounds.fisher_matrix(rig, n_shots, seed)))
    arms = {
        'fiber_no_monitor': (RIG_WAVELENGTHS_UM, None),
        'fiber_monitor_through_target': (four, None),
        'fiber_dichroic_monitor': (four, MONITOR_UM),
    }
    for arm, (lam, monitor) in arms.items():
        for disp in DISPERSIONS_PER_UM:
            cfg = dataclasses.replace(
                rig, wavelengths_um=lam, fiber=_fiber(4, 0.0, disp, monitor)
            )
            j = bounds.fisher_matrix(cfg, n_shots, seed)
            info = _profiled(j)
            rows.append(
                {'arm': arm, 'dispersion_per_um': disp, 'info': info, 'cond': _cond(j)}
            )
    for row in rows:
        row['info_rel_free_space'] = row['info'] / ref
    return {'rows': rows, 'monitor_um': MONITOR_UM, 'n_shots': n_shots}


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a, b]).double())[0, 1])


def q32(n_draws: int, seed: int) -> dict:
    """corr(P_perp, |kappa|^2) over static fiber draws, per mode count."""
    rows = []
    for n in MODES:
        gen = torch.Generator().manual_seed(seed)
        cfg = _fiber(n, 0.0, 0.0, None)
        draw = simulate_fiber(cfg, (0.635,), n_draws, 1, 1.0, CPU, gen)
        k2, perp = draw.kappa.abs().flatten() ** 2, draw.p_perp.flatten()
        r = _corr(perp, k2)
        rows.append(
            {
                'n_modes': n,
                'corr': r,
                # Large-sample (delta-method) standard error of a Pearson r.
                'corr_se': (1 - r * r) / math.sqrt(n_draws),
                'corr_times_sqrt_n': r * math.sqrt(n),
                'mean_kappa2': float(k2.mean()),
                'coe_mean_kappa2': 2 / (2 * n + 1),
                'mean_p_perp': float(perp.mean()),
            }
        )
    return {'rows': rows, 'n_draws': n_draws}


def drift_scan(n_modes: int, n_shots: int, seed: int) -> dict:
    """|corr(kappa(start), kappa(end))| over one rig shot, per drift rate."""
    rig = SimConfig.nominal(
        STEMLAB_125_14, RIG_WAVELENGTHS_UM, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
    )
    length = STEMLAB_125_14.buffer_size
    rows = []
    for drift in DRIFTS_RAD_PER_SQRT_S:
        gen = torch.Generator().manual_seed(seed)
        cfg = _fiber(n_modes, drift, 0.0, None)
        k = simulate_fiber(cfg, (0.635,), n_shots, length, rig.sample_rate, CPU, gen)
        a, b = k.kappa[:, 0, 0], k.kappa[:, 0, -1]
        num = (a * b.conj()).mean().abs()
        den = ((a.abs() ** 2).mean() * (b.abs() ** 2).mean()).sqrt()
        rows.append({'drift_rad_per_sqrt_s': drift, 'end_corr': float(num / den)})
    return {'n_modes': n_modes, 'shot_s': length / rig.sample_rate, 'rows': rows}


def main() -> None:
    """Parse arguments and write the fiber analytics artifact."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fisher-shots', type=int, default=16)
    parser.add_argument('--draws', type=int, default=4000)
    parser.add_argument('--drift-shots', type=int, default=400)
    parser.add_argument('--drift-modes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = {
        'q3_1': q31(args.fisher_shots, args.seed),
        'q3_2': q32(args.draws, args.seed),
        'drift': drift_scan(args.drift_modes, args.drift_shots, args.seed),
    }
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
