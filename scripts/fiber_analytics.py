"""Q3.1 and Q3.2 analytic tier: fiber phase as a Fisher nuisance, and P_perp.

Q3.1: profiled velocity information and condition number of the local Fisher
matrix (``bounds.fisher_matrix`` with ``SimConfig.fiber``) for free space, a
fiber with no monitor, a fourth color through the fiber to the target, and
that color returned by a dichroic at the distal face (one color, and three),
against the dispersion of the fiber-phase drift; all fading-free upper bounds
(see ``q31``). Q3.2: corr(P_perp, |kappa|^2) against mode count over static
fully mixed fibers -- about -1.3 / sqrt(N) for N >= 8 -- with its Spearman
rank form, R^2 (the fraction of fading variance P_perp explains), the same
against |kappa| (what C scales with), and
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

import torch
from scipy.stats import pearsonr, spearmanr

from smi.core.acquisition import STEMLAB_125_14
from smi.core.paths import RESULTS_DIR
from smi.data.frozen import RIG_WAVELENGTHS_UM
from smi.experiments.analytics import correlation_condition
from smi.theory import bounds
from smi.theory.fiber import FiberConfig, simulate_fiber
from smi.theory.forward import NOMINAL_ALPHA, NOMINAL_FEEDBACK_C, SimConfig

DEFAULT_OUTPUT = RESULTS_DIR / 'fiber_results.json'
# Fourth color for the monitor arms: a common diode line clear of the rig's.
MONITOR_UM = 0.785
# Three dichroic monitor colors, as many as the target set (diode lines).
MONITORS_3_UM = (0.785, 0.808, 0.850)
MODES = (1, 2, 4, 8, 16, 32, 64)
DISPERSIONS_PER_UM = (0.0, 1e-3, 1e-2, 1e-1)
DRIFTS_RAD_PER_SQRT_S = (1.0, 3.0, 10.0, 30.0, 100.0)
CPU = torch.device('cpu')


def _fiber(
    n_modes: int, drift: float, dispersion: float, monitor: tuple[float, ...]
) -> FiberConfig:
    return FiberConfig(
        n_modes=n_modes,
        drift_rad_per_sqrt_s=drift,
        path_drift_um_per_sqrt_s=0.0,
        dispersion_per_um=dispersion,
        spectral_corr_width_um=0.01,
        monitor_wavelengths_um=monitor,
        pbs_channel=False,
    )


def _monitor_info(
    rig: SimConfig, monitors: tuple[float, ...], n_shots: int, seed: int
) -> float:
    """Fiber-rate information M of the monitor colors alone, u profiled.

    The v column is zero for a monitor-only set, so it is dropped first.
    """
    cfg = dataclasses.replace(
        rig, wavelengths_um=monitors, fiber=_fiber(4, 0.0, 0.0, monitors)
    )
    j = bounds.fisher_matrix(cfg, n_shots, seed)[1:, 1:]
    return bounds.profiled_information(j, -1)


def q31(n_shots: int, seed: int) -> dict:
    """Fisher table for the monitor arms against dispersion.

    At zero dispersion the target colors measure v + u with information S
    (the free-space value) and the monitors measure u alone with M, so the
    profiled velocity information is the parallel combination S M / (S + M),
    a fraction M / (S + M) of free space: ~1/4 for one monitor color beside
    three targets, rising toward 1/2 as the monitor set matches the target
    set (``predicted_rel_free_space``). Every fiber arm holds |kappa| = 1
    (genie, fading-free), so the monitor rows are upper bounds; and a
    static-rate u stands in for the path drift: a random walk would make the
    no-monitor degeneracy hold sample by sample (stronger still) and leave
    the monitor rows more optimistic.
    """
    rig = SimConfig.nominal(
        STEMLAB_125_14, RIG_WAVELENGTHS_UM, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
    )
    four = (*RIG_WAVELENGTHS_UM, MONITOR_UM)
    ref = bounds.profiled_information(bounds.fisher_matrix(rig, n_shots, seed), 0)
    rows = [{'arm': 'free_space', 'dispersion_per_um': None, 'info': ref}]
    rows[0].update(cond=correlation_condition(bounds.fisher_matrix(rig, n_shots, seed)))
    arms = {
        'fiber_no_monitor': (RIG_WAVELENGTHS_UM, ()),
        'fiber_monitor_through_target': (four, ()),
        'fiber_dichroic_monitor': (four, (MONITOR_UM,)),
        'fiber_dichroic_monitor_x3': (
            (*RIG_WAVELENGTHS_UM, *MONITORS_3_UM),
            MONITORS_3_UM,
        ),
    }
    predicted = {}
    for arm, (_, monitor) in arms.items():
        if monitor:
            m = _monitor_info(rig, monitor, n_shots, seed)
            predicted[arm] = {'monitor_info': m, 'rel': m / (ref + m)}
    for arm, (lam, monitor) in arms.items():
        for disp in DISPERSIONS_PER_UM:
            cfg = dataclasses.replace(
                rig, wavelengths_um=lam, fiber=_fiber(4, 0.0, disp, monitor)
            )
            j = bounds.fisher_matrix(cfg, n_shots, seed)
            info = bounds.profiled_information(j, 0)
            row = {
                'arm': arm,
                'dispersion_per_um': disp,
                'info': info,
                'cond': correlation_condition(j),
            }
            if arm in predicted and disp == 0.0:
                row['predicted_rel_free_space'] = predicted[arm]['rel']
                row['monitor_info'] = predicted[arm]['monitor_info']
            rows.append(row)
    for row in rows:
        row['info_rel_free_space'] = row['info'] / ref
    return {
        'rows': rows,
        'monitor_um': MONITOR_UM,
        'monitors_x3_um': MONITORS_3_UM,
        'n_shots': n_shots,
        'note': 'fading-free (|kappa| = 1) upper bounds; ramp-rate path drift',
    }


def q32(n_draws: int, seed: int) -> dict:
    """corr(P_perp, |kappa|^2) over static fiber draws, per mode count."""
    rows = []
    for n in MODES:
        gen = torch.Generator().manual_seed(seed)
        cfg = _fiber(n, 0.0, 0.0, ())
        draw = simulate_fiber(cfg, (0.635,), n_draws, 1, 1.0, CPU, gen)
        k2, perp = draw.kappa.abs().flatten() ** 2, draw.p_perp.flatten()
        k2_64, perp_64 = k2.double().numpy(), perp.double().numpy()
        r = float(pearsonr(perp_64, k2_64).statistic)
        r_amp = float(pearsonr(perp_64, k2.sqrt().double().numpy()).statistic)
        rows.append(
            {
                'n_modes': n,
                'corr': r,
                # Large-sample (delta-method) standard error of a Pearson r.
                'corr_se': (1 - r * r) / math.sqrt(n_draws),
                'corr_times_sqrt_n': r * math.sqrt(n),
                'spearman': float(spearmanr(perp_64, k2_64).statistic),
                # Fraction of fading (|kappa|^2) variance a linear P_perp fit explains.
                'r2': r * r,
                # |kappa| is what C and the visibility scale with.
                'corr_abs_kappa': r_amp,
                'r2_abs_kappa': r_amp * r_amp,
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
        cfg = _fiber(n_modes, drift, 0.0, ())
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
