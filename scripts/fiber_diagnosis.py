"""Diagnose the q3_fiber zero-predictor scores and persist per-shot RMSEs.

Writes ``fiber_diagnosis.json`` (drift-free checkpoint probes; the
linear-Gaussian perfectly-demodulated reference RMSE per arm, with each fiber
impairment knocked out in turn, and its MSE ratio to the free-space reference;
check A, a classical inverse-fit decode of static-fiber shots; check B and the
remedy test, checkpoints rescored on per-shot rescaled inputs) and
``per_shot_rmse.json`` (per-shot velocity RMSE by
cell key, so paired standard errors are computable). CPU by default.

Usage::

    pixi run python scripts/fiber_diagnosis.py --help
"""

import argparse
import itertools
import json
from pathlib import Path

from smi.core.paths import GRID_RUNS_DIR, RESULTS_DIR
from smi.data.frozen import DEFAULT_ROOT
from smi.experiments.diagnostics import (
    Knockouts,
    channel_stats,
    classical_static_check,
    drift_free_probe,
    fiber_reference,
    per_shot_table,
    rescale_probe,
    static_fiber,
)
from smi.experiments.grids import GRIDS
from smi.experiments.runner import expand_grid, trained_checkpoint

# The nominal N = 3 / 4 cells the fiber arms and the R1 count are paired with.
_R1_KEYS = [
    f'{fam}__{ch}__nominal__seed0'
    for fam in ('tcan-44k', 'unettransformer-offline-44k')
    for ch in ('ch635-675-515', 'ch635-675-515-785')
]


_CAVEATS = [
    'Not a bound in either direction: perfect demodulation (fringe order known)'
    ' is optimistic.',
    'Linear-stationary-Gaussian estimation is pessimistic for non-Gaussian'
    ' speckle and the non-stationary chirp; arm separations rest on the'
    " simulator's white dp/dt prior against the 1 Hz-1 kHz drive band.",
    'Fading enters only via the shot-averaged |kappa|^2: blind to in-shot fades'
    ' and cycle slips, and to P_perp (so fiber-pbs equals fiber).',
]
# Stats draw seed, disjoint from the scored shots' seed.
_STATS_SEED = 7
# Nominal-trained q1_r1_count cell key per fiber arm with matching channels.
_NOMINAL_FOR = {'fiber': 'ch635-675-515', 'fiber-4color': 'ch635-675-515-785'}


def checks(n_shots: int, seed: int, device: str) -> dict:
    """Checks A (classical decode), B (visibility rescale) and the remedy test."""
    r1 = {c.key: c for c in expand_grid(GRIDS['q1_r1_count'])}
    cells = list(expand_grid(GRIDS['q3_fiber']))
    nominal_sim = next(c.sim for c in cells if c.sim.fiber is None)
    check_a = {'nominal': classical_static_check(nominal_sim, n_shots, seed)}
    check_b: dict[str, dict] = {}
    remedy = {}
    for cell in cells:
        if cell.sim.fiber is None:
            continue
        own_stats = channel_stats(cell.sim, n_shots, _STATS_SEED)
        ckpt = trained_checkpoint(cell, 'q3_fiber', GRID_RUNS_DIR)
        remedy[cell.key] = rescale_probe(
            ckpt, cell.sim, own_stats, n_shots, seed, device
        )
        if cell.sim_name not in _NOMINAL_FOR:
            continue
        static = static_fiber(cell.sim)
        if cell.sim_name not in check_a:
            check_a[cell.sim_name] = classical_static_check(static, n_shots, seed)
        arm = check_b.setdefault(cell.sim_name, {})
        arm[f'fiber-trained:{cell.key}'] = rescale_probe(
            ckpt, static, own_stats, n_shots, seed, device
        )
        nom = r1[f'{cell.family}__{_NOMINAL_FOR[cell.sim_name]}__nominal__seed0']
        nom_ckpt = trained_checkpoint(nom, 'q1_r1_count', GRID_RUNS_DIR)
        arm[f'nominal-trained:{nom.key}'] = rescale_probe(
            nom_ckpt,
            static,
            channel_stats(nom.sim, n_shots, _STATS_SEED),
            n_shots,
            seed,
            device,
        )
    return {
        'rescale': 'per shot and channel: subtract the shot mean, divide by the'
        ' shot std, multiply by the pooled per-channel std and add the pooled'
        ' mean of the network training sim (seed 7 draw)',
        'check_a_inverse_fit_static': check_a,
        'check_b_static_rescaled': check_b,
        'remedy_as_trained_rescaled': remedy,
    }


def diagnose(n_shots: int, seed: int, device: str) -> dict:
    """Probe every fiber checkpoint and compute every q3_fiber arm's reference."""
    probes = {}
    bounds = {}
    for cell in expand_grid(GRIDS['q3_fiber']):
        if cell.sim.fiber is not None:
            ckpt = trained_checkpoint(cell, 'q3_fiber', GRID_RUNS_DIR)
            probes[cell.key] = drift_free_probe(ckpt, cell.sim, n_shots, seed, device)
        if cell.sim_name in bounds:
            continue
        flags = [(True, True, True)]
        if cell.sim.fiber is not None:
            flags = list(itertools.product((True, False), repeat=3))
        bounds[cell.sim_name] = {
            f'fading={f}_speckle={s}_path={p}': fiber_reference(
                cell.sim, Knockouts(fading=f, speckle=s, path=p), n_shots, seed
            )
            for f, s, p in flags
        }
    free = bounds['nominal']['fading=True_speckle=True_path=True']['rmse']
    for arm in bounds.values():
        for rec in arm.values():
            rec['mse_ratio_to_free_space'] = (free / rec['rmse']) ** 2
    return {
        'n_shots': n_shots,
        'seed': seed,
        'probes': probes,
        'reference_name': 'linear-Gaussian perfectly-demodulated reference RMSE',
        'reference_caveats': _CAVEATS,
        'reference_rmse': bounds,
        **checks(n_shots, seed, device),
    }


def main() -> None:
    """Parse arguments and write both artifacts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--shots', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--skip-per-shot', action='store_true')
    # The q3_fiber_sweep cells finish hours after the diagnosis is settled.
    parser.add_argument('--per-shot-only', action='store_true')
    args = parser.parse_args()
    if not args.per_shot_only:
        out = diagnose(args.shots, args.seed, args.device)
        diag = RESULTS_DIR / 'fiber_diagnosis.json'
        diag.write_text(json.dumps(out, indent=2) + '\n')
    if not args.skip_per_shot:
        cells = {
            'q3_fiber': [],
            'q1_r1_count': _R1_KEYS,
            'q2_r4_noise': [],
            'q3_fiber_sweep': [],
        }
        table = per_shot_table(cells, GRID_RUNS_DIR, args.frozen_root, args.device)
        (RESULTS_DIR / 'per_shot_rmse.json').write_text(
            json.dumps(table, indent=1) + '\n'
        )


if __name__ == '__main__':
    main()
