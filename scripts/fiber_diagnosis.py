"""Diagnose the q3_fiber zero-predictor scores and persist per-shot RMSEs.

Writes ``fiber_diagnosis.json`` (drift-free checkpoint probes and the
linear-Gaussian per-sample bound per arm, with each fiber impairment knocked
out in turn) and ``per_shot_rmse.json`` (per-shot velocity RMSE by cell key,
so paired standard errors are computable). CPU by default.

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
    drift_free_probe,
    fiber_bound,
    per_shot_table,
)
from smi.experiments.grids import GRIDS
from smi.experiments.runner import expand_grid, trained_checkpoint

# The nominal N = 3 / 4 cells the fiber arms and the R1 count are paired with.
_R1_KEYS = [
    f'{fam}__{ch}__nominal__seed0'
    for fam in ('tcan-44k', 'unettransformer-offline-44k')
    for ch in ('ch635-675-515', 'ch635-675-515-785')
]


def diagnose(n_shots: int, seed: int, device: str) -> dict:
    """Probe every fiber checkpoint and bound every q3_fiber arm."""
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
            f'fading={f}_speckle={s}_path={p}': fiber_bound(
                cell.sim, Knockouts(fading=f, speckle=s, path=p), n_shots, seed
            )
            for f, s, p in flags
        }
    return {'n_shots': n_shots, 'seed': seed, 'probes': probes, 'bounds': bounds}


def main() -> None:
    """Parse arguments and write both artifacts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--shots', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--skip-per-shot', action='store_true')
    args = parser.parse_args()
    out = diagnose(args.shots, args.seed, args.device)
    (RESULTS_DIR / 'fiber_diagnosis.json').write_text(json.dumps(out, indent=2) + '\n')
    if not args.skip_per_shot:
        cells = {'q3_fiber': [], 'q1_r1_count': _R1_KEYS, 'q2_r4_noise': []}
        table = per_shot_table(cells, GRID_RUNS_DIR, args.frozen_root, args.device)
        (RESULTS_DIR / 'per_shot_rmse.json').write_text(
            json.dumps(table, indent=1) + '\n'
        )


if __name__ == '__main__':
    main()
