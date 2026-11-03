"""CLI for :mod:`smi.experiments.shift_eval`; see that module for what it does.

Usage::

    pixi run python scripts/shift_eval.py --knob feedback_c --values 0.05 0.2 0.5
"""

import argparse
import sys
from pathlib import Path

from smi.core.paths import FROZEN_DIR, GRID_RUNS_DIR, RESULTS_DIR
from smi.experiments.grids import grid as get_grid
from smi.experiments.shift_eval import FAMILIES, GRID, KNOBS, evaluate_shifts


def main() -> None:
    """Parse arguments, score the shifts and print one line per record."""
    parser = argparse.ArgumentParser(description='Shifted-test-set evaluation.')
    parser.add_argument('--knob', required=True, choices=KNOBS)
    parser.add_argument('--values', type=float, nargs='+', required=True)
    parser.add_argument('--grid', default=GRID)
    parser.add_argument('--families', nargs='+', default=list(FAMILIES))
    parser.add_argument('--out', type=Path, default=RESULTS_DIR / 'shift_results.json')
    parser.add_argument('--device', default='mps')
    parser.add_argument(
        '--test-seed', type=int, help="frozen test draw; default the grid's own"
    )
    args = parser.parse_args()
    test_seed = (
        get_grid(args.grid).test_set.seed if args.test_seed is None else args.test_seed
    )
    records = evaluate_shifts(
        args.grid,
        tuple(args.families),
        args.knob,
        tuple(args.values),
        args.out,
        GRID_RUNS_DIR,
        FROZEN_DIR,
        args.device,
        test_seed,
    )
    for r in records:
        sys.stdout.write(f'{r.cell_key}\t{r.metrics["velocity_rmse"]["mean"]:.5f}\n')


if __name__ == '__main__':
    main()
