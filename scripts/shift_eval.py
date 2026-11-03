"""CLI for :mod:`smi.experiments.shift_eval`; see that module for what it does.

Usage::

    pixi run python scripts/shift_eval.py --knob feedback_c --values 0.05 0.2 0.5
"""

import argparse
import sys
from pathlib import Path

from smi.core.paths import FROZEN_DIR, GRID_RUNS_DIR, RESULTS_DIR
from smi.experiments.shift_eval import evaluate_shifts

# Best two noise-free smi-a3 families of q3_s1_families by velocity RMSE.
BEST_TWO = ('tcan-44k-T90', 'unettransformer-offline-44k-T90')


def main() -> None:
    """Parse arguments, score the shifts and print one line per record."""
    parser = argparse.ArgumentParser(description='Shifted-test-set evaluation.')
    parser.add_argument(
        '--knob',
        required=True,
        choices=('feedback_c', 'wavelength_scale', 'phase_offset'),
    )
    parser.add_argument('--values', type=float, nargs='+', required=True)
    parser.add_argument('--grid', default='q3_s1_families')
    parser.add_argument('--families', nargs='+', default=list(BEST_TWO))
    parser.add_argument('--out', type=Path, default=RESULTS_DIR / 'shift_results.json')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    records = evaluate_shifts(
        args.grid,
        tuple(args.families),
        args.knob,
        tuple(args.values),
        args.out,
        GRID_RUNS_DIR,
        FROZEN_DIR,
        args.device,
    )
    for r in records:
        sys.stdout.write(f'{r.cell_key}\t{r.metrics["velocity_rmse"]["mean"]:.5f}\n')


if __name__ == '__main__':
    main()
