"""CLI for :mod:`smi.experiments.runner`; see that module for what it does.

Usage::

    pixi run python scripts/run_experiments.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

from smi.core.paths import GRID_RUNS_DIR, RESULTS_DIR
from smi.data.frozen import DEFAULT_ROOT
from smi.experiments import runner
from smi.experiments.grids import grid as shipped_grid

DEFAULT_OUT = RESULTS_DIR / 'synthetic_results.json'
DEFAULT_LOCK = RESULTS_DIR / '.runner.lock'


def main() -> int:
    """Run a grid or re-score recorded cells; 1 when another runner holds the lock."""
    parser = argparse.ArgumentParser(description=runner.__doc__.split('\n')[0])
    parser.add_argument(
        'grid',
        nargs='?',
        help='grid name, a module of smi.experiments.grids (omit with '
        '--rescore-metrics)',
    )
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        '--only', action='append', help='glob over cell keys; repeatable (OR)'
    )
    parser.add_argument(
        '--runs-dir',
        type=Path,
        help='grid run: its checkpoint directory (default runs/grids/<grid>); '
        '--rescore-metrics: the directory of grid directories (default runs/grids)',
    )
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        '--rescore-metrics',
        action='store_true',
        help='add RESCORED_METRICS to recorded cells instead of running a grid',
    )
    parser.add_argument(
        '--include-oracle',
        action='store_true',
        help='with --rescore-metrics, also re-score the inverse_fit oracle',
    )
    parser.add_argument(
        '--device', default='cpu', help='with --rescore-metrics: evaluation device'
    )
    args = parser.parse_args()
    if args.rescore_metrics == (args.grid is not None):
        parser.error('give either a grid or --rescore-metrics, not both or neither')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    try:
        if args.rescore_metrics:
            n = runner.rescore_metrics(
                args.out,
                runs_dir=args.runs_dir or GRID_RUNS_DIR,
                lock_path=args.lock,
                frozen_root=args.frozen_root,
                device=args.device,
                include_oracle=args.include_oracle,
            )
            sys.stdout.write(f'rescored {n} records\n')
            return 0
        n = runner.run_grid(
            shipped_grid(args.grid),
            args.out,
            runs_dir=args.runs_dir or GRID_RUNS_DIR / args.grid,
            lock_path=args.lock,
            frozen_root=args.frozen_root,
            only=args.only,
        )
    except runner.RunnerLockedError as e:
        sys.stderr.write(f'refusing to start: {e}\n')
        return 1
    sys.stdout.write(f'ran {n} cells\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
