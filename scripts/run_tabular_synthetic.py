"""CLI for :mod:`smi.experiments.tabular_synthetic`; see that module for what it does.

Usage::

    pixi run python scripts/run_tabular_synthetic.py q3_s1_families
"""

import argparse
import logging
from pathlib import Path

from smi.core.paths import RESULTS_DIR
from smi.data.frozen import DEFAULT_ROOT
from smi.experiments import tabular_synthetic

DEFAULT_OUT = RESULTS_DIR / 'synthetic_results.json'
DEFAULT_LOCK = RESULTS_DIR / '.runner.lock'


def main() -> None:
    """Fit and score the tabular models on each named grid's sims."""
    doc = tabular_synthetic.__doc__
    assert doc is not None
    parser = argparse.ArgumentParser(description=doc.split('\n')[0])
    parser.add_argument('grids', nargs='+', help='grid names (smi.experiments.grids)')
    parser.add_argument('--models', nargs='+', default=list(tabular_synthetic.MODELS))
    parser.add_argument('--window', type=int, default=91)
    parser.add_argument('--n-train-shots', type=int, default=512)
    # 0, as the grids' network seeds; the test set is seed 1000.
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--frozen-root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    for name in args.grids:
        tabular_synthetic.run_tabular(
            name,
            tuple(args.models),
            args.window,
            args.out,
            n_train_shots=args.n_train_shots,
            train_seed=args.train_seed,
            frozen_root=args.frozen_root,
            lock_path=args.lock,
        )


if __name__ == '__main__':
    main()
