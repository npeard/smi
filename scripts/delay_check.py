"""CLI for :mod:`smi.experiments.delay_check`; see that module for what it does.

Usage::

    pixi run python scripts/delay_check.py --help
"""

import argparse
import logging

from smi.core.paths import DATA_DIR, DATASETS
from smi.experiments import delay_check


def main() -> None:
    """Parse arguments and report the measured lag for every dataset."""
    parser = argparse.ArgumentParser(description='Velocity/photodiode delay check.')
    parser.add_argument('--n-shots', type=int, default=200)
    parser.add_argument('--max-lag', type=int, default=512)
    parser.add_argument('--smooth-samples', type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    delay_check.report_lags(
        DATASETS, DATA_DIR, args.n_shots, args.max_lag, args.smooth_samples
    )


if __name__ == '__main__':
    main()
