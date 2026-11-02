"""CLI for :mod:`smi.experiments.calibrate_sim`; see that module for what it does.

Usage::

    pixi run python scripts/calibrate_sim.py --help
"""

import argparse
import logging
from pathlib import Path

from smi.core.acquisition import BOARDS
from smi.experiments import calibrate_sim


def main() -> None:
    """Parse arguments and write the calibration JSON."""
    parser = argparse.ArgumentParser(description=calibrate_sim.__doc__.splitlines()[0])
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--n-shots', type=int, required=True)
    parser.add_argument('--units', choices=('signal', 'volts'), required=True)
    parser.add_argument('--board', choices=sorted(BOARDS), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    calibrate_sim.write_calibration(
        args.dataset, args.out, args.n_shots, args.units, BOARDS[args.board]
    )


if __name__ == '__main__':
    main()
