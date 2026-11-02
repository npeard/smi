"""CLI for :mod:`smi.experiments.q1_binned`; see that module for what it does.

Usage::

    pixi run python scripts/q1_binned.py --help
"""

import argparse
from pathlib import Path

from smi.core.paths import RESULTS_DIR
from smi.experiments import q1_binned

DEFAULT_OUT = RESULTS_DIR / 'q1_binned_errors.json'


def main() -> None:
    """Parse arguments and write the Q1 binned-error JSON."""
    parser = argparse.ArgumentParser(description=q1_binned.__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    q1_binned.run_q1_binned(parser.parse_args().out)


if __name__ == '__main__':
    main()
