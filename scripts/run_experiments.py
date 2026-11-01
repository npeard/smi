"""CLI for :mod:`smi.experiments.runner`; see that module for what it does.

Usage::

    pixi run python scripts/run_experiments.py --help
"""

import sys

from smi.experiments.runner import main

if __name__ == '__main__':
    sys.exit(main())
