"""Appendix: a low-excursion synthetic shot, free space beside fiber."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _example_shots import plot_synthetic
from _figcommon import parse_args


def main() -> None:
    plot_synthetic('low', parse_args(__doc__).output)


if __name__ == '__main__':
    main()
