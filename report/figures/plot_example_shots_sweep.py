"""Appendix: the three synthetic shot types through a low- and a high-fading fiber."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _example_shots import plot_sweep
from _figcommon import parse_args


def main() -> None:
    plot_sweep(parse_args(__doc__).output)


if __name__ == '__main__':
    main()
