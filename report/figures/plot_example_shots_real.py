"""Appendix: real free-space shots, or a placeholder while they are blinded."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _example_shots import plot_real
from _figcommon import parse_args


def main() -> None:
    plot_real(parse_args(__doc__).output)


if __name__ == '__main__':
    main()
