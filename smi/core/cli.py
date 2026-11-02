"""Command-line pieces shared by the entry-point scripts."""

import argparse
import logging


def add_verbosity(parser: argparse.ArgumentParser) -> None:
    """Add the ``--verbosity`` logging-level flag."""
    parser.add_argument(
        '--verbosity',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging verbosity level (default: INFO)',
    )


def setup_logging(level: str) -> None:
    """Configure root logging at ``level`` with the repo's timestamped format."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
