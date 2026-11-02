"""CLI for :mod:`smi.experiments.baselines`; see that module for what it does.

Usage::

    pixi run python scripts/run_baselines.py --help
"""

import argparse
import logging
from pathlib import Path

import torch

from smi.core.paths import DATASETS, RESULTS_DIR
from smi.estimators.phase_fit import DEFAULT_WINDOW_SAMPLES
from smi.experiments import baselines

DEFAULT_OUTPUT = RESULTS_DIR / 'baseline_results.json'


def main() -> None:
    """Parse arguments and regenerate the baseline artifact."""
    # These defaults are the settings the committed artifact was produced
    # with, so a bare invocation reproduces it. They previously differed
    # (cpu, 1201 grid points), which meant the documented regeneration
    # command silently returned different numbers than the file it overwrote.
    parser = argparse.ArgumentParser(description=baselines.__doc__.splitlines()[0])
    parser.add_argument(
        '--n-shots',
        type=int,
        default=1000,
        help='Shots for method 1, which costs about a millisecond per shot.',
    )
    parser.add_argument(
        '--n-shots-method2',
        type=int,
        default=200,
        help='Shots for method 2. Kept well below --n-shots because the '
        'Viterbi decode is ~5.85 s per shot on cuda:0.',
    )
    parser.add_argument(
        '--windowing',
        choices=sorted(baselines.WINDOWING_MODES),
        default='fringe-matched',
        help="How method 1's windowed measure picks its window length. "
        "'fringe-matched' (the default, and what the committed artifact "
        "uses) scales W by each shot's own fringe rate; 'fixed' uses "
        '--window-samples for every shot.',
    )
    parser.add_argument(
        '--window-samples',
        type=int,
        default=DEFAULT_WINDOW_SAMPLES,
        help="Window length for method 1's windowed refit under "
        '--windowing=fixed. Pass 0 to report only the global fit.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Decode device. The Viterbi pass is a long sequential loop: '
        'seconds per shot on a GPU, minutes on CPU.',
    )
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grid-points', type=int, default=1601)
    parser.add_argument('--phase-grid-points', type=int, default=8)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    baselines.run_baselines(
        n_shots=args.n_shots,
        n_shots_method2=args.n_shots_method2,
        device=args.device,
        batch_size=args.batch_size,
        output=args.output,
        datasets=DATASETS,
        grid_points=args.grid_points,
        phase_grid_points=args.phase_grid_points,
        window_samples=args.window_samples or None,
        windowing=args.windowing,
    )


if __name__ == '__main__':
    main()
