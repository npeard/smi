"""CLI for :mod:`smi.experiments.gain_sweep`; see that module for what it does.

Usage::

    pixi run python scripts/gain_sweep.py --help
"""

import argparse
import logging
from pathlib import Path

from smi.core.paths import DATA_DIR, DATASETS, RESULTS_DIR
from smi.estimators.gain_sweep import gain_grid
from smi.estimators.phase_fit import DEFAULT_WINDOW_SAMPLES
from smi.experiments import gain_sweep

DEFAULT_OUTPUT = RESULTS_DIR / 'gain_sweep_results.json'


def main() -> None:
    """Parse arguments and regenerate the gain-sweep artifact.

    The defaults are the settings the committed artifact was produced with, so
    a bare invocation reproduces it.
    """
    parser = argparse.ArgumentParser(description=gain_sweep.__doc__.splitlines()[0])
    parser.add_argument(
        '--n-shots',
        type=int,
        default=1000,
        help='Shots per dataset. This is method 1, sub-millisecond per fit, '
        "so 1000 costs ~9 min per acquisition -- do not copy method 2's 200, "
        'which exists for a 5.85 s/shot GPU decode.',
    )
    parser.add_argument('--gain-low', type=float, default=gain_sweep.DEFAULT_GAIN_LOW)
    parser.add_argument(
        '--gain-high',
        type=float,
        default=gain_sweep.DEFAULT_GAIN_HIGH,
        help='Top of the swept range. The default reaches the 2024 '
        "calibration's k=33.42 so the same curve adjudicates it.",
    )
    parser.add_argument('--gain-step', type=float, default=gain_sweep.DEFAULT_GAIN_STEP)
    parser.add_argument(
        '--window-samples',
        type=int,
        default=DEFAULT_WINDOW_SAMPLES,
        help='Window length for the windowed measure. Pass 0 to report only '
        'the global fit, which is about twice as fast.',
    )
    parser.add_argument(
        '--delay-max-shift',
        type=int,
        default=8,
        help='Half-width in samples of the drive-to-photodiode delay check, '
        'swept at the winning gain. Pass 0 to skip it.',
    )
    parser.add_argument(
        '--no-per-shot',
        action='store_true',
        help='Skip the per-shot optimal-gain fit. That fit is the evidence '
        'that the residual is not a scale error, so the artifact is weaker '
        'without it.',
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    gain_sweep.run_gain_sweep(
        n_shots=args.n_shots,
        gains=gain_grid(args.gain_low, args.gain_high, args.gain_step),
        window_samples=args.window_samples or None,
        output=args.output,
        datasets=DATASETS,
        delay_max_shift=args.delay_max_shift,
        include_per_shot=not args.no_per_shot,
        data_dir=DATA_DIR,
    )


if __name__ == '__main__':
    main()
