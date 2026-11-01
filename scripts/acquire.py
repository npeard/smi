"""Acquire shots from the Red Pitaya rig into an HDF5 dataset.

Usage::

    pixi run python scripts/acquire.py --num_samples 5000 --dataset_name my-data.h5
"""

import argparse
import logging

from stemlab import STEMLAB_125_14_SPEC

from smi.acquisition.rig import (
    HOSTS,
    PRIMARY,
    SHOT_DELAY_S,
    SHOT_TIMEOUT_S,
    acquisition_config,
    connect,
)
from smi.acquisition.shots import run_shots
from smi.core.cli import add_verbosity, setup_logging
from smi.core.paths import DATA_DIR
from smi.theory.coil_driver import CoilDriver
from smi.theory.waveform import Waveform

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--num_samples', type=int, required=True, help='Number of shots to acquire.'
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Dataset filename under data/ (e.g. "my-data.h5").',
    )
    add_verbosity(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbosity)
    name = args.dataset_name
    if not name.endswith('.h5'):
        name += '.h5'
    path = DATA_DIR / name
    logger.info(f'Acquiring {args.num_samples} shots into {path}')

    manager = connect(HOSTS)
    # The drive timing comes from the board stemlab drives, not a copy of it.
    acquisition = acquisition_config(STEMLAB_125_14_SPEC)
    try:
        for board in range(len(HOSTS)):
            manager.blink_led(board, led=0, n_blinks=3, period_s=0.5)
        manager.reset()
        manager.configure_daisy_chain(PRIMARY)
        run_shots(
            manager,
            Waveform(start_freq=1, end_freq=1000, acquisition=acquisition),
            CoilDriver(),
            args.num_samples,
            path,
            SHOT_TIMEOUT_S,
            SHOT_DELAY_S,
        )
    finally:
        manager.close()


if __name__ == '__main__':
    main()
