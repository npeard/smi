"""Train a model from a YAML config.

Usage::

    pixi run python scripts/train.py --config configs/models/tcn-config.yaml

Deliberately imports nothing from ``smi.acquisition``, so training does not
pull in the Red Pitaya stack.
"""

import argparse
import logging
import time

import lightning as lightning_module

from smi.core.paths import CONFIGS_DIR
from smi.learning.training_interface import TrainingConfig, TrainingInterface

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--config',
        type=str,
        default=str(CONFIGS_DIR / 'models' / 'tcn-config.yaml'),
        help='Path to YAML config file for training.',
    )
    parser.add_argument(
        '--verbosity',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging verbosity level (default: INFO)',
    )
    return parser.parse_args()


def train_model(config_path: str) -> None:
    """Train and test the model described by the YAML config at ``config_path``."""
    train_config = TrainingConfig.from_yaml(config_path)

    # Seed everything (Python random, NumPy, PyTorch, CUDA) from time
    seed = int(time.time())
    lightning_module.seed_everything(seed)
    # Note that get_data_loaders in datasets.py uses a hardcoded seed of 42 for
    # deterministic splits
    logger.info(f'Using random seed: {seed}')

    trainer = TrainingInterface(
        config=train_config,
        experiment_name=train_config.training_config['experiment_name'],
    )
    trainer.train()
    trainer.test()
    # Close the wandb logger if it was configured
    if trainer.config.training_config['use_logging']:
        trainer.trainer.loggers[0].experiment.finish()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.verbosity,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    train_model(args.config)


if __name__ == '__main__':
    main()
