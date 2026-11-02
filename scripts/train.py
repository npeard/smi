"""Train a model from a named config in smi.learning.configs.

Usage::

    pixi run python scripts/train.py --config matched/tcn-44k

Deliberately imports nothing from ``smi.acquisition``, so training does not
pull in the Red Pitaya stack.
"""

import argparse
import logging
import time

import lightning as lightning_module

from smi.core.cli import add_verbosity, setup_logging
from smi.learning.configs import MODEL_CONFIGS, model_config
from smi.learning.training_interface import TrainingConfig, TrainingInterface

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--config',
        required=True,
        choices=sorted(MODEL_CONFIGS),
        metavar='NAME',
        help='Registry name of the training config, e.g. matched/tcn-44k.',
    )
    add_verbosity(parser)
    return parser.parse_args()


def train_model(train_config: TrainingConfig) -> None:
    """Train and test the model ``train_config`` describes."""
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
    setup_logging(args.verbosity)
    train_model(model_config(args.config))


if __name__ == '__main__':
    main()
