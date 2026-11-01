import argparse
import logging
import sys
import time

import lightning as lightning_module

from smi.core.paths import CONFIGS_DIR
from smi.learning.training_interface import TrainingConfig, TrainingInterface


def setup_logging(verbosity: str) -> None:
    """Setup logging configuration based on verbosity level."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }

    level = level_map.get(verbosity.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a model or acquire data from Red Pitaya'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=str(CONFIGS_DIR / 'models' / 'tcn-config.yaml'),
        help='Path to YAML config file for training.',
    )
    parser.add_argument(
        '--acquire_dataset',
        action='store_true',
        help='Acquire real data from Red Pitaya for training, validation, and test '
        'datasets',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        help='Number of samples to acquire. Required with --acquire_dataset.',
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        help='Dataset filename (e.g., "my-data.h5"). Required with --acquire_dataset.',
    )
    parser.add_argument(
        '--verbosity',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging verbosity level (default: INFO)',
    )
    # IP address is now handled by RedPitayaManager defaults
    return parser.parse_args()


def acquire_dataset(
    num_samples: int, dataset_name: str, logger: logging.Logger
) -> None:
    """Acquire real data from Red Pitaya hardware.

    Args:
        num_samples: Number of samples to acquire
        dataset_name: Filename for the dataset (e.g., "my-data.h5")
        logger: Logger instance for output
    """
    # The SMI acquisition loop was split out of the Red Pitaya manager when it
    # became the standalone stemlab library; smi.acquisition rebuilds it on top.
    raise NotImplementedError('rebuilt in smi.acquisition')


def train_model(config_path: str, logger: logging.Logger) -> None:
    """Train a model using the specified configuration.

    Args:
        config_path: Path to the YAML config file
        logger: Logger instance for output
    """
    train_config = TrainingConfig.from_yaml(config_path)

    # Seed everything (Python random, NumPy, PyTorch, CUDA) from time
    seed = int(time.time())
    lightning_module.seed_everything(seed)
    # Note that get_data_loaders in datasets.py uses a hardcoded seed of 42 for
    # deterministic splits
    logger.info(f'Using random seed: {seed}')

    # Create trainer
    trainer = TrainingInterface(
        config=train_config,
        experiment_name=train_config.training_config['experiment_name'],
    )

    # Start training
    trainer.train()
    trainer.test()
    # Close the wandb logger if it was configured
    if trainer.config.training_config['use_logging']:
        trainer.trainer.loggers[0].experiment.finish()


def main():
    args = parse_args()

    # Setup logging based on verbosity argument
    setup_logging(args.verbosity)
    logger = logging.getLogger(__name__)

    # Mode 1: Dataset Acquisition (no TrainingConfig needed)
    if args.acquire_dataset:
        if not args.num_samples:
            raise ValueError(
                '--num_samples argument is required when using --acquire_dataset. '
                'Please specify the number of samples to acquire.'
            )
        if not args.dataset_name:
            raise ValueError(
                '--dataset_name argument is required when using --acquire_dataset. '
                'Please specify the filename for the dataset.'
            )

        acquire_dataset(args.num_samples, args.dataset_name, logger)
        sys.exit()

    # Mode 2: Training (requires TrainingConfig)
    if not args.config:
        raise ValueError('Config file is required for training mode')

    train_model(args.config, logger)


if __name__ == '__main__':
    # Training new model:
    # python main.py --config path/to/config.yaml

    # Acquiring real data from Red Pitaya:
    # python main.py --acquire_dataset --num_samples 5000 --dataset_name my-data.h5

    main()
