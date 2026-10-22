import argparse
import logging
import sys
import time
from pathlib import Path

import lightning as lightning_module

from smi.analysis.generate_data import generate_dataset_from_rp
from smi.analysis.training_interface import TrainingConfig, TrainingInterface
from smi.analysis.tune_search import export_best_model, run_search
from smi.redpitaya.manager import RedPitayaManager


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
        default='./analysis/models/configs/tcn-config.yaml',
        help='Path to YAML config file for training.',
    )
    parser.add_argument(
        '--acquire_dataset',
        action='store_true',
        help='Acquire real data from Red Pitaya for training, validation, and test '
        'datasets',
    )
    parser.add_argument(
        '--search',
        action='store_true',
        help='Run a Ray Tune hyperparameter/architecture search over the config '
        '(list-valued fields become tune.choice search dimensions) instead of a '
        'single training run.',
    )
    parser.add_argument(
        '--num_search_samples',
        type=int,
        default=10,
        help='Number of configurations to sample during --search (default: 10).',
    )
    parser.add_argument(
        '--gpu_fraction',
        type=float,
        default=0.0,
        help='Fraction of a GPU each search trial requests (0 for CPU; '
        '~1/gpu_fraction trials pack per GPU on the rig).',
    )
    parser.add_argument(
        '--max_concurrent_trials',
        type=int,
        default=None,
        help='Cap on simultaneous search trials (limits CUDA-context overhead).',
    )
    parser.add_argument(
        '--search_alg',
        type=str,
        default='random',
        choices=['random', 'optuna'],
        help='Search algorithm for --search (default: random).',
    )
    parser.add_argument(
        '--export_best',
        type=str,
        default=None,
        help='After --search, export the best single-model trial to this '
        'TorchScript .pt path.',
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
    # Ensure dataset_name has .h5 extension
    if not dataset_name.endswith('.h5'):
        dataset_name += '.h5'

    # Set data directory using pathlib
    data_dir = Path(__file__).parent / 'analysis' / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Acquiring dataset from Red Pitaya using default connection')
    logger.info(f'Number of samples: {num_samples}')
    logger.info(f'Dataset filename: {dataset_name}')

    # Create Red Pitaya Manager with default connection settings
    rp_manager = RedPitayaManager(
        ['rp-f0c04a.local', 'rp-f0c026.local'],
        blink_on_connect=True,
        data_save_path=str(data_dir),
    )

    try:
        # Configure the Red Pitaya for acquisition
        logger.info('Configuring Red Pitaya for data acquisition...')
        rp_manager.reset_all()

        # Configure daisy chain for synchronized triggering
        rp_manager.configure_daisy_chain()

        # Create dataset
        generate_dataset_from_rp(
            rp_manager=rp_manager,
            output_dir=str(data_dir),
            num_samples=num_samples,
            dataset_filename=dataset_name,
        )
        logger.info('Dataset acquisition complete!')
    finally:
        # Ensure we close the connection to the Red Pitaya
        rp_manager.close_all()


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

    # Mode 2: Ray Tune hyperparameter/architecture search
    if args.search:
        if not args.config:
            raise ValueError('Config file is required for search mode')
        logger.info('Starting Ray Tune search with config: %s', args.config)
        best = run_search(
            args.config,
            num_samples=args.num_search_samples,
            gpu_fraction=args.gpu_fraction,
            max_concurrent_trials=args.max_concurrent_trials,
            search_alg=args.search_alg,
        )
        logger.info('Best config: %s', best.config)
        logger.info('Best metrics: %s', best.metrics)
        if args.export_best:
            path = export_best_model(best, args.export_best)
            logger.info('Exported best model to %s', path)
        sys.exit()

    # Mode 3: Training (requires TrainingConfig)
    if not args.config:
        raise ValueError('Config file is required for training mode')

    train_model(args.config, logger)


if __name__ == '__main__':
    # Training new model:
    # python main.py --config path/to/config.yaml

    # Acquiring real data from Red Pitaya:
    # python main.py --acquire_dataset --num_samples 5000 --dataset_name my-data.h5

    main()
