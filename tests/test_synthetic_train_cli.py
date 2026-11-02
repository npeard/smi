"""End-to-end CLI path for synthetic training, including the test stage."""

import importlib.util

from smi.core.paths import REPO_ROOT
from smi.learning.configs import model_config

# scripts/ is not a package, so load the training script by path.
_spec = importlib.util.spec_from_file_location(
    'train_script', REPO_ROOT / 'scripts' / 'train.py'
)
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)

CONFIG = 'synthetic-tcn-config'


def test_synthetic_main_runs_test_stage():
    cfg = model_config(CONFIG)
    cfg.model_config.update(temporal_channels=[4, 4], dropout=0.0)
    cfg.training_config.update(
        max_epochs=1,
        batch_size=2,
        accelerator='cpu',
        precision='32-true',
        use_logging=False,
    )
    assert cfg.synthetic_config is not None
    cfg.synthetic_config.update(steps_per_epoch=2, val_steps=1)
    cfg.data_config['split_ratios'] = [80, 10, 10]
    train_script.train_model(cfg)
