"""End-to-end CLI path for synthetic training, including the test stage."""

import importlib.util

import yaml

from smi.core.paths import CONFIGS_DIR, REPO_ROOT

# scripts/ is not a package, so load the training script by path.
_spec = importlib.util.spec_from_file_location(
    'train_script', REPO_ROOT / 'scripts' / 'train.py'
)
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)

CONFIG = CONFIGS_DIR / 'models' / 'synthetic-tcn-config.yaml'


def test_synthetic_main_runs_test_stage(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg['model'].update(temporal_channels=[4, 4], dropout=0.0)
    cfg['training'].update(
        max_epochs=1,
        batch_size=2,
        accelerator='cpu',
        precision='32-true',
        use_logging=False,
    )
    cfg['synthetic'].update(steps_per_epoch=2, val_steps=1)
    cfg['data']['split_ratios'] = [80, 10, 10]
    path = tmp_path / 'tiny.yaml'
    path.write_text(yaml.safe_dump(cfg))
    train_script.train_model(str(path))
