"""R8 flow matching: train a FlowLitModule config, or score the trained flows.

``scripts/flow.py`` is the CLI. Not a ``TrainingInterface`` mode: that
class's convergence contract (velocity-RMSE monitor, zero-predictor plateau
check) has no counterpart for a flow loss, which has no zero-output plateau.
Training here is a fixed epoch budget with the best checkpoint by
``val/flow_loss``.

Budget (CPU, 2 torch threads, others' jobs running): denoise 0.45 s per
crop-4096 step; the global posterior 0.72 s per full-shot training step and
0.53 s per validation step (median of 5), at 200 steps plus 20 validation
steps per epoch. With ``FLOW_EPOCHS`` that is ~1.6 h for the denoiser and
~1.7 h per posterior, ~5 h. Scoring is ~2.4 h: sampling costs 0.07 s
(denoiser) and 0.03 s (posterior) per draw-step per full shot, over 64 shots,
for the Q3.3 sampling check's 32 draws x 16 steps plus 16 x 32 and each
posterior's 64 x 16.
"""

import json
import logging
from pathlib import Path

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from smi.core.paths import GRID_RUNS_DIR, RUNS_DIR
from smi.experiments.flow_eval import (
    Sampling,
    score_audit,
    score_denoise,
    score_posterior,
    score_posterior_sign,
)
from smi.learning.configs import model_config
from smi.learning.flow_lit_module import FlowLitModule
from smi.learning.synthetic_lit_module import SyntheticIndexDataset

logger = logging.getLogger(__name__)

FLOW_RUNS = RUNS_DIR / 'flow'
GRID = 'q3_fiber'
DENOISE = 'flow/denoise-unettransformer-offline-44k'
POSTERIORS = {
    'nominal': 'flow/posterior-free-unettransformer-global-44k',
    'fiber': 'flow/posterior-fiber-unettransformer-global-44k',
}
# The conditioning-encoder flows (smi.learning.flow), scored under the same
# result keys with this suffix so the encoder-free scores stay as recorded.
ENCODER_SUFFIX = '_enc'
DENOISE_ENC = 'flow/denoise-unettransformer-offline-enc'
POSTERIORS_ENC = {
    'nominal': 'flow/posterior-free-unettransformer-global-enc',
    'fiber': 'flow/posterior-fiber-unettransformer-global-enc',
}
VARIANTS = {'': (DENOISE, POSTERIORS), ENCODER_SUFFIX: (DENOISE_ENC, POSTERIORS_ENC)}
# The denoiser's 32 draws are the top of score_denoise's draw-count check
# (8/16/32 are prefixes of them); the band and modality scores need enough
# draws per marginal for a 2-component BIC to mean something.
# The audit scores the posterior mean on two full sets per arm; 16 draws keep
# that under ~10 min per set, and both sets share it so their gap is fair.
AUDIT_SAMPLING = Sampling(n_samples=16, n_steps=16, per_call=8, seed=0)
DENOISE_SAMPLING = Sampling(n_samples=32, n_steps=16, per_call=4, seed=0)
POSTERIOR_SAMPLING = Sampling(n_samples=64, n_steps=16, per_call=8, seed=0)


def checkpoint(name: str) -> Path:
    """Where ``train`` leaves the best checkpoint of config ``name``."""
    return FLOW_RUNS / f'{name.replace("/", "__")}.ckpt'


def train(name: str, accelerator: str, seed: int) -> Path:
    """Train config ``name`` for its fixed epoch budget; return the checkpoint."""
    cfg = model_config(name)
    syn = cfg.synthetic_config
    assert syn is not None
    hp = cfg.training_config
    lightning.seed_everything(seed, workers=True)
    module = FlowLitModule(
        cfg.model_config,
        hp,
        syn['steps_per_epoch'],
        sim=syn['sim'],
        **syn['flow'],
        train_crop_length=syn['train_crop_length'],
        fit_norm_stats=True,
    )
    batch = hp['batch_size']
    best = ModelCheckpoint(
        dirpath=FLOW_RUNS / name.replace('/', '__'),
        filename='{epoch}',
        monitor='val/flow_loss',
        mode='min',
        save_top_k=1,
    )
    trainer = lightning.Trainer(
        max_epochs=hp['max_epochs'],
        callbacks=[best],
        logger=False,
        check_val_every_n_epoch=1,
        accelerator=accelerator,
        devices=1,
        precision='32-true',
    )
    trainer.fit(
        module,
        train_dataloaders=DataLoader(
            SyntheticIndexDataset(syn['steps_per_epoch'] * batch), batch_size=batch
        ),
        val_dataloaders=DataLoader(
            SyntheticIndexDataset(syn['val_steps'] * batch), batch_size=batch
        ),
    )
    out = checkpoint(name)
    out.write_bytes(Path(best.best_model_path).read_bytes())
    logger.info(f'{name}: best val/flow_loss {best.best_model_score} -> {out}')
    return out


def _load(name: str, device: str) -> FlowLitModule:
    module = FlowLitModule.load_from_checkpoint(
        checkpoint(name), map_location=device, fit_norm_stats=False
    )
    return module.eval()


def score(device: str, frozen_root: Path, out: Path, variant: str) -> None:
    """Score one ``VARIANTS`` entry's flows on the q3_fiber frozen sets.

    Written under that variant's suffixed keys, merged into ``out`` if it
    exists, so scoring one variant leaves the other's scores as recorded.
    """
    denoise, posteriors = VARIANTS[variant]
    results = json.loads(out.read_text()) if out.exists() else {}
    if not variant:
        results |= {
            'grid': GRID,
            'device': device,
            'torch_threads': torch.get_num_threads(),
        }
    results |= {
        f'q3_3_denoise{variant}': {
            'config': denoise,
            'device': device,
            **score_denoise(
                _load(denoise, device),
                GRID,
                GRID_RUNS_DIR,
                frozen_root,
                DENOISE_SAMPLING,
                device,
            ),
        },
        f'q3_4_posterior{variant}': {
            arm: {
                'config': name,
                'device': device,
                **score_posterior(
                    _load(name, device),
                    GRID,
                    arm,
                    GRID_RUNS_DIR,
                    frozen_root,
                    POSTERIOR_SAMPLING,
                    device,
                ),
            }
            for arm, name in posteriors.items()
        },
    }
    out.write_text(json.dumps(results, indent=2) + '\n')
    logger.info(f'wrote {out}')


def score_sign(device: str, frozen_root: Path, out: Path, variant: str) -> None:
    """Add one variant's sign-blind and sign-fixed posterior scores to ``out``.

    Merged under a new key so the original scores stay as recorded.
    """
    results = json.loads(out.read_text())
    results[f'q3_4_posterior_sign{variant}'] = {
        arm: {
            'config': name,
            'device': device,
            **score_posterior_sign(
                _load(name, device),
                GRID,
                arm,
                GRID_RUNS_DIR,
                frozen_root,
                POSTERIOR_SAMPLING,
                device,
            ),
        }
        for arm, name in VARIANTS[variant][1].items()
    }
    out.write_text(json.dumps(results, indent=2) + '\n')
    logger.info(f'wrote {out}')


def audit(device: str, frozen_root: Path, out: Path) -> None:
    """Add the encoder posteriors' audit (``flow_eval.score_audit``) to ``out``."""
    results = json.loads(out.read_text())
    results['q3_4_audit'] = {
        arm: {
            'config': name,
            'device': device,
            **score_audit(
                _load(name, device),
                GRID,
                arm,
                GRID_RUNS_DIR,
                frozen_root,
                AUDIT_SAMPLING,
                device,
            ),
        }
        for arm, name in POSTERIORS_ENC.items()
    }
    out.write_text(json.dumps(results, indent=2) + '\n')
    logger.info(f'wrote {out}')
