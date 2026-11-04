"""Named training configs, one per architecture family and budget.

``MODEL_CONFIGS`` maps a name (e.g. ``'matched/tcn-44k'``, ``'tcn-config'``)
to a :class:`TrainingConfig`. The names are the stems, relative to
``configs/models``, of the YAML files these entries replaced, and each entry
resolves to the dicts ``yaml.safe_load`` gave for its file, except that the
stale ``data_dir`` is gone (real-data runs find files under
``smi.core.paths.DATA_DIR``) and the YAML-1.1 string numerics (``'1e-3'``)
are floats. That is load-bearing, not cosmetic: experiment cells fingerprint
the resolved dicts and resume matches recorded cells by fingerprint, so an
edit here moves the fingerprint of every grid cell that uses the entry.

Three groups:

* ``matched/<family>-{19k,44k}``: the synthetic Q1-Q3 families, every one at
  a 19,329- or 44,000-parameter budget (+/-10%, tests/test_matched_configs.py)
  and, where it has a finite field, a matched context: 85 samples of one-sided
  lookback (causal column) or Barland's 91-sample window (offline column).
* ``displacement/<family>-44k``: matched entries retargeted to displacement.
* ``matched/*-matched*``, ``tcn-repeated``, ``tcn-varying`` and the
  ``*-config`` entries: real-data runs at the older 57,377-parameter Barland
  budget, and the legacy Ray Tune search spaces, whose list-valued fields are
  ``tune.choice`` dimensions.
"""

import copy
import dataclasses
import difflib
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from smi.core.acquisition import STEMLAB_125_14
from smi.learning.training_interface import TrainingConfig
from smi.theory.fiber import FiberConfig
from smi.theory.forward import SimConfig

SEQUENCE_LENGTH = 16384
# Must exceed the receptive field with margin (test_matched_configs.MARGIN).
MATCHED_CROP = 4096
VELOCITY_MONITOR = 'val/velocity_rmse_um_per_ms'
DISPLACEMENT_MONITOR = 'val/displacement_rmse_um'
LEGACY_DATASETS = ['free-space-synchro_10k.h5', 'mmfiber-synchro_10k.h5']
REAL_DATASET = 'free-space-synchro_10k.h5'


def _sim(feedback_c: float, alpha: float) -> dict[str, Any]:
    """Every SimConfig field, no defaults (smi.theory.forward.SimConfig).

    The experiment runner replaces the whole block per cell.
    """
    return {
        'acquisition': 'STEMLAB_125_14',  # smi.core.acquisition.BOARDS
        'wavelengths_um': [0.635, 0.6748, 0.515],
        'sample_rate': 488281.25,  # 125 MHz / decimation 256
        'start_freq_hz': 1.0,
        'end_freq_hz': 1000.0,
        'drive_gain': 0.002454,  # NOMINAL_DRIVE_GAIN
        'feedback_c': feedback_c,
        'alpha': alpha,
        'snr_db': None,
        'shot_noise': False,
        'visibility_range': [1.0, 1.0],
        'dc_offset_range': [0.0, 0.0],
        'amplitude_drift_rel': 0.0,
        'phase_drift_rad_per_sqrt_s': 0.0,
        'contrast_drift_rel': 0.0,
        'pd_bandwidth_hz': None,
        'adc_bits': None,
        'adc_full_scale': 1.0,
    }


def _training(**overrides: Any) -> dict[str, Any]:
    """The matched training block, with ``overrides`` replacing its fields.

    ``experiment_name`` is an override, or :func:`_named` sets it from the
    registry key.

    The experiment grid overrides device and precision, and the runner derives
    ``convergence.monitor`` from the loss and target.
    """
    return {
        'max_epochs': 30,
        'check_val_every_n_epoch': 1,  # validate every epoch; patience counts these
        # Stop on a plateau; max_epochs stays the cap.
        'convergence': {
            'monitor': VELOCITY_MONITOR,
            'patience_epochs': 10,
            'min_rel_delta': 0.01,  # improvement below 1% of the best counts as none
            # No stop sooner: S1 zero-output stalls lasted ~15 epochs, then escaped.
            'min_epochs': 30,
        },
        'warmup_epochs': 0,
        'T_0': 30,  # initial restart period (epochs for the first cosine cycle)
        'T_mult': 1,  # 1 = constant restart period, 2 = doubling
        'batch_size': 16,
        'optimizer': 'Adam',  # 'Adam', 'SGD' or 'Muon'
        'weight_decay': 1e-6,  # Adam only
        'momentum': 0.9,  # SGD only
        'learning_rate': 1e-3,
        'eta_min': 1e-6,
        'accelerator': 'cpu',
        'precision': '32-true',
        'devices': 1,
        'use_logging': False,
        'wandb_project': 'smi',
        'target': 'velocity',  # what the model predicts: velocity or displacement
        **overrides,
    }


def _loss(**overrides: Any) -> dict[str, Any]:
    # dynamic: dynamic loss weighting, which overrides the weights; usually
    # not better than fixed ones. The experiment runner sets sign_invariant
    # per sim.
    return {
        'dynamic': False,
        'velocity_loss_weight': 1.0,
        'displacement_loss_weight': 0.0,
        'sign_invariant': False,
        **overrides,
    }


def _matched(model: dict[str, Any], train_crop_length: int | None) -> TrainingConfig:
    """A synthetic matched-budget entry.

    ``model['in_channels']`` is a placeholder: synthetic training replaces it
    with ``len(synthetic.sim.wavelengths_um)``.
    """
    return TrainingConfig(
        model_config=model,
        training_config=_training(),
        data_config={
            'dataset_file': 'unused',  # not used in synthetic mode
            'split_ratios': [80, 10, 10],
            'num_workers': 0,
        },
        loss_config=_loss(),
        synthetic_config={
            'use_synthetic_training': True,
            'steps_per_epoch': 200,
            'train_crop_length': train_crop_length,
            'val_steps': 20,
            'sim': _sim(feedback_c=0.2, alpha=3.0),
        },
    )


def _displacement(config: TrainingConfig) -> TrainingConfig:
    """``config`` retargeted to displacement: only the target, its monitor and
    the two loss weights change, so the velocity-target S1 cells keep their
    fingerprints and keys.
    """
    out = copy.deepcopy(config)
    out.training_config['target'] = 'displacement'
    out.training_config['convergence']['monitor'] = DISPLACEMENT_MONITOR
    out.loss_config.update(velocity_loss_weight=0.0, displacement_loss_weight=1.0)
    return out


def _real(model: dict[str, Any]) -> TrainingConfig:
    """A real-data entry at the 57,377-parameter Barland budget."""
    return TrainingConfig(
        model_config=model,
        training_config=_training(accelerator='gpu', devices=[0]),
        data_config={
            'dataset_file': REAL_DATASET,
            'split_ratios': [80, 10, 10],
            'num_workers': 7,
        },
        loss_config=_loss(),
    )


def _legacy(
    experiment_name: str,
    model: dict[str, Any],
    training: dict[str, Any],
    loss: dict[str, Any],
) -> TrainingConfig:
    """A legacy real-data search space: 200 epochs on both datasets, logged."""
    return TrainingConfig(
        model_config=model,
        training_config=_training(
            experiment_name=experiment_name,
            max_epochs=200,
            T_0=200,
            accelerator='gpu',
            use_logging=True,
            **training,
        ),
        data_config={
            'dataset_file': LEGACY_DATASETS,
            'split_ratios': [[80, 10, 10]],  # train, val, test percentages
            'num_workers': 7,
        },
        loss_config=_loss(**loss),
    )


def _named(entries: dict[str, TrainingConfig]) -> dict[str, TrainingConfig]:
    """``entries`` with each one logging under its key, dashes as underscores."""
    for stem, config in entries.items():
        config.training_config['experiment_name'] = stem.replace('-', '_')
    return entries


def _conv(type_: str, kernel_size: int, temporal_channels: list[int]) -> dict[str, Any]:
    """A matched TCN or SCNN; dilations 1-2-4 for three blocks."""
    return {
        'type': type_,
        'sequence_length': SEQUENCE_LENGTH,
        'in_channels': 3,
        'activation': 'GELU',
        'use_layer_norm': True,
        'use_weight_norm': False,
        'kernel_size': kernel_size,
        'temporal_channels': temporal_channels,
        'dilation_base': 2,
        'dropout': 0.0,
    }


def _lstm(hidden_size: int, bidirectional: bool) -> dict[str, Any]:
    return {
        'type': 'LSTM',
        'sequence_length': SEQUENCE_LENGTH,
        'in_channels': 3,
        'hidden_size': hidden_size,
        'num_layers': 2,
        'dropout': 0.0,
        'bidirectional': bidirectional,
    }


def _mamba(d_model: int, d_state: int, expand: int, num_layers: int) -> dict[str, Any]:
    return {
        'type': 'Mamba',
        'sequence_length': SEQUENCE_LENGTH,
        'in_channels': 3,
        'd_model': d_model,  # internal feature dimension per Mamba block
        'd_state': d_state,  # SSM state dimension (N in the paper)
        'd_conv': 4,  # causal depthwise conv kernel inside each block
        'expand': expand,  # inner dim = d_model * expand
        'num_layers': num_layers,  # stacked Mamba blocks
        'use_layer_norm': True,
    }


def _mlp(
    window_size: int, hidden_sizes: list[int], window_stride: int
) -> dict[str, Any]:
    return {
        'type': 'MLP',
        'window_size': window_size,
        'in_channels': 3,
        'hidden_sizes': hidden_sizes,
        'activation': 'GELU',
        'dropout': 0.0,
        'window_stride': window_stride,
    }


def _barland(channels: list[int]) -> dict[str, Any]:
    # Stride 8, not the window size. At stride == window_size the overlap
    # average in SlidingWindowModel.forward has nothing to average, so the
    # output is a staircase of 256-sample constants; the lowest velocity RMSE
    # any such predictor can reach on the real split is 0.1866 um/ms, which is
    # what the earlier stride-256 row measured instead of the architecture. The
    # floor at stride 8 is 0.0065 um/ms, 17x below the TCN's 0.1116, so it
    # cannot bind the comparison. Stride 1 would be exact but costs ~6 min per
    # epoch (memory-bound on the unfold).
    return {
        'type': 'Barland',
        'window_size': 256,
        'in_channels': 3,
        'activation': 'LeakyReLU',
        'dropout': 0.0,
        'use_weight_norm': False,
        'window_stride': 8,
        'channels': channels,
    }


def _neural_operator(width: int, kernel_len: int | None) -> dict[str, Any]:
    return {
        'type': 'NeuralOperator',
        'sequence_length': SEQUENCE_LENGTH,
        'in_channels': 3,
        'width': width,
        'n_layers': 4,
        'modes': 8,
        'kernel_len': kernel_len,
    }


def _patch_transformer(
    d_model: int, ffn_dim: int, attention_span: int, causal: bool
) -> dict[str, Any]:
    return {
        'type': 'PatchTransformer',
        'sequence_length': SEQUENCE_LENGTH,
        'in_channels': 3,
        'patch_len': 16,
        'stride': 10,
        'd_model': d_model,
        'n_heads': 4,
        'n_layers': 2,
        'ffn_dim': ffn_dim,
        'attention_span': attention_span,
        'dropout': 0.0,
        'causal': causal,
    }


def _unet_transformer(
    channels: list[int], ffn_dim: int, attention_span: int, causal: bool
) -> dict[str, Any]:
    return {
        'type': 'UNetTransformer',
        'sequence_length': SEQUENCE_LENGTH,
        'in_channels': 3,
        'channels': channels,
        'kernel_size': 4,
        'pool': 2,
        'n_heads': 4,
        'n_layers': 2,
        'ffn_dim': ffn_dim,
        'attention_span': attention_span,
        'dropout': 0.0,
        'causal': causal,
    }


def _tcan(siamese_channels: list[int], decoder_channels: list[int]) -> dict[str, Any]:
    # Encoder field 37 + decoder field 49 - 1 shared centre = 85.
    return {
        'type': 'TCAN',
        'sequence_length': SEQUENCE_LENGTH,
        'in_channels': 3,
        'activation': 'GELU',
        'use_layer_norm': True,
        'use_weight_norm': False,
        'dropout': 0.0,
        'siamese_kernel_size': 7,
        'siamese_channels': siamese_channels,
        'siamese_dilation_base': 2,
        'atten_heads': 4,  # must divide siamese_channels[-1]
        'decoder_kernel_size': 7,
        'decoder_channels': decoder_channels,
        'decoder_dilation_base': 3,
    }


_MATCHED: dict[str, TrainingConfig] = _named(
    {
        # Causal column: one-sided lookback of 85 samples, the tcn-44k value
        # (kernel 7, dilations 1-2-4).
        'tcn-19k': _matched(_conv('TCN', 7, [16, 16, 32]), MATCHED_CROP),
        'tcn-44k': _matched(_conv('TCN', 7, [16, 32, 48]), MATCHED_CROP),
        'tcan-19k': _matched(_tcan([16, 16], [16, 20]), MATCHED_CROP),
        'tcan-44k': _matched(_tcan([24, 24], [24, 32]), MATCHED_CROP),
        # Offline column: symmetric field of 85 samples (kernel 7, dilations
        # 1-2-4), within 10% of Barland's 91, which is fixed by the published
        # network. An odd kernel is required: symmetric padding cannot preserve
        # length otherwise. Widths as the TCN of the same budget.
        'scnn-19k': _matched(_conv('SCNN', 7, [16, 16, 32]), MATCHED_CROP),
        'scnn-44k': _matched(_conv('SCNN', 7, [16, 32, 48]), MATCHED_CROP),
        # Offline column: one window spans 91 samples, set by the published
        # network's fixed depth and kernels, and every other offline family's
        # field is matched to that window. Averaging every window that covers an
        # output widens Barland's true dependency span to 498 samples (its
        # receptive_field).
        'barland-19k': _matched(_barland([24, 24, 24, 32]), MATCHED_CROP),
        'barland-44k': _matched(_barland([24, 24, 48, 64]), MATCHED_CROP),
        # Offline column: one window spans 91 samples (the shared offline field),
        # matched to Barland's published window. Averaging every window that
        # covers an output widens the MLP's true dependency span to 181 samples
        # (its receptive_field). Stride 1: point-to-point, see mlp-matched.
        'mlp-19k': _matched(_mlp(91, [66, 16], 1), MATCHED_CROP),
        'mlp-44k': _matched(_mlp(91, [130, 64], 1), MATCHED_CROP),
        # LSTM and Mamba: lookback is unbounded (recurrent state), so context
        # cannot be matched; only the parameter budget is; the model reports
        # receptive_field None. A unidirectional LSTM and Mamba are causal; a
        # bidirectional LSTM is offline.
        'lstm-19k': _matched(_lstm(39, bidirectional=False), MATCHED_CROP),
        'lstm-44k': _matched(_lstm(59, bidirectional=False), MATCHED_CROP),
        'lstm-bidir-19k': _matched(_lstm(24, bidirectional=True), MATCHED_CROP),
        'lstm-bidir-44k': _matched(_lstm(36, bidirectional=True), MATCHED_CROP),
        'mamba-19k': _matched(_mamba(24, 8, 2, 2), MATCHED_CROP),
        'mamba-44k': _matched(_mamba(48, 16, 1, 3), MATCHED_CROP),
        # Causal column: one-sided lookback of 85 samples, the tcn-44k value:
        # n_layers (kernel_len - 1) + 1 = 4 * 21 + 1. Each layer's kernel lives on
        # lags 0..21 and is the inverse DFT of 8 complex coefficients per channel
        # pair, so the budget is set by modes, not kernel_len. Applied by
        # zero-padded FFT (no circular wrap).
        'neuraloperator-causal-19k': _matched(_neural_operator(17, 22), MATCHED_CROP),
        'neuraloperator-causal-44k': _matched(_neural_operator(26, 22), MATCHED_CROP),
        # Offline column, unbounded context: the spectral convolution spans the
        # whole input (8 retained modes), so every output sees every sample and
        # context cannot be matched; only the parameter budget is (like
        # lstm-bidir); the model reports receptive_field None. Same width, depth
        # and modes as the causal variant, hence the same parameter count.
        # Trains on full shots (crop None, as the model's requires_full_length
        # demands; the experiment runner sets that itself from the model, whatever
        # its grid says), so a step carries ~4x the samples of the 4096-crop
        # siblings: a training-budget confound, to be quoted with the "matched on
        # budget only" caveat.
        'neuraloperator-offline-19k': _matched(_neural_operator(17, None), None),
        'neuraloperator-offline-44k': _matched(_neural_operator(26, None), None),
        # Causal column: worst-phase one-sided lookback of 85 samples, the tcn-44k
        # value: (n_layers * attention_span + 1) * stride + patch_len - 1
        # = (2 * 3 + 1) * 10 + 16 - 1.
        'patchtransformer-causal-19k': _matched(
            _patch_transformer(32, 56, 3, causal=True), MATCHED_CROP
        ),
        'patchtransformer-causal-44k': _matched(
            _patch_transformer(48, 104, 3, causal=True), MATCHED_CROP
        ),
        # Offline column: symmetric field of 96 samples, +5.5% of Barland's 91:
        # 2 * n_layers * attention_span * stride + patch_len = 2 * 4 * 10 + 16.
        # Same patches as the causal variant; span 2, not 3, since both sides are
        # attended.
        'patchtransformer-offline-19k': _matched(
            _patch_transformer(32, 56, 2, causal=False), MATCHED_CROP
        ),
        'patchtransformer-offline-44k': _matched(
            _patch_transformer(48, 104, 2, causal=False), MATCHED_CROP
        ),
        # Causal column: worst-phase one-sided lookback of 85 samples, the tcn-44k
        # value: (K - 1)(G + S) + (N + 2) P - 1 with K = 4, pool p = 2, D = 2
        # levels (P = 4, G = 3, S = 7) and N = n_layers * attention_span = 12:
        # 3 * 10 + 14 * 4 - 1. Pooling latency makes it phase-dependent: 82 at
        # best.
        'unettransformer-causal-19k': _matched(
            _unet_transformer([12, 16, 28], 48, 6, causal=True), MATCHED_CROP
        ),
        'unettransformer-causal-44k': _matched(
            _unet_transformer([16, 28, 40], 80, 6, causal=True), MATCHED_CROP
        ),
        # Offline column: total width 81..85 over phase, right-heavy by 8..12
        # samples (even K); the worst phase, 85, is -6.6% of Barland's 91:
        # (ceil((K - 1) G / P) + 1 + 2N) P + (K - 1) S = (3 + 1 + 12) * 4 + 3 * 7,
        # with the causal variant's K, pool and levels and N = 2 * 3. Span 3, not
        # 6, since both sides are attended; the window (7 tokens) and so the
        # parameter count equal the causal variant's.
        'unettransformer-offline-19k': _matched(
            _unet_transformer([12, 16, 28], 48, 3, causal=False), MATCHED_CROP
        ),
        'unettransformer-offline-44k': _matched(
            _unet_transformer([16, 28, 40], 80, 3, causal=False), MATCHED_CROP
        ),
        # Real-data runs at the Barland budget itself: 57,377 parameters at
        # in_channels=3, receptive field 91. Barland's use_weight_norm is false
        # (unlike barland-config): the weight-norm reparametrization adds 176
        # g-scalars, and the 57,377 ceiling every other config is held to is the
        # unnormalized count.
        'barland-matched': _real(_barland([16, 32, 64, 64])),
        # Matched to the Barland budget on BOTH axes: 44,049 parameters (under
        # the ceiling) and a receptive field of 85 samples. Matching on parameters
        # alone would confound capacity with context -- at a fixed budget the TCN
        # field ranges from 37 to 373 samples. The SCNN counterpart differs only
        # in type, so causality is the only variable on that leg.
        'tcn-matched': _real(_conv('TCN', 7, [16, 32, 48])),
        'scnn-matched': _real(_conv('SCNN', 7, [16, 32, 48])),
        # Second capacity point at the SAME receptive field (85 samples), at
        # 19,329 parameters instead of 44,049. Holding the field fixed while
        # moving only the width is what separates "needs capacity" from "needs
        # context".
        'tcn-matched-small': _real(_conv('TCN', 7, [16, 16, 32])),
        'scnn-matched-small': _real(_conv('SCNN', 7, [16, 16, 32])),
        # All-varying widths [16, 32, 64] (27,985 parameters) against a repeated
        # first width [32, 32, 64] (35,585), both at receptive field 29: whether
        # repeating a channel width earns its place.
        'tcn-varying': _real(_conv('TCN', 3, [16, 32, 64])),
        'tcn-repeated': _real(_conv('TCN', 3, [32, 32, 64])),
        # The simple learned floor at the matched budget: 51,329 parameters, close
        # under the ceiling; receptive field is the window exactly (256). Stride 1
        # makes it a point-to-point map from local input to velocity -- "a CNN
        # with kernel size 1" -- which is the question the MLP row answers: how
        # much does windowed context buy over none? At stride 256 the output was a
        # staircase of 256-sample constants and the row measured that
        # discretization rather than the model.
        'mlp-matched': _real(_mlp(256, [64, 32], 1)),
    }
)

# Search-space TCN/SCNN: lists are tune.choice dimensions.
_LEGACY_CONV = {
    'sequence_length': SEQUENCE_LENGTH,
    'activation': 'GELU',
    'use_layer_norm': True,
    'use_weight_norm': True,
    'dropout': [0, 0.05],
    'in_channels': 3,
    'kernel_size': [3, 5, 7],
    'temporal_channels': [[16, 16, 32, 32, 64, 64]],
    'dilation_base': [2, 3, 4],
}

_LEGACY: dict[str, TrainingConfig] = {
    'tcn-config': _legacy(
        'tcn_pareto_recep_scan_2',
        {'type': 'TCN', **_LEGACY_CONV},
        {'batch_size': 128, 'optimizer': 'Muon'},
        {'displacement_loss_weight': [0, 0.1]},
    ),
    'scnn-config': _legacy(
        'scnn_pareto_recep_scan_2',
        {'type': 'SCNN', **_LEGACY_CONV},
        {'batch_size': 128, 'optimizer': 'Muon'},
        {'displacement_loss_weight': [0, 0.1]},
    ),
    'tcan-config': _legacy(
        'tcan_pareto_setup',
        {
            'type': 'TCAN',
            'sequence_length': SEQUENCE_LENGTH,
            'in_channels': 3,
            'activation': 'GELU',
            'use_layer_norm': True,
            'use_weight_norm': True,
            'dropout': 0.0,
            # Siamese encoder: shared-weight TCN applied to each input channel.
            'siamese_kernel_size': 7,
            'siamese_channels': [[16, 16, 16, 16], [8, 8, 16, 16]],
            'siamese_dilation_base': 3,
            # Cross-channel attention at each time step.
            'atten_heads': [4, 8],  # must divide siamese_channels[-1]
            # Decoder: one TCN from all attended channel features to 1 output.
            'decoder_kernel_size': 7,
            'decoder_channels': [[16, 16, 8, 8], [16, 16, 8, 8]],
            'decoder_dilation_base': 2,
        },
        {'batch_size': 64, 'optimizer': 'Muon'},
        {},
    ),
    # The baseline model from Barland et al. 2021.
    'barland-config': _legacy(
        'barland_pareto_test',
        {
            'type': 'Barland',
            'window_size': 256,
            'activation': 'LeakyReLU',
            'in_channels': 3,
            'dropout': 0.1,
            'use_weight_norm': True,
            'window_stride': 256,
            'channels': [16, 32, 64, 64],
        },
        {'batch_size': 256, 'optimizer': ['Adam', 'Muon']},
        {},
    ),
    'lstm-config': _legacy(
        'lstm_setup',
        {
            'type': 'LSTM',
            'sequence_length': SEQUENCE_LENGTH,
            'in_channels': 3,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': False,  # true for non-causal inference
        },
        {'batch_size': 64, 'optimizer': 'Muon', 'learning_rate': 1e-2},
        {'displacement_loss_weight': [0.0, 0.1]},
    ),
    'mamba-config': _legacy(
        'mamba_baseline', _mamba(32, 16, 1, 3), {'batch_size': 32}, {}
    ),
    # Windowed MLP baseline, the simplest learned model in the comparison:
    # windowed exactly as BarlandCNN is (256 samples in, one velocity out), so
    # its number answers "architecture vs architecture". One dataset, unlogged.
    'mlp-config': TrainingConfig(
        model_config=_mlp(256, [64, 32], 256),
        training_config=_training(
            experiment_name='mlp_baseline',
            max_epochs=200,
            T_0=200,
            batch_size=128,
            accelerator='gpu',
        ),
        data_config={
            'dataset_file': REAL_DATASET,
            'split_ratios': [80, 10, 10],
            'num_workers': 7,
        },
        loss_config=_loss(),
    ),
    # Synthetic, Michelson, noise-free: each detector knob at its
    # exact-identity value. in_channels is derived from the sim's wavelengths.
    'synthetic-tcn-config': TrainingConfig(
        model_config={
            'type': 'TCN',
            'sequence_length': SEQUENCE_LENGTH,
            'activation': 'GELU',
            'use_layer_norm': True,
            'use_weight_norm': True,
            'dropout': 0.05,
            'kernel_size': 3,
            'temporal_channels': [[16, 16, 32, 32, 64, 64]],
            'dilation_base': 2,
        },
        training_config=_training(
            experiment_name='synthetic_tcn',
            max_epochs=200,
            warmup_epochs=5,
            T_0=200,
            batch_size=128,
        ),
        data_config={
            'dataset_file': 'unused',
            'split_ratios': [[80, 10, 10]],
            'num_workers': 0,
        },
        loss_config=_loss(displacement_loss_weight=0.1),
        synthetic_config={
            'use_synthetic_training': True,
            'steps_per_epoch': 1000,
            'train_crop_length': None,  # the full 16384-sample shot
            'val_steps': 200,
            'sim': _sim(feedback_c=0.0, alpha=0.0),
        },
    ),
}

# The R7 fiber operating point (the plan's Task-3 grid point): N = 8 modes,
# speckle drift 3 rad/sqrt(s), path drift 2.8 um/sqrt(s), no monitor. The
# q3_fiber grid's arms and the R8 flows share this one definition; configs
# cannot import the grid (learning sits below experiments).
R8_FIBER = FiberConfig(
    n_modes=8,
    drift_rad_per_sqrt_s=3.0,
    path_drift_um_per_sqrt_s=2.8,
    dispersion_per_um=0.0,
    spectral_corr_width_um=0.01,
    monitor_wavelengths_um=(),
    pbs_channel=False,
)


def _flow(
    task: str,
    model: dict[str, Any],
    fiber: FiberConfig | None,
    crop: int | None,
    encoder_channels: int,
) -> TrainingConfig:
    """Flow-matching entry (``FlowLitModule``) on an offline UNetTransformer.

    Offline because a posterior over a trace may use both sides of each sample.
    The sim is the q3_fiber grid's nominal rig (its ``nominal`` or ``fiber``
    arm, pinned in tests/test_flow.py), so the flows are scored on that grid's
    frozen sets against its TCANs. ``in_channels`` / ``out_channels`` are
    placeholders the module sets from the task. ``encoder_channels`` > 0
    adds the conditioning encoder (smi.learning.flow), a second copy of the
    backbone, at ``FLOW_AUX_WEIGHT``.
    """
    config = _matched({**model, 'out_channels': 1}, crop)
    assert config.synthetic_config is not None
    sim = SimConfig.nominal(STEMLAB_125_14, (0.635, 0.6748, 0.515), 0.2, 3.0)
    config.synthetic_config['sim'] = dataclasses.asdict(
        dataclasses.replace(sim, fiber=fiber)
    )
    config.synthetic_config['flow'] = {'task': task, 'n_time_features': 8}
    if encoder_channels:
        config.synthetic_config['flow'] |= {
            'encoder_channels': encoder_channels,
            'aux_weight': FLOW_AUX_WEIGHT,
        }
    # Flow loss is noisy and has no zero-predictor plateau: a fixed budget,
    # best checkpoint by val/flow_loss (smi.experiments.flow_runner).
    epochs = FLOW_EPOCHS_ENC if encoder_channels else FLOW_EPOCHS
    config.training_config['max_epochs'] = epochs[task]
    return config


#: Epochs per task, sized to the ~8 h R8 budget (flow_runner docstring).
FLOW_EPOCHS = {'denoise': 60, 'posterior': 40}
# The encoder doubles the step cost (CPU, 4 threads, two jobs running: 1.52 s
# per full-shot posterior step, 1.04 s per denoise step), so the original
# epochs would take ~3.7 h per flow; these fit the three in parallel plus
# scoring inside the ~6 h retry budget (~2.2 h training each).
FLOW_EPOCHS_ENC = {'denoise': 35, 'posterior': 25}
# Unit weight: both losses are MSEs in the same unit-variance x_1 coordinates,
# and the flow loss's floor (the x_0 variance it cannot predict) is ~1, so the
# regression term is of the same order without dominating.
FLOW_AUX_WEIGHT = 1.0
# E: the x_1 guess (channel 0) plus spare features for the field.
FLOW_ENCODER_CHANNELS = 8
_LOCAL = _unet_transformer([16, 28, 40], 80, 3, causal=False)
# Global context for Q3.4: the shot-global quantities (path drift, the +/-v
# sign pair) need joint samples coherent over the whole shot, which the 85-sample
# local field cannot give. Pool 16 puts a 16384-sample shot on 64 bottleneck
# tokens and span 64 lets every token attend all of them (field 66,867), at
# 45.7k parameters, inside the 44k +/-10% band; trained on full shots so that
# span is exercised.
_GLOBAL = {**_unet_transformer([16, 28, 40], 80, 64, causal=False), 'pool': 16}

_FLOW: dict[str, TrainingConfig] = {
    # Q3.3: fiber signals -> the paired free-space signals.
    'flow/denoise-unettransformer-offline-44k': _flow(
        'denoise', _LOCAL, R8_FIBER, MATCHED_CROP, 0
    ),
    # Q3.4: signals -> velocity trajectory, free space and fiber.
    'flow/posterior-free-unettransformer-global-44k': _flow(
        'posterior', _GLOBAL, None, None, 0
    ),
    'flow/posterior-fiber-unettransformer-global-44k': _flow(
        'posterior', _GLOBAL, R8_FIBER, None, 0
    ),
    # The same flows with the conditioning encoder: the ones above learned
    # to ignore their conditioning (smi.learning.flow). Twice the parameters,
    # since the encoder is a second backbone.
    'flow/denoise-unettransformer-offline-enc': _flow(
        'denoise', _LOCAL, R8_FIBER, MATCHED_CROP, FLOW_ENCODER_CHANNELS
    ),
    'flow/posterior-free-unettransformer-global-enc': _flow(
        'posterior', _GLOBAL, None, None, FLOW_ENCODER_CHANNELS
    ),
    'flow/posterior-fiber-unettransformer-global-enc': _flow(
        'posterior', _GLOBAL, R8_FIBER, None, FLOW_ENCODER_CHANNELS
    ),
}

_DISPLACEMENT_STEMS = (
    'tcn-44k',
    'lstm-44k',
    'lstm-bidir-44k',
    'neuraloperator-offline-44k',
)

MODEL_CONFIGS: Mapping[str, TrainingConfig] = MappingProxyType(
    {
        **{f'matched/{stem}': config for stem, config in _MATCHED.items()},
        **{
            f'displacement/{stem}': _displacement(_MATCHED[stem])
            for stem in _DISPLACEMENT_STEMS
        },
        **_LEGACY,
        **_FLOW,
    }
)


def model_config(name: str) -> TrainingConfig:
    """A private copy of the entry ``name``, safe to mutate.

    Training writes ``in_channels`` into its model config and callers override
    budgets in place, so the registry itself is never handed out.

    Raises:
        KeyError: For an unknown name, listing the closest known ones.
    """
    if name not in MODEL_CONFIGS:
        near = difflib.get_close_matches(name, MODEL_CONFIGS, n=5, cutoff=0.4)
        raise KeyError(f'unknown model config {name!r}; closest: {near}')
    return copy.deepcopy(MODEL_CONFIGS[name])
