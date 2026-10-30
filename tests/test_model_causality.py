#!/usr/bin/env python3
"""Tests for temporal causality of TCN models, and non-causality of SCNN.

A causal model must not respond to inputs from the future: when a block of
ones is placed at position ``step_pos`` in an otherwise-zero sequence, the
model output at every time index *before* ``step_pos`` should be
indistinguishable from its response to the all-zero sequence.

SCNN is non-causal by design and is tested to confirm it *does* exhibit
lookahead (i.e. it has causal violations).
"""

import pytest
import torch

from smi.analysis.models.base import Model
from smi.analysis.models.factory import MODEL_REGISTRY, create_model, is_causal
from smi.analysis.models.lstm import LSTM, LSTMConfig
from smi.analysis.models.mamba import Mamba, MambaConfig
from smi.analysis.models.scnn import SCNN, SCNNConfig
from smi.analysis.models.tcan import TCAN, TCANConfig
from smi.analysis.models.tcn import TCN, TCNConfig
from tests.test_receptive_field import phase_period, roundoff_tolerance

SEQUENCE_LENGTH = 256
IN_CHANNELS = 3
BLOCK_SIZE = 32
# Positions where the step block is placed; chosen to give good coverage
# without making the test suite slow. A step of 31 walks the in-stride phase
# of every strided model below (test_step_positions_cover_every_phase): a
# leak at one phase only is invisible to a probe that always lands on another.
STEP_POSITIONS = list(range(BLOCK_SIZE, SEQUENCE_LENGTH - BLOCK_SIZE, 31))
# A response before the step is a violation only if it exceeds this fraction
# of the maximum post-step response magnitude.
VIOLATION_THRESHOLD = 0.0


@pytest.fixture
def tcn_model():
    config = TCNConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=7,
        temporal_channels=[16, 32, 64, 32, 16],
        dilation_base=2,
        dropout=0.0,
    )
    return TCN(config).eval()


@pytest.fixture
def scnn_model():
    config = SCNNConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=7,
        temporal_channels=[16, 32, 64, 32, 16],
        dilation_base=2,
        dropout=0.0,
    )
    return SCNN(config).eval()


def _get_step_outputs(model, step_positions, sequence_length, in_channels, block_size):
    """Run the model on step-function inputs and a zero baseline.

    Returns:
        baseline: model output for the all-zero input, shape [sequence_length]
        outputs: model outputs for each step position, shape [n_steps, sequence_length]
    """
    device = next(model.parameters()).device

    zero_input = torch.zeros(1, in_channels, sequence_length, device=device)
    with torch.no_grad():
        baseline = model(zero_input)[0, 0]  # [sequence_length]

    outputs = []
    for step_pos in step_positions:
        x = torch.zeros(1, in_channels, sequence_length, device=device)
        x[:, :, step_pos : step_pos + block_size] = 1.0
        with torch.no_grad():
            out = model(x)[0, 0]  # [sequence_length]
        outputs.append(out)

    return baseline, torch.stack(outputs)  # [n_steps, sequence_length]


def _causal_violation_rate(baseline, outputs, step_positions, block_size, threshold):
    """Compute the fraction of pre-step time-steps that violate causality.

    A time step t < step_pos is a violation when
        |output[t] - baseline[t]| > threshold * max(|output[step_pos:]|)
    """
    total_violations = 0
    total_positions = 0

    for i, step_pos in enumerate(step_positions):
        if step_pos == 0:
            continue

        post_step_max = outputs[i, step_pos:].abs().max().item()
        if post_step_max < 1e-6:
            # No meaningful response at all -- skip to avoid false positives
            continue

        pre_step_deviation = (outputs[i, :step_pos] - baseline[:step_pos]).abs()
        violations = pre_step_deviation > threshold * post_step_max

        total_violations += int(violations.sum().item())
        total_positions += step_pos

    violation_rate = total_violations / total_positions if total_positions > 0 else 0.0
    return violation_rate, total_violations, total_positions


@pytest.fixture
def lstm_model():
    config = LSTMConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        hidden_size=32,
        num_layers=2,
        dropout=0.0,
        bidirectional=False,
    )
    return LSTM(config).eval()


@pytest.fixture
def lstm_bidirectional_model():
    config = LSTMConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        hidden_size=32,
        num_layers=2,
        dropout=0.0,
        bidirectional=True,
    )
    return LSTM(config).eval()


class TestLSTMCausality:
    """Unidirectional LSTM must be strictly causal."""

    def test_causal_violation_rate(self, lstm_model):
        baseline, outputs = _get_step_outputs(
            lstm_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        violation_rate, n_violations, n_positions = _causal_violation_rate(
            baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, VIOLATION_THRESHOLD
        )
        assert violation_rate == 0.0, (
            f'LSTM has {n_violations}/{n_positions} causal violations '
            f'({violation_rate * 100:.2f}%)'
        )

    def test_output_unchanged_before_step(self, lstm_model):
        """Output before the step must equal the baseline for every step position."""
        baseline, outputs = _get_step_outputs(
            lstm_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        for i, step_pos in enumerate(STEP_POSITIONS):
            if step_pos == 0:
                continue
            pre_step_diff = (
                (outputs[i, :step_pos] - baseline[:step_pos]).abs().max().item()
            )
            post_step_max = outputs[i, step_pos:].abs().max().item()
            if post_step_max < 1e-6:
                continue
            assert pre_step_diff <= VIOLATION_THRESHOLD * post_step_max, (
                f'LSTM step_pos={step_pos}: max pre-step deviation '
                f'{pre_step_diff:.4e} exceeds threshold '
                f'{VIOLATION_THRESHOLD * post_step_max:.4e}'
            )


@pytest.fixture
def mamba_model():
    config = MambaConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        d_model=16,
        d_state=8,
        d_conv=4,
        expand=2,
        num_layers=2,
        use_layer_norm=True,
    )
    return Mamba(config).eval()


class TestMambaCausality:
    """Mamba uses causal depthwise conv and sequential state updates -- must be causal.

    The causal conv and sequential SSM updates guarantee no lookahead.
    """

    def test_causal_violation_rate(self, mamba_model):
        baseline, outputs = _get_step_outputs(
            mamba_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        violation_rate, n_violations, n_positions = _causal_violation_rate(
            baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, VIOLATION_THRESHOLD
        )
        assert violation_rate == 0.0, (
            f'Mamba has {n_violations}/{n_positions} causal violations '
            f'({violation_rate * 100:.2f}%)'
        )

    def test_output_unchanged_before_step(self, mamba_model):
        """Output before the step must equal the baseline for every step position."""
        baseline, outputs = _get_step_outputs(
            mamba_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        for i, step_pos in enumerate(STEP_POSITIONS):
            if step_pos == 0:
                continue
            pre_step_diff = (
                (outputs[i, :step_pos] - baseline[:step_pos]).abs().max().item()
            )
            post_step_max = outputs[i, step_pos:].abs().max().item()
            if post_step_max < 1e-6:
                continue
            assert pre_step_diff <= VIOLATION_THRESHOLD * post_step_max, (
                f'Mamba step_pos={step_pos}: max pre-step deviation '
                f'{pre_step_diff:.4e} exceeds threshold '
                f'{VIOLATION_THRESHOLD * post_step_max:.4e}'
            )


class TestLSTMBidirectionalNonCausality:
    """Bidirectional LSTM uses future context and must exhibit causal violations."""

    def test_has_causal_violations(self, lstm_bidirectional_model):
        """Bidirectional LSTM output must change before the step."""
        baseline, outputs = _get_step_outputs(
            lstm_bidirectional_model,
            STEP_POSITIONS,
            SEQUENCE_LENGTH,
            IN_CHANNELS,
            BLOCK_SIZE,
        )
        violation_rate, _n_violations, _ = _causal_violation_rate(
            baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, VIOLATION_THRESHOLD
        )
        assert violation_rate > 0.0, (
            'Bidirectional LSTM shows no causal violations -- '
            'backward pass may not be working'
        )


class TestSCNNNonCausality:
    """SCNN uses symmetric padding and must exhibit lookahead (causal violations)."""

    def test_has_causal_violations(self, scnn_model):
        """SCNN output must change before the step for at least one step position."""
        baseline, outputs = _get_step_outputs(
            scnn_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        violation_rate, _n_violations, _ = _causal_violation_rate(
            baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, VIOLATION_THRESHOLD
        )
        assert violation_rate > 0.0, (
            'SCNN shows no causal violations -- symmetric padding may not be working'
        )


class TestTCNCausality:
    def test_causal_violation_rate(self, tcn_model):
        baseline, outputs = _get_step_outputs(
            tcn_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        violation_rate, n_violations, n_positions = _causal_violation_rate(
            baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, VIOLATION_THRESHOLD
        )
        assert violation_rate == 0.0, (
            f'TCN has {n_violations}/{n_positions} causal violations '
            f'({violation_rate * 100:.2f}%)'
        )

    def test_output_unchanged_before_step(self, tcn_model):
        """Output before the step must equal the baseline for every step position."""
        baseline, outputs = _get_step_outputs(
            tcn_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        for i, step_pos in enumerate(STEP_POSITIONS):
            if step_pos == 0:
                continue
            pre_step_diff = (
                (outputs[i, :step_pos] - baseline[:step_pos]).abs().max().item()
            )
            post_step_max = outputs[i, step_pos:].abs().max().item()
            if post_step_max < 1e-6:
                continue
            assert pre_step_diff <= VIOLATION_THRESHOLD * post_step_max, (
                f'TCN step_pos={step_pos}: max pre-step deviation '
                f'{pre_step_diff:.4e} exceeds threshold '
                f'{VIOLATION_THRESHOLD * post_step_max:.4e}'
            )


@pytest.fixture
def stemtcan_model():
    config = TCANConfig(
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        siamese_kernel_size=7,
        siamese_channels=[8, 8, 16],
        siamese_dilation_base=2,
        atten_heads=4,
        decoder_kernel_size=7,
        decoder_channels=[32, 16],
        decoder_dilation_base=2,
        dropout=0.0,
    )
    return TCAN(config).eval()


class TestTCANCausality:
    def test_causal_violation_rate(self, stemtcan_model):
        baseline, outputs = _get_step_outputs(
            stemtcan_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        violation_rate, n_violations, n_positions = _causal_violation_rate(
            baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, VIOLATION_THRESHOLD
        )
        assert violation_rate == 0.0, (
            f'TCAN has {n_violations}/{n_positions} causal violations '
            f'({violation_rate * 100:.2f}%)'
        )

    def test_output_unchanged_before_step(self, stemtcan_model):
        """Output before the step must equal the baseline for every step position."""
        baseline, outputs = _get_step_outputs(
            stemtcan_model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
        )
        for i, step_pos in enumerate(STEP_POSITIONS):
            if step_pos == 0:
                continue
            pre_step_diff = (
                (outputs[i, :step_pos] - baseline[:step_pos]).abs().max().item()
            )
            post_step_max = outputs[i, step_pos:].abs().max().item()
            if post_step_max < 1e-6:
                continue
            assert pre_step_diff <= VIOLATION_THRESHOLD * post_step_max, (
                f'TCAN step_pos={step_pos}: max pre-step deviation '
                f'{pre_step_diff:.4e} exceeds threshold '
                f'{VIOLATION_THRESHOLD * post_step_max:.4e}'
            )


# Smallest viable hparams per registry type, by the same probe the classes
# above use: is_causal must agree with what the network actually does.
REGISTRY_HPARAMS: dict[str, dict] = {
    'Barland': dict(
        type='Barland',
        window_size=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='LeakyReLU',
        dropout=0.0,
        use_weight_norm=False,
        window_stride=SEQUENCE_LENGTH // 2,
        channels=[16, 32, 64, 64],
    ),
    'TCN': dict(
        type='TCN',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=3,
        temporal_channels=[8, 8],
        dilation_base=2,
        dropout=0.0,
    ),
    'SCNN': dict(
        type='SCNN',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        kernel_size=3,
        temporal_channels=[8, 8],
        dilation_base=2,
        dropout=0.0,
    ),
    'TCAN': dict(
        type='TCAN',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        activation='GELU',
        use_layer_norm=True,
        use_weight_norm=False,
        siamese_kernel_size=3,
        siamese_channels=[8, 8],
        siamese_dilation_base=2,
        atten_heads=4,
        decoder_kernel_size=3,
        decoder_channels=[8, 8],
        decoder_dilation_base=2,
        dropout=0.0,
    ),
    'LSTM': dict(
        type='LSTM',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        hidden_size=8,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ),
    'MLP': dict(
        type='MLP',
        window_size=SEQUENCE_LENGTH // 2,
        in_channels=IN_CHANNELS,
        hidden_sizes=[16],
        activation='GELU',
        dropout=0.0,
        window_stride=SEQUENCE_LENGTH // 4,
    ),
    'PatchTransformer': dict(
        type='PatchTransformer',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        patch_len=8,
        stride=5,
        d_model=16,
        n_heads=2,
        n_layers=2,
        ffn_dim=32,
        attention_span=2,
        dropout=0.0,
        causal=True,
    ),
    'UNetTransformer': dict(
        type='UNetTransformer',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        channels=[8, 8, 16],
        kernel_size=4,
        pool=2,
        n_heads=2,
        n_layers=1,
        ffn_dim=32,
        attention_span=2,
        dropout=0.0,
        causal=True,
    ),
    'NeuralOperator': dict(
        type='NeuralOperator',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        width=8,
        n_layers=2,
        modes=4,
        kernel_len=9,
        causal=True,
    ),
    'Mamba': dict(
        type='Mamba',
        sequence_length=SEQUENCE_LENGTH,
        in_channels=IN_CHANNELS,
        d_model=16,
        d_state=8,
        d_conv=4,
        expand=2,
        num_layers=1,
        use_layer_norm=True,
    ),
}
LSTM_BIDIRECTIONAL = {**REGISTRY_HPARAMS['LSTM'], 'bidirectional': True}
PATCH_OFFLINE = {**REGISTRY_HPARAMS['PatchTransformer'], 'causal': False}
UNET_OFFLINE = {**REGISTRY_HPARAMS['UNetTransformer'], 'causal': False}
NEURAL_OPERATOR_OFFLINE = {
    **REGISTRY_HPARAMS['NeuralOperator'],
    'kernel_len': None,
    'causal': False,
}


def test_registry_hparams_cover_every_model_type():
    assert set(REGISTRY_HPARAMS) == set(MODEL_REGISTRY)


@pytest.mark.parametrize(
    'hparams',
    [
        *REGISTRY_HPARAMS.values(),
        LSTM_BIDIRECTIONAL,
        PATCH_OFFLINE,
        UNET_OFFLINE,
        NEURAL_OPERATOR_OFFLINE,
    ],
    ids=[
        *REGISTRY_HPARAMS,
        'LSTM-bidirectional',
        'PatchTransformer-offline',
        'UNetTransformer-offline',
        'NeuralOperator-offline',
    ],
)
def test_is_causal_agrees_with_the_empirical_probe(hparams):
    model = create_model(hparams).eval()
    baseline, outputs = _get_step_outputs(
        model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
    )
    rate, _, _ = _causal_violation_rate(
        baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, roundoff_tolerance(hparams)
    )
    assert is_causal(hparams) == (rate == 0.0)


def test_is_causal_covers_the_tabular_baselines():
    assert not is_causal({'type': 'LightGBM'})
    assert not is_causal({'type': 'Ridge'})


STRIDED = [
    REGISTRY_HPARAMS['PatchTransformer'],
    PATCH_OFFLINE,
    REGISTRY_HPARAMS['UNetTransformer'],
    UNET_OFFLINE,
]
STRIDED_IDS = [
    'PatchTransformer-causal',
    'PatchTransformer-offline',
    'UNetTransformer-causal',
    'UNetTransformer-offline',
]


@pytest.mark.parametrize('hparams', STRIDED, ids=STRIDED_IDS)
def test_step_positions_cover_every_phase(hparams):
    period = phase_period(hparams)
    assert period > 1
    assert {s % period for s in STEP_POSITIONS} == set(range(period))


@pytest.mark.parametrize(
    'hparams',
    [*STRIDED, REGISTRY_HPARAMS['NeuralOperator'], NEURAL_OPERATOR_OFFLINE],
    ids=[*STRIDED_IDS, 'NeuralOperator-causal', 'NeuralOperator-offline'],
)
def test_wrapped_two_way_model_leaks_only_when_offline(hparams):
    """The probe runs on the full ``Model``, not the inner network.

    Tokenisation or pooling, positional bias, skip connections, the
    per-sample upsampling and the FFT padding are each a place a future
    sample could enter; only the wrapped model shows them all. Non-trivial
    normalization stats make sure the baked buffers are included.
    """
    model = Model(
        create_model(hparams),
        input_mean=torch.tensor([0.1, -0.2, 0.3]),
        input_std=torch.tensor([1.5, 0.5, 2.0]),
        output_mean=0.4,
        output_std=3.0,
    ).eval()
    baseline, outputs = _get_step_outputs(
        model, STEP_POSITIONS, SEQUENCE_LENGTH, IN_CHANNELS, BLOCK_SIZE
    )
    rate, n_violations, _ = _causal_violation_rate(
        baseline, outputs, STEP_POSITIONS, BLOCK_SIZE, roundoff_tolerance(hparams)
    )
    if hparams['causal']:
        assert n_violations == 0
    else:
        assert rate > 0.0
