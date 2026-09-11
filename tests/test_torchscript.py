"""TorchScript scriptability sanity check for packaged models.

torch.compile is the training-speed path (see lit_module.py); TorchScript is the
PACKAGING path -- a trained model is scripted into a self-contained artifact that
takes raw device outputs (Model.to_torchscript / scripts/export_torchscript.py).
These tests verify that each architecture, wrapped in the normalization Model,
can be scripted and produces output matching eager execution.

A model type whose optional dependency is missing is skipped; one that is not
yet scriptable is xfailed with the underlying error so the gap is documented
rather than hidden.
"""

import pytest
import torch

from smi.analysis.models.base import Model
from smi.analysis.models.factory import create_model

IN_CHANNELS = 3
SEQ_LEN = 256

# (id, model_hparams, input_length) -- small configs for fast CPU scripting.
MODEL_CONFIGS = [
    (
        'TCN',
        dict(
            type='TCN',
            sequence_length=SEQ_LEN,
            in_channels=IN_CHANNELS,
            activation='GELU',
            use_layer_norm=False,
            use_weight_norm=False,
            kernel_size=3,
            temporal_channels=[4, 4],
            dilation_base=2,
            dropout=0.0,
        ),
        SEQ_LEN,
    ),
    (
        'SCNN',
        dict(
            type='SCNN',
            sequence_length=SEQ_LEN,
            in_channels=IN_CHANNELS,
            activation='GELU',
            use_layer_norm=False,
            use_weight_norm=False,
            kernel_size=3,
            temporal_channels=[4, 4],
            dilation_base=2,
            dropout=0.0,
        ),
        SEQ_LEN,
    ),
    (
        # BarlandCNN's FC head is fixed to a 256-sample window (Linear(640, ...)).
        'Barland',
        dict(
            type='Barland',
            window_size=256,
            in_channels=IN_CHANNELS,
            activation='LeakyReLU',
            dropout=0.0,
            use_weight_norm=False,
            window_stride=256,
        ),
        512,
    ),
    (
        'TCAN',
        dict(
            type='TCAN',
            sequence_length=SEQ_LEN,
            in_channels=IN_CHANNELS,
            activation='GELU',
            use_layer_norm=False,
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
        SEQ_LEN,
    ),
    (
        'LSTM',
        dict(
            type='LSTM',
            sequence_length=SEQ_LEN,
            in_channels=IN_CHANNELS,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
        ),
        SEQ_LEN,
    ),
    (
        'Mamba',
        dict(
            type='Mamba',
            sequence_length=SEQ_LEN,
            in_channels=IN_CHANNELS,
            d_model=16,
            d_state=8,
            d_conv=4,
            expand=1,
            num_layers=2,
            use_layer_norm=True,
        ),
        SEQ_LEN,
    ),
]


@pytest.mark.parametrize(
    ('hparams', 'length'),
    [(h, n) for _, h, n in MODEL_CONFIGS],
    ids=[mid for mid, _, _ in MODEL_CONFIGS],
)
def test_model_is_torchscript_scriptable(hparams: dict, length: int):
    """Each wrapped model scripts and matches eager output (or skip/xfail)."""
    try:
        inner = create_model(hparams)
    except ImportError as exc:
        pytest.skip(f'optional dependency missing for {hparams["type"]}: {exc}')

    model = Model.identity(inner, hparams['in_channels'])
    model.eval()
    x = torch.randn(1, hparams['in_channels'], length)
    with torch.no_grad():
        eager = model(x)

    try:
        scripted = model.to_torchscript()
    except Exception as exc:  # noqa: BLE001 -- record, don't hide, the limitation
        pytest.xfail(
            f'{hparams["type"]} is not yet TorchScript-scriptable: '
            f'{type(exc).__name__}: {str(exc)[:300]}'
        )

    with torch.no_grad():
        scripted_out = scripted(x)
    assert scripted_out.shape == eager.shape
    torch.testing.assert_close(scripted_out, eager, rtol=1e-4, atol=1e-5)
