"""Empirical receptive-field and causality tests.

The property under test is a formula, so asserting it against another
formula proves nothing. These tests measure the field by backpropagating
from a single output timestep and counting which input timesteps receive
gradient -- ground truth the formula cannot talk its way around.
"""

import pytest
import torch

from smi.analysis.models.neural_operator import NeuralOperator, NeuralOperatorConfig
from smi.analysis.models.patch_transformer import (
    PatchTransformer,
    PatchTransformerConfig,
)
from smi.analysis.models.scnn import SCNN, SCNNConfig
from smi.analysis.models.tcan import TCAN, TCANConfig
from smi.analysis.models.tcn import TCN, TCNConfig
from smi.analysis.models.unet_transformer import UNetTransformer, UNetTransformerConfig


def measure_receptive_field(
    model: torch.nn.Module, in_channels: int, length: int
) -> tuple[int, bool]:
    """Measure the true receptive field by gradient flow at the centre sample.

    Exact: for the time-domain families only (see ``probe_rel_tol``).

    Returns:
        (field_width, sees_future) where field_width is the number of input
        timesteps with nonzero gradient, and sees_future reports whether any
        timestep strictly after the probed output index received gradient.
    """
    return measure_receptive_field_at(
        model, in_channels, length, length // 2, rel_tol=0.0
    )


def measure_worst_phase_field(
    model: torch.nn.Module, in_channels: int, length: int, period: int, rel_tol: float
) -> tuple[int, bool]:
    """Widest measured field over one period of output positions.

    A strided model is periodic, not translation invariant: its field depends
    on where the probed sample falls inside a stride, so one probe can land on
    a narrow phase. The reported field is the worst case over the period.
    """
    probes = [
        measure_receptive_field_at(
            model, in_channels, length, length // 2 + j, rel_tol=rel_tol
        )
        for j in range(period)
    ]
    return max(w for w, _ in probes), any(f for _, f in probes)


def phase_period(hparams: dict) -> int:
    """Output positions a probe must sweep to see every phase of the field.

    Strided models repeat their geometry once per total stride; the
    convolutional and recurrent families are translation invariant.
    """
    if hparams['type'] == 'PatchTransformer':
        return hparams['stride']
    if hparams['type'] == 'UNetTransformer':
        return hparams['pool'] ** (len(hparams['channels']) - 1)
    return 1


def measure_receptive_field_at(
    model: torch.nn.Module, in_channels: int, length: int, probe: int, rel_tol: float
) -> tuple[int, bool]:
    """Measure the receptive field of output sample ``probe`` by gradient flow.

    An input sample counts as touched when its gradient exceeds ``rel_tol``
    times the largest one (the model's ``probe_rel_tol``).
    """
    model.eval()
    x = torch.zeros(1, in_channels, length, requires_grad=True)
    out = model(x)
    model.zero_grad()
    out[0, 0, probe].backward()

    grad = x.grad
    assert grad is not None
    magnitude = grad.abs().sum(dim=1)[0]
    touched = (magnitude > rel_tol * magnitude.max()).nonzero().flatten()
    field_width = int(touched.numel())
    sees_future = bool((touched > probe).any().item())
    return field_width, sees_future


def _config(kernel_size: int, dilation_base: int, channels: list[int]) -> TCNConfig:
    return TCNConfig(
        sequence_length=2048,
        in_channels=3,
        activation='GELU',
        use_layer_norm=False,
        use_weight_norm=False,
        kernel_size=kernel_size,
        temporal_channels=channels,
        dilation_base=dilation_base,
        dropout=0.0,
    )


@pytest.mark.parametrize(
    ('kernel_size', 'dilation_base', 'channels'),
    [(3, 2, [8, 8, 16]), (5, 2, [8, 16]), (3, 3, [8, 8, 16]), (7, 2, [8, 16])],
)
def test_receptive_field_matches_measurement(kernel_size, dilation_base, channels):
    """The reported receptive field equals the measured one."""
    model = TCN(_config(kernel_size, dilation_base, channels))
    measured, _ = measure_receptive_field(model, in_channels=3, length=2048)
    assert model.receptive_field == measured


def test_causal_tcn_never_sees_the_future():
    """A causal TCN must receive no gradient from samples after the probe."""
    model = TCN(_config(3, 2, [8, 8, 16]))
    _, sees_future = measure_receptive_field(model, in_channels=3, length=2048)
    assert not sees_future


def test_scnn_does_see_the_future():
    """SCNN is non-causal by construction; that is the whole difference."""
    config = SCNNConfig(
        sequence_length=2048,
        in_channels=3,
        activation='GELU',
        use_layer_norm=False,
        use_weight_norm=False,
        kernel_size=3,
        temporal_channels=[8, 8, 16],
        dilation_base=2,
        dropout=0.0,
    )
    _, sees_future = measure_receptive_field(SCNN(config), in_channels=3, length=2048)
    assert sees_future


def test_tcan_receptive_field_does_not_double_count_the_centre():
    """TCAN sums two TCN fields, each of which already includes its centre.

    Without the correction the reported field is one larger than the measured
    one -- an off-by-one that is invisible unless the field is measured.
    """
    config = TCANConfig(
        sequence_length=2048,
        in_channels=3,
        activation='GELU',
        use_layer_norm=False,
        use_weight_norm=False,
        siamese_kernel_size=3,
        siamese_channels=[8, 16],
        siamese_dilation_base=2,
        atten_heads=2,
        decoder_kernel_size=3,
        decoder_channels=[16, 16],
        decoder_dilation_base=2,
        dropout=0.0,
    )
    model = TCAN(config)
    measured, _ = measure_receptive_field(model, in_channels=3, length=2048)
    assert model.receptive_field == measured


def test_config_rejects_dilation_base_that_outruns_the_kernel():
    """A stack whose dilation outruns one block's span must not construct.

    kernel_size=2 / dilation_base=4 reports a field of 43 but measures 27: the
    dilated coverage has gaps, so the property over-reports. A config that
    silently misstates its own receptive field is worse than one that refuses.
    """
    with pytest.raises(ValueError, match='dilation_base 4 exceeds'):
        _config(2, 4, [8, 16, 32])


@pytest.mark.parametrize(
    ('kernel_size', 'dilation_base'),
    # (3, 4) is the case that matters: tcn-config.yaml and scnn-config.yaml
    # both search kernel_size [3,5,7] x dilation_base [2,3,4], so a tighter
    # bound than 2*(k-1)+1 would crash those searches mid-run on a
    # configuration that is in fact exact.
    [(2, 3), (3, 4), (3, 5), (5, 2), (7, 7)],
)
def test_config_accepts_dilation_base_within_the_kernel_span(
    kernel_size, dilation_base
):
    """Inside the valid regime the config constructs and the field is exact."""
    model = TCN(_config(kernel_size, dilation_base, [8, 16]))
    measured, _ = measure_receptive_field(model, in_channels=3, length=2048)
    assert model.receptive_field == measured


def _patch_transformer(
    patch_len: int, stride: int, n_layers: int, span: int, causal: bool
) -> PatchTransformer:
    return PatchTransformer(
        PatchTransformerConfig(
            sequence_length=2048,
            in_channels=3,
            patch_len=patch_len,
            stride=stride,
            d_model=8,
            n_heads=2,
            n_layers=n_layers,
            ffn_dim=16,
            attention_span=span,
            dropout=0.0,
            causal=causal,
        )
    )


PATCH_GEOMETRIES = [(16, 10, 2, 3), (8, 8, 1, 1), (8, 5, 2, 2), (11, 10, 2, 2)]


@pytest.mark.parametrize(('patch_len', 'stride', 'n_layers', 'span'), PATCH_GEOMETRIES)
@pytest.mark.parametrize('causal', [True, False], ids=['causal', 'offline'])
def test_patch_transformer_field_matches_worst_phase_measurement(
    patch_len, stride, n_layers, span, causal
):
    """The property is the worst case over the stride, and it is finite."""
    model = _patch_transformer(patch_len, stride, n_layers, span, causal)
    measured, sees_future = measure_worst_phase_field(
        model, in_channels=3, length=2048, period=stride, rel_tol=0.0
    )
    assert model.receptive_field == measured
    assert sees_future == (not causal)


def test_patch_transformer_rejects_gapped_patches():
    """A stride longer than the patch skips samples no token ever sees."""
    with pytest.raises(ValueError, match='stride 9 exceeds patch_len 8'):
        _patch_transformer(8, 9, 1, 1, causal=True)


def _unet(
    channels: list[int],
    kernel_size: int,
    pool: int,
    n_layers: int,
    span: int,
    causal: bool,
) -> UNetTransformer:
    return UNetTransformer(
        UNetTransformerConfig(
            sequence_length=2048,
            in_channels=3,
            channels=channels,
            kernel_size=kernel_size,
            pool=pool,
            n_heads=2,
            n_layers=n_layers,
            ffn_dim=16,
            attention_span=span,
            dropout=0.0,
            causal=causal,
        )
    )


# (channels, kernel_size, pool, n_layers, span): the matched geometry first,
# then odd kernels, a single level, a deeper stack and a pool of three.
UNET_GEOMETRIES = [
    ([4, 6, 8], 4, 2, 2, 6),
    ([4, 6, 8], 3, 2, 1, 1),
    ([4, 8], 5, 3, 2, 2),
    ([4, 4, 4, 8], 2, 2, 2, 3),
    ([4, 6, 8], 3, 3, 1, 2),
]


@pytest.mark.parametrize(
    ('channels', 'kernel_size', 'pool', 'n_layers', 'span'), UNET_GEOMETRIES
)
@pytest.mark.parametrize('causal', [True, False], ids=['causal', 'offline'])
def test_unet_transformer_field_matches_worst_phase_measurement(
    channels, kernel_size, pool, n_layers, span, causal
):
    """The property is the worst case over the total stride, and finite."""
    model = _unet(channels, kernel_size, pool, n_layers, span, causal)
    period = pool ** (len(channels) - 1)
    measured, sees_future = measure_worst_phase_field(
        model, in_channels=3, length=2048, period=period, rel_tol=0.0
    )
    assert model.receptive_field == measured
    assert sees_future == (not causal)


@pytest.mark.parametrize(
    ('channels', 'kernel_size', 'pool', 'n_layers', 'span'), UNET_GEOMETRIES
)
def test_causal_unet_transformer_lookback_spans_one_stride(
    channels, kernel_size, pool, n_layers, span
):
    """Pooling latency moves the lookback by exactly one total stride.

    The docstring's minimum, receptive_field - (period - 1), is reached at
    one phase of every period; the matched comparison holds the maximum.
    """
    model = _unet(channels, kernel_size, pool, n_layers, span, causal=True)
    period = pool ** (len(channels) - 1)
    widths = [
        measure_receptive_field_at(model, 3, 2048, 1024 + j, rel_tol=0.0)[0]
        for j in range(period)
    ]
    assert min(widths) == model.receptive_field - (period - 1)


def _neural_operator(
    n_layers: int, modes: int, kernel_len: int | None
) -> NeuralOperator:
    # The 1e-5 round-off floor is relative to a random init's output scale, so
    # an unseeded init could flip a gate; fix it.
    torch.manual_seed(0)
    return NeuralOperator(
        NeuralOperatorConfig(
            sequence_length=2048,
            in_channels=3,
            width=8,
            n_layers=n_layers,
            modes=modes,
            kernel_len=kernel_len,
        )
    )


# (n_layers, modes, kernel_len): the matched geometry, a single long kernel,
# a fully free kernel (modes = (K + 1) // 2) and a pointwise-only one.
NEURAL_OPERATOR_GEOMETRIES = [(4, 8, 22), (1, 8, 85), (3, 4, 7), (2, 1, 1)]


@pytest.mark.parametrize(
    ('n_layers', 'modes', 'kernel_len'), NEURAL_OPERATOR_GEOMETRIES
)
def test_causal_neural_operator_lookback_matches_measurement(
    n_layers, modes, kernel_len
):
    """Stacked kernels on lags 0..K-1 reach back n_layers (K - 1) + 1 samples."""
    model = _neural_operator(n_layers, modes, kernel_len)
    measured, sees_future = measure_worst_phase_field(
        model, in_channels=3, length=2048, period=1, rel_tol=model.probe_rel_tol
    )
    assert model.receptive_field == measured
    assert not sees_future


def test_offline_neural_operator_field_is_the_whole_input():
    """A spectral convolution over the whole input reaches every sample.

    That is why the property reports None (unbounded) rather than a width:
    the measured field is the probe length, whatever the probe length is.
    """
    model = _neural_operator(n_layers=2, modes=8, kernel_len=None)
    assert model.receptive_field is None
    for length in (512, 2048):
        measured, sees_future = measure_worst_phase_field(
            model, in_channels=3, length=length, period=1, rel_tol=model.probe_rel_tol
        )
        assert measured == length
        assert sees_future
