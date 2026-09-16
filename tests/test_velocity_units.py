"""Velocity is um/ms in one convention, across every boundary that once disagreed.

Three places in the pipeline used to carry velocity in different units:
``CoilDriver`` (and hence the dataset) emitted um/s, ``LitModule.predict_step``
returned whatever the dataset stored, and ``LitModule.loss_function`` silently
rescaled by ``1e-3`` before the MSE so the logged loss was um/ms. Comparing a
training log to a reported baseline was therefore a 1000x error.

These tests pin the convention at each of those boundaries. Where a unit is
asserted, the reference is built from first principles -- displacement in um
and a time step derived from the sample rate in Hz -- rather than from the
module constant that performs the conversion. Comparing the conversion against
itself would only prove self-consistency, which stays true if the conversion is
removed; anchoring to the physical definitions is what makes re-inserting a
``1e-3``, or dropping the um/ms conversion, fail loudly.
"""

import numpy as np
import torch

from smi.analysis.features import default_registry
from smi.analysis.features.normalization import NORMALIZATION_STATS
from smi.analysis.lit_module import LitModule
from smi.synthetic.coil_driver import MS_PER_S, CoilDriver
from smi.synthetic.interferometers import MichelsonInterferometer
from smi.synthetic.waveform import Waveform

# Velocity RMS on the real acquisitions is ~0.35 um/ms. Anything above this
# bound is um/s (which would be ~350) and anything far below it is um/us.
MAX_PLAUSIBLE_VELOCITY_UM_PER_MS = 10.0
SEQ_LEN = 256
IN_CHANNELS = 3


def _drive_velocity_and_displacement() -> tuple[np.ndarray, np.ndarray, float]:
    """A CoilDriver velocity/displacement pair from a deterministic drive waveform.

    Returns:
        Tuple of (velocity in um/ms, displacement in um, sample rate in Hz).
    """
    rng = np.random.default_rng(0)
    waveform = Waveform(start_freq=10, end_freq=1000, gen_dec=8192, acq_dec=256)
    t, _voltage, _spectrum = waveform.sample()
    sample_rate = float(1 / (t[1] - t[0]))
    # A fixed drive rather than the randomized sample, so the magnitudes this
    # test asserts on do not move between runs.
    drive = 0.5 * np.sin(2 * np.pi * 300.0 * t) + 0.05 * rng.standard_normal(t.size)
    velocity, displacement = CoilDriver().get_velocity_and_displacement(
        drive, sample_rate
    )
    return velocity, displacement, sample_rate


def _tiny_lit_module() -> LitModule:
    """A small velocity-target LitModule that runs on CPU."""
    return LitModule(
        model_hparams={
            'type': 'TCN',
            'sequence_length': SEQ_LEN,
            'in_channels': IN_CHANNELS,
            'activation': 'GELU',
            'use_layer_norm': False,
            'use_weight_norm': False,
            'kernel_size': 3,
            'temporal_channels': [4, 4],
            'dilation_base': 2,
            'dropout': 0.0,
        },
        loss_hparams={
            'dynamic': False,
            'velocity_loss_weight': 1.0,
            'displacement_loss_weight': 1.0,
        },
        training_hparams={'target': 'velocity'},
    )


def test_coil_driver_velocity_is_um_per_ms():
    """The stored target unit: velocity is the um/ms derivative of the um displacement.

    The reference derivative is computed here from first principles -- a plain
    central difference over a time step in milliseconds -- rather than by
    calling ``derivative_displacement``. That matters: both ``get_velocity``
    and ``derivative_displacement`` share the module's ``MS_PER_S`` constant,
    so comparing them to each other only proves they agree, not that either is
    in um/ms. Anchoring to ``displacement`` (definitively um) and
    ``sample_rate`` (definitively Hz) is what actually pins the unit.
    """
    velocity, displacement, sample_rate = _drive_velocity_and_displacement()

    # dt in milliseconds, spelled out rather than imported, so this assertion
    # is independent of the constant under test.
    dt_ms = 1000.0 / sample_rate
    expected = np.gradient(displacement, dt_ms)

    # The spectral derivative inside get_velocity and a central difference are
    # different operators, so they agree to a few percent rather than exactly,
    # and the central difference's truncation error is worst near a velocity
    # zero crossing. Hence rtol for the bulk plus an atol tied to the signal
    # scale for the crossings. A um/s velocity misses by 1000x and fails both.
    assert np.allclose(
        velocity[1:-1], expected[1:-1], rtol=0.05, atol=0.01 * float(np.std(velocity))
    )


def test_integrate_velocity_uses_a_millisecond_time_step():
    """``integrate_velocity`` treats its input as um/ms, not um/s.

    Like the test above, the reference is built from first principles rather
    than from the module's own constant, so reverting the conversion cannot
    keep this assertion self-consistently true.
    """
    _velocity, _displacement, sample_rate = _drive_velocity_and_displacement()
    dt_ms = 1000.0 / sample_rate
    # A constant 1 um/ms velocity must cover exactly dt_ms microns per sample.
    unit_velocity = np.ones(64)
    integrated = CoilDriver.integrate_velocity(unit_velocity, sample_rate)
    assert np.isclose(float(integrated[1] - integrated[0]), dt_ms, rtol=1e-9)


def test_velocity_feature_agrees_with_coil_driver():
    """The registry's velocity target carries the same unit as CoilDriver's."""
    _, _, sample_rate = _drive_velocity_and_displacement()
    rng = np.random.default_rng(1)
    raw = {
        'RP1_CH1': rng.standard_normal(1024).astype(np.float64),
        'RP1_CH2': rng.standard_normal(1024).astype(np.float32),
        'RP2_CH1': rng.standard_normal(1024).astype(np.float32),
        'RP2_CH2': rng.standard_normal(1024).astype(np.float32),
        'sample_rate': sample_rate,
    }
    computed = default_registry.compute(raw, ['velocity'])
    expected, _ = CoilDriver().get_velocity_and_displacement(
        raw['RP1_CH1'], sample_rate
    )
    assert np.allclose(computed['velocity'], expected)


def test_normalization_velocity_std_is_um_per_ms():
    """The baked-in normalization buffer is on the same scale as the target.

    ``NORMALIZATION_STATS`` is generated from the dataset, so a stale um/s
    stats file would make the model's output de-normalization wrong by 1000x
    while every forward pass still ran without error.
    """
    std = NORMALIZATION_STATS['velocity']['std']
    assert 0.0 < std < MAX_PLAUSIBLE_VELOCITY_UM_PER_MS
    # Displacement is um and velocity is um/ms precisely so the two targets sit
    # on one numeric scale; a multi-task loss weight then means what it says.
    disp_std = NORMALIZATION_STATS['displacement']['std']
    assert 0.1 < std / disp_std < 10.0


def test_predict_step_velocity_is_stored_units():
    """``predict_step`` returns velocity in the unit the dataset stores.

    Two things are checked, both of which a stray rescale would break: the
    ground-truth target is handed back exactly as supplied, and the prediction
    equals the model's own raw-units output. ``predict_step`` must not apply a
    conversion that ``loss_function`` does not, which is precisely the
    divergence the old ``1e-3`` created.

    The prediction's *magnitude* is deliberately not asserted here: the network
    is untrained, so its output scale reflects random weights rather than the
    unit convention. The velocity scale is pinned against real data in
    :func:`test_normalization_velocity_std_is_um_per_ms` instead.
    """
    module = _tiny_lit_module()
    module.eval()
    torch.manual_seed(0)
    signals = torch.randn(2, IN_CHANNELS, SEQ_LEN)
    velocity_target = torch.randn(2, SEQ_LEN) * 0.35
    displacement_target = torch.randn(2, SEQ_LEN)

    with torch.no_grad():
        direct = module(signals)
        velocity_hat, returned_target, _disp_hat, _disp_target, _signals = (
            module.predict_step(
                (signals, velocity_target, displacement_target), batch_idx=0
            )
        )

    # The target passes through untouched: no hidden rescale on the way out.
    torch.testing.assert_close(returned_target, velocity_target)
    # The reported velocity is the model's raw-units output, unconverted.
    torch.testing.assert_close(velocity_hat, direct)


def test_velocity_loss_is_mse_in_stored_units():
    """The logged velocity loss is the plain MSE of the stored-unit velocities.

    This is the assertion the old ``1e-3`` rescale inside ``loss_function``
    broke: it made ``{stage}/velocity_loss`` a factor of 1e-6 smaller than the
    MSE of the quantities ``predict_step`` reports.
    """
    module = _tiny_lit_module()
    torch.manual_seed(0)
    velocity_hat = torch.randn(2, SEQ_LEN) * 0.35
    velocity_target = torch.randn(2, SEQ_LEN) * 0.35
    displacement_target = torch.randn(2, SEQ_LEN)

    expected = torch.mean((velocity_hat - velocity_target) ** 2)
    losses = module.loss_function(
        velocity_hat.clone(), velocity_target.clone(), displacement_target.clone()
    )
    torch.testing.assert_close(losses['velocity'], expected)


def test_integrate_velocity_round_trips_um_per_ms():
    """Integrating a um/ms velocity yields displacement in um.

    ``integrate_velocity`` is the physics-informed bridge inside the loss, so
    if it kept the um/s time step the auxiliary displacement term would be
    1000x off and the multi-task weights would be meaningless.
    """
    velocity, displacement, sample_rate = _drive_velocity_and_displacement()
    integrated = CoilDriver.integrate_velocity(velocity, sample_rate)
    assert np.allclose(
        integrated - integrated[0],
        displacement - displacement[0],
        atol=0.05 * float(np.ptp(displacement)),
        rtol=0.0,
    )


def test_interferometer_simulated_buffer_velocity_is_um_per_ms():
    """``get_simulated_buffer`` returns velocity in um/ms, not um/s.

    This method differentiates displacement itself rather than routing through
    ``CoilDriver``, so it is a second place the unit can drift. Its own
    docstrings and plot axes declare um/ms; this pins the value to match.

    The existing interferometer tests cannot catch a scale error here:
    ``test_velocity_displacement_consistency`` exercises the
    ``sample_simulated`` path (which does go through ``CoilDriver``), and its
    other checks compare by correlation, which is scale-invariant.
    """
    sample_rate = 100_000.0
    n = 512
    time = np.arange(n) / sample_rate
    # Constant 1 um/ms ramp: displacement in um must advance by dt_ms per step.
    dt_ms = MS_PER_S / sample_rate
    displacement = np.arange(n) * dt_ms

    interferometer = MichelsonInterferometer(wavelength=0.635, phase=0.0)
    _, _, _, velocity = interferometer.get_simulated_buffer(displacement, time)

    assert np.allclose(velocity, 1.0, rtol=1e-9), (
        f'expected 1 um/ms, got {velocity[1]:.6g} -- a factor of '
        f'{velocity[1]:.6g} suggests the time step is still in seconds'
    )
