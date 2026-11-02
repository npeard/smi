"""Simulator calibration: validated only by a synthetic round trip (blinded).

A frozen set made with known knobs is calibrated through the real-data schema
alone; every identifiable knob must come back within the method's acceptance
bar (20%, SNR within 1 dB), and every knob the data cannot identify must come
back as None rather than as a number that would pass by accident.
"""

import dataclasses
import json
import math

import h5py
import numpy as np
import pytest

from smi.core.acquisition import STEMLAB_125_14
from smi.data.frozen import RIG_WAVELENGTHS_UM, write_frozen_set
from smi.experiments.calibrate_sim import calibrate, write_calibration
from smi.theory.forward import (
    NOMINAL_ALPHA,
    NOMINAL_DRIVE_GAIN,
    NOMINAL_FEEDBACK_C,
    NOMINAL_SWEEPS,
    SimConfig,
)

N_SHOTS = 256
REL_TOL = 0.2
SNR_TOL_DB = 1.0
# No estimator meets the bar at the published phase drift for these; see the
# calibrate_sim module doc.
UNCALIBRATED = (
    'amplitude_drift_rel',
    'contrast_drift_rel',
    'dc_offset_range',
    'shot_noise',
)
KNOBS = (
    'drive_gain',
    'snr_db',
    'phase_drift_rad_per_sqrt_s',
    'feedback_c',
    'alpha',
    'pd_bandwidth_hz',
    'visibility_range',
)
# The volts test's DC level; any positive value works.
DC_VOLTS = 0.5


def _known(pd_bandwidth_hz: float, **changes) -> SimConfig:
    # Nominal, except a visibility top below 1 so both bounds are tested.
    nominal = SimConfig.nominal(
        STEMLAB_125_14, RIG_WAVELENGTHS_UM, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
    )
    return dataclasses.replace(
        nominal, visibility_range=(0.5, 0.9), pd_bandwidth_hz=pd_bandwidth_hz, **changes
    )


# Off-nominal points, each moving one knob (100 kHz PD, read from the knee).
OFF_NOMINAL = {
    'gain_x2': {'drive_gain': 2 * NOMINAL_DRIVE_GAIN},
    'snr_20': {'snr_db': 20.0},
    'snr_40': {'snr_db': 40.0},
    'drift_21.5': {'phase_drift_rad_per_sqrt_s': 21.5},
    'drift_58': {'phase_drift_rad_per_sqrt_s': 58.0},
    'c_0.4': {'feedback_c': 0.4},
    'alpha_5': {'alpha': 5.0},
}
# (point, knob) that miss the bar, with the measured worst per-channel error
# (256 shots, seed 0). Recorded as limits, not widened tolerances.
LIMITS = {
    ('snr_20', 'pd_bandwidth_hz'): 'None: the noise-band knee is buried',
    ('drift_58', 'feedback_c'): 'worst -32% (0.135 for 0.2)',
    ('drift_58', 'alpha'): 'worst +23% (3.7 for 3)',
    ('drift_58', 'pd_bandwidth_hz'): 'worst +59% (158.8 kHz for 100 kHz)',
    ('c_0.4', 'pd_bandwidth_hz'): 'worst +59% (158.8 kHz for 100 kHz)',
}


@pytest.fixture(scope='module')
def calibrated(tmp_path_factory):
    """Calibrate each off-nominal frozen set once, however many tests read it."""
    cache = {}

    def get(point):
        if point not in cache:
            cfg = _known(100e3, **OFF_NOMINAL[point])
            path = tmp_path_factory.mktemp(point) / 'known.h5'
            write_frozen_set(cfg, 0, N_SHOTS, path)
            cache[point] = (cfg, calibrate(path, N_SHOTS, 'signal', STEMLAB_125_14))
        return cache[point]

    return get


# 100 kHz is read from the noise-band knee; 5 kHz from the amplitude roll-off.
@pytest.fixture(scope='module', params=[100e3, 5e3], ids=['knee', 'rolloff'])
def known_set(request, tmp_path_factory):
    cfg = _known(request.param)
    path = tmp_path_factory.mktemp('cal') / 'known.h5'
    return cfg, write_frozen_set(cfg, 0, N_SHOTS, path)


def _close(got: float | None, want: float) -> bool:
    return got is not None and abs(got - want) <= REL_TOL * abs(want)


def _assert_knob(name: str, cfg: SimConfig, result: dict) -> None:
    """One knob within the bar on every channel (drive_gain is file-level)."""
    if name == 'drive_gain':
        assert _close(result[name], cfg.drive_gain), result[name]
        return
    for key, ch in result['channels'].items():
        got, want = ch[name], getattr(cfg, name)
        if name == 'snr_db':
            assert got is not None
            assert abs(got - want) <= SNR_TOL_DB, (key, got)
        elif name == 'visibility_range':
            assert got is not None
            assert all(_close(g, w) for g, w in zip(got, want, strict=True)), (key, got)
        else:
            assert _close(got, want), (key, name, got)


def test_round_trip_recovers_known_knobs(known_set):
    cfg, path = known_set
    result = calibrate(path, N_SHOTS, 'signal', STEMLAB_125_14)
    for name in KNOBS:
        _assert_knob(name, cfg, result)
    assert result['adc_bits'] == cfg.adc_bits
    for ch in result['channels'].values():
        for name in UNCALIBRATED:
            assert ch[name] is None, f'{name} has no estimator, got {ch[name]}'
        assert ch['adc_full_scale'] is None, 'a signal-units file carries no volts'

    rebuilt = SimConfig.from_calibration(STEMLAB_125_14, result, cfg.wavelengths_um)
    assert abs(rebuilt.snr_db - cfg.snr_db) <= SNR_TOL_DB
    for name in ('drive_gain', 'feedback_c', 'alpha', 'pd_bandwidth_hz'):
        assert _close(getattr(rebuilt, name), getattr(cfg, name)), name
    # Uncalibrated knobs fall back to the nominal (here equal to the truth).
    for name in UNCALIBRATED:
        assert getattr(rebuilt, name) == getattr(cfg, name)


@pytest.mark.parametrize(
    ('point', 'knob'),
    [(p, k) for p in OFF_NOMINAL for k in KNOBS if (p, k) not in LIMITS],
)
def test_round_trip_tracks_off_nominal_knob(calibrated, point, knob):
    """Each knob holds the bar at moved settings, not only at the nominal."""
    _assert_knob(knob, *calibrated(point))


@pytest.mark.parametrize(
    ('point', 'knob'), [pytest.param(p, k, id=f'{p}-{k}') for (p, k) in LIMITS]
)
def test_round_trip_known_limits(calibrated, point, knob, request):
    request.applymarker(
        pytest.mark.xfail(strict=True, reason=f'{point}: {LIMITS[(point, knob)]}')
    )
    _assert_knob(knob, *calibrated(point))


def test_volts_file_calibrates_like_its_signal_units(known_set, tmp_path):
    """A DC-coupled acquisition (volts) maps onto the same knobs."""
    _, path = known_set
    volts = tmp_path / 'volts.h5'
    with h5py.File(path, 'r') as src, h5py.File(volts, 'w') as dst:
        dst.attrs['sample_rate'] = src.attrs['sample_rate']
        # Real files carry no pd_wavelengths_um: the rig constants apply.
        for key in ('RP1_CH1', 'RP1_CH2', 'RP2_CH1', 'RP2_CH2'):
            data = src[key][:]
            dst[key] = data if key == 'RP1_CH1' else DC_VOLTS * (1 + data)
    signal = calibrate(path, N_SHOTS, 'signal', STEMLAB_125_14)['channels']
    result = calibrate(volts, N_SHOTS, 'volts', STEMLAB_125_14)['channels']
    for key, ch in result.items():
        # The mean of a shot is the DC level only up to the fringe's own small
        # DC term, so the scale (hence visibility) matches to a few percent.
        assert ch['snr_db'] == pytest.approx(signal[key]['snr_db'], abs=0.5)
        assert ch['feedback_c'] == pytest.approx(signal[key]['feedback_c'], abs=0.01)
        np.testing.assert_allclose(
            ch['visibility_range'], signal[key]['visibility_range'], rtol=0.05
        )
        assert ch['adc_full_scale'] == pytest.approx(1.0 / DC_VOLTS, rel=0.05)


def test_write_calibration_writes_json(known_set, tmp_path):
    _, path = known_set
    out = tmp_path / 'sub' / 'cal.json'
    write_calibration(path, out, 8, 'signal', STEMLAB_125_14)
    assert json.loads(out.read_text())['n_shots'] == 8


def test_nominal_sits_inside_every_sweep():
    nominal = SimConfig.nominal(
        STEMLAB_125_14, RIG_WAVELENGTHS_UM, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA
    )
    names = {f.name for f in dataclasses.fields(SimConfig)}
    for name, sweep in NOMINAL_SWEEPS.items():
        assert name in names, name
        value = getattr(nominal, name)
        assert any(
            v == value or (isinstance(v, float) and math.isclose(v, value))
            for v in sweep
        ), (name, value, sweep)
        # Every sweep point must be a valid config.
        for v in sweep:
            dataclasses.replace(nominal, **{name: v})
