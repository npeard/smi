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

from smi.analysis.calibrate_sim import calibrate, main
from smi.synthetic.forward import (
    NOMINAL_ALPHA,
    NOMINAL_FEEDBACK_C,
    NOMINAL_SWEEPS,
    SimConfig,
)
from smi.synthetic.frozen import write_frozen_set

RIG_WAVELENGTHS = (0.635, 0.6748, 0.515)
N_SHOTS = 256
REL_TOL = 0.2
SNR_TOL_DB = 1.0
# Not identifiable from the real schema at the published phase drift; see the
# calibrate_sim module doc for why each one is None.
NOT_IDENTIFIABLE = (
    'amplitude_drift_rel',
    'contrast_drift_rel',
    'dc_offset_range',
    'shot_noise',
)
# The volts test's DC level; any positive value works.
DC_VOLTS = 0.5


def _known(pd_bandwidth_hz: float) -> SimConfig:
    # Nominal, except a visibility top below 1 so both bounds are tested.
    nominal = SimConfig.nominal(RIG_WAVELENGTHS, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA)
    return dataclasses.replace(
        nominal, visibility_range=(0.5, 0.9), pd_bandwidth_hz=pd_bandwidth_hz
    )


# 100 kHz is read from the noise-band knee; 5 kHz from the amplitude roll-off.
@pytest.fixture(scope='module', params=[100e3, 5e3], ids=['knee', 'rolloff'])
def known_set(request, tmp_path_factory):
    cfg = _known(request.param)
    path = tmp_path_factory.mktemp('cal') / 'known.h5'
    return cfg, write_frozen_set(cfg, 0, N_SHOTS, path)


def _close(got: float | None, want: float) -> bool:
    return got is not None and abs(got - want) <= REL_TOL * abs(want)


def test_round_trip_recovers_known_knobs(known_set):
    cfg, path = known_set
    result = calibrate(path, N_SHOTS, 'signal')
    assert _close(result['drive_gain'], cfg.drive_gain), result['drive_gain']
    assert result['adc_bits'] == cfg.adc_bits
    for key, ch in result['channels'].items():
        assert ch['snr_db'] is not None
        assert abs(ch['snr_db'] - cfg.snr_db) <= SNR_TOL_DB, (key, ch['snr_db'])
        for name in (
            'phase_drift_rad_per_sqrt_s',
            'feedback_c',
            'alpha',
            'pd_bandwidth_hz',
        ):
            assert _close(ch[name], getattr(cfg, name)), (key, name, ch[name])
        for got, want in zip(ch['visibility_range'], cfg.visibility_range, strict=True):
            assert _close(got, want), (key, ch['visibility_range'])
        for name in NOT_IDENTIFIABLE:
            assert ch[name] is None, f'{name} is not identifiable, got {ch[name]}'
        assert ch['adc_full_scale'] is None, 'a signal-units file carries no volts'

    rebuilt = SimConfig.from_calibration(result, cfg.wavelengths_um)
    assert abs(rebuilt.snr_db - cfg.snr_db) <= SNR_TOL_DB
    for name in ('drive_gain', 'feedback_c', 'alpha', 'pd_bandwidth_hz'):
        assert _close(getattr(rebuilt, name), getattr(cfg, name)), name
    # Unidentified knobs fall back to the nominal (here equal to the truth).
    for name in NOT_IDENTIFIABLE:
        assert getattr(rebuilt, name) == getattr(cfg, name)


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
    signal = calibrate(path, N_SHOTS, 'signal')['channels']
    result = calibrate(volts, N_SHOTS, 'volts')['channels']
    for key, ch in result.items():
        # The mean of a shot is the DC level only up to the fringe's own small
        # DC term, so the scale (hence visibility) matches to a few percent.
        assert ch['snr_db'] == pytest.approx(signal[key]['snr_db'], abs=0.5)
        assert ch['feedback_c'] == pytest.approx(signal[key]['feedback_c'], abs=0.01)
        np.testing.assert_allclose(
            ch['visibility_range'], signal[key]['visibility_range'], rtol=0.05
        )
        assert ch['adc_full_scale'] == pytest.approx(1.0 / DC_VOLTS, rel=0.05)


def test_cli_writes_json(known_set, tmp_path):
    _, path = known_set
    out = tmp_path / 'cal.json'
    main(
        [
            '--dataset',
            str(path),
            '--out',
            str(out),
            '--n-shots',
            '8',
            '--units',
            'signal',
        ]
    )
    assert json.loads(out.read_text())['n_shots'] == 8


def test_nominal_sits_inside_every_sweep():
    nominal = SimConfig.nominal(RIG_WAVELENGTHS, NOMINAL_FEEDBACK_C, NOMINAL_ALPHA)
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
