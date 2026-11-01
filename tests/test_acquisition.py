"""smi.acquisition against stemlab's FakeTransport (no hardware)."""

import numpy as np
import pytest
from stemlab import STEMLAB_125_14_SPEC, RedPitayaManager
from stemlab.testing import FakeTransport

from smi.acquisition import shots
from smi.acquisition.rig import PRIMARY, acquisition_config
from smi.acquisition.shots import live_view, run_shots
from smi.core.acquisition import NOMINAL_ACQ_DECIMATION, STEMLAB_125_14
from smi.data.schema import CHANNEL_KEYS, SAMPLE_RATE_ATTR, read_shots
from smi.theory.coil_driver import CoilDriver
from smi.theory.waveform import Waveform

BUFFER = np.linspace(-0.5, 0.5, STEMLAB_125_14.buffer_size)


class NeverTriggers(FakeTransport):
    def _answer(self, msg: str) -> str | None:
        return None if msg.upper() == 'ACQ:TRIG:STAT?' else super()._answer(msg)


def make_manager(transport=FakeTransport) -> RedPitayaManager:
    manager = RedPitayaManager(
        ['rp-a', 'rp-b'], lambda host: transport(host, BUFFER), STEMLAB_125_14_SPEC
    )
    manager.configure_daisy_chain(PRIMARY)
    return manager


@pytest.fixture
def waveform() -> Waveform:
    return Waveform(start_freq=1, end_freq=1000, acquisition=STEMLAB_125_14, seed=0)


def test_acquisition_config_matches_core():
    assert acquisition_config(STEMLAB_125_14_SPEC) == STEMLAB_125_14


def test_run_shots_writes_every_shot(tmp_path, waveform):
    manager = make_manager()
    path = run_shots(
        manager, waveform, CoilDriver(), 3, tmp_path / 'sub' / 'd.h5', 1.0, 0.0
    )
    arrays, attrs = read_shots(path)
    assert set(arrays) == set(CHANNEL_KEYS)
    for values in arrays.values():
        assert values.shape == (3, STEMLAB_125_14.buffer_size)
    np.testing.assert_allclose(arrays['RP1_CH1'][0], BUFFER, atol=1e-6)
    assert (
        attrs[SAMPLE_RATE_ATTR]
        == STEMLAB_125_14.sample_rate_dec1_hz / NOMINAL_ACQ_DECIMATION
    )
    assert 'creation_time' in attrs
    # The drive loads on board 0 only, and the rig is reset (output off) at the end.
    assert any(c.startswith('SOUR1:') for c in manager.devices[0].sent)
    assert not any(c.startswith('SOUR') for c in manager.devices[1].sent)
    assert manager.devices[0].sent[-2:] == ['GEN:RST', 'ACQ:RST']


def test_run_shots_appends_to_existing_file(tmp_path, waveform):
    path = tmp_path / 'd.h5'
    for _ in range(2):
        run_shots(make_manager(), waveform, CoilDriver(), 1, path, 1.0, 0.0)
    arrays, _ = read_shots(path)
    assert arrays['RP1_CH1'].shape[0] == 2


def test_timeout_aborts_without_fabricating_data(tmp_path, waveform):
    manager = make_manager(NeverTriggers)
    path = tmp_path / 'd.h5'
    with pytest.raises(RuntimeError, match='consecutive shots failed'):
        run_shots(manager, waveform, CoilDriver(), 10, path, 0.0, 0.0)
    assert not path.exists()


def test_timeout_shot_is_skipped_then_run_continues(tmp_path, waveform, monkeypatch):
    manager = make_manager()
    real = manager.acquire
    calls = []

    def flaky(timeout_s):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError('no trigger')
        return real(timeout_s)

    sleeps = []
    monkeypatch.setattr(manager, 'acquire', flaky)
    monkeypatch.setattr('smi.acquisition.shots.time.sleep', sleeps.append)
    path = run_shots(manager, waveform, CoilDriver(), 3, tmp_path / 'd.h5', 1.0, 7.5)
    arrays, _ = read_shots(path)
    assert arrays['RP1_CH1'].shape[0] == 2
    # A pause follows every shot but the last, the skipped one included.
    assert [s for s in sleeps if s == 7.5] == [7.5, 7.5]


def test_out_of_range_drive_shot_is_skipped_then_run_continues(
    tmp_path, waveform, monkeypatch
):
    manager = make_manager()
    real = manager.configure_generation
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError('samples must lie in [-1, 1]')
        return real(*args, **kwargs)

    monkeypatch.setattr(manager, 'configure_generation', flaky)
    path = run_shots(manager, waveform, CoilDriver(), 3, tmp_path / 'd.h5', 1.0, 0.0)
    arrays, _ = read_shots(path)
    assert arrays['RP1_CH1'].shape[0] == 2


def test_output_is_off_while_each_shot_is_reconfigured(tmp_path, waveform):
    manager = make_manager()
    run_shots(manager, waveform, CoilDriver(), 2, tmp_path / 'd.h5', 1.0, 0.0)
    sent = [c.upper() for c in manager.devices[PRIMARY].sent]
    off = [i for i, c in enumerate(sent) if c == 'OUTPUT1:STATE OFF']
    on = [i for i, c in enumerate(sent) if c == 'OUTPUT1:STATE ON']
    arm = [i for i, c in enumerate(sent) if c == 'ACQ:TRIG CH1_PE']
    load = [i for i, c in enumerate(sent) if c.startswith('SOUR1:TRAC:DATA:DATA')]
    trig = [i for i, c in enumerate(sent) if c == 'SOUR1:TRIG:INT']
    assert len(off) == len(on) == len(arm) == len(load) == 2
    for shot in range(2):
        assert off[shot] < load[shot] < arm[shot] < on[shot] < trig[shot]


def test_live_view_runs_forever_view_on_a_driven_rig(waveform, monkeypatch):
    seen = {}

    class StubView:
        def __init__(self, manager, histogram, timeout_s):
            seen.update(manager=manager, histogram=histogram, timeout_s=timeout_s)

        def run(self, n_frames):
            seen['n_frames'] = n_frames

    monkeypatch.setattr(shots, 'LiveView', StubView)
    manager = make_manager()
    live_view(manager, waveform, CoilDriver(), 2.5)
    assert seen['manager'] is manager
    assert (seen['timeout_s'], seen['n_frames']) == (2.5, None)
    assert 'OUTPUT1:STATE ON' in manager.devices[0].sent
