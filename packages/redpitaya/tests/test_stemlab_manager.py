import subprocess
import sys

import matplotlib
import numpy as np
import pytest
from stemlab import STEMLAB_125_14_SPEC, AcquisitionTrigger, RedPitayaManager
from stemlab.live_view import LiveView
from stemlab.testing import FakeTransport

matplotlib.use('Agg')

BUFFER = np.linspace(-0.5, 0.5, STEMLAB_125_14_SPEC.buffer_size)
TRIG = AcquisitionTrigger(source='CH1_PE', delay_samples=8192, level_v=0.1)


@pytest.fixture
def manager():
    return RedPitayaManager(
        ['rp-a', 'rp-b'], lambda host: FakeTransport(host, BUFFER), STEMLAB_125_14_SPEC
    )


def test_generation_sends_samples(manager):
    samples = np.array([0.0, 0.123456789, -1.0, 0.5])
    manager.configure_generation(
        0, 2, samples, sample_rate_hz=1e6, amplitude_v=0.8, offset_v=0.1, trigger='INT'
    )
    sent = manager.devices[0].sent
    assert sent[0] == 'GEN:RST'
    assert 'SOUR2:FUNC ARBITRARY' in sent
    assert 'SOUR2:VOLT 0.8' in sent
    assert 'SOUR2:VOLT:OFFS 0.1' in sent
    assert 'SOUR2:FREQ:FIX 250000.0' in sent
    assert 'SOUR2:TRIG:SOUR INT' in sent
    (data_cmd,) = [c for c in sent if c.startswith('SOUR2:TRAC:DATA:DATA ')]
    sent_values = np.array(data_cmd.split(' ', 1)[1].split(','), dtype=float)
    np.testing.assert_allclose(sent_values, samples, atol=5e-6)
    assert manager.devices[1].sent == []


def test_generation_rejects_out_of_range(manager):
    with pytest.raises(ValueError, match=r'\[-1, 1\]'):
        manager.configure_generation(0, 1, np.array([1.5]), 1e6, 1.0, 0.0, 'INT')
    with pytest.raises(ValueError, match='1-D'):
        manager.configure_generation(0, 1, np.zeros(16385), 1e6, 1.0, 0.0, 'INT')


def test_acquire_returns_arrays_and_metadata(manager):
    manager.configure_acquisition(0, 64, TRIG, n_samples=1000, channels=[1, 2])
    manager.start_acquisition()
    acq = manager.acquire(timeout_s=1.0)
    assert set(acq.data) == {(0, 1), (0, 2)}
    for trace in acq.data.values():
        assert trace.shape == (1000,)
    assert acq.metadata.decimation == {0: 64}
    assert acq.metadata.sample_rate_hz == {0: 125e6 / 64}
    assert acq.metadata.trigger == {0: TRIG}
    assert acq.metadata.board_ids == ('rp-a', 'rp-b')
    sent = manager.devices[0].sent
    assert 'ACQ:DEC 64' in sent
    assert 'ACQ:TRIG:DLY 8192' in sent
    assert 'ACQ:TRIG:LEV 0.1' in sent
    assert 'ACQ:TRig CH1_PE' in sent
    assert 'ACQ:SOUR1:DATA:OLD:N? 1000' in sent


def test_full_buffer_read_uses_plain_query(manager):
    manager.configure_acquisition(0, 1, TRIG, n_samples=16384, channels=[1])
    manager.start_acquisition()
    acq = manager.acquire(timeout_s=1.0)
    assert acq.data[0, 1].shape == (16384,)
    assert 'ACQ:SOUR1:DATA?' in manager.devices[0].sent


def test_acquire_times_out(manager):
    manager.configure_acquisition(0, 1, TRIG, n_samples=10, channels=[1])
    manager.devices[0]._answer = lambda _msg: 'WAIT'
    with pytest.raises(TimeoutError):
        manager.acquire(timeout_s=0.05)


def test_daisy_chain_configures_secondary_boards(manager):
    manager.configure_daisy_chain(primary=0)
    for device in manager.devices:
        assert device.sent == [
            'DAISY:SYNC:TRig ON',
            'DAISY:SYNC:CLK ON',
            'DAISY:TRIG_O:SOUR ADC',
            'DIG:PIN LED5,1',
        ]
    secondary = AcquisitionTrigger(source='EXT_NE', delay_samples=0, level_v=0.0)
    manager.configure_acquisition(0, 8, TRIG, n_samples=10, channels=[1])
    manager.configure_acquisition(1, 8, secondary, n_samples=10, channels=[2])
    for device in manager.devices:
        device.sent.clear()
    manager.start_acquisition()
    # The secondary must be armed before the primary can forward its trigger.
    assert manager.devices[1].sent == ['ACQ:START', 'ACQ:TRig EXT_NE']
    assert manager.devices[0].sent == ['ACQ:START', 'ACQ:TRig CH1_PE']
    acq = manager.acquire(timeout_s=1.0)
    assert set(acq.data) == {(0, 1), (1, 2)}


def test_output_and_reset_commands(manager):
    manager.enable_output(1, 2, enable=True)
    manager.trigger_generation(1, 2)
    manager.reset()
    assert manager.devices[1].sent == [
        'OUTPUT2:STATE ON',
        'SOUR2:TRig:INT',
        'GEN:RST',
        'ACQ:RST',
    ]
    with pytest.raises(ValueError, match='out of range'):
        manager.enable_output(2, 1, enable=True)


def test_live_view_draws_each_channel(manager):
    manager.configure_acquisition(0, 64, TRIG, n_samples=100, channels=[1, 2])
    view = LiveView(manager, histogram=True, timeout_s=1.0)
    acq = view.run(n_frames=2)
    assert acq.data[0, 1].shape == (100,)
    assert view.fig is not None
    assert len(view.fig.axes) == 4


def test_import_has_no_side_effects():
    probe = (
        'import socket\n'
        'def refuse(*a, **k): raise AssertionError("socket opened on import")\n'
        'socket.socket = refuse\n'
        'socket.create_connection = refuse\n'
        'import stemlab, stemlab.live_view, stemlab.testing\n'
    )
    subprocess.run([sys.executable, '-c', probe], check=True)  # noqa: S603
