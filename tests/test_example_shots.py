"""The appendix example shots: shot-type selection and the committed index."""

import importlib.util
import json

import numpy as np
import pytest

from smi.core.paths import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    'example_shots', REPO_ROOT / 'scripts' / 'example_shots.py'
)
assert _spec is not None
assert _spec.loader is not None
example_shots = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example_shots)


def test_select_shots_orders_by_excursion():
    p2p = np.array([5.0, 0.1, 2.0, 9.0, 1.0, 3.0, 0.5, 7.0, 4.0, 6.0])
    picks = example_shots.select_shots(p2p, example_shots.SHOT_QUANTILES)
    assert list(picks) == ['low', 'mid', 'high']
    assert p2p[picks['low']] < p2p[picks['mid']] < p2p[picks['high']]


def test_select_shots_refuses_a_collision():
    with pytest.raises(ValueError, match='collide'):
        example_shots.select_shots(np.array([1.0, 1.0, 1.0]), {'a': 0.1, 'b': 0.9})


def test_zoom_slice_stays_inside_the_shot():
    v = np.zeros(1000)
    v[995] = 1.0
    z = example_shots.zoom_slice(v, 400)
    assert (z.start, z.stop) == (600, 1000)


def test_committed_index_matches_arrays():
    index = json.loads(example_shots.INDEX.read_text())
    arrays = np.load(example_shots.NPZ)
    syn = index['synthetic']
    assert list(syn['arms']) == list(example_shots.ARMS)
    # Paired arms: the same shots, ordered by excursion.
    shots = [
        {k: s['shot'] for k, s in arm['shots'].items()} for arm in syn['arms'].values()
    ]
    assert shots[0] == shots[1]
    p2p = [s['displacement_p2p_um'] for s in syn['arms']['nominal']['shots'].values()]
    assert p2p == sorted(p2p)
    for arm, info in syn['arms'].items():
        for kind, entry in info['shots'].items():
            tag = f'{arm}/{kind}'
            assert arrays[f'{tag}/signals'].shape[0] == len(info['wavelengths_um'])
            assert arrays[f'{tag}/velocity_zoom'].size == index['zoom_samples']
            for model in entry['models']:
                assert arrays[f'{tag}/{model}/velocity'].shape == (
                    arrays[f'{tag}/velocity'].shape
                )
    assert index['real']['status'] in ('blinded', 'present')
