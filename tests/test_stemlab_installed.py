"""stemlab is a separate distribution; every env that runs smi must install it."""

import importlib.metadata

import stemlab


def test_stemlab_installed():
    assert importlib.metadata.distribution('stemlab').metadata['Name'] == 'stemlab'
    assert stemlab.STEMLAB_125_14_SPEC.buffer_size == 16384
