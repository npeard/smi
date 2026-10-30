import sys

import lightgbm  # noqa: F401
import numpy as np
import pytest
import sklearn  # noqa: F401
import torch


def test_single_openmp_runtime_survives_corrcoef():
    data = np.random.default_rng(0).normal(size=(3, 1000))
    assert np.isfinite(np.corrcoef(data)).all()


@pytest.mark.skipif(sys.platform != 'darwin', reason='MPS only on macOS')
def test_mps_available():
    assert torch.backends.mps.is_available()
