import subprocess
import sys
import textwrap

import pytest
import torch

# Runs in a subprocess because a second OpenMP runtime aborts the process
# ("OMP: Error #15") rather than raising. Each library must start a parallel
# region, since merely importing them (or np.corrcoef, which uses Accelerate on
# macOS) never initializes a runtime. Results are compared to a one-thread run
# to catch silent corruption as well as aborts.
_PROBE = textwrap.dedent(
    """
    import lightgbm
    import numpy as np
    import torch
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8))
    y = X[:, 0] + 0.1 * rng.normal(size=2000)

    a = torch.tensor(rng.normal(size=(512, 512)), dtype=torch.float32)

    def run(n):
        torch.set_num_threads(n)
        mm = (a @ a).numpy()
        gbm = lightgbm.LGBMRegressor(n_estimators=10, n_jobs=n, verbose=-1)
        gbm_pred = gbm.fit(X, y).predict(X)
        rf = RandomForestRegressor(n_estimators=10, n_jobs=n, random_state=0)
        rf_pred = rf.fit(X, y).predict(X)
        return mm, gbm_pred, rf_pred

    for one, two in zip(run(1), run(2), strict=True):
        assert np.isfinite(two).all()
        np.testing.assert_allclose(two, one, rtol=1e-3, atol=1e-3)
    print("ok")
    """
)


def test_torch_lightgbm_sklearn_threads_coexist():
    result = subprocess.run(
        [sys.executable, '-c', _PROBE], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'


@pytest.mark.skipif(sys.platform != 'darwin', reason='MPS only on macOS')
def test_mps_available():
    assert torch.backends.mps.is_available()
