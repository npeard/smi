"""Non-torch reference models the learned architectures are measured against.

These deliberately sit outside ``smi.analysis.models``: a LightGBM ``Booster``
and a scikit-learn ``Ridge`` are not ``nn.Module``s, and routing them through
``create_model`` -- whose contract every training caller relies on -- would
distort that contract to accommodate two objects that never train on a GPU.
"""

from .tabular import (
    SupportsPredict,
    fit_lightgbm,
    fit_ridge,
    predict_windows,
    windowize,
)

__all__ = [
    'SupportsPredict',
    'fit_lightgbm',
    'fit_ridge',
    'predict_windows',
    'windowize',
]
