"""Array-in/array-out estimators of displacement and velocity.

Nothing here reads a file or sees a ``FeatureRegistry``: callers in
``smi.experiments`` load the data and hand over arrays.

``tabular`` holds the non-torch reference models (LightGBM, Ridge). They sit
outside ``smi.analysis.models`` deliberately: a ``Booster`` and a ``Ridge`` are
not ``nn.Module``s, and routing them through ``create_model`` would distort the
contract every training caller relies on.
"""
