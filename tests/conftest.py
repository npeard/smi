"""Pytest-wide configuration.

Pin native-math thread counts to 1 for the test session. On this Intel-mac /
conda setup the pypi torch wheel bundles its own OpenMP runtime while conda
numpy/scipy/h5py use the conda ``llvm-openmp``; with multiple threads the two
runtimes collide and segfault the suite once several modules are exercised in a
single process (each test file passes in isolation, but the full run crashes).
Forcing a single thread sidesteps the collision. The tests are small so the
speed cost is negligible, and training is unaffected (it does not import this
file). OpenMP/BLAS read these variables when their thread pool first
initializes, so setting them here -- before any test runs a heavy numeric op --
is early enough even though numpy is already imported.
"""

import os

for _var in (
    'OMP_NUM_THREADS',
    'MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
):
    os.environ.setdefault(_var, '1')
