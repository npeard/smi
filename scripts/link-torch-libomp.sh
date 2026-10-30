#!/bin/sh
# lightgbm's macOS wheel loads @rpath/libomp.dylib. A separate libomp (conda's
# llvm-openmp) next to torch's own deadlocks or segfaults the first threaded
# lightgbm fit (OMP Error #179), so point the rpath copy at the one inside the
# torch wheel: dyld then loads a single runtime for both.
torch_omp="$CONDA_PREFIX/lib/python3.12/site-packages/torch/lib/libomp.dylib"
[ -f "$torch_omp" ] && ln -sf "$torch_omp" "$CONDA_PREFIX/lib/libomp.dylib"
