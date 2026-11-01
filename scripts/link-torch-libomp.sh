#!/bin/sh
# lightgbm's macOS wheel loads @rpath/libomp.dylib. A separate libomp (conda's
# llvm-openmp) next to torch's own deadlocks or segfaults the first threaded
# lightgbm fit (OMP Error #179), so point the rpath copy at the one inside the
# torch wheel: dyld then loads a single runtime for both.
# The python minor version is globbed, not named, so a bump of the env's
# python does not silently stop the link. Pixi sources this file on
# activation, so it must not `exit`.
linked=
for torch_omp in "$CONDA_PREFIX"/lib/python*/site-packages/torch/lib/libomp.dylib; do
    if [ -f "$torch_omp" ]; then
        ln -sf "$torch_omp" "$CONDA_PREFIX/lib/libomp.dylib"
        linked=1
        break
    fi
done
if [ -z "$linked" ]; then
    echo "link-torch-libomp: no torch libomp.dylib under $CONDA_PREFIX/lib/python*/site-packages;" \
        "threaded lightgbm fits may crash (OMP Error #179)" >&2
fi
