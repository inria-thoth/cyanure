#!/bin/bash

set -e
set -x

printenv

# OpenMP is not present on macOS by default
if [[ "$RUNNER_OS" == "macOS" ]]; then
    # Make sure to use a libomp version binary compatible with the oldest
    # supported version of the macos SDK as libomp will be vendored into the
    # scikit-learn wheels for macos.

    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        export MACOSX_DEPLOYMENT_TARGET=10.9
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
        OPENBLAS_URL="https://anaconda.org/conda-forge/libopenblas/0.3.21/download/osx-64/libopenblas-0.3.21-openmp_h429af6e_3.tar.bz2"
        UNWIND_URL="https://anaconda.org/conda-forge/libosxunwind/0.0.6/download/osx-64/libosxunwind-0.0.6-h940c156_0.tar.bz2"
    else
        # arm64 builds must cross compile because CI is on x64
        export PYTHON_CROSSENV=1
        # SciPy requires 12.0 on arm to prevent kernel panics
        # https://github.com/scipy/scipy/issues/14688
        # We use the same deployment target to match SciPy.
        export MACOSX_DEPLOYMENT_TARGET=12.0
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
        OPENBLAS_URL="https://anaconda.org/conda-forge/libopenblas/0.3.21/download/osx-arm64/libopenblas-0.3.21-openmp_hc731615_3.tar.bz2"
        UNWIND_URL="https://anaconda.org/conda-forge/libosxunwind/0.0.6/download/osx-64/libosxunwind-0.0.6-hc021e02_0.tar.bz2"
    fi

    sudo conda create -n build_openmp $OPENMP_URL
    sudo conda create -n build_blas $OPENBLAS_URL
    sudo conda create -n build_unwind $UNWIND_URL

    sudo cp "/usr/local/miniconda/envs/build_blas/lib/libopenblas.0.dylib" "/usr/local/miniconda/envs/build_blas/lib/libopenblas.dylib"
    sudo cp "/usr/local/miniconda/envs/build_unwind/lib/libosxunwind.dylib" "/usr/local/miniconda/envs/build_unwind/lib/libunwind.dylib"

    ls "/usr/local/miniconda/envs/build_unwind/lib"
fi