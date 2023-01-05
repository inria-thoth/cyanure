#!/bin/bash

set -e
set -x

# OpenMP is not present on macOS by default
if [[ "$RUNNER_OS" == "macOS" ]]; then
    # Make sure to use a libomp version binary compatible with the oldest
    # supported version of the macos SDK as libomp will be vendored into the
    # scikit-learn wheels for macos.

    if [[ "$ARCHFLAGS" == "-arch arm64" ]]; then
        # arm64 builds must cross compile because CI is on x64
        export PYTHON_CROSSENV=1
        # SciPy requires 12.0 on arm to prevent kernel panics
        # https://github.com/scipy/scipy/issues/14688
        # We use the same deployment target to match SciPy.
        export MACOSX_DEPLOYMENT_TARGET=12.0

        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
        OPENBLAS_URL="https://anaconda.org/conda-forge/libopenblas/0.3.21/download/osx-arm64/libopenblas-0.3.21-openmp_hc731615_3.tar.bz2"
        GFORTRAN_URL="https://anaconda.org/conda-forge/libgfortran5/11.3.0/download/osx-arm64/libgfortran5-11.3.0-hdaf2cc0_26.tar.bz2"
        LLVM_URL="https://anaconda.org/conda-forge/llvm-openmp/15.0.5/download/osx-arm64/llvm-openmp-15.0.5-h7cfbb63_0.tar.bz2"

        sudo conda create -n build $LLVM_URL $OPENBLAS_URL $GFORTRAN_URL
        
    else
        export MACOSX_DEPLOYMENT_TARGET=10.9
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
        OPENBLAS_URL="https://anaconda.org/conda-forge/libopenblas/0.3.21/download/osx-64/libopenblas-0.3.21-openmp_h429af6e_3.tar.bz2"
        GFORTRAN_URL="https://anaconda.org/conda-forge/libgfortran5/11.3.0/download/osx-64/libgfortran5-11.3.0-h082f757_26.tar.bz2"

        sudo conda create -n build $OPENMP_URL $OPENBLAS_URL $GFORTRAN_URL
    fi

    

    PREFIX="$CONDA_HOME/envs/build"

    export CC=/usr/bin/clang
    export CXX=/usr/local/opt/llvm/bin/clang++
    export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
    export CFLAGS="$CFLAGS -I$PREFIX/include"
    export CXXFLAGS="$CXXFLAGS -I$PREFIX/include"
    export LDFLAGS="$LDFLAGS -Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"

    sudo cp "/usr/local/miniconda/envs/build/lib/libopenblas.0.dylib" "/usr/local/miniconda/envs/build/lib/libopenblas.dylib"

    sudo ls /usr/local/miniconda/envs/build

fi