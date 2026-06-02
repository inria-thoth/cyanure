# Changes

- Fix macOS build under `conda-build`: when `CONDA_BUILD` is set, use
  the conda-provided toolchain (`$CXX`, `$PREFIX`-rooted OpenBLAS,
  `libomp` via `-lomp`) instead of the hardcoded GitHub-runner paths.
  Unlocks conda-forge packaging of the macOS wheel.

# Known issues:

- Floating point rounding errors with certain solvers.
- The preprocessing is not working if the array is already in fortran format.
