# Changes

- Fix Windows build under `conda-build`: detect `CONDA_BUILD` and pick
  up OpenBLAS from `%LIBRARY_PREFIX%\include\openblas` and
  `%LIBRARY_PREFIX%\lib`, link against `openblas` (conda-forge's import
  library name) instead of the hardcoded `libopenblas`, and drop the
  unused `/PIC` flag. Completes conda-forge packaging support
  alongside the macOS fix shipped in 1.2.1.

# Known issues:

- Floating point rounding errors with certain solvers.
- The preprocessing is not working if the array is already in fortran format.
