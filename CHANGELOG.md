# Changes

- Fix several issues concerning memory consumption
- Fix several issues concerning execution time
- Fix the number of threads parameter with MKL and Openblas 
- Changes in the handling of dependencies when installing from source

# Known issues:

- Floating point rounding errors with certain solvers.
- The preprocessing is not working if the array is already in fortran format.