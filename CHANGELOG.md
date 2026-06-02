# Changes

- Fix out-of-bounds read in `OptimInfo::replace` that corrupted per-class
  optimization info for multinomial training (caused intermittent
  segfaults under multithreading).
- Fix data race in `SpMatrix::XAt` (scatter writes from concurrent OpenMP
  workers); use per-thread buffers and a sequential reduction, matching
  the existing `AAt` pattern.
- Fix `_xold` initialization in `Solver::solve` so the stopping criterion
  does not dereference an unsized vector when `max_iter == 1` (the
  duality-gap interval is clamped to `max_iter` by the Python wrapper).
- Fix `_warm_start` to use a scalar intercept when restoring a 1D
  initial-weight vector.
- Fix `npyToSpMatrix` DECREF ordering so the underlying numpy arrays
  stay alive until after `SpMatrix::setData` stores their data pointers.
- Honor `OMP_NUM_THREADS` in `init_omp`: an explicit environment value
  now wins over the auto-detected default. When OpenBLAS is used, its
  thread count is coordinated with OpenMP via `openblas_set_num_threads`
  to avoid uncontrolled thread oversubscription.
- Guard nested OpenMP setup: `init_omp` is now a no-op when called from
  inside an existing parallel region, preventing it from overriding the
  thread count set by the enclosing solver.
- Linux wheels now target `manylinux_2_28` (glibc 2.28+), required so
  scipy ships precompiled wheels for the test step.
- macOS wheels now require macOS 11.0+ and bundle a newer `libomp`
  (>= 15) so the symbol `__kmpc_dispatch_deinit` is available; the
  extension is linked with `-Wl,-headerpad_max_install_names` so
  `delocate` can rewrite install names.
- Drop `scipy` and `scikit-learn` from `[build-system].requires`; they
  are runtime dependencies, not build-time ones.
- Pin `setuptools>=78.1.1` and `wheel>=0.38.1` in `[build-system].requires`
  to address Dependabot security advisories.
- Drop `-DNDEBUG` so internal assertions catch errors during testing.
- New test: `test_optimization_info_per_class` checks that the
  per-class iteration counter in `optimization_info_` is a valid
  strictly-increasing positive sequence.
- CI: updated `actions/checkout` and `actions/setup-python` to versions
  compatible with the Node.js 24 runtime; added a `workflow_dispatch`
  trigger to the develop workflow.

# Known issues:

- Floating point rounding errors with certain solvers.
- The preprocessing is not working if the array is already in fortran format.
