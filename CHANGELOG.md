# Changes

- Add CI 
- Package released on PyPi and conda-forge
- API compliant with scikit-learn
- Fix some bugs: 
    - Issues related to int type on Windows
    - Issues related to uninitialized variables

# Known issues:

- There are issues concerning the utilization of CPU cores.
- It seems that there are some inconsistencies in the execution speed depending on the BLAS implementation.
- The preprocessing is not working if the array is already in fortran format.