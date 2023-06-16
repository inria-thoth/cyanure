# Cyanure
Cyanure: An Open-Source Toolbox for Empirical Risk Minimization for Python,
C++, and soon more.

The library should be fully compatible with scikit-learn.

Cyanure is distributed under BSD-3-Clause license. 
Go to the [project website](https://inria-thoth.github.io/cyanure/welcome.html) for more information including information concerning the API.

Installation
============

The recommanded installation procedure is to use either conda or pip.

The package available on pip are shipped with OpenBLAS and OpenMP except for Windows which does not use OpenMP.
For conda the package is available on conda forge and support every BLAS implementation provided by conda forge.

You can either install with:

 `conda install -c conda-forge cyanure`

 or 

 `pip install cyanure`

You can also install Cyanure from the sources. However, the setup file is not mature enough to work with a variety of BLAS configuration (especially on Windows). Do not hesitate to open an issue if you encounter difficulties installing the package.

On top of that, you can not use default compiler to compile on MacOS with OpenMP.


Installation from source
========================

It is likely that during compilation the implementation of BLAS will not be find automatically.
In this case you can set 2 different environment variables:
- MKL_PATH
- OPENBLAS_PATH 

The MKL variable has the priority over Openblas.
It has to contain the path towards the Blas library.

(e.g: /usr/local/opt/openblas/lib)


Create a new release
====================

When you wish to create a new version of the library you should open a merge 
request to merge on the master branch.

You should update the version of the library by incrementing the number version
in the __VERSION__ file.
Major version is dedicated to breaking changes.
Minor version to new features.
Fixes version to bug fixes release.

You should also update the __CHANGELOG__ file to pinpoint the modifications impacting the users.

Once the merge request is merged a Github and PyPi release will be created. The commit will also be tagged.

Once this part of CI is finished, you should update the conda forge recipe.
Please refer to the following link. For a "simple" release you need to update the version of the recipe and the hash corresponding to the archive which will be downloaded from github. If the build number is different of 0 you should set it to 0.

https://conda-forge.org/docs/maintainer/updating_pkgs.html#updating-recipes



Contribution
============

If you want to contribute to the library you should verify that actual CI is still passing. The CI will trigger automatically on push. It will build the package on the different OS and execute tests on all the wheels.

If you add features in the library, please write tests. A CI job is dedicated to verify that code coverage does not decrease.

Default Parameters
==================

The default values has been changed compared to the one of the initial Cyanure to respect scikit-learn value.

The regularization parameter in Cyanure is different from the one in scikit-learn. It is the inverse from the one in sklearn. It means that larger parameter implies stronger regularization.

Reproductibility
================

It is possible to have different results with the same seed due to some underlying floating point rounding errors.

 



