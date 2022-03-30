Welcome to Cyanure's documentation!
===================================
Cyanure is an open-source C++ software package with a Python 3 interface. 
The goal of Cyanure is to provide state-of-the-art solvers for learning linear models,
based on stochastic variance-reduced stochastic optimization with
acceleration mechanisms and Quasi-Newton principles.
Cyanure can handle a large variety of loss functions (logistic, square,
squared hinge, multinomial logistic) and regularization functions (:math:`\ell_2`,
:math:`\ell_1`, elastic-net, fused Lasso, multi-task group Lasso).
It provides a simple Python API, which is very close to that of scikit-learn,
which should be extended to other languages such as R or Matlab in a near future.

The code is written by `Julien Mairal <http://julien.mairal.org>`_ (Inria, Univ. Grenoble-Alpes) and Thomas Ryckeboer (Inria), and 
a documentation is provided in pdf in the following arXiv document

* Julien Mairal. `Cyanure: An Open-Source Toolbox for Empirical Risk Minimization for Python, C++, and soon more <https://arxiv.org/abs/1912.08165>`_ arXiv:1912.08165.  2019 

Main features
-------------
Cyanure is build upon several goals and principles:
   * **Cyanure is memory efficient**. In case of dense matrix, it will
     never make a copy of it except if you are not using floating type. Matrices can be provided in double or single
     precision. Sparse matrices (scipy/CSR format for Python, CSC for C++) can
     be provided with integers coded in 32 or 64-bits. When fitting an
     intercept, there is no need to add a column of 1's and there is no matrix
     copy as well. 
   * **Cyanure implements fast algorithms.** Cyanure builds upon two algorithmic principles: (i) variance-reduced stochastic optimization; (ii) Nesterov of Quasi-Newton acceleration. Variance-reduced stochastic optimization algorithms are now popular, but tend to perform poorly when the objective function is badly conditioned. We observe large gains when combining these approaches with Quasi-Newton. 
   * **Cyanure only depends on your BLAS implementation.** Cyanure depends on usual machine learning libraries,  BLAS library, numpy, scipy and scikit-learn for Python. We show how to link with OpenBLAS and Intel MKL in the python package, but any other BLAS implementation will do. By default pre-compiled packages are shipped with OpenBLAS.
   * **Cyanure can handle many combinations of loss and regularization functions.** Cyanure can handle a vast combination of loss functions (logistic, square, squared hinge, multiclass logistic) with regularization functions (:math:`\ell_2`, :math:`\ell_1`, elastic-net, fused lasso, multi-task group lasso).
   * **Cyanure provides optimization guarantees.** We believe that reproducibility is important in research. For this reason, knowing if you have solved your problem when the algorithm stops is important. Cyanure provides such a guarantee with a mechanism called duality gap.
   * **Cyanure is easy to use.** All the classes of the library are compatible with the [SKLEARN]_'s API, in order to use Cyanure with minimum effort.
   * **Cyanure should not be only for Python.** A python interface is provided for the C++ code, but it should be feasible to develop an interface for any language with a C++ API, such as R or Matlab. We are planning to develop such interfaces in the future.

Besides all these nice features, Cyanure has also probably some drawbacks, which we will let you discover by yourself.  


License
=======
Cyanure is distributed under BSD-3-Clause license. Even though this is non-legally binding, the author kindly ask users to cite the previous arXiv document in their publications, as well as the publication related to the algorithm they have chosen (see References section). 
Note that if you have chosen the solver 'auto', you are likely to use [QNING]_ or [CATALYST]_ combined with [MISO]_.


Installation
============

The recommanded installation procedure is to use either conda or pip.

The package available on conda and pip are shipped with OpenBLAS and OpenMP except for Windows which does not use OpenMP.
For conda the package is available on conda forge.

You can either install with:

 `conda install -c conda-forge cyanure`

 or 

 `pip install cyanure`

You can also install Cyanure from the sources. However, the setup file is not mature enough to work with a variety of BLAS configuration (especially on Windows).

On top of that, you can not use default compiler to compile on MacOS with OpenMP.

.. image:: logo-inria-scientifique-couleur.jpg 
   :width: 35%
.. image:: erc-logo.gif
   :width: 15%
.. image:: logo_miai.jpg
   :width: 20%
