Welcome to Arsenic's documentation!
===================================
Arsenic is an open-source C++ software package with a Python interface. 
The goal of Arsenic is to provide state-of-the-art solvers for learning linear models,
based on stochastic variance-reduced stochastic optimization with
acceleration mechanisms and Quasi-Newton principles.
Arsenic can handle a large variety of loss functions (logistic, square,
squared hinge, multinomial logistic) and regularization functions (:math:`\ell_2`,
:math:`\ell_1`, elastic-net, fused Lasso, multi-task group Lasso).
It provides a simple Python API, which is very close to that of scikit-learn,
which should be extended to other languages such as R or Matlab in a near future.

The code is written by `Julien Mairal <http://julien.mairal.org>`_ (Inria, Univ. Grenoble-Alpes), and 
a documentation is provided in pdf in the following arXiv document

* Julien Mairal. Arsenic: An Open-Source Toolbox for Empirical Risk Minimization. arXiv:  2019 

Main features
-------------
Arsenic is build upon several goals and principles:
   * **Arsenic is memory efficient**. If Arsenic accepts your dataset, it will
     never make a copy of it. Matrices can be provided in double or single
     precision. Sparse matrices (scipy/CSR format for Python, CSC for C++) can
     be provided with integers coded in 32 or 64-bits. When fitting an
     intercept, there is no need to add a column of 1's and there is no matrix
     copy as well. 
   * **Arsenic implements fast algorithms.** Arsenic builds upon two algorithmic principles: (i) variance-reduced stochastic optimization; (ii) Nesterov of Quasi-Newton acceleration. Variance-reduced stochastic optimization algorithms are now popular, but tend to perform poorly when the objective function is badly conditioned. We observe large gains when combining these approaches with Quasi-Newton. 
   * **Arsenic only depends on your BLAS implementation.** Arsenic does not depend on external libraries, except a BLAS library and numpy for Python. We show how to link with OpenBlas and Intel MKL in the python package, but any other BLAS implementation will do.
   * **Arsenic can handle many combinations of loss and regularization functions.** Arsenic can handle a vast combination of loss functions (logistic, square, squared hinge, multiclass logistic) with regularization functions (:math:`\ell_2`, :math:`\ell_1`, elastic-net, fused lasso, multi-task group lasso).
   * **Arsenic provides optimization guarantees.** We believe that reproducibility is important in research. For this reason, knowing if you have solved your problem when the algorithm stops is important. Arsenic provides such a guarantee with a mechanism called duality gap.
   * **Arsenic is easy to use.** We have developed a very simple API, relatively close to scikit-learn's API, and provide also compatibility functions with scikit-learn in order to use Arsenic with minimum effort.
   * **Arsenic should not be only for Python.** A python interface is provided for the C++ code, but it should be feasible to develop an interface for any language with a C++ API, such as R or Matlab. We are planning to develop such interfaces in the future.

Besides all these nice features, Arsenic has also probably some drawbacks, which we will let you discover by yourself.  


License
=======
Arsenic is distributed under BSD-3-Clause license. Even though this is non-legally binding, the author kindly ask users to cite the previous arXiv document in their publications, as well as the publication related to the algorithm they have chosen (see References section). 
Note that if you have chosen the solver 'auto', you are likely to use [QNING]_ or [CATALYST]_ combined with [MISO]_.


Installation
============

If you are using Anaconda and have the package mkl installed, simply type 
::
   python setup_arsenic_mkl install

If instead your numpy relies on openblas, you should use
::
   python setup_arsenic_openblas install
