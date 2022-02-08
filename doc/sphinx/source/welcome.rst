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

The code is written by `Julien Mairal <http://julien.mairal.org>`_ (Inria, Univ. Grenoble-Alpes), and 
a documentation is provided in pdf in the following arXiv document

* Julien Mairal. `Cyanure: An Open-Source Toolbox for Empirical Risk Minimization for Python, C++, and soon more <https://arxiv.org/abs/1912.08165>`_ arXiv:1912.08165.  2019 

Main features
-------------
Cyanure is build upon several goals and principles:
   * **Cyanure is memory efficient**. If Cyanure accepts your dataset, it will
     never make a copy of it. Matrices can be provided in double or single
     precision. Sparse matrices (scipy/CSR format for Python, CSC for C++) can
     be provided with integers coded in 32 or 64-bits. When fitting an
     intercept, there is no need to add a column of 1's and there is no matrix
     copy as well. 
   * **Cyanure implements fast algorithms.** Cyanure builds upon two algorithmic principles: (i) variance-reduced stochastic optimization; (ii) Nesterov of Quasi-Newton acceleration. Variance-reduced stochastic optimization algorithms are now popular, but tend to perform poorly when the objective function is badly conditioned. We observe large gains when combining these approaches with Quasi-Newton. 
   * **Cyanure only depends on your BLAS implementation.** Cyanure does not depend on external libraries, except a BLAS library and numpy for Python. We show how to link with OpenBlas and Intel MKL in the python package, but any other BLAS implementation will do.
   * **Cyanure can handle many combinations of loss and regularization functions.** Cyanure can handle a vast combination of loss functions (logistic, square, squared hinge, multiclass logistic) with regularization functions (:math:`\ell_2`, :math:`\ell_1`, elastic-net, fused lasso, multi-task group lasso).
   * **Cyanure provides optimization guarantees.** We believe that reproducibility is important in research. For this reason, knowing if you have solved your problem when the algorithm stops is important. Cyanure provides such a guarantee with a mechanism called duality gap.
   * **Cyanure is easy to use.** We have developed a very simple API, relatively close to scikit-learn's API, and provide also compatibility functions with scikit-learn in order to use Cyanure with minimum effort.
   * **Cyanure should not be only for Python.** A python interface is provided for the C++ code, but it should be feasible to develop an interface for any language with a C++ API, such as R or Matlab. We are planning to develop such interfaces in the future.

Besides all these nice features, Cyanure has also probably some drawbacks, which we will let you discover by yourself.  


License
=======
Cyanure is distributed under BSD-3-Clause license. Even though this is non-legally binding, the author kindly ask users to cite the previous arXiv document in their publications, as well as the publication related to the algorithm they have chosen (see References section). 
Note that if you have chosen the solver 'auto', you are likely to use [QNING]_ or [CATALYST]_ combined with [MISO]_.


Installation
============

You can either install Cyanure from its source on `its github repository <https://github.com/jmairal/cyanure>`_, or on PyPI. 
Cyanure requires Python 3 and was tested on Anaconda on Linux.

* If you want to use Github on Linux, simply clone the repository, and then

   If you are using Anaconda and have the package mkl installed, simply type 
   ::
      python setup_cyanure_mkl.py install

   If instead your numpy relies on openblas, you should use
   ::
      python setup_cyanure_openblas.py install

* If you prefer to use PyPI on Linux, 
  
   Simply replace the two previous commands by
   ::
      pip install cyanure-mkl 

   or by the following command
   ::
      pip install cyanure-openblas

* If you work on Mac, you need a C++ compiler, provided for instance with XCode, which is unlikely to support openmp. If your compiler supports it, follow the Linux instructions. Otherwise, use the "no_openmp" variants.

   if you download the sources on github, 
   ::
      python setup_cyanure_mkl_no_openmp.py install    or     python setup_cyanure_openblas_no_openmp.py install

   or if you use PyPI
   ::
      pip install cyanure-mkl-no-openmp    or       pip install cyanure-openblas-no-openmp   

.. image:: logo-inria-scientifique-couleur.jpg 
   :width: 35%
.. image:: erc-logo.gif
   :width: 15%
.. image:: logo_miai.jpg
   :width: 20%
