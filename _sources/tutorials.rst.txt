Tutorials
=========

.. note:: Many of the datasets I used are available `here <http://pascal.inrialpes.fr/data2/mairal/data/>`_.  I am waiting for the authorization from Criteo before putting their dataset online as well in .npz format.

Examples for binary classification
----------------------------------
The following code performs binary classification with :math:`\ell_2`-regularized logistic regression, with no intercept, on the criteo dataset (21Gb, huge sparse matrix)::

   import cyanure as cyan
   import scipy.sparse
   import numpy as np
   #load criteo dataset 21Gb, n=45840617, p=999999
   dataY=np.load('criteo_y.npz',allow_pickle=True); y=dataY['y']
   X = scipy.sparse.load_npz('criteo_X.npz')
   #normalize the rows of X in-place, without performing any copy
   cyan.preprocess(X,normalize=True,columns=False) 
   #declare a binary classifier for l2-logistic regression
   classifier=cyan.BinaryClassifier(loss='logistic',penalty='l2')
   # uses the auto solver by default, performs at most 500 epochs
   classifier.fit(X,y,lambd=0.1/X.shape[0],max_epochs=500,tol=1e-3,it0=5) 

Before we comment the previous choices, let us 
run the above code on a regular three-years-old quad-core workstation with 32Gb of memory
(Intel Xeon CPU E5-1630 v4, 400\$ retail price). ::

   Matrix X, n=45840617, p=999999
   *********************************
   Catalyst Accelerator
   MISO Solver
   Incremental Solver with uniform sampling
   Lipschitz constant: 0.250004
   Logistic Loss is used
   L2 regularization
   Epoch: 5, primal objective: 0.456014, time: 92.5784
   Best relative duality gap: 14383.9
   Epoch: 10, primal objective: 0.450885, time: 227.593
   Best relative duality gap: 1004.69
   Epoch: 15, primal objective: 0.450728, time: 367.939
   Best relative duality gap: 6.50049
   Epoch: 20, primal objective: 0.450724, time: 502.954
   Best relative duality gap: 0.068658
   Epoch: 25, primal objective: 0.450724, time: 643.323
   Best relative duality gap: 0.00173208
   Epoch: 30, primal objective: 0.450724, time: 778.363
   Best relative duality gap: 0.00173207
   Epoch: 35, primal objective: 0.450724, time: 909.426
   Best relative duality gap: 9.36947e-05
   Time elapsed : 928.114

The solver used was *catalyst-miso*; the problem was solved up to
accuracy tol=0.001 in about 15mn after 35 epochs (without taking into account
the time to load the dataset from the hard drive). The regularization
parameter was chosen to be :math:`\lambda=\frac{1}{10n}`, which is close to the
optimal one given by cross-validation.  Even though performing a grid search with
cross-validation would be more costly, it nevertheless shows that processing such 
a large dataset does not necessarily require to massively invest in Amazon EC2 credits,
GPUs, or distributed computing architectures.

In the next example, we use the squared hinge loss with
:math:`\ell_1`-regularization, choosing the regularization parameter such that the
obtained solution has about 10\% non-zero coefficients.
We also fit an intercept. As shown below, the solution is obtained in 26s on a
laptop with a quad-core i7-8565U CPU (specifically a dell XPS 13 9380).::

   import cyanure as cyan
   import numpy as np
   import scipy.sparse
   #load rcv1 dataset about 1Gb, n=781265, p=47152
   data = np.load('rcv1.npz',allow_pickle=True); y=data['y']; X=data['X']
   X = scipy.sparse.csc_matrix(X.all()).T # n x p matrix, csr format 
   #normalize the rows of X in-place, without performing any copy
   cyan.preprocess(X,normalize=True,columns=False) 
   #declare a binary classifier for squared hinge loss + l1 regularization
   classifier=cyan.BinaryClassifier(loss='sqhinge',penalty='l2')
   # uses the auto solver by default, performs at most 500 epochs
   classifier.fit(X,y,lambd=0.000005,max_epochs=500,tol=1e-3) 

which yields::

   Matrix X, n=781265, p=47152
   Memory parameter: 20
   *********************************
   QNing Accelerator
   MISO Solver
   Incremental Solver with uniform sampling
   Lipschitz constant: 1
   Squared Hinge Loss is used
   L1 regularization
   Epoch: 10, primal objective: 0.0915524, time: 7.33038
   Best relative duality gap: 0.387338
   Epoch: 20, primal objective: 0.0915441, time: 15.524
   Best relative duality gap: 0.00426003
   Epoch: 30, primal objective: 0.0915441, time: 25.738
   Best relative duality gap: 0.000312145
   Time elapsed : 26.0225
   Total additional line search steps: 8
   Total skipping l-bfgs steps: 0

Multiclass classification
-------------------------
Let us now do something a bit more involved and perform multinomial logistic regression on the
*ckn_mnist* dataset (10 classes, n=60000, p=2304, dense matrix), with multi-task group lasso regularization,
using the same laptop as previously, and choosing a regularization parameter that yields a solution with 5\% non zero coefficients.::

   from cyanure.estimators import MultiClassifier
    from cyanure.data_processing import preprocess
    import numpy as np


    #load ckn_mnist dataset 10 classes, n=60000, p=2304
    data=np.load('dataset/ckn_mnist.npz')
    y=data['y'].astype("float64")
    y = np.squeeze(y)
    X=data['X'].astype("float64")

    #center and normalize the rows of X in-place, without performing any copy
    preprocess(X,centering=True,normalize=True,columns=False)
    #declare a multinomial logistic classifier with group Lasso regularization
    classifier=MultiClassifier(loss='multiclass-logistic',penalty='l1l2',lambda_1=0.0001,max_iter=500,tol=1e-3,duality_gap_interval=5, verbose=True)
    # uses the auto solver by default, performs at most 500 epochs
    classifier.fit(X,y) 

which produces::

    Info : Matrix X, n=60000, p=2304
    Info : Memory parameter: 20
    Info : *********************************
    Info : QNing Accelerator
    Info : MISO Solver
    Info : Incremental Solver 
    Info : with uniform sampling
    Info : Lipschitz constant: 0.25
    Info : Multiclass logistic Loss is used
    Info : Mixed L1-L2 norm regularization
    Info : Epoch: 5, primal objective: 0.334992, time: 39.9396
    Info : Best relative duality gap: 0.0922704
    Info : Epoch: 10, primal objective: 0.332324, time: 75.1757
    Info : Best relative duality gap: 0.0268615
    Info : Epoch: 15, primal objective: 0.332051, time: 103.535
    Info : Best relative duality gap: 0.0155829
    Info : Epoch: 20, primal objective: 0.331984, time: 138.484
    Info : Best relative duality gap: 0.0049912
    Info : Epoch: 25, primal objective: 0.331973, time: 174.755
    Info : Best relative duality gap: 0.00232949
    Info : Epoch: 30, primal objective: 0.331972, time: 207.706
    Info : Best relative duality gap: 0.00141096
    Info : Epoch: 35, primal objective: 0.331972, time: 235.578
    Info : Best relative duality gap: 0.000769221
    Info : Time elapsed : 236.065
    Info : Total additional line search steps: 3
    Info : Total skipping l-bfgs steps: 0


Learning the multiclass classifier took about 3mn and 35s. To conclude, we provide a last more classical example
of learning l2-logistic regression classifiers on the same dataset, in a one-vs-all fashion.::

   from cyanure.estimators import MultiClassifier
   from cyanure.data_processing import preprocess
   import numpy as np


   #load ckn_mnist dataset 10 classes, n=60000, p=2304
   data=np.load('dataset/ckn_mnist.npz')
   y=data['y'].astype("float64")
   y = np.squeeze(y)
   X=data['X'].astype("float64")

   #center and normalize the rows of X in-place, without performing any copy
   preprocess(X,centering=True,normalize=True,columns=False)
   #declare a multinomial logistic classifier with group Lasso regularization
   classifier=MultiClassifier(loss='logistic',penalty='l2',lambda_1=0.01/X.shape[0],max_iter=500,tol=1e-3, multi_class="ovr",verbose=True)
   # uses the auto solver by default, performs at most 500 epochs
   classifier.fit(X,y)

Then, the 10 classifiers are learned in parallel using the four cpu cores
(still on the same laptop), which gives the following output after about 1mn::
    
    Info : Matrix X, n=60000, p=2304
    Info : Solver 7 has terminated after 30 epochs in 20.3963 seconds
    Info :    Primal objective: 0.0105672, relative duality gap: 0.000304014
    Info : Solver 4 has terminated after 30 epochs in 22.6081 seconds
    Info :    Primal objective: 0.0087735, relative duality gap: 0.000198454
    Info : Solver 0 has terminated after 35 epochs in 23.8481 seconds
    Info :    Primal objective: 0.00577782, relative duality gap: 0.000326908
    Info : Solver 8 has terminated after 30 epochs in 22.1619 seconds
    Info :    Primal objective: 0.0150245, relative duality gap: 0.000373297
    Info : Solver 5 has terminated after 30 epochs in 20.5889 seconds
    Info :    Primal objective: 0.00900673, relative duality gap: 0.000429762
    Info : Solver 1 has terminated after 35 epochs in 24.4471 seconds
    Info :    Primal objective: 0.00487408, relative duality gap: 0.000178935
    Info : Solver 9 has terminated after 25 epochs in 19.4686 seconds
    Info :    Primal objective: 0.0161172, relative duality gap: 0.000812255
    Info : Solver 6 has terminated after 30 epochs in 21.1479 seconds
    Info :    Primal objective: 0.00687949, relative duality gap: 0.000246356
    Info : Solver 2 has terminated after 35 epochs in 20.7598 seconds
    Info :    Primal objective: 0.0104325, relative duality gap: 0.000125021
    Info : Solver 3 has terminated after 25 epochs in 10.4471 seconds
    Info :    Primal objective: 0.0080502, relative duality gap: 0.000679728
    Info : Time for the one-vs-all strategy
    Info : Time elapsed : 81.5488


Note that the toolbox also provides the classes LinearSVC and LogisticRegression that are near-compatible with scikit-learn's API. 
