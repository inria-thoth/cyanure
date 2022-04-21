Tutorials
=========

.. note:: Many of the datasets I used are available `here <http://pascal.inrialpes.fr/data2/mairal/data/>`_.

Examples for binary classification
----------------------------------
The following code performs binary classification with :math:`\ell_2`-regularized logistic regression, with no intercept, on the criteo dataset (21Gb, huge sparse matrix)::

   from cyanure.estimators import Classifier
    from cyanure.data_processing import preprocess
    import scipy.sparse
    import numpy as np

    y_path = "dataset/criteo_y.npz"
    x_path = "dataset/criteo_X.npz"


    #load criteo dataset 21Gb, n=45840617, p=999999
    dataY=np.load(y_path, allow_pickle=True)
    y=dataY['arr_0']
    X = scipy.sparse.load_npz(x_path)

    #normalize the rows of X in-place, without performing any copy
    preprocess(X,normalize=True,columns=False) 
    #declare a binary classifier for l2-logistic regression  uses the auto solver by default, performs at most 500 epochs
    classifier=Classifier(loss='logistic',penalty='l2',lambda_1=0.1/X.shape[0],max_iter=500,tol=1e-3,duality_gap_interval=5, verbose=True, fit_intercept=False)
    classifier.fit(X,y)

Before we comment the previous choices, let us 
run the above code on a regular three-years-old quad-core workstation with 32Gb of memory
(Intel Xeon CPU E5-1620 v3). ::

    Info : Matrix X, n=45840617, p=999999
    Info : *********************************
    Info : Catalyst Accelerator
    Info : MISO Solver
    Info : Incremental Solver 
    Info : with uniform sampling
    Info : Lipschitz constant: 0.250004
    Info : Logistic Loss is used
    Info : L2 regularization
    Info : Epoch: 5, primal objective: 0.456014, time: 124.345
    Info : Best relative duality gap: 14383.9
    Info : Epoch: 10, primal objective: 0.450885, time: 288.115
    Info : Best relative duality gap: 1004.69
    Info : Epoch: 15, primal objective: 0.450728, time: 456.976
    Info : Best relative duality gap: 6.50049
    Info : Epoch: 20, primal objective: 0.450724, time: 620.227
    Info : Best relative duality gap: 0.068658
    Info : Epoch: 25, primal objective: 0.450724, time: 789.031
    Info : Best relative duality gap: 0.00173208
    Info : Epoch: 30, primal objective: 0.450724, time: 951.987
    Info : Best relative duality gap: 0.00173207
    Info : Epoch: 35, primal objective: 0.450724, time: 1120.44
    Info : Best relative duality gap: 9.36947e-05
    Info : Time elapsed : 1130.85

The solver used was *catalyst-miso*; the problem was solved up to
accuracy tol=0.001 in about 20mn after 35 epochs (without taking into account
the time to load the dataset from the hard drive). The regularization
parameter was chosen to be :math:`\lambda=\frac{1}{10n}`, which is close to the
optimal one given by cross-validation.  Even though performing a grid search with
cross-validation would be more costly, it nevertheless shows that processing such 
a large dataset does not necessarily require to massively invest in Amazon EC2 credits,
GPUs, or distributed computing architectures.

In the next example, we use the squared hinge loss with
:math:`\ell_1`-regularization, choosing the regularization parameter such that the
obtained solution has about 10\% non-zero coefficients.
We also fit an intercept.::

    from cyanure.estimators import Classifier
    from cyanure.data_processing import preprocess
    import numpy as np
    import scipy.sparse

    #load rcv1 dataset about 1Gb, n=781265, p=47152
    data = np.load('dataset/rcv1.npz',allow_pickle=True)
    y=data['y']
    y = np.squeeze(y)
    X=data['X']

    X = scipy.sparse.csc_matrix(X.all()).T # n x p matrix, csr format

    #normalize the rows of X in-place, without performing any copy
    preprocess(X,normalize=True,columns=False)
    #declare a binary classifier for squared hinge loss + l1 regularization
    classifier=Classifier(loss='sqhinge',penalty='l1',lambda_1=0.000005,max_iter=500,tol=1e-3, duality_gap_interval=10, verbose=True, fit_intercept=True)
    # uses the auto solver by default, performs at most 500 epochs
    classifier.fit(X,y) 

which yields::

    Info : Matrix X, n=781265, p=47152
    Info : Memory parameter: 20
    Info : *********************************
    Info : QNing Accelerator
    Info : MISO Solver
    Info : Incremental Solver 
    Info : with uniform sampling
    Info : Lipschitz constant: 1
    Info : Squared Hinge Loss is used
    Info : L1 regularization
    Info : Epoch: 10, primal objective: 0.0915524, time: 7.3084
    Info : Best relative duality gap: 0.387338
    Info : Epoch: 20, primal objective: 0.0915441, time: 15.5574
    Info : Best relative duality gap: 0.00426003
    Info : Epoch: 30, primal objective: 0.0915441, time: 25.6204
    Info : Best relative duality gap: 0.000312145
    Info : Time elapsed : 25.7541
    Info : Total additional line search steps: 8
    Info : Total skipping l-bfgs steps: 0


Multiclass classification
-------------------------
Let us now do something a bit more involved and perform multinomial logistic regression on the
*ckn_mnist* dataset (10 classes, n=60000, p=2304, dense matrix), with multi-task group lasso regularization,
using a laptop with a Intel i7-6500U CPU, and choosing a regularization parameter that yields a solution with 5\% non zero coefficients.::

    from cyanure.estimators import Classifier
    from cyanure.data_processing import preprocess
    import numpy as np


    #load ckn_mnist dataset 10 classes, n=60000, p=2304
    data=np.load('dataset/ckn_mnist.npz')
    y=data['y']
    y = np.squeeze(y)
    X=data['X']

    #center and normalize the rows of X in-place, without performing any copy
    preprocess(X,centering=True,normalize=True,columns=False)
    #declare a multinomial logistic classifier with group Lasso regularization
    classifier=Classifier(loss='multiclass-logistic',penalty='l1l2',lambda_1=0.0001,max_iter=500,tol=1e-3,duality_gap_interval=5, verbose=True, fit_intercept=False)
    # uses the auto solver by default, performs at most 500 epochs
    classifier.fit(X,y)) 

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
    Info : Epoch: 5, primal objective: 0.340267, time: 38.5398
    Info : Best relative duality gap: 0.332094
    Info : Epoch: 10, primal objective: 0.337645, time: 81.2934
    Info : Best relative duality gap: 0.0696317
    Info : Epoch: 15, primal objective: 0.337337, time: 117.92
    Info : Best relative duality gap: 0.0174771
    Info : Epoch: 20, primal objective: 0.337294, time: 153.597
    Info : Best relative duality gap: 0.0103667
    Info : Epoch: 25, primal objective: 0.337285, time: 214.605
    Info : Best relative duality gap: 0.0042598
    Info : Epoch: 30, primal objective: 0.337284, time: 273.082
    Info : Best relative duality gap: 0.00133971
    Info : Epoch: 35, primal objective: 0.337284, time: 326.157
    Info : Best relative duality gap: 0.000633275
    Info : Time elapsed : 326.667
    Info : Total additional line search steps: 7
    Info : Total skipping l-bfgs steps: 0



Learning the multiclass classifier took about 5mn and 26s. To conclude, we provide a last more classical example
of learning l2-logistic regression classifiers on the same dataset, in a one-vs-all fashion.::

    from cyanure.estimators import Classifier
    from cyanure.data_processing import preprocess
    import numpy as np


    #load ckn_mnist dataset 10 classes, n=60000, p=2304
    data=np.load('dataset/ckn_mnist.npz')
    y=data['y']
    y = np.squeeze(y)
    X=data['X']

    #center and normalize the rows of X in-place, without performing any copy
    preprocess(X,centering=True,normalize=True,columns=False)
    #declare a multinomial logistic classifier with group Lasso regularization
    classifier=Classifier(loss='logistic',penalty='l2',lambda_1=0.01/X.shape[0],max_iter=500,tol=1e-3,duality_gap_interval=10, multi_class="ovr",verbose=True, fit_intercept=False)
    # uses the auto solver by default, performs at most 500 epochs
    classifier.fit(X,y)

Then, the 10 classifiers are learned in parallel using the four cpu cores
(still on the same laptop), which gives the following output after about 18 sec::
    
    Info : Matrix X, n=60000, p=2304
    Info : Solver 6 has terminated after 30 epochs in 4.81277 seconds
    Info :    Primal objective: 0.00696712, relative duality gap: 0.000259394
    Info : Solver 3 has terminated after 40 epochs in 6.66197 seconds
    Info :    Primal objective: 0.00806731, relative duality gap: 6.45332e-05
    Info : Solver 0 has terminated after 40 epochs in 6.79647 seconds
    Info :    Primal objective: 0.00581547, relative duality gap: 0.000814821
    Info : Solver 8 has terminated after 50 epochs in 8.89198 seconds
    Info :    Primal objective: 0.0154151, relative duality gap: 0.000295737
    Info : Solver 4 has terminated after 30 epochs in 5.006 seconds
    Info :    Primal objective: 0.00892158, relative duality gap: 0.000320581
    Info : Solver 1 has terminated after 30 epochs in 5.0446 seconds
    Info :    Primal objective: 0.00555609, relative duality gap: 0.000898035
    Info : Solver 7 has terminated after 50 epochs in 9.10058 seconds
    Info :    Primal objective: 0.0105672, relative duality gap: 5.41137e-05
    Info : Solver 9 has terminated after 30 epochs in 5.34556 seconds
    Info :    Primal objective: 0.0162128, relative duality gap: 0.000435193
    Info : Solver 5 has terminated after 40 epochs in 5.39682 seconds
    Info :    Primal objective: 0.00918654, relative duality gap: 0.000391729
    Info : Solver 2 has terminated after 50 epochs in 6.20206 seconds
    Info :    Primal objective: 0.010768, relative duality gap: 0.000145563
    Info : Time for the one-vs-all strategy
    Info : Time elapsed : 18.5133



