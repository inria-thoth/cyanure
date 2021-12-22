Tutorials
=========

Examples for binary classification
----------------------------------
The following code performs binary classification with :math:`\ell_2`-regularized logistic regression, with no intercept, on the criteo dataset (21Gb, huge sparse matrix)
::
   import arsenic as ars
   #load criteo dataset 21Gb, n=45840617, p=999999
   dataY=np.load('criteo_y.npz',allow_pickle=True); y=dataY['y']
   X = scipy.sparse.load_npz('criteo_X.npz')
   #normalize the rows of X in-place, without performing any copy
   ars.preprocess(X,normalize=True,columns=False) 
   #declare a binary classifier for l2-logistic regression
   classifier=ars.BinaryClassifier(loss='logistic',penalty='l2')
   # uses the auto solver by default, performs at most 500 epochs
   classifier.fit(X,y,lambda_1=0.1/X.shape[0],max_iter=500,tol=1e-3,duality_gap_interval=5) 

Before we comment the previous choices, let us 
run the above code on a regular three-years-old quad-core workstation with 32Gb of memory
(Intel Xeon CPU E5-1630 v4, 400\$ retail price). 
::
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
accuracy tol=0.001 in about $15$mn after 35 epochs (without taking into account
the time to load the dataset from the hard drive). The regularization
parameter was chosen to be :math:`\lambda_1=\frac{1}{10n}`, which is close to the
optimal one given by cross-validation.  Even though performing a grid search with
cross-validation would be more costly, it nevertheless shows that processing such 
a large dataset does not necessarily require to massively invest in Amazon EC2 credits,
GPUs, or distributed computing architectures.

In the next example, we use the squared hinge loss with
:math:`\ell_1`-regularization, choosing the regularization parameter such that the
obtained solution has about 10\% non-zero coefficients.
We also fit an intercept. As shown below, the solution is obtained in 26s on a
laptop with a quad-core i7-8565U CPU (specifically a dell XPS 13 9380).
::
   import arsenic as ars
   #load rcv1 dataset about 1Gb, n=781265, p=47152
   data = np.load('rcv1.npz',allow_pickle=True); y=data['y']; X=data['X']
   X = scipy.sparse.csc_matrix(X.all()).T # n x p matrix, csr format 
   #normalize the rows of X in-place, without performing any copy
   ars.preprocess(X,normalize=True,columns=False) 
   #declare a binary classifier for squared hinge loss + l1 regularization
   classifier=ars.BinaryClassifier(loss='sqhinge',penalty='l2')
   # uses the auto solver by default, performs at most 500 epochs
   classifier.fit(X,y,lambda_1=0.000005,max_iter=500,tol=1e-3) 
which yields
::
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
using the same laptop as previously, and choosing a regularization parameter that yields a solution with 5\% non zero coefficients.
::
   import arsenic as ars
   #load ckn_mnist dataset 10 classes, n=60000, p=2304
   data=np.load('ckn_mnist.npz'); y=data['y']; X=data['X']
   #center and normalize the rows of X in-place, without performing any copy
   ars.preprocess(X,centering=True,normalize=True,columns=False) 
   #declare a multinomial logistic classifier with group Lasso regularization
   classifier=ars.MultiClassifier(loss='multiclass-logistic',penalty='l1l2')
   # uses the auto solver by default, performs at most 500 epochs
   classifier.fit(X,y,lambda_1=0.0001,max_iter=500,tol=1e-3,duality_gap_interval=5) 
which produces
::
   Matrix X, n=60000, p=2304
   Memory parameter: 20
   *********************************
   QNing Accelerator
   MISO Solver
   Incremental Solver with uniform sampling
   Lipschitz constant: 0.25
   Multiclass logistic Loss is used
   Mixed L1-L2 norm regularization
   Epoch: 5, primal objective: 0.340267, time: 30.2643
   Best relative duality gap: 0.332051
   Epoch: 10, primal objective: 0.337646, time: 62.0562
   Best relative duality gap: 0.0695877
   Epoch: 15, primal objective: 0.337337, time: 93.9541
   Best relative duality gap: 0.0172626
   Epoch: 20, primal objective: 0.337293, time: 125.683
   Best relative duality gap: 0.0106066
   Epoch: 25, primal objective: 0.337285, time: 170.044
   Best relative duality gap: 0.00409663
   Epoch: 30, primal objective: 0.337284, time: 214.419
   Best relative duality gap: 0.000677961
   Time elapsed : 215.074
   Total additional line search steps: 4
   Total skipping l-bfgs steps: 0

Learning the multiclass classifier took about 3mn and 35s. To conclude, we provide a last more classical example
of learning l2-logistic regression classifiers on the same dataset, in a one-vs-all fashion.
::
   import arsenic as ars
   #load ckn_mnist dataset 10 classes, n=60000, p=2304
   data=np.load('ckn_mnist.npz'); y=data['y']; X=data['X']
   #center and normalize the rows of X in-place, without performing any copy
   ars.preprocess(X,centering=True,normalize=True,columns=False) 
   #declare a multinomial logistic classifier with group Lasso regularization
   classifier=ars.MultiClassifier(loss='logistic',penalty='l2')
   # uses the auto solver by default, performs at most 500 epochs
   classifier.fit(X,y,lambda_1=0.01/X.shape[0],max_iter=500,tol=1e-3) 

Then, the $10$ classifiers are learned in parallel using the four cpu cores
(still on the same laptop), which gives the following output after about $1$mn
::
   Matrix X, n=60000, p=2304
   Solver 4 has terminated after 30 epochs in 36.3953 seconds
      Primal objective: 0.00877348, relative duality gap: 8.54385e-05
   Solver 8 has terminated after 30 epochs in 37.5156 seconds
      Primal objective: 0.0150244, relative duality gap: 0.000311491
   Solver 9 has terminated after 30 epochs in 38.4993 seconds
      Primal objective: 0.0161167, relative duality gap: 0.000290268
   Solver 7 has terminated after 30 epochs in 39.5971 seconds
      Primal objective: 0.0105672, relative duality gap: 6.49337e-05
   Solver 0 has terminated after 40 epochs in 45.1612 seconds
      Primal objective: 0.00577768, relative duality gap: 3.6291e-05
   Solver 6 has terminated after 40 epochs in 45.8909 seconds
      Primal objective: 0.00687928, relative duality gap: 0.000175357
   Solver 2 has terminated after 40 epochs in 45.9899 seconds
      Primal objective: 0.0104324, relative duality gap: 1.63646e-06
   Solver 5 has terminated after 40 epochs in 47.1608 seconds
      Primal objective: 0.00900643, relative duality gap: 3.42144e-05
   Solver 3 has terminated after 30 epochs in 12.8874 seconds
      Primal objective: 0.00804966, relative duality gap: 0.000200631
   Solver 1 has terminated after 40 epochs in 15.8949 seconds
      Primal objective: 0.00487406, relative duality gap: 0.000584138
   Time for the one-vs-all strategy
   Time elapsed : 62.9996

Note that the toolbox also provides the classes LinearSVC and LogisticRegression that are near-compatible with scikit-learn's API. 
