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
classifier=Classifier(loss='sqhinge',penalty='l1',lambda_1=0.000005,max_iter=500,tol=1e-3, duality_gap_interval=10, verbose=True, fit_intercept=False)
# uses the auto solver by default, performs at most 500 epochs
classifier.fit(X,y)