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