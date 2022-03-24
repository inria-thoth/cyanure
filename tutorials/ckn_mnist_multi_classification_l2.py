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