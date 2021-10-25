import cyanure.cyanure as ars
import numpy as np
import scipy.sparse
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset");
parser.add_argument("--penalty");
parser.add_argument("--solver");
parser.add_argument("--loss");
parser.add_argument("--lambd",type=float);
args=parser.parse_args();

dataset='ckn_mnist';
penalty='l1l2';
solver='auto';
lambd=0.0001;
loss='multiclass-logistic';

nthreads=4
datapath='/scratch/clear/mairal/large_datasets/'
datapath='/scratch/tryckebo/Documents/Cyanure/dataset/'
logfiles='/scratch/clear/mairal/logs_cyanure/'
normalize=True
centering=True
it0=10
intercept=False
multiclass=False
classif=True


if dataset=='ckn_mnist':
    data=np.load(datapath+dataset+'.npz')
    y=data['y']
    X=data['X'].astype('float64')
    y=np.squeeze(np.float64(y))
    multiclass=True

if dataset=='svhn':
    data=np.load(datapath+dataset+'.npz')
    y=data['arr_1']
    X=data['arr_0']
    multiclass=True

if dataset=='rcv1':
    data = np.load(datapath+'rcv1.npz',allow_pickle=True)
    y=data['y']
    X=data['X']
    X = scipy.sparse.csc_matrix(X.all()).T # n x p matrix, csr format 
    X=X.astype('float64')

if dataset=='alpha' or dataset=='covtype' or dataset=='epsilon' or dataset=='ocr':
    data=np.load(datapath+dataset+'.npz')
    y=data['arr_1']
    X=data['arr_0']
    y=np.squeeze(y)

if dataset=='real-sim' or dataset=='webspam' or dataset=='kddb' or dataset=='criteo':
    dataY=np.load(datapath+dataset+'_y.npz',allow_pickle=True)
    y=dataY['arr_0']
    X = scipy.sparse.load_npz(datapath+dataset+'_X.npz')
    y=np.squeeze(y)

ars.preprocess(X,centering=centering,normalize=normalize,columns=False) 

if classif:
    if multiclass:
        classifier=ars.MultiClassifier(loss=loss,penalty=penalty,fit_intercept=intercept)
    else:
        classifier=ars.BinaryClassifier(loss=loss,penalty=penalty,fit_intercept=intercept)
else:
    classifier=ars.Regression(loss=loss,penalty=penalty,fit_intercept=intercept)


if penalty=='l2':
    lambd=lambd/(X.shape[0])
    
classifier.fit(X,y,it0=it0,lambd=lambd,lambd2=lambd,nthreads=nthreads,tol=1e-3,solver=solver,restart=False,seed=0,max_epochs=100)
sparsity=np.count_nonzero(classifier.w.ravel())/len(classifier.w.ravel())
print(sparsity)

