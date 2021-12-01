import os
import argparse

import numpy as np
import scipy.sparse

import cyanure.cyanure as ars

def get_data(datapath, dataset):
    multiclass = False

    if dataset=='ckn_mnist':
        data=np.load(os.path.join(datapath, dataset+'.npz'))
        y=data['y']
        X=data['X'].astype('float64')
        y=np.squeeze(np.float64(y))
        multiclass=True

    if dataset=='svhn':
        data=np.load(os.path.join(datapath, dataset+'.npz'))
        y=data['arr_1']
        X=data['arr_0']
        multiclass=True

    if dataset=='rcv1':
        data = np.load(datapath+'rcv1.npz',allow_pickle=True)
        y=data['y']
        X=data['X']
        # n x p matrix, csr format
        X = scipy.sparse.csc_matrix(X.all()).T
        X=X.astype('float64')

    if dataset in ('alpha', 'covtype', 'epsilon', 'ocr'):
        data=np.load(os.path.join(datapath, dataset+'.npz'))
        y=data['arr_1']
        X=data['arr_0']
        y=np.squeeze(y)

    if dataset in ('real-sim', 'webspam', 'kddb', 'criteo'):
        dataY=np.load(datapath+dataset+'_y.npz',allow_pickle=True)
        y=dataY['arr_0']
        X = scipy.sparse.load_npz(datapath+dataset+'_X.npz')
        y=np.squeeze(y)

    return X, y, multiclass

def process(arguments, X, y, multiclass):
    ars.preprocess(X,centering=arguments.centering,normalize=arguments.normalize,columns=False)

    if arguments.classif:
        if multiclass:
            classifier=ars.MultiClassifier(loss=arguments.loss,penalty=arguments.penalty,fit_intercept=arguments.intercept)
        else:
            classifier=ars.BinaryClassifier(loss=arguments.loss,penalty=arguments.penalty,fit_intercept=arguments.intercept)
    else:
        classifier=ars.Regression(loss=arguments.loss,penalty=arguments.penalty,fit_intercept=arguments.intercept)

    lambd = arguments.lambd
    if arguments.penalty=='l2':
        lambd=arguments.lambd/(X.shape[0])

    classifier.fit(X,y,duality_gap_interval=arguments.duality_gap_interval,lambd=lambd,lambd2=lambd,n_threads=arguments.n_threads,tol=1e-3,solver=arguments.solver,restart=False,random_state=0,max_epochs=100)
    sparsity=np.count_nonzero(classifier.w_.ravel())/len(classifier.w_.ravel())
    print(sparsity)

def main(arguments):
    X, y, multiclass = get_data(arguments.datapath, arguments.dataset)

    process(arguments, X, y, multiclass)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ckn_mnist")
    parser.add_argument("--penalty", default="l1l2")
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--loss", default="multiclass-logistic")
    parser.add_argument("--lambd", type=float, default=0.0001)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--duality_gap_interval", type=int, default=10)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--centering", type=bool, default=True)
    parser.add_argument("--intercept", type=bool, default=False)
    parser.add_argument("--multiclass", type=bool, default=False)
    parser.add_argument("--classif", type=bool, default=True)
    parser.add_argument("--datapath", type=str, required=True)
    args=parser.parse_args()

    main(args)
