import os
import argparse
import subprocess

import numpy as np
import pandas as pd 
import scipy.sparse

from cyanure.estimators import Classifier, Regression
from cyanure.data_processing import preprocess

def get_data(datapath, dataset):

    if dataset in ('alpha', 'covtype', 'epsilon', 'ocr'):
        data=np.load(os.path.join(datapath, dataset + '.npz'))
        y=data['arr_1']
        X=data['arr_0']
        y=np.squeeze(y)

    return X, y

def get_name(arguments, lambda_1):

    blas_impl = os.environ.get('CONDA_PREFIX')

    filename = ""

    try:
        nb_cores = subprocess.check_output(['oarprint', 'core']).count(b'\n')
        filename += str(nb_cores) + "_"
    except:
        pass

    if "mkl" in blas_impl:
        filename += "mkl_"
    elif "openblas" in blas_impl:
        filename += "openblas_"
    elif "netlib" in blas_impl:
        filename += "netlib_"
    elif "blis" in blas_impl:
        filename += "blis_"

    filename += arguments.penalty + "_" + '{0:.3E}'.format(lambda_1) + "_" + str(arguments.duality_gap_interval) + "_" + str(arguments.n_threads)  + "_" 

    filename += str(arguments.solver) + "_" + str(arguments.tol) + "_" + str(arguments.dataset)

    filename += ".csv"

    return filename


def process(arguments, X, y):
    preprocess(X,centering=arguments.centering,normalize=arguments.normalize,columns=False)

    lambda_1 = arguments.lambda_1
    if arguments.penalty=='l2':
        lambda_1=arguments.lambda_1/(X.shape[0])

    if arguments.classif:
        classifier=Classifier(loss=arguments.loss,penalty=arguments.penalty,fit_intercept=arguments.intercept,duality_gap_interval=arguments.duality_gap_interval,lambda_1=lambda_1,lambda_2=lambda_1,n_threads=arguments.n_threads,solver=arguments.solver, verbose=False)
    else:
        classifier=Regression(loss=arguments.loss,penalty=arguments.penalty,fit_intercept=arguments.intercept,duality_gap_interval=arguments.duality_gap_interval,lambda_1=lambda_1,lambda_2=lambda_1,n_threads=arguments.n_threads,solver=arguments.solver, verbose=False)

    estimator = classifier.fit(X,y)
    estimator.optimization_info_ = np.squeeze(estimator.optimization_info_).T
    
    
    info_dataframe = pd.DataFrame(estimator.optimization_info_, columns=['Epoch', 'Primal objective', 'Best dual', 'Relative duality gap', 'Diff???', 'Time(s)'])
    filename = get_name(arguments, lambda_1)
    info_dataframe.to_csv("csv/" + filename, header=['Epoch', 'Primal objective', 'Best dual', 'Relative duality gap', 'Diff???', 'Time(s)'], index=False)


def main(arguments):
    X, y = get_data(arguments.datapath, arguments.dataset)

    process(arguments, X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="covtype")
    parser.add_argument("--penalty", default="l2")
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--loss", default="logistic")
    parser.add_argument("--lambda_1", type=float, default=0.001)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--duality_gap_interval", type=int, default=5)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--centering", type=bool, default=True)
    parser.add_argument("--intercept", type=bool, default=False)
    parser.add_argument("--classif", type=bool, default=True)
    parser.add_argument("--datapath", type=str, default="datasets")
    args=parser.parse_args()

    main(args)
