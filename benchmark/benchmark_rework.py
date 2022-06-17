import os
import argparse
import subprocess

import scipy
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from cyanure.estimators import Classifier, Regression
from cyanure.data_processing import preprocess

def get_data(datapath, dataset):

    if 'ckn_mnist' in dataset:
        data=np.load(os.path.join(datapath, dataset + '.npz'))
        y=data['y']
        X=data['X'].astype('float64')
        y=np.squeeze(np.float64(y))

    if 'svhn' in dataset:
        data=np.load(os.path.join(datapath, dataset + '.npz'))
        y=data['arr_1']
        X=data['arr_0']

    if 'rcv1' in dataset:
        data = np.load(os.path.join(datapath, 'rcv1.npz'), allow_pickle=True)
        y=data['y']
        X=data['X']
        X = scipy.sparse.csc_matrix(X.all()).T # n x p matrix, csr format 
        X=X.astype('float64')

    if dataset=='kddb' or dataset=='criteo':
        dataY=np.load(os.path.join(datapath, dataset + '_y.npz'), allow_pickle=True)
        y=dataY['arr_0']
        X = scipy.sparse.load_npz(os.path.join(datapath, dataset + '_X.npz'))
        y=np.squeeze(y)

    if dataset=='real-sim' or dataset=='webspam' :
        data=np.load(os.path.join(datapath, dataset + '.npz'), allow_pickle=True)
        y=data['arr_1']
        print(data['arr_0'].view())
        print(type(data['arr_0']))
        print(data['arr_0'].dtype)
        X=scipy.sparse.csr_matrix(data['arr_0'])
        print(X)
        y=np.squeeze(y)


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

    return filename

def compute_relative_optimality_gap(estimator):
    min_eval=100
    max_dual=-100
    if len(estimator.optimization_info_.shape) > 1 :
        min_eval=min(min_eval,np.min(estimator.optimization_info_[1,]))
        max_dual=max(max_dual,np.max(estimator.optimization_info_[2,]))
        info = np.array(np.maximum((estimator.optimization_info_[1,]-max_dual)/min_eval,1e-9))
    
    else:
        min_eval=min(min_eval,np.min(estimator.optimization_info_[1]))
        max_dual=max(max_dual,np.max(estimator.optimization_info_[2]))
        info = np.array(np.maximum((estimator.optimization_info_[1]-max_dual)/min_eval,1e-9))

    return info


def process(arguments, X, y):
    if arguments.intercept:
        arguments.normalize = False
        arguments.centering = False
    else:
        arguments.normalize = True
        arguments.centering = True

    preprocess(X,centering=arguments.centering,normalize=arguments.normalize,columns=False)

    lambda_1 = arguments.lambda_1
    if arguments.penalty=='l2':
        lambda_1=arguments.lambda_1/(X.shape[0])

    if arguments.classif:
        classifier=Classifier(loss=arguments.loss,penalty=arguments.penalty,fit_intercept=arguments.intercept,duality_gap_interval=arguments.duality_gap_interval,lambda_1=lambda_1,lambda_2=lambda_1,n_threads=arguments.n_threads,solver=arguments.solver, verbose=False)
    else:
        classifier=Regression(loss=arguments.loss,penalty=arguments.penalty,fit_intercept=arguments.intercept,duality_gap_interval=arguments.duality_gap_interval,lambda_1=lambda_1,lambda_2=lambda_1,n_threads=arguments.n_threads,solver=arguments.solver, verbose=False)

    estimator = classifier.fit(X,y)
    estimator.optimization_info_ = np.squeeze(estimator.optimization_info_)
    relative_optimality_gap = compute_relative_optimality_gap(estimator)
    client = MlflowClient() 
    experiment_id = "0"
    tags = {"mlflow.runName": get_name(arguments, lambda_1)}
    run = client.create_run(experiment_id, tags=tags)
    try:
        client.log_param(run.info.run_id, "nb_cores", int(subprocess.check_output(['oarprint', 'core']).count(b'\n')))
    except:
        pass
    client.log_param(run.info.run_id, "environement", os.environ.get('CONDA_PREFIX').split('/')[-1])
    client.log_param(run.info.run_id, "penalty", arguments.penalty)
    client.log_param(run.info.run_id, "lambda", float('{0:.3E}'.format(lambda_1)))
    client.log_param(run.info.run_id, "Duality gap interval", int(arguments.duality_gap_interval))
    client.log_param(run.info.run_id, "nb_threads", int(arguments.n_threads))
    client.log_param(run.info.run_id, "solver", arguments.solver)
    client.log_param(run.info.run_id, "Tolerance", float(arguments.tol))
    client.log_param(run.info.run_id, "Dataset", arguments.dataset)

    # Log the metric. Unlike mlflow.log_metric this method
    # does not start a run if one does not exist. It will log
    # the metric for the run id in the backend store.
    if len(estimator.optimization_info_.shape) > 1 :
        for i in range(len(estimator.optimization_info_[0])):
            timestamp = int(estimator.optimization_info_[5][i] * 1000)
            epoch = int(estimator.optimization_info_[0][i])
            client.log_metric(run.info.run_id, "Primal objective", estimator.optimization_info_[1][i], timestamp, epoch)
            client.log_metric(run.info.run_id, "Best dual", estimator.optimization_info_[2][i], timestamp, epoch)
            client.log_metric(run.info.run_id, "Relative duality gap", estimator.optimization_info_[3][i], timestamp, epoch)
            client.log_metric(run.info.run_id, "Diff", estimator.optimization_info_[4][i], timestamp, epoch)
            client.log_metric(run.info.run_id, "Relative optimality gap", relative_optimality_gap[i], timestamp, epoch)
            print(relative_optimality_gap[i])
    else:
        timestamp = int(estimator.optimization_info_[5] * 1000)
        epoch = int(estimator.optimization_info_[0])
        client.log_metric(run.info.run_id, "Primal objective", estimator.optimization_info_[1], timestamp, epoch)
        client.log_metric(run.info.run_id, "Best dual", estimator.optimization_info_[2], timestamp, epoch)
        client.log_metric(run.info.run_id, "Relative duality gap", estimator.optimization_info_[3], timestamp, epoch)
        client.log_metric(run.info.run_id, "Diff", estimator.optimization_info_[4], timestamp, epoch)
        client.log_metric(run.info.run_id, "Relative optimality gap", relative_optimality_gap, timestamp, epoch)
    client.set_terminated(run.info.run_id)



def main(arguments):
    X, y = get_data(arguments.datapath, arguments.dataset)

    process(arguments, X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="kddb")
    parser.add_argument("--penalty", default="l2")
    parser.add_argument("--solver", default="qning-miso")
    parser.add_argument("--loss", default="logistic")
    parser.add_argument("--lambda_1", type=float, default=0.0001)
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
