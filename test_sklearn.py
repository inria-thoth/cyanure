# -*- coding: utf-8 -*-
import os
import cyanure as cyan
import numpy as np
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import argparse
from timeit import default_timer as timer

CLASSIFIERS = {'sqhinge': cyan.LinearSVC,
               'logistic': cyan.LogisticRegression}
CLASSIFIERS_SK = {'sqhinge': LinearSVC,
                  'logistic': LogisticRegression}


def load_args():
    parser = argparse.ArgumentParser(
        description='Test Cyanure compatibility with scikit-learn',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datapath", type=str, default='./data')
    parser.add_argument("--dataset", type=str, default='ckn_mnist',
                        help='dataset')
    parser.add_argument("--loss", type=str, default='sqhinge',
                        choices=['sqhinge', 'logistic'],
                        help='loss function')
    parser.add_argument("--penalty", type=str, default='l2',
                        choices=['none', 'l2', 'l1'])
    parser.add_argument("--C", type=float, default=1.0,
                        help='regularization parameter')
    parser.add_argument("--intercept", action='store_true')
    args = parser.parse_args()
    return args

def load_data(datapath, dataset, ext='.npz'):
    if dataset == 'ckn_mnist':
        data = np.load(os.path.join(datapath, dataset + ext))
        y = data['y']
        y = np.squeeze(y.astype('float64'))
        X = data['X'].astype('float64')
    elif dataset == 'covtype':
        data = np.load(os.path.join(datapath, dataset + ext))
        y = data['arr_1']
        X = data['arr_0']
        y = np.squeeze(y)
    return X, y

def test_grid(args, X, y):
    clf = CLASSIFIERS[args.loss](
        penalty=args.penalty,
        C=args.C,
        max_iter=10,
        fit_intercept=args.intercept
    )
    grid_clf = GridSearchCV(clf,
                           {'C' : np.logspace(-2, 2, 3)},
                           cv=5,
                           n_jobs=-1, verbose=0, return_train_score=True)
    grid_clf.fit(X, y)
    print(grid_clf.cv_results_)

def benchmark(args, X, y):
    clf1 = CLASSIFIERS[args.loss](
        penalty=args.penalty,
        C=args.C,
        fit_intercept=args.intercept
    )
    clf2 = CLASSIFIERS_SK[args.loss](
        penalty=args.penalty,
        C=args.C,
        fit_intercept=args.intercept
    )

    tic = timer()
    clf1.fit(X, y)
    score = clf1.score(X, y)
    toc = timer()
    print("Cyanure finished! elapsed time: {:.2f}s, accuracy: {:.2f}%".format(toc - tic, score * 100))

    tic = timer()
    clf2.fit(X, y)
    score = clf2.score(X, y)
    toc = timer()
    print("Scikit-learn finished! elapsed time: {:.2f}s, accuracy: {:.2f}%".format(toc - tic, score * 100))

def main():
    args = load_args()
    X, y = load_data(args.datapath, args.dataset)
    print(X)
    print(y)
    cyan.preprocess(X, centering=True, normalize=True, columns=False)

    benchmark(args, X, y)
    test_grid(args, X, y)
    # clf = CLASSIFIERS[args.loss](
    #     penalty=args.penalty,
    #     C=args.C,
    #     max_iter=10,
    #     fit_intercept=args.intercept
    # )


    # grid_clf = GridSearchCV(clf,
    #                        {'C' : np.logspace(-5, 5, 3)},
    #                        cv=5,
    #                        n_jobs=-1, verbose=0, return_train_score=True)
    # grid_clf.fit(X, y)
    # print(grid_clf.cv_results_)
    
    # clf = LinearSVC(C=args.C)
    # clf.fit(X, y)
    # print(clf.predict(X))
    # print(clf.score(X, y))



if __name__ == "__main__":
    main()
