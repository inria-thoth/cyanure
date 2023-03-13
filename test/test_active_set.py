import pytest


from sklearn.exceptions import ConvergenceWarning
from libsvmdata import fetch_libsvm
import scipy.sparse
import numpy as np
import warnings

from cyanure.estimators import L1Logistic, Lasso, Regression, fit_large_feature_number


@pytest.fixture
def dataset_finance():
    X, y = fetch_libsvm('finance')
    X = scipy.sparse.csr_matrix(X)
    return X, y

@pytest.fixture
def dataset_rcv1():
    X, y = fetch_libsvm('rcv1.binary')
    if (scipy.sparse.issparse(X) and
                scipy.sparse.isspmatrix_csc(X)):
        X = scipy.sparse.csr_matrix(X)
    return X, y

@pytest.mark.parametrize(
    "estimator",
    [Lasso(verbose=False)]
)
def test_active_set_finance(estimator, dataset_finance):
   
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    X, y = dataset_finance[0], dataset_finance[1]

    n_samples = X.shape[0]
    estimator.lambda_1= 2697 / n_samples
    estimator.fit_intercept = False

    estimator.max_iter = 500

    estimator.fit(X, y)

@pytest.mark.parametrize(
    "estimator",
    [ L1Logistic(verbose=False)]
)
def test_active_set_rcv1(estimator, dataset_rcv1):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    X, y = dataset_rcv1[0], dataset_rcv1[1]

    n_samples = X.shape[0]
    estimator.lambda_1= 2697 / n_samples
    estimator.fit_intercept = False

    estimator.max_iter = 500

    estimator.fit(X, y)

def test_active_set_finance_without_subprocess(dataset_finance):
   
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    X, y = dataset_finance[0], dataset_finance[1]

    n_samples = X.shape[0]

    primary = Regression(loss='square', penalty='l1',
                             fit_intercept=False,
                             lambda_1=2697 / n_samples,
                             max_iter=500, verbose=False)

    secondary = Regression(loss='square', penalty='l1',
                             fit_intercept=False,
                             lambda_1=2697 / n_samples,
                             max_iter=500, verbose=False)

    type(primary).__name__ = "Lasso"
    type(secondary).__name__ = "Lasso"

    fit_large_feature_number(primary, secondary, X, y)

def test_active_set_finance_without_subprocess_intercept(dataset_finance):
   
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    X, y = dataset_finance[0], dataset_finance[1]

    n_samples = X.shape[0]

    primary = Regression(loss='square', penalty='l1',
                             fit_intercept=True,
                             lambda_1=0.1 / n_samples,
                             max_iter=10, verbose=False)

    secondary = Regression(loss='square', penalty='l1',
                             fit_intercept=True,
                             lambda_1=0.1 / n_samples,
                             max_iter=10, verbose=False)

    type(primary).__name__ = "Lasso"
    type(secondary).__name__ = "Lasso"

    fit_large_feature_number(primary, secondary, X, y)

def test_fit_large_feature_number():
    # Test case with small number of features
    X = np.random.rand(3, 2)
    labels = np.array([0, 1, 0])
    estimator = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    aux = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    type(estimator).__name__ = "Lasso"
    type(aux).__name__ = "Lasso"
    result = fit_large_feature_number(estimator, aux, X, labels)
    assert isinstance(result, Regression)
    assert result.coef_.shape == (2,)
    assert result.intercept_ is not None
    assert result.intercept_ == 0

    # Test case with large number of features and dense matrix
    X = np.random.rand(100, 1001)
    labels = np.random.randint(2, size=100)
    estimator = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    aux = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    type(estimator).__name__ = "Lasso"
    type(aux).__name__ = "Lasso"
    result = fit_large_feature_number(estimator, aux, X, labels)
    assert isinstance(result, Regression)
    assert result.coef_.shape == (1001,)
    assert result.intercept_ is not None
    assert result.intercept_ != 0

    # Test case with large number of features and sparse matrix
    X = scipy.sparse.csr_matrix(np.random.rand(100, 1001))
    labels = np.random.randint(2, size=100)
    estimator = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    aux = Regression(lambda_1=0.1, fit_intercept=True, tol=1e-4, verbose=True)
    type(estimator).__name__ = "Lasso"
    type(aux).__name__ = "Lasso"
    result = fit_large_feature_number(estimator, aux, X, labels)
    assert isinstance(result, Regression)
    assert result.coef_.shape == (1001,)
    assert result.intercept_ is not None
    assert result.intercept_ != 0
