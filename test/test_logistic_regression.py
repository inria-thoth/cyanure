import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.datasets import load_iris

import scipy.sparse as sp
from scipy import linalg, optimize, sparse

from cyanure.cyanure import LogisticRegression

import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal
import scipy.sparse as sp
from scipy import linalg, optimize, sparse

import pytest

from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _IS_32BIT
from sklearn.utils._testing import ignore_warnings
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import scale
from sklearn.utils._testing import skip_if_no_parallel

from sklearn.exceptions import ConvergenceWarning

import pytest

X = np.array([[-1, 0], [0, 1], [1, 1]]).astype("float64")
X_sp = sp.csr_matrix(X)
Y1 = np.array([0, 1, 1]).astype("float64")
Y2 = np.array([2, 1, 0]).astype("float64")
iris = load_iris()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(iris.target_names[iris.target])

solvers = ('ista', 'fista', 'catalyst-ista', 'qning-ista', 'svrg', 'catalyst-svrg', 
            'qning-svrg', 'acc-svrg', 'miso', 'catalyst-miso', 'qning-miso')


def check_predictions(clf, X, y):
    """Check that the model is able to fit the classification data"""
    n_samples = len(y)
    classes = np.unique(y)
    n_classes = classes.shape[0]

    predicted = clf.fit(X, y).predict(X)
    assert_array_equal(clf.classes_, classes)

    assert predicted.shape == (n_samples,)
    assert_array_equal(predicted, y)

    probabilities = clf.predict_proba(X)
    assert probabilities.shape == (n_samples, n_classes)
    assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))
    assert_array_equal(probabilities.argmax(axis=1), y)


def test_predict_2_classes():
    # Simple sanity check on a 2 classes dataset
    # Make sure it predicts the correct result on simple datasets.
    check_predictions(LogisticRegression(), X, Y1)
    check_predictions(LogisticRegression(), X_sp, Y1)

    check_predictions(LogisticRegression(lambda_1=0.001), X, Y1)
    check_predictions(LogisticRegression(lambda_1=0.001), X_sp, Y1)

    check_predictions(LogisticRegression(fit_intercept=False), X, Y1)
    check_predictions(LogisticRegression(fit_intercept=False), X_sp, Y1)


def test_error():
    # Test for appropriate exception on errors
    msg = "Penalty term must be positive"

    with pytest.raises(ValueError, match=msg):
        LogisticRegression(lambda_1=-1).fit(X, Y1)

    with pytest.raises(ValueError, match=msg):
        LogisticRegression(lambda_1="test").fit(X, Y1)

    for LR in [LogisticRegression]:
        msg = "Tolerance for stopping criteria must be positive"

        with pytest.raises(ValueError, match=msg):
            LR(tol=-1).fit(X, Y1)

        with pytest.raises(ValueError, match=msg):
            LR(tol="test").fit(X, Y1)

        msg = "Maximum number of iteration must be positive"

        with pytest.raises(ValueError, match=msg):
            LR(max_iter=-1).fit(X, Y1)

        with pytest.raises(ValueError, match=msg):
            LR(max_iter="test").fit(X, Y1)


def test_predict_3_classes():
    check_predictions(LogisticRegression(lambda_1=0.1), X, Y2)
    check_predictions(LogisticRegression(lambda_1=0.1), X_sp, Y2)


def test_predict_iris():
    # Test logistic regression with the iris dataset
    n_samples, n_features = iris.data.shape

    target = le.transform(iris.target_names[iris.target])

    # Test that both multinomial and OvR solvers handle
    # multiclass data correctly and give good accuracy
    # score (>0.95) for the training data.
    for clf in [
        LogisticRegression(lambda_1=1/(2*len(iris.data)*iris.data.shape[0]), solver="catalyst-miso"),
        LogisticRegression(lambda_1=1/(2*len(iris.data)*iris.data.shape[0]), solver="qning-ista"),
        LogisticRegression(
            lambda_1=1/(2*len(iris.data)*iris.data.shape[0]), solver="catalyst-svrg"),
        LogisticRegression(
            lambda_1=1/(2*len(iris.data)*iris.data.shape[0]), solver="catalyst-miso", tol=1e-2, random_state=42
        ),
        LogisticRegression(
            lambda_1=1/(2*len(iris.data)*iris.data.shape[0]),
            solver="qning-ista",
            tol=1e-2,
            random_state=42,
        ),
    ]:
        clf.fit(iris.data, target)
        assert_array_equal(np.unique(target), clf.classes_)

        pred = clf.predict(iris.data)
        assert np.mean(pred == target) > 0.95

        probabilities = clf.predict_proba(iris.data)
        assert_array_almost_equal(probabilities.sum(axis=1), np.ones(n_samples))

        pred = probabilities.argmax(axis=1)
        assert np.mean(pred == target) > 0.95


def test_sparsify():
    # Test sparsify and densify members.
    target = iris.target_names[iris.target]
    clf = LogisticRegression(random_state=0).fit(iris.data, target)

    pred_d_d = clf.decision_function(iris.data)

    clf.sparsify()
    assert sp.issparse(clf.coef_)
    pred_s_d = clf.decision_function(iris.data)

    sp_data = sp.coo_matrix(iris.data)
    pred_s_s = clf.decision_function(sp_data)

    clf.densify()
    pred_d_s = clf.decision_function(sp_data)

    assert_array_almost_equal(pred_d_d, pred_s_d)
    assert_array_almost_equal(pred_d_d, pred_s_s)
    assert_array_almost_equal(pred_d_d, pred_d_s)

def test_inconsistent_input():
    # Test that an exception is raised on inconsistent input
    rng = np.random.RandomState(0)
    X_ = rng.random_sample((5, 10))
    y_ = np.ones(X_.shape[0])
    y_[0] = 0

    clf = LogisticRegression(random_state=0)

    # Wrong dimensions for training data
    y_wrong = y_[:-1]

    with pytest.raises(ValueError):
        clf.fit(X, y_wrong)

    # Wrong dimensions for test data
    with pytest.raises(ValueError):
        clf.fit(X_, y_).predict(rng.random_sample((3, 12)))

def test_write_parameters():
    # Test that we can write to coef_ and intercept_
    clf = LogisticRegression(random_state=0)
    clf.fit(X, Y1)
    clf.coef_[:] = 0
    clf.intercept_[:] = 0
    assert_array_almost_equal(clf.decision_function(X), 0)


def test_nan():
    # Test proper NaN handling.
    # Regression test for Issue #252: fit used to go into an infinite loop.
    Xnan = np.array(X, dtype=np.float64)
    Xnan[0, 1] = np.nan
    logistic = LogisticRegression(random_state=0)

    with pytest.raises(ValueError):
        logistic.fit(Xnan, Y1)

def test_logistic_regression_solvers():
    X, y = make_classification(n_features=10, n_informative=5, random_state=0)

    params = dict(fit_intercept=False, random_state=42)
    ncg = LogisticRegression(solver="ista", **params)
    lbf = LogisticRegression(solver="svrg", **params)
    lib = LogisticRegression(solver="miso", **params)
    ncg.fit(X, y)
    lbf.fit(X, y)
    lib.fit(X, y)
    assert_array_almost_equal(ncg.coef_, lib.coef_, decimal=2)
    assert_array_almost_equal(lib.coef_, lbf.coef_, decimal=2)
    assert_array_almost_equal(ncg.coef_, lbf.coef_, decimal=2)


def test_logistic_regression_solvers_multiclass():
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10, n_classes=3, random_state=0
    )
    params = dict(fit_intercept=False, random_state=42)
    ncg = LogisticRegression(solver="ista", **params)
    lbf = LogisticRegression(solver="svrg", **params)
    lib = LogisticRegression(solver="miso", **params)
    ncg.fit(X, y)
    lbf.fit(X, y)
    lib.fit(X, y)
    assert_array_almost_equal(ncg.coef_, lib.coef_, decimal=2)
    assert_array_almost_equal(lib.coef_, lbf.coef_, decimal=2)
    assert_array_almost_equal(ncg.coef_, lbf.coef_, decimal=2)


@pytest.mark.parametrize("solver", solvers)
def test_logistic_regression_multinomial(solver):
    # TODO Voir avec Julien
    # Tests for the multinomial option in logistic regression

    # Some basic attributes of Logistic Regression
    n_samples, n_features, n_classes = 50, 20, 3
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_classes=n_classes,
        random_state=0,
    )

    X = StandardScaler(with_mean=False).fit_transform(X)    
    # 'qning-miso' is used as a referenced
    solver_ref = "catalyst-svrg"
    ref_i = LogisticRegression(solver=solver, multi_class="multinomial")
    ref_w = LogisticRegression(
        solver=solver, multi_class="multinomial", fit_intercept=False)
    ref_i.fit(X, y)
    ref_w.fit(X, y)
    assert ref_i.coef_.shape == (n_features, n_classes)
    assert ref_w.coef_.shape == (n_features, n_classes)
    clf_i = LogisticRegression(
        solver=solver,
        multi_class="multinomial",
        random_state=42,
        max_iter=2000,
        tol=1e-7
    )
    clf_w = LogisticRegression(
        solver=solver,
        multi_class="multinomial",
        random_state=42,
        max_iter=2000,
        tol=1e-7,
        fit_intercept=False
    )
    clf_i.fit(X, y)
    clf_w.fit(X, y)
    assert clf_i.coef_.shape == (n_features, n_classes)
    assert clf_w.coef_.shape == (n_features, n_classes)

    # Compare solutions between lbfgs and the other solvers
    assert_allclose(ref_i.coef_, clf_i.coef_, rtol=1e-2)
    assert_allclose(ref_w.coef_, clf_w.coef_, rtol=1e-2)
    assert_allclose(ref_i.intercept_, clf_i.intercept_, rtol=1e-2)


def test_liblinear_decision_function_zero():
    # Test negative prediction when decision_function values are zero.
    # Liblinear predicts the positive class when decision_function values
    # are zero. This is a test to verify that we do not do the same.
    # See Issue: https://github.com/scikit-learn/scikit-learn/issues/3600
    # and the PR https://github.com/scikit-learn/scikit-learn/pull/3623
    X, y = make_classification(n_samples=5, n_features=5, random_state=0)
    clf = LogisticRegression(fit_intercept=False, solver="ista")
    clf.fit(X, y)

    # Dummy data such that the decision function becomes zero.
    X = np.zeros((5, 5))
    assert_array_equal(clf.predict(X), np.zeros(5))

def test_logreg_l1():
    # Because liblinear penalizes the intercept and saga does not, we do not
    # fit the intercept to make it possible to compare the coefficients of
    # the two models at convergence.
    rng = np.random.RandomState(42)
    n_samples = 50
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    X_noise = rng.normal(size=(n_samples, 3))
    X_constant = np.ones(shape=(n_samples, 2))
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    lr_liblinear = LogisticRegression(
        penalty="l1",
        lambda_1=1.0,
        solver="miso",
        fit_intercept=False,
        multi_class="ovr",
        tol=1e-10,
    )
    lr_liblinear.fit(X, y)

    lr_saga = LogisticRegression(
        penalty="l1",
        lambda_1=1.0,
        solver="svrg",
        fit_intercept=False,
        multi_class="ovr",
        max_iter=1000,
        tol=1e-10,
    )
    lr_saga.fit(X, y)
    assert_array_almost_equal(lr_saga.coef_, lr_liblinear.coef_)

    # Noise and constant features should be regularized to zero by the l1
    # penalty
    assert_array_almost_equal(lr_liblinear.coef_[-5:, 0], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[-5:, 0], np.zeros(5))


def test_logreg_l1_sparse_data():
    # Because liblinear penalizes the intercept and saga does not, we do not
    # fit the intercept to make it possible to compare the coefficients of
    # the two models at convergence.
    rng = np.random.RandomState(42)
    n_samples = 50
    X, y = make_classification(n_samples=n_samples, n_features=20, random_state=0)
    X_noise = rng.normal(scale=0.1, size=(n_samples, 3))
    X_constant = np.zeros(shape=(n_samples, 2))
    X = np.concatenate((X, X_noise, X_constant), axis=1)
    X[X < 1] = 0
    X = sparse.csr_matrix(X)

    lr_miso = LogisticRegression(
        penalty="l1",
        lambda_1=1.0,
        solver="miso",
        fit_intercept=False,
        multi_class="ovr",
        tol=1e-10,
    )
    lr_miso.fit(X, y)

    lr_saga = LogisticRegression(
        penalty="l1",
        lambda_1=1.0,
        solver="ista",
        fit_intercept=False,
        multi_class="ovr",
        max_iter=1000,
        tol=1e-10,
    )
    lr_saga.fit(X, y)
    assert_array_almost_equal(lr_saga.coef_, lr_miso.coef_)
    # Noise and constant features should be regularized to zero by the l1
    # penalty
    assert_array_almost_equal(lr_miso.coef_[-5:, 0], np.zeros(5))
    assert_array_almost_equal(lr_saga.coef_[-5:, 0], np.zeros(5))

    # Check that solving on the sparse and dense data yield the same results
    lr_saga_dense = LogisticRegression(
        penalty="l1",
        lambda_1=1.0,
        solver="ista",
        fit_intercept=False,
        multi_class="ovr",
        max_iter=1000,
        tol=1e-10,
    )
    lr_saga_dense.fit(X.toarray(), y)
    assert_array_almost_equal(lr_saga.coef_, lr_saga_dense.coef_)

def test_logreg_predict_proba_multinomial():
    X, y = make_classification(
        n_samples=10, n_features=100, random_state=0, n_classes=3, n_informative=30
    )

    # Predicted probabilities using the true-entropy loss should give a
    # smaller loss than those using the ovr method.
    clf_multi = LogisticRegression(multi_class="multinomial", solver="svrg")
    clf_multi.fit(X, y)
    clf_multi_loss = log_loss(y, clf_multi.predict_proba(X))
    clf_ovr = LogisticRegression(multi_class="ovr", solver="svrg")
    clf_ovr.fit(X, y)
    clf_ovr_loss = log_loss(y, clf_ovr.predict_proba(X))
    assert clf_ovr_loss > clf_multi_loss

@pytest.mark.parametrize("max_iter", np.arange(1, 5))
@pytest.mark.parametrize(
    "solver, message",
    [
        (
            "ista",
            "The max_iter was reached which means the coef_ did not converge",
        ),
        (
            "fista",
            "The max_iter was reached which means the coef_ did not converge",
        ),
        ("miso", "The max_iter was reached which means the coef_ did not converge"),
        ("svrg", "The max_iter was reached which means the coef_ did not converge"),
        ("catalyst-ista", "The max_iter was reached which means the coef_ did not converge"),
        ("qning-ista", "The max_iter was reached which means the coef_ did not converge"),
        ("catalyst-svrg", "The max_iter was reached which means the coef_ did not converge"),
        ("qning-svrg", "The max_iter was reached which means the coef_ did not converge"),
        ("acc-svrg", "The max_iter was reached which means the coef_ did not converge"),
        ("catalyst-miso", "The max_iter was reached which means the coef_ did not converge"),
        ("qning-miso", "The max_iter was reached which means the coef_ did not converge"),
    ],
)
def test_max_iter(max_iter, solver, message):
    # Test that the maximum number of iteration is reached
    X, y_bin = iris.data, iris.target.copy()
    y_bin[y_bin == 2] = 0

    lr = LogisticRegression(
        max_iter=max_iter,
        tol=1e-15,
        random_state=0,
        solver=solver,
    )
    with pytest.warns(ConvergenceWarning, match=message):
        lr.fit(X, y_bin)

    assert lr.n_iter_[0] == max_iter


@pytest.mark.parametrize("solver", solvers)
def test_n_iter(solver):
    # Test that self.n_iter_ has the correct format.
    X, y = iris.data, iris.target

    y_bin = y.copy()
    y_bin[y_bin == 2] = 0

    # OvR case
    n_classes = np.unique(y).shape[0]
    clf = LogisticRegression(
        tol=1e-2, solver=solver, lambda_1=1.0, random_state=42
    )
    clf.fit(X, y)
    assert clf.n_iter_.shape == (n_classes,)

    # multinomial case

@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("warm_start", (True, False))
@pytest.mark.parametrize("fit_intercept", (True, False))
@pytest.mark.parametrize("multi_class", ["ovr", "multinomial"])
def test_warm_start(solver, warm_start, fit_intercept, multi_class):
    # A 1-iteration second fit on same data should give almost same result
    # with warm starting, and quite different result without warm starting.
    # Warm starting does not work with liblinear solver.
    X, y = iris.data, iris.target

    clf = LogisticRegression(
        tol=1e-4,
        multi_class=multi_class,
        warm_start=warm_start,
        solver=solver,
        random_state=42,
        fit_intercept=fit_intercept,
    )
    with ignore_warnings(category=ConvergenceWarning):
        clf.fit(X, y)
        coef_1 = clf.coef_

        clf.max_iter = 1
        clf.fit(X, y)
    cum_diff = np.sum(np.abs(coef_1 - clf.coef_))
    msg = (
        "Warm starting issue with %s solver in %s mode "
        "with fit_intercept=%s and warm_start=%s"
        % (solver, multi_class, str(fit_intercept), str(warm_start))
    )
    if warm_start:
        assert 2.0 > cum_diff, msg
    else:
        assert cum_diff > 2.0, msg


# alpha=1e-3 is time consuming
@pytest.mark.parametrize("penalty", ["l1", "l2"])
@pytest.mark.parametrize("alpha", np.logspace(-1, 1, 3))
def test_ista_vs_svrg(penalty, alpha):
    iris = load_iris()
    X, y = iris.data, iris.target
    X = np.concatenate([X] * 3)
    y = np.concatenate([y] * 3)

    X_bin = X[y <= 1]
    y_bin = y[y <= 1] * 2 - 1

    X_sparse, y_sparse = make_classification(
        n_samples=50, n_features=20, random_state=0
    )
    X_sparse = sparse.csr_matrix(X_sparse)

    for (X, y) in ((X_bin, y_bin), (X_sparse, y_sparse)):
        n_samples = X.shape[0]
        saga = LogisticRegression(
            lambda_1=1 / (n_samples * alpha),
            solver="ista",
            max_iter=1000,
            fit_intercept=True,
            penalty=penalty,
            random_state=0,
            tol=1e-24,
            n_threads=-1,
        )

        liblinear = LogisticRegression(
            lambda_1=1 / (n_samples * alpha),
            solver="svrg",
            max_iter=1000,
            fit_intercept=True,
            penalty=penalty,
            random_state=0,
            tol=1e-24,
            n_threads=-1,
        )

        saga.fit(X, y)
        liblinear.fit(X, y)
        # Convergence for alpha=1e-3 is very slow
        assert_array_almost_equal(saga.coef_, liblinear.coef_, 2)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_dtype_match(solver, fit_intercept):
    # Test that np.float32 input data is not cast to np.float64 when possible
    # and that the output is approximately the same no matter the input format.


    out32_type = np.float32

    X_32 = np.array(X).astype(np.float32)
    y_32 = np.array(Y1).astype(np.float32)
    X_64 = np.array(X).astype(np.float64)
    y_64 = np.array(Y1).astype(np.float64)
    X_sparse_32 = sp.csr_matrix(X, dtype=np.float32)
    X_sparse_64 = sp.csr_matrix(X, dtype=np.float64)
    solver_tol = 5e-4

    lr_templ = LogisticRegression(
        solver=solver,
        random_state=42,
        tol=solver_tol,
        fit_intercept=fit_intercept,
    )

    # Check 32-bit type consistency
    lr_32 = clone(lr_templ)
    lr_32.fit(X_32, y_32)
    assert lr_32.coef_.dtype == out32_type

    # Check 32-bit type consistency with sparsity
    lr_32_sparse = clone(lr_templ)
    lr_32_sparse.fit(X_sparse_32, y_32)
    
    assert lr_32_sparse.coef_.dtype == out32_type

    # Check 64-bit type consistency
    lr_64 = clone(lr_templ)
    lr_64.fit(X_64, y_64)
    assert lr_64.coef_.dtype == np.float64

    # Check 64-bit type consistency with sparsity
    lr_64_sparse = clone(lr_templ)
    lr_64_sparse.fit(X_sparse_64, y_64)
    assert lr_64_sparse.coef_.dtype == np.float64

    # solver_tol bounds the norm of the loss gradient
    # dw ~= inv(H)*grad ==> |dw| ~= |inv(H)| * solver_tol, where H - hessian
    #
    # See https://github.com/scikit-learn/scikit-learn/pull/13645
    #
    # with  Z = np.hstack((np.ones((3,1)), np.array(X)))
    # In [8]: np.linalg.norm(np.diag([0,2,2]) + np.linalg.inv((Z.T @ Z)/4))
    # Out[8]: 1.7193336918135917

    # factor of 2 to get the ball diameter
    atol = 2 * 1.72 * solver_tol
    if True: # os.name == "nt" and _IS_32BIT:
        # FIXME from scikit-learn test
        atol = 1e-1

    # Check accuracy consistency
    # TODO Qning divergence + small divergence ista/fista
    assert_allclose(lr_32.coef_, lr_64.coef_.astype(np.float32), atol=atol)

    assert_allclose(lr_32.coef_, lr_32_sparse.coef_, atol=atol)
    assert_allclose(lr_64.coef_, lr_64_sparse.coef_, atol=atol)

def test_warm_start_converge_LR():
    # Test to see that the logistic regression converges on warm start,
    # with multi_class='multinomial'. Non-regressive test for #10836

    rng = np.random.RandomState(0)
    X = np.concatenate((rng.randn(100, 2) + [1, 1], rng.randn(100, 2)))
    y = np.array([1] * 100 + [-1] * 100)
    lr_no_ws = LogisticRegression(
        multi_class="multinomial", solver="miso", warm_start=False, random_state=0
    )
    lr_ws = LogisticRegression(
        multi_class="multinomial", solver="miso", warm_start=True, random_state=0
    )

    lr_no_ws_loss = log_loss(y, lr_no_ws.fit(X, y).predict_proba(X))
    for i in range(5):
        lr_ws.fit(X, y)
    lr_ws_loss = log_loss(y, lr_ws.predict_proba(X))
    assert_allclose(lr_no_ws_loss, lr_ws_loss, rtol=1e-4)


def test_elastic_net_coeffs():
    # make sure elasticnet penalty gives different coefficients from l1 and l2
    # with saga solver (l1_ratio different from 0 or 1)
    X, y = make_classification(random_state=0)

    alpha = 2
    n_samples = 100
    lambda_1 = 1 / (n_samples * alpha)
    lambda_2 = 1 / (n_samples * alpha)
    coeffs = list()
    for penalty in ("elasticnet", "l1", "l2"):
        if penalty in ["l1", "l2"]:
            lambda_2 = 0
        lr = LogisticRegression(
            penalty=penalty, lambda_1=lambda_1, solver="qning-miso", random_state=0, lambda_2=lambda_2
        )
        lr.fit(X, y)
        coeffs.append(lr.coef_)

    elastic_net_coeffs, l1_coeffs, l2_coeffs = coeffs
    # make sure coeffs differ by at least .1
    assert not np.allclose(elastic_net_coeffs, l1_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(elastic_net_coeffs, l2_coeffs, rtol=0, atol=0.1)
    assert not np.allclose(l2_coeffs, l1_coeffs, rtol=0, atol=0.1)


@pytest.mark.parametrize("alpha", [0.001, 0.1, 1, 10, 100, 1000, 1e6])
@pytest.mark.parametrize("penalty, lambda_1, lambda_2", [("l1", 1, 0), ("l2", 0, 1)])
def test_elastic_net_l1_l2_equivalence(alpha, penalty, lambda_1, lambda_2):
    # Make sure elasticnet is equivalent to l1 when l1_ratio=1 and to l2 when
    # l1_ratio=0.
    X, y = make_classification(random_state=0)

    lr_enet = LogisticRegression(
        penalty="elasticnet", lambda_1=lambda_1*alpha, lambda_2=lambda_2*alpha, solver="qning-miso", random_state=0
    )
    lr_expected = LogisticRegression(
        penalty=penalty, lambda_1=alpha, solver="qning-miso", random_state=0
    )
    lr_enet.fit(X, y)
    lr_expected.fit(X, y)

    assert_array_almost_equal(lr_enet.coef_, lr_expected.coef_)


@pytest.mark.parametrize("lambda_1", [0.001, 1, 100, 1e6])
def test_elastic_net_vs_l1_l2(lambda_1):
    # Make sure that elasticnet with grid search on l1_ratio gives same or
    # better results than just l1 or just l2.

    X, y = make_classification(500, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    param_grid = {"lambda_1": np.linspace(0, 0.1, 5), "lambda_2": 1 - np.linspace(0, 0.1, 5)}

    enet_clf = LogisticRegression(
        penalty="elasticnet", lambda_1=lambda_1, solver="qning-miso", random_state=0
    )
    gs = GridSearchCV(enet_clf, param_grid, refit=True)

    l1_clf = LogisticRegression(penalty="l1", lambda_1=lambda_1, solver="qning-miso", random_state=0)
    l2_clf = LogisticRegression(penalty="l2", lambda_1=lambda_1, solver="qning-miso", random_state=0)

    for clf in (gs, l1_clf, l2_clf):
        clf.fit(X_train, y_train)

    assert gs.score(X_test, y_test) >= l1_clf.score(X_test, y_test)
    assert gs.score(X_test, y_test) >= l2_clf.score(X_test, y_test)

@pytest.mark.parametrize("C", np.logspace(-2, 2, 4))
@pytest.mark.parametrize("multiplier", [0.1, 0.5, 0.9])
def test_LogisticRegression_elastic_net_objective(C, multiplier):
    # Check that training with a penalty matching the objective leads
    # to a lower objective.
    # Here we train a logistic regression with l2 (a) and elasticnet (b)
    # penalties, and compute the elasticnet objective. That of a should be
    # greater than that of b (both objectives are convex).
    n_samples=1000
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=20,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        random_state=0,
    )
    X = scale(X)
    lambda_1 = 1.0 / C / n_samples

    lr_enet = LogisticRegression(
        penalty="elasticnet",
        solver="qning-miso",
        random_state=0,
        lambda_1=lambda_1 * multiplier,
        lambda_2=lambda_1 * (1 - multiplier),
        fit_intercept=False,
    )
    lr_l2 = LogisticRegression(
        penalty="l2", solver="qning-miso", random_state=0, lambda_1=lambda_1, fit_intercept=False
    )
    lr_enet.fit(X, y)
    lr_l2.fit(X, y)

    def enet_objective(lr):
        coef = lr.coef_.ravel()
        obj = lambda_1 * log_loss(y, lr.predict_proba(X))
        obj += multiplier * np.sum(np.abs(coef))
        obj += (1.0 - multiplier) * 0.5 * np.dot(coef, coef)
        return obj

    assert enet_objective(lr_enet) < enet_objective(lr_l2)

@pytest.mark.parametrize("C", np.logspace(-2, 2, 4))
@pytest.mark.parametrize("multiplier", [0.1, 0.5, 0.9])
def test_elastic_net_versus_sgd(C, multiplier):
    # Compare elasticnet penalty in LogisticRegression() and SGD(loss='log')
    n_samples = 500
    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=5,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        random_state=1,
    )
    X = scale(X)
    lambda_1 = 1.0 / C / n_samples

    sgd = SGDClassifier(
        penalty="elasticnet",
        random_state=1,
        fit_intercept=False,
        tol=-np.inf,
        max_iter=2000,
        l1_ratio=multiplier,
        alpha=lambda_1,
        loss="log",
    )
    log = LogisticRegression(
        penalty="elasticnet",
        random_state=1,
        fit_intercept=False,
        tol=1e-5,
        max_iter=2000,
        lambda_1=lambda_1 * multiplier,
        lambda_2=lambda_1 * (1 - multiplier),
        solver="qning-svrg",
    )

    sgd.fit(X, y)
    log.fit(X, y)
    assert_array_almost_equal(sgd.coef_, np.transpose(log.coef_), decimal=1)

@pytest.mark.parametrize(
    "est",
    [
        LogisticRegression(random_state=0, max_iter=500),
    ],
    ids=lambda x: x.__class__.__name__,
)
@pytest.mark.parametrize("solver", solvers)
def test_logistic_regression_multi_class_auto(est, solver):
    # check multi_class='auto' => multi_class='ovr' iff binary y or liblinear

    def fit(X, y, **kw):
        return clone(est).set_params(**kw).fit(X, y)

    scaled_data = scale(iris.data)
    X = scaled_data[::10]
    X2 = scaled_data[1::10]
    y_multi = iris.target[::10]
    y_bin = y_multi == 0
    est_auto_bin = fit(X, y_bin, multi_class="auto", solver=solver)
    est_ovr_bin = fit(X, y_bin, multi_class="ovr", solver=solver)
    assert_allclose(est_auto_bin.coef_, est_ovr_bin.coef_)
    assert_allclose(est_auto_bin.predict_proba(X2), est_ovr_bin.predict_proba(X2))

    est_auto_multi = fit(X, y_multi, multi_class="auto", solver=solver)
    est_multi_multi = fit(X, y_multi, multi_class="multinomial", solver=solver)
    assert_allclose(est_auto_multi.coef_, est_multi_multi.coef_)
    assert_allclose(
        est_auto_multi.predict_proba(X2), est_multi_multi.predict_proba(X2)
    )

    # Make sure multi_class='ovr' is distinct from ='multinomial'
    assert not np.allclose(
        est_auto_bin.coef_,
        fit(X, y_bin, multi_class="multinomial", solver=solver).coef_,
    )
    assert not np.allclose(
        est_auto_bin.coef_,
        fit(X, y_multi, multi_class="multinomial", solver=solver).coef_,
    )



@pytest.mark.parametrize("solver", solvers)
def test_penalty_none(solver):
    # - Make sure warning is raised if penalty='none' and lambda_1 is set to a
    #   non-default value.
    # - Make sure setting penalty='none' is equivalent to setting lambda_1=np.inf with
    #   l2 penalty.
    X, y = make_classification(n_samples=1000, random_state=0)

    msg = "Setting penalty='none' will ignore the lambda_1"
    lr = LogisticRegression(penalty="none", solver=solver, lambda_1=4)
    with pytest.warns(UserWarning, match=msg):
        lr.fit(X, y)

    lr_none = LogisticRegression(penalty="none", solver=solver, random_state=0)
    lr_l2_C_inf = LogisticRegression(
        penalty="l2", lambda_1=1/np.inf, solver=solver, random_state=0
    )
    pred_none = lr_none.fit(X, y).predict(X)
    pred_l2_C_inf = lr_l2_C_inf.fit(X, y).predict(X)
    assert_array_equal(pred_none, pred_l2_C_inf)


@pytest.mark.parametrize("solver", solvers)
def test_large_sparse_matrix(solver):
    # Solvers either accept large sparse matrices, or raise helpful error.
    # Non-regression test for pull-request #21093.

    # generate sparse matrix with int64 indices
    X = sp.rand(20, 10, format="csr")
    for attr in ["indices", "indptr"]:
        setattr(X, attr, getattr(X, attr).astype("int64"))
    y = np.random.randint(2, size=X.shape[0])

    LogisticRegression(solver=solver).fit(X, y)