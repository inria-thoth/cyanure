import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.datasets import load_iris

import scipy.sparse as sp
from scipy import linalg, optimize, sparse

from cyanure.cyanure import LogisticRegression

import pytest

X = np.array([[-1, 0], [0, 1], [1, 1]]).astype("float64")
X_sp = sp.csr_matrix(X)
Y1 = np.array([0, 1, 1]).astype("float64")
Y2 = np.array([2, 1, 0]).astype("float64")
iris = load_iris()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(iris.target_names[iris.target])


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

    check_predictions(LogisticRegression(lambd=0.001), X, Y1)
    check_predictions(LogisticRegression(lambd=0.001), X_sp, Y1)

    check_predictions(LogisticRegression(fit_intercept=False), X, Y1)
    check_predictions(LogisticRegression(fit_intercept=False), X_sp, Y1)


def test_error():
    # Test for appropriate exception on errors
    msg = "Penalty term must be positive"

    with pytest.raises(ValueError, match=msg):
        LogisticRegression(lambd=-1).fit(X, Y1)

    with pytest.raises(ValueError, match=msg):
        LogisticRegression(lambd="test").fit(X, Y1)

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
    check_predictions(LogisticRegression(lambd=0.1), X, Y2)
    check_predictions(LogisticRegression(lambd=0.1), X_sp, Y2)


def test_predict_iris():
    # Test logistic regression with the iris dataset
    n_samples, n_features = iris.data.shape

    target = le.transform(iris.target_names[iris.target])

    # Test that both multinomial and OvR solvers handle
    # multiclass data correctly and give good accuracy
    # score (>0.95) for the training data.
    for clf in [
        LogisticRegression(lambd=1/(2*len(iris.data)*iris.data.shape[0]), solver="catalyst-miso"),
        LogisticRegression(lambd=1/(2*len(iris.data)*iris.data.shape[0]), solver="qning-ista"),
        LogisticRegression(
            lambd=1/(2*len(iris.data)*iris.data.shape[0]), solver="catalyst-svrg"),
        LogisticRegression(
            lambd=1/(2*len(iris.data)*iris.data.shape[0]), solver="catalyst-miso", tol=1e-2, random_state=42
        ),
        LogisticRegression(
            lambd=1/(2*len(iris.data)*iris.data.shape[0]),
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
