"""
import pytest

from sklearn.utils.estimator_checks import check_estimator

from cyanure.estimators import LogisticRegression, Regression, Classifier, LinearSVC, L1Logistic, Lasso

from sklearn.datasets import make_blobs
from sklearn.utils.estimator_checks import _pairwise_estimator_convert_X, _enforce_estimator_tags_y
from sklearn.utils._testing import set_random_state, create_memmap_backed_data
from sklearn.base import clone

estimator_orig = LogisticRegression()

def test_debug():
    """Check if self is returned when calling fit."""
    X, y = make_blobs(random_state=0, n_samples=21)
    # some want non-negative input
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator

def test_debug_2():
    """Check if self is returned when calling fit."""
    X, y = make_blobs(random_state=0, n_samples=21)
    # some want non-negative input
    X -= X.min()
    X = _pairwise_estimator_convert_X(X, estimator_orig)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    X, y = create_memmap_backed_data([X, y], aligned=True)

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator

@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(), Regression(), Classifier(), LinearSVC(), Lasso(), L1Logistic()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
"""