import pytest

from sklearn.utils.estimator_checks import check_estimator

from cyanure.cyanure import LogisticRegression, BinaryClassifier, Regression


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(), BinaryClassifier(), Regression()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
