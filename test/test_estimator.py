import pytest

from sklearn.utils.estimator_checks import check_estimator

from cyanure.cyanure import LogisticRegression, BinaryClassifier, Regression, MultiClassifier


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(), BinaryClassifier(), Regression(), MultiClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
