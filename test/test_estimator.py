import pytest

from sklearn.utils.estimator_checks import check_estimator

from cyanure.cyanure import LogisticRegression, Regression, MultiClassifier, LinearSVC, L1Logistic, Lasso


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(), Regression(), MultiClassifier(), LinearSVC(), Lasso(), L1Logistic()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
