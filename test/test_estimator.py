import pytest

from sklearn.utils.estimator_checks import check_estimator

from cyanure.estimators import LogisticRegression, Regression, Classifier, LinearSVC, L1Logistic, Lasso


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(), Regression(), Classifier(), LinearSVC(), Lasso(), L1Logistic()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
