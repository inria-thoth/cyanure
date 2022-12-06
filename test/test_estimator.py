import pytest

from sklearn.utils.estimator_checks import check_estimator

from cyanure.estimators import LogisticRegression, Regression, Classifier, LinearSVC, L1Logistic, Lasso


@pytest.mark.parametrize(
    "estimator",
    [LogisticRegression(verbose=False), Regression(verbose=False), Classifier(verbose=False), LinearSVC(verbose=False), Lasso(verbose=False), L1Logistic(verbose=False)]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
