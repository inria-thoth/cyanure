import warnings

import numpy as np
import scipy.sparse

from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

import cyanure_lib


def preprocess(X, centering=False, normalize=True, columns=False):
    """Perform in-place centering or normalization, either of columns or rows
    of the input matrix X

    Parameters
    ----------
    X : numpy array, or scipy sparse CSR matrix
        input matrix

    centering : boolean, default=False
        perform a centering operation

    normalize : boolean, default=True
        l2-normalization

    columns : boolean, default=False
        operates on rows (False) or columns (True).
    """

    if scipy.sparse.issparse(X):
        training_data_fortran = X.T
    else:
        training_data_fortran = np.asfortranarray(X.T)
    return cyanure_lib.preprocess_(training_data_fortran, centering, normalize, not columns)


def check_labels(y, estimator):
    le = None

    # TODO check if relevant
    if estimator._estimator_type == "classifier":
        y_type = type_of_target(y)
        if y_type not in [
            "binary",
            "multiclass",
            "multiclass-multioutput",
            "multilabel-indicator",
            "multilabel-sequences",
        ]:
            raise ValueError("Unknown label type: %r" % y_type)

        if np.issubdtype(type(y[0]), np.str_):
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)
        elif np.isfinite(y[0]) and (type(y[0]) == "float64" or type(y[0]) == "float32"):
            raise ValueError("Unknown label type: " + str(y.dtype))
        elif np.isfinite(y[0]) and (type(y[0]) != "int64"):
            y = y.astype("int64")
    else:
        y = np.asfortranarray(y, 'float64')

    if False in np.isfinite(y):
        raise ValueError(
            "Input contains NaN, infinity or a value too large for dtype('float64').")

    if len(np.unique(y)) == 1:
        raise ValueError("There is only one class in the y.")

    return y, le


def check_input_type(X, y, estimator):
    le = None

    if np.iscomplexobj(X) or np.iscomplexobj(y):
        raise ValueError("Complex data not supported")

    if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(y):
        # TODO Flexible dtype
        X = np.asfortranarray(X, 'float64')

        y, le = check_labels(y, estimator)

        if False in np.isfinite(X):
            raise ValueError(
                "Input contains NaN, infinity or a value too large for dtype('float64').")

    else:
        if scipy.sparse.issparse(X) and X.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")
        if scipy.sparse.issparse(y) and y.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")

    return X, y, le


def check_positive_parameter(parameter, message):
    if not isinstance(parameter, (int, float)):
        raise ValueError(message)

    if isinstance(parameter, (int, float)) and parameter < 0:
        raise ValueError(message)


def check_parameters(estimator):

    check_positive_parameter(
        estimator.tol, "Tolerance for stopping criteria must be positive")
    check_positive_parameter(estimator.max_iter,
                             "Maximum number of iteration must be positive")
    check_positive_parameter(estimator.lambd,
                             "Penalty term must be positive")


def check_input(X, y, estimator):
    if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(y):
        X = np.array(X)
        y = np.array(y)

    if X.ndim == 1:
        raise ValueError("The training array has only one dimension.")

    if X.shape[0] == 0:
        raise ValueError("Empty training array")

    if len(X.shape) > 1 and X.shape[1] == 0:
        raise ValueError("0 feature(s) (shape=(" + str(X.shape[0]) + ", 0)) while a minimum of " + str(
            X.shape[0]) + " is required.")

    if y.shape[0] != X.shape[0]:
        raise ValueError(
            "X and y should have the same number of observations")

    if X.shape[0] == 1:
        raise ValueError("There should have more than 1 sample")

    if not estimator._get_tags()["multioutput"] and not estimator._get_tags()["multioutput_only"] and y.ndim > 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected", DataConversionWarning)

    X, y, le = check_input_type(
        X, y, estimator)

    check_parameters(estimator)

    return X, y, le
