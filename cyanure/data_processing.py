import warnings
import numbers
import platform

import numpy as np
import scipy.sparse

from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

import cyanure_lib

from cyanure.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


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
        if platform.system() == "Windows":
            training_data_fortran.indptr = training_data_fortran.indptr.astype(np.float64).astype(np.intc)
            training_data_fortran.indices = training_data_fortran.indices.astype(np.float64).astype(np.intc)
    else:
        training_data_fortran = np.asfortranarray(X.T)
    return cyanure_lib.preprocess_(training_data_fortran, centering, normalize, not columns)


def check_labels(y, estimator):
    le = None

    if estimator._estimator_type == "classifier":
        y_type = type_of_target(y)
        if y_type not in [
            "binary",
            "multiclass"
        ]:
            raise ValueError("Unknown label type: %r" % y_type)

        if np.issubdtype(type(y[0]), np.str_):
            le = LabelEncoder()
            le.fit(y)
            y = le.transform(y)
    else:
        if type(y[0]) != np.float32 and type(y[0]) != np.float64:
            logger.info("The labels have been converted in float64")
            y = y.astype('float64')

    if False in np.isfinite(y):
        raise ValueError(
            "Input contains NaN, infinity or a value too large for dtype('float64').")

    if len(np.unique(y)) == 1:
        raise ValueError("There is only one class in the y.")

    return y, le

def get_element(array):
    element = array[0]
    for i in range(len(array.shape) - 1):
        element = element[0]
    return element

def check_input_type(X, y, estimator):
    le = None

    if np.iscomplexobj(X) or np.iscomplexobj(y):
        raise ValueError("Complex data not supported")

    if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(y):
        x_element = get_element(X)
        if type(x_element) != np.float32 and type(x_element) != np.float64:
            
            logger.info("The features have been converted in float64")
            X = np.asfortranarray(X, 'float64')
        else:
            X = np.asfortranarray(X)

        y, le = check_labels(y, estimator)

        if False in np.isfinite(X):
            raise ValueError(
                "Input contains NaN, infinity or a value too large for dtype('float64').")

    else:
        if scipy.sparse.issparse(X) and X.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")
        if scipy.sparse.issparse(y) and y.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")

        if platform.system() == "Windows":
            if scipy.sparse.issparse(X):
                X.indptr = X.indptr.astype(np.float64).astype(np.intc)
                X.indices = X.indices.astype(np.float64).astype(np.intc)
            if scipy.sparse.issparse(y):
                y.indptr = y.indptr.astype(np.float64).astype(np.intc)
                y.indices = y.indices.astype(np.float64).astype(np.intc)

        

    return X, y, le


def check_positive_parameter(parameter, message):
    if not isinstance(parameter, numbers.Number):
        raise ValueError(message)

    if isinstance(parameter, numbers.Number) and parameter < 0:
        raise ValueError(message)


def check_parameters(estimator):

    check_positive_parameter(
        estimator.tol, "Tolerance for stopping criteria must be positive")
    check_positive_parameter(estimator.max_iter,
                             "Maximum number of iteration must be positive")
    check_positive_parameter(estimator.lambda_1,
                             "Penalty term must be positive")

    # Verify that it is not the default value
    if (estimator.penalty is None or estimator.penalty == "none") and estimator.lambda_1 != 0.1:
        warnings.warn("Setting penalty='none' will ignore the lambda_1")


def check_input_fit(X, y, estimator):
    if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(y):
        X = np.array(X)
        y = np.array(y)

    if X.ndim == 1:
        raise ValueError("The training array has only one dimension.")

    if X.shape[0] == 0:
        raise ValueError("Empty training array")

    if y is None or True in np.array(np.equal(y, None)):
        raise ValueError("y should be a 1d array")

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

    X, y, le = check_input_type(X, y, estimator)
    check_parameters(estimator)

    return X, y, le


def check_input_inference(X, estimator):
    if not scipy.sparse.issparse(X):
        X = np.array(X)
        if X.dtype != "float32" or X.dtype != "float64":
            X = np.asfortranarray(X, dtype="float64")

        if False in np.isfinite(X):
            raise ValueError("NaN of inf values in the training array(s)")

    if X.ndim == 1:
        raise ValueError("Reshape your data")

    if X.shape[1] != estimator.n_features_in_:
        raise ValueError("X has %d features per sample; expecting %d"
                            % (X.shape[1], estimator.n_features_in_))

    return X
