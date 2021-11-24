import warnings

import numpy as np
import scipy.sparse

from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

import cyanure_lib


def preprocess(training_data, centering=False, normalize=True, columns=False):
    """Perform in-place centering or normalization, either of columns or rows
    of the input matrix training_data

    Parameters
    ----------
    training_data : numpy array, or scipy sparse CSR matrix
        input matrix

    centering : boolean, default=False
        perform a centering operation

    normalize : boolean, default=True
        l2-normalization

    columns : boolean, default=False
        operates on rows (False) or columns (True).
    """

    if scipy.sparse.issparse(training_data):
        training_data_fortran = training_data.T
    else:
        training_data_fortran = np.asfortranarray(training_data.T)
    return cyanure_lib.preprocess_(training_data_fortran, centering, normalize, not columns)


def check_input_type(training_data, labels, estimator):
    le = None

    if np.iscomplexobj(training_data) or np.iscomplexobj(labels):
        raise ValueError("Complex data not supported")

    if not scipy.sparse.issparse(training_data) and not scipy.sparse.issparse(labels):
        if np.iscomplexobj(training_data) or np.iscomplexobj(labels):
            raise ValueError(
                "Complex data not supported!")

        if len(training_data) == 0:
            raise ValueError("Empty training array")

        # TODO Flexible dtype
        training_data = np.asfortranarray(training_data, 'float64')

        # TODO check if relevant
        if estimator._estimator_type == "classifier":
            y_type = type_of_target(labels)
            if y_type not in [
                "binary",
                "multiclass",
                "multiclass-multioutput",
                "multilabel-indicator",
                "multilabel-sequences",
            ]:
                raise ValueError("Unknown label type: %r" % y_type)

            if np.issubdtype(type(labels[0]), np.str_):
                le = LabelEncoder()
                le.fit(labels)
                labels = le.transform(labels)
            elif np.isfinite(labels[0]) and (type(labels[0]) == "float64" or type(labels[0]) == "float32"):
                raise ValueError("Unknown label type: " + str(labels.dtype))
            elif np.isfinite(labels[0]) and (type(labels[0]) != "int64"):
                labels = labels.astype("int64")
        else:

            labels = labels.astype("float64")

        if False in np.isfinite(training_data) or False in np.isfinite(labels):
            raise ValueError(
                "Input contains NaN, infinity or a value too large for dtype('float64').")

        if len(np.unique(labels)) == 1:
            raise ValueError("There is only one class in the labels.")
    else:
        if scipy.sparse.issparse(training_data) and training_data.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")
        if scipy.sparse.issparse(labels) and labels.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")

    return training_data, labels, le


def check_positive_parameter(parameter, message):
    if not isinstance(parameter, (int, float)):
        raise ValueError(message)

    if isinstance(parameter, (int, float)) and parameter <= 0:
        raise ValueError(message)


def check_parameters(estimator):

    check_positive_parameter(
        estimator.tol, "Tolerance for stopping criteria must be positive")
    check_positive_parameter(estimator.max_iter,
                             "Maximum number of iteration must be positive")


def check_input(training_data, labels, estimator):
    if not scipy.sparse.issparse(training_data) and not scipy.sparse.issparse(labels):
        training_data = np.array(training_data)
        labels = np.array(labels)

    if training_data.ndim == 1:
        raise ValueError("The training array has only one dimension.")

    if training_data.shape[0] == 0:
        raise ValueError("Empty training array")

    if len(training_data.shape) > 1 and training_data.shape[1] == 0:
        raise ValueError("0 feature(s) (shape=(" + str(training_data.shape[0]) + ", 0)) while a minimum of " + str(
            training_data.shape[0]) + " is required.")

    if labels.shape[0] != training_data.shape[0]:
        raise ValueError(
            "training_data and y should have the same number of observations")

    if training_data.shape[0] == 1:
        raise ValueError("There should have more than 1 sample")

    if not estimator._get_tags()["multioutput"] and not estimator._get_tags()["multioutput_only"] and labels.ndim > 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected", DataConversionWarning)

    training_data, labels, le = check_input_type(training_data, labels, estimator)

    check_parameters(estimator)

    return training_data, labels, le
