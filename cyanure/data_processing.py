"""Contain the functions concerning the processing of data."""

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
    """
    Preprocess features training data.

    Perform in-place centering or normalization, either of columns or rows
    of the input matrix X.

    Parameters
    ----------
        X (numpy array or scipy sparse CSR matrix):
            Input matrix

        centering (boolean) : default=False
            Perform a centering operation

        normalize (boolean): default=True
            l2-normalization

        columns (boolean): default=False
            Operates on rows (False) or columns (True).
    """
    if scipy.sparse.issparse(X):
        training_data_fortran = X.T
        if platform.system() == "Windows":
            training_data_fortran.indptr = training_data_fortran.indptr.astype(np.float64).astype(np.intc)
            training_data_fortran.indices = training_data_fortran.indices.astype(np.float64).astype(np.intc)
    else:
        training_data_fortran = np.asfortranarray(X.T)
    return cyanure_lib.preprocess_(training_data_fortran, centering, normalize, not columns)


def check_labels(labels, estimator):
    """
    Verify the format of labels depending on the type of the estimator.

    Can convert labels in some cases.

    Parameters
    ----------
        labels (numpy array or scipy sparse CSR matrix):
            Numpy array containing labels

        estimator (ERM):
            The estimator which will be fitted

    Raises
    ------
        ValueError:
            Format of the labels does not respect the format supported by Cyanure classifiers.

        ValueError:
            Labels have an non finite value

        ValueError:
            Problem has only one class

    Returns
    -------
        labels (numpy array or scipy sparse CSR matrix):
            Converted labels if required by the estimator.

        label_encoder (sklearn.LabelEncoder):
            Convert text labels if needed
    """
    label_encoder = None

    if estimator._estimator_type == "classifier":
        y_type = type_of_target(labels)
        if y_type not in [
            "binary",
            "multiclass"
        ]:
            raise ValueError("Unknown label type: %r" % y_type)

        if np.issubdtype(type(labels[0]), np.str_):
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            labels = label_encoder.transform(labels)
    else:
        if type(labels[0]) not in (np.float32, np.float64):
            logger.info("The labels have been converted in float64")
            labels = labels.astype('float64')

    if False in np.isfinite(labels):
        raise ValueError(
            "Input contains NaN, infinity or a value too large for dtype('float64').")

    if len(np.unique(labels)) == 1:
        raise ValueError("There is only one class in the labels.")

    return labels, label_encoder


def get_element(array):
    """
    Get an element from an array of any depth.

    Args
    ----
        array (Type of the element):
            Array we want to get an element

    Returns
    -------
        Type of the element:
            One of the element of the array
    """
    element = array[0]
    for i in range(len(array.shape) - 1):
        element = element[i]
    return element


def check_input_type(X, labels, estimator):
    """
    Verify the format of labels and features depending on the type of the estimator.

    Can convert labels in some cases.

    Parameters
    ----------
        X (numpy array or scipy sparse CSR matrix):
            Numpy array containing features

        labels (numpy array or scipy sparse CSR matrix):
            Numpy array containing labels

        estimator (ERM):
            The estimator which will be fitted

    Raises
    ------
        ValueError:
            Data are complex

        ValueError:
            Data contains non finite value

        TypeError:
            Sparsed features are not CSR

        TypeError:
            Sparsed labels are not CSR

    Returns
    -------
        X (numpy array or scipy sparse CSR matrix):
            Converted features if required by the estimator.

        labels (numpy array or scipy sparse CSR matrix):
            Converted labels if required by the estimator.

        label_encoder (sklearn.LabelEncoder):
            Convert text labels if needed
    """
    label_encoder = None

    if np.iscomplexobj(X) or np.iscomplexobj(labels):
        raise ValueError("Complex data not supported")

    if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(labels):
        x_element = get_element(X)
        if type(x_element) not in (np.float32, np.float64):

            logger.info("The features have been converted in float64")
            X = np.asfortranarray(X, 'float64')
        else:
            X = np.asfortranarray(X)

        labels, label_encoder = check_labels(labels, estimator)

        if False in np.isfinite(X):
            raise ValueError(
                "Input contains NaN, infinity or a value too large for dtype('float64').")

    else:
        if scipy.sparse.issparse(X) and X.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")
        if scipy.sparse.issparse(labels) and labels.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")

        if platform.system() == "Windows":
            if scipy.sparse.issparse(X):
                X.indptr = X.indptr.astype(np.float64).astype(np.intc)
                X.indices = X.indices.astype(np.float64).astype(np.intc)
            if scipy.sparse.issparse(labels):
                labels.indptr = labels.indptr.astype(np.float64).astype(np.intc)
                labels.indices = labels.indices.astype(np.float64).astype(np.intc)

    return X, labels, label_encoder


def check_positive_parameter(parameter, message):
    """
    Check that a parameter if a number and positive.

    Parameters
    ----------
        parameter (Any):
            Parameter to verify

        message (string):
            Message of the exception

    Raises
    ------
        ValueError:
            Parameter is not a number

        ValueError:
            Parameter is not positive
    """
    if not isinstance(parameter, numbers.Number):
        raise ValueError(message)

    if isinstance(parameter, numbers.Number) and parameter < 0:
        raise ValueError(message)


def check_parameters(estimator):
    """
    Verify that the different parameters of an estimator respect the constraints.

    Parameters
    ----------
        estimator (ERM):
            Estimator to veriffy
    """
    check_positive_parameter(
        estimator.tol, "Tolerance for stopping criteria must be positive")
    check_positive_parameter(estimator.max_iter,
                             "Maximum number of iteration must be positive")
    check_positive_parameter(estimator.lambda_1,
                             "Penalty term must be positive")

    # Verify that it is not the default value
    if (estimator.penalty is None or estimator.penalty == "none") and estimator.lambda_1 != 0.1:
        warnings.warn("Setting penalty='none' will ignore the lambda_1")


def check_input_fit(X, labels, estimator):
    """
    Check the different input arrays required for training according to the estimator type.

    Can convert data if necessary.

    Parameters
    ----------
        X (numpy array or scipy sparse CSR matrix):
            Numpy array containing features

        labels (numpy array or scipy sparse CSR matrix):
            Numpy array containing labels

        estimator (ERM):
            The estimator which will be fitted

    Raises
    ------
        ValueError:
            There is only one feature.

        ValueError:
            There is no sample.

        ValueError:
            An observation has no label.

        ValueError:
            Feature array has no feature

        ValueError:
            Features and labels does not have the same number of observations.

        ValueError:
            There is only one sample.

    Returns
    -------
        X (numpy array or scipy sparse CSR matrix):
            Converted features if required by the estimator.

        labels (numpy array or scipy sparse CSR matrix):
            Converted labels if required by the estimator.

        label_encoder (sklearn.LabelEncoder):
            Convert text labels if needed
    """
    if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(labels):
        X = np.array(X)
        labels = np.array(labels)

    if X.ndim == 1:
        raise ValueError("The training array has only one dimension.")

    if X.shape[0] == 0:
        raise ValueError("Empty training array")

    if labels is None or True in np.array(np.equal(labels, None)):
        raise ValueError("y should be a 1d array")

    if len(X.shape) > 1 and X.shape[1] == 0:
        raise ValueError("0 feature(s) (shape=(" + str(X.shape[0]) + ", 0)) while a minimum of "
              + str(X.shape[0]) + " is required.")

    if labels.shape[0] != X.shape[0]:
        raise ValueError(
            "X and labels should have the same number of observations")

    if X.shape[0] == 1:
        raise ValueError("There should have more than 1 sample")

    if not estimator._get_tags()["multioutput"] and \
       not estimator._get_tags()["multioutput_only"] and labels.ndim > 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected", DataConversionWarning)

    X, labels, label_encoder = check_input_type(X, labels, estimator)
    check_parameters(estimator)

    return X, labels, label_encoder


def check_input_inference(X, estimator):
    """
    Check the format of the array which will be used for inference. Input array can be converted.

    Parameters
    ----------
        X (numpy array or scipy sparse CSR matrix):
            Array which will be used for inference

        estimator (ERM):
            Estimator which will be used

    Raises
    ------
        ValueError:
            One of the value is not finite

        ValueError:
            Shape of features is not correct

        ValueError:
            Shape of features does not correspond to estimators shape

    Returns
    -------
        X (numpy array or scipy sparse CSR matrix):
            Potentially converted array (if converted as numpy.float64)

    """
    if not scipy.sparse.issparse(X):
        X = np.array(X)
        if X.dtype != "float32" or X.dtype != "float64":
            X = np.asfortranarray(X, dtype="float64")

        if False in np.isfinite(X):
            raise ValueError("NaN of inf values in the training array(s)")

    if X.ndim == 1:
        raise ValueError("Reshape your data")

    if X.shape[1] != estimator.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features per sample; \
                           expecting {estimator.n_features_in_}")

    return X
