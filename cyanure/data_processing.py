"""Contain the functions concerning the processing of data."""

import warnings
import numbers
import platform

import math

import numpy as np
import scipy.sparse

from scipy.sparse import issparse
from scipy.sparse import dok_matrix
from scipy.sparse import lil_matrix

from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._array_api import get_namespace
from sklearn.utils.validation import check_array, _assert_all_finite

from collections.abc import Sequence

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

def sklearn_catch_warnings(y):
    with warnings.catch_warnings():
            warnings.simplefilter("error", np.VisibleDeprecationWarning)
            if not issparse(y):
                try:
                    y = check_array(y, dtype=None, **check_y_kwargs)
                except (np.VisibleDeprecationWarning, ValueError) as e:
                    if str(e).startswith("Complex data not supported"):
                        raise

                    # dtype=object should be provided explicitly for ragged arrays,
                    # see NEP 34
                    y = check_array(y, dtype=object, **check_y_kwargs)

# Code from scikit-learn
def type_of_target(y, input_name=""):
    """Determine the type of data indicated by the target.
    Note that this type is the most specific type that can be inferred.
    For example:
        * ``binary`` is more specific but compatible with ``multiclass``.
        * ``multiclass`` of integers is more specific but compatible with
          ``continuous``.
        * ``multilabel-indicator`` is more specific but compatible with
          ``multiclass-multioutput``.
    Parameters
    ----------
    y : {array-like, sparse matrix}
        Target values. If a sparse matrix, `y` is expected to be a
        CSR/CSC matrix.
    input_name : str, default=""
        The data name used to construct the error message.
        .. versionadded:: 1.1.0
    Returns
    -------
    target_type : str
        One of:
        * 'continuous': `y` is an array-like of floats that are not all
          integers, and is 1d or a column vector.
        * 'continuous-multioutput': `y` is a 2d array of floats that are
          not all integers, and both dimensions are of size > 1.
        * 'binary': `y` contains <= 2 discrete values and is 1d or a column
          vector.
        * 'multiclass': `y` contains more than two discrete values, is not a
          sequence of sequences, and is 1d or a column vector.
        * 'multiclass-multioutput': `y` is a 2d array that contains more
          than two discrete values, is not a sequence of sequences, and both
          dimensions are of size > 1.
        * 'multilabel-indicator': `y` is a label indicator matrix, an array
          of two dimensions with at least two columns, and at most 2 unique
          values.
        * 'unknown': `y` is array-like but none of the above, such as a 3d
          array, sequence of sequences, or an array of non-sequence objects.
    Examples
    --------
    >>> from sklearn.utils.multiclass import type_of_target
    >>> import numpy as np
    >>> type_of_target([0.1, 0.6])
    'continuous'
    >>> type_of_target([1, -1, -1, 1])
    'binary'
    >>> type_of_target(['a', 'b', 'a'])
    'binary'
    >>> type_of_target([1.0, 2.0])
    'binary'
    >>> type_of_target([1, 0, 2])
    'multiclass'
    >>> type_of_target([1.0, 0.0, 3.0])
    'multiclass'
    >>> type_of_target(['a', 'b', 'c'])
    'multiclass'
    >>> type_of_target(np.array([[1, 2], [3, 1]]))
    'multiclass-multioutput'
    >>> type_of_target([[1, 2]])
    'multilabel-indicator'
    >>> type_of_target(np.array([[1.5, 2.0], [3.0, 1.6]]))
    'continuous-multioutput'
    >>> type_of_target(np.array([[0, 1], [1, 1]]))
    'multilabel-indicator'
    """
    xp, is_array_api = get_namespace(y)
    valid = (
        (isinstance(y, Sequence) or issparse(y) or hasattr(y, "__array__"))
        and not isinstance(y, str)
        or is_array_api
    )

    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), got %r" % y
        )

    sparse_pandas = y.__class__.__name__ in ["SparseSeries", "SparseArray"]
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if is_multilabel(y):
        return "multilabel-indicator"

    # DeprecationWarning will be replaced by ValueError, see NEP 34
    # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
    # We therefore catch both deprecation (NumPy < 1.24) warning and
    # value error (NumPy >= 1.24).
    check_y_kwargs = dict(
        accept_sparse=True,
        allow_nd=True,
        force_all_finite=False,
        ensure_2d=False,
        ensure_min_samples=0,
        ensure_min_features=0,
    )

   sklearn_catch_warnings(y)

    # The old sequence of sequences format
    try:
        if (
            not hasattr(y[0], "__array__")
            and isinstance(y[0], Sequence)
            and not isinstance(y[0], str)
        ):
            raise ValueError(
                "Sequence of sequences are not"
                " supported; use a binary array or sparse"
                " matrix instead."
            )
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim not in (1, 2):
        # Number of dimension greater than 2: [[[1, 2]]]
        return "unknown"
    if not min(y.shape):
        # Empty ndarray: []/[[]]
        if y.ndim == 1:
            # 1-D empty array: []
            return "binary"  # []
        # 2-D empty array: [[]]
        return "unknown"
    if not issparse(y) and y.dtype == object and not isinstance(y.flat[0], str):
        # [obj_1] and not ["label_1"]
        return "unknown"

    # Check if multioutput
    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # Check float and contains non-integer float values
    if y.dtype.kind == "f":
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        data = y.data if issparse(y) else y
        if xp.any(data != np.floor(data)):
            _assert_all_finite(data, input_name=input_name)
            return "continuous" + suffix

    # Check multiclass
    first_row = y[0] if not issparse(y) else y.getrow(0).data
    if xp.unique_values(y).shape[0] > 2 or (y.ndim == 2 and len(first_row) > 1):
        # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
        return "multiclass" + suffix
    else:
        return "binary"  # [1, 2] or [["a"], ["b"]]


# Code from scikit-learn
def is_multilabel(y):
    """Check if ``y`` is in a multilabel format.
    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.
    Returns
    -------
    out : bool
        Return ``True``, if ``y`` is in a multilabel format, else ```False``.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.multiclass import is_multilabel
    >>> is_multilabel([0, 1, 0, 1])
    False
    >>> is_multilabel([[1], [0, 2], []])
    False
    >>> is_multilabel(np.array([[1, 0], [0, 0]]))
    True
    >>> is_multilabel(np.array([[1], [0], [0]]))
    False
    >>> is_multilabel(np.array([[1, 0, 0]]))
    True
    """
    xp, is_array_api = get_namespace(y)
    if hasattr(y, "__array__") or isinstance(y, Sequence) or is_array_api:
        # DeprecationWarning will be replaced by ValueError, see NEP 34
        # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
        check_y_kwargs = dict(
            accept_sparse=True,
            allow_nd=True,
            force_all_finite=False,
            ensure_2d=False,
            ensure_min_samples=0,
            ensure_min_features=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.VisibleDeprecationWarning)
            try:
                y = check_array(y, dtype=None, **check_y_kwargs)
            except (np.VisibleDeprecationWarning, ValueError) as e:
                if str(e).startswith("Complex data not supported"):
                    raise

                # dtype=object should be provided explicitly for ragged arrays,
                # see NEP 34
                y = check_array(y, dtype=object, **check_y_kwargs)

    if not (hasattr(y, "shape") and y.ndim == 2 and y.shape[1] > 1):
        return False

    if issparse(y):
        if isinstance(y, (dok_matrix, lil_matrix)):
            y = y.tocsr()
        labels = xp.unique_values(y.data)
        return (
            len(y.data) == 0
            or (labels.size == 1 or (labels.size == 2) and (0 in labels))
            and (y.dtype.kind in "biu" or _is_integral_float(labels))  # bool, int, uint
        )
    else:
        labels = xp.unique_values(y)

        return len(labels) < 3 and (
            y.dtype.kind in "biu" or _is_integral_float(labels)  # bool, int, uint
        )


# Code from scikit-learn
def _is_integral_float(y):
    return y.dtype.kind == "f" and np.all(y.astype(int) == y)


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

        if np.issubdtype(type(labels[0]), np.str_):
            label_encoder = LabelEncoder()
            label_encoder.fit(labels)
            labels = label_encoder.transform(labels)

        y_type = type_of_target(labels)
        if y_type not in [
            "binary",
            "multiclass"
        ]:
            raise ValueError("Unknown label type: %r" % y_type)

    else:
        if type(labels[0]) not in (np.float32, np.float64):
            logger.info("The labels have been converted in float64")
            labels = labels.astype('float64')

    check_is_finite(labels)

    if estimator._estimator_type == "classifier" and len(np.unique(labels)) == 1:
        raise ValueError("Classifier can't train when only one class is present.")

    return labels, label_encoder


def check_is_finite(array_to_test):
    if len(array_to_test.shape) == 1:
        for value in array_to_test:
            if not math.isfinite(value):
                raise ValueError(
                    "Input contains NaN, infinity or a value too large for dtype('float64').")
    else:
        for sub_array_to_test in array_to_test:
            check_is_finite(sub_array_to_test)


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
            X = X.astype('float64')

        labels, label_encoder = check_labels(labels, estimator)

        check_is_finite(X)

    else:
        if scipy.sparse.issparse(X) and X.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")
        if scipy.sparse.issparse(labels) and labels.getformat() != "csr":
            raise TypeError("The library only supports CSR sparse data.")

        X, labels = windows_conversion(X, labels)

    return X, labels, label_encoder


def windows_conversion(X, labels):

    if platform.system() == "Windows":
        if scipy.sparse.issparse(X):
            X.indptr = X.indptr.astype(np.float64).astype(np.intc)
            X.indices = X.indices.astype(np.float64).astype(np.intc)
        if scipy.sparse.issparse(labels):
            labels.indptr = labels.indptr.astype(np.float64).astype(np.intc)
            labels.indices = labels.indices.astype(np.float64).astype(np.intc)

    return X, labels


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

    instance_class = estimator.__class__
    instance = instance_class()

    # Verify that it is not the default value
    if (estimator.penalty is None or estimator.penalty == "none") and estimator.lambda_1 != instance.lambda_1:
        warnings.warn("Setting penalty='none' will ignore the lambda_1")


def convert_to_array(X, labels):
    if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(labels):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

    return X, labels


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

    X, labels = convert_to_array(X, labels)

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
            X = X.astype("float64")

        check_is_finite(X)

    if X.ndim == 1:
        raise ValueError("Reshape your data")

    if X.shape[1] != estimator.n_features_in_:
        raise ValueError(f"X has {X.shape[1]} features per sample; \
                           expecting {estimator.n_features_in_}")

    return X
