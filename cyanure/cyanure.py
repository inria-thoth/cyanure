# Author: Julien Mairal <julien.mairal@inria.fr>
#
# License: BSD 3 clause

from abc import abstractmethod, ABC
import numpy as np
import scipy.sparse
from scipy import sparse
import warnings
import cyanure_lib
import math

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot, softmax
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target


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
        Xf = X.T
    else:
        Xf = np.asfortranarray(X.T)
    return cyanure_lib.preprocess_(Xf, centering, normalize, not columns)


def check_input(X, y, estimator):
        le = None

        if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(y):
            X = np.array(X)
            y = np.array(y)

        if X.ndim == 1:
            raise ValueError("The training array has only one dimension.")

        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("Complex data not supported")

        if X.shape[0] == 0:
            raise ValueError("Empty training array")

        if len(X.shape) > 1 and X.shape[1] == 0:
            raise ValueError("0 feature(s) (shape=(" + str(X.shape[0]) + ", 0)) while a minimum of " + str(
                X.shape[0]) + " is required.")

        if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(y):
            if np.iscomplexobj(X) or np.iscomplexobj(y):
                raise ValueError("Complex data not supported")

            if len(X) == 0:
                raise ValueError("Empty training array")

            # TODO Flexible dtype
            X = np.asfortranarray(X, 'float64')
           
            #TODO check if relevant
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

                 y = y.astype("float64")
            
            if False in np.isfinite(X) or False in np.isfinite(y):
                raise ValueError("Input contains NaN, infinity or a value too large for dtype('float64').")

            if len(np.unique(y)) == 1:
                raise ValueError("There is only one class in the labels.")
        else:
            if scipy.sparse.issparse(X) and X.getformat() != F"csr":
                raise TypeError("The library only supports CSR sparse data.")
            if  scipy.sparse.issparse(y) and y.getformat() != "csr":
                raise TypeError("The library only supports CSR sparse data.")

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "X and y should have the same number of observations")

        if X.shape[0] == 1:
            raise ValueError("There should have more than 1 sample")

        if not (type(estimator.tol) == int or type(estimator.tol) == float):
            raise ValueError(
                "Tolerance for stopping criteria must be positive")

        if not (type(estimator.max_epochs) == int or type(estimator.max_epochs) == float):
            raise ValueError("Maximum number of iteration must be positive")
        
        if y.ndim > 1:
             warnings.warn("A column-vector y was passed when a 1d array was expected", DataConversionWarning) 

        return X, y, le

        

class ERM(BaseEstimator, ABC):
    """The generic class for empirical risk minimization problems.
    For univariates problems, minimizes

        min_{w,b} (1/n) sum_{i=1}^n L( y_i, <w, x_i> + b)   + psi(w)

    """

    def __init__(self, loss='square', penalty='l2', fit_intercept=False, dual_variable=None, tol=1e-3, solver="auto", random_state=0, max_epochs=500):
        """Initialization function of the ERM class.

        Parameters
        ----------
        loss : string, default='square'
            Loss function to be used. Possible choices are
            - 'square' =>  L(y,z) = 0.5 ( y-z)^2
            - 'logistic' => L(y,z) = log(1 + e^{-y z} )
            - 'sqhinge' or 'squared_hinge' => L(y,z) = 0.5 max( 0, 1- y z)^2
            - 'safe-logistic' => L(y,z) = e^{ yz - 1 } - y z  if yz <= 1
                                 and 0 otherwise
            - 'multiclass-logistic' => multinomial logistic (see Latex
                                       documentation).
            Note that for binary classification, we assume the labels to be of
            the form {-1,+1}

        penalty: string, default='none'
            Regularization function psi. Possible choices are

            For univariate problems
            - 'none' => psi(w) = 0
            - 'l2' =>  psi{w) = (lambd/2) ||w||_2^2
            - 'l1' =>  psi{w) = lambd ||w||_1
            - 'elastic-net' =>  psi{w) = lambd ||w||_1 + (lambd2/2)||w||_2^2
            - 'fused-lasso' => psi(w) = lambd sum_{i=2}^p |w[i]-w[i-1]|
                                      + lambd2||w||_1 + (lambd3/2)||w||_2^2
            - 'l1-ball'     => encodes the constraint ||w||_1 <= lambd
            - 'l2-ball'     => encodes the constraint ||w||_2 <= lambd

            For multivariate problems, the previous penalties operate on each
            individual (e.g., class) predictor.
            In addition, multitask-group Lasso penalties are provided for
            multivariate problems (w is then a matrix)
            - 'l1l2' or 'l1linf', see Latex documentation

        fit_intercept: boolean, default='False'
            learns an unregularized intercept b  (or several intercepts for
            multivariate problems)
        """

        self.loss = loss
        if loss == 'squared_hinge':
            self.loss = 'sqhinge'
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.dual_variable = dual_variable
        self.solver = solver
        self.tol = tol
        self.random_state = random_state
        self.max_epochs = max_epochs

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, univariate=True, nthreads=-1, random_state=0):
        """
        The fitting function (the one that does the job)

        Parameters
        ----------

        X : numpy array, or scipy sparse CSR matrix
            input n x p numpy matrix; the samples are on the rows

        y : labels, numpy array.
            - vector of size n with real values for regression
            - vector of size n with {-1,+1} labels for binary classification,
              which will be automatically converted if labels in {0,1} are
              provided
            - matrix of size n x k for multivariate regression
            - vector of size n with entries in {0,1,k-1} for classification
              with k classes

        lambd: float, default=0
            first regularization parameter

        lambd2: float, default=0
            second regularization parameter, if needed

        lambd3: float, default=0
            third regularization parameter, if needed

        solver: string, default='auto'
            Optimization solver. Possible choices are
            - 'ista'
            - 'fista'
            - 'catalyst-ista'
            - 'qning-ista'  (proximal quasi-Newton method)
            - 'svrg'
            - 'catalyst-svrg' (accelerated SVRG with Catalyst)
            - 'qning-svrg'  (quasi-Newton SVRG)
            - 'acc-svrg'    (SVRG with direct acceleration)
            - 'miso'
            - 'catalyst-miso' (accelerated MISO with Catalyst)
            - 'qning-miso'  (quasi-Newton MISO)
            - 'auto'
            see the Latex documentation for more details.
            If you are unsure, use 'auto'

         tol: float, default='1e-3'
            Tolerance parameter. For almost all combinations of loss and
            penalty functions, this parameter is based on a duality gap.
            Assuming the (non-negative) objective function is "f" and its
            optimal value is "f^*", the algorithm stops with the guarantee

            f(x_t) - f^*  <=  tol f(x_t)

         max_epochs: int, default=500
            Maximum number of iteration of the algorithm in terms of passes
            over the data

         it0: int, default=10
            Frequency of duality-gap computation

         verbose: boolean, default=True
            Display information or not

         nthreads: int, default=-1
            maximum number of cores the method may use (-1 = all cores).
            Note that more cores is not always better.

         seed: int, default=0
            random seed

         restart: boolean, default=False
            use a restart strategy (useful for computing regularization path)

         univariate: boolean, default=True
            univariate or multivariate problems

         l_qning: int, default=20
            memory paramter for the qning method

         f_restart: int, default=50
            restart strategy for fista


        Returns
        -------

        test returns a numpy array carrying information about the optimization
        process (number of iterations, objective function values, duality gap)
        will be documented in the future if people ask me,
        """
        if X.ndim == 1:
            raise ValueError("The training array has only one dimension.")

        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("Complex data not supported")

        if X.shape[0] == 0:
            raise ValueError("Empty training array")

        if len(X.shape) > 1 and X.shape[1] == 0:
            raise ValueError("0 feature(s) (shape=(" + str(X.shape[0]) + ", 0)) while a minimum of " + str(
                X.shape[0]) + " is required.")

        if not scipy.sparse.issparse(X) and not scipy.sparse.issparse(y):
            if np.iscomplexobj(X) or np.iscomplexobj(y):
                raise ValueError("Complex data not supported")

            if len(X) == 0:
                raise ValueError("Empty training array")

            # TODO Flexible dtype
            X = np.asfortranarray(X, 'float64')
            y = np.asfortranarray(y, 'float64')

            if False in np.isfinite(X) or False in np.isfinite(y):
                raise ValueError("NaN of inf values in the training array(s)")
        else:
            if scipy.sparse.issparse(X) and X.getformat() != "csr":
                raise TypeError("The library only supports CSR sparse data.")
            if  scipy.sparse.issparse(y) and y.getformat() != "csr":
                raise TypeError("The library only supports CSR sparse data.")

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "X and y should have the same number of observations")

        if X.shape[0] == 1:
            raise ValueError("There should have more than 1 sample")

        if not (type(self.tol) == int or type(self.tol) == float):
            raise ValueError(
                "Tolerance for stopping criteria must be positive")

        if not (type(max_epochs) == int or type(max_epochs) == float):
            raise ValueError("Maximum number of iteration must be positive")

        Xf = X.T if scipy.sparse.issparse(X) else np.asfortranarray(X.T)
        p = X.shape[1] + 1 if self.fit_intercept else X.shape[1]
        y = np.squeeze(y)

        if univariate:
            w0 = np.zeros(p, dtype=Xf.dtype)
            yf = np.squeeze(y.astype(Xf.dtype))
        else:
            if y.squeeze().ndim > 1:
                nclasses = y.squeeze().shape[1]
                yf = np.asfortranarray(y.T)
            else:
                nclasses = int(np.max(y) + 1)
                yf = np.squeeze(np.int32(y))
            w0 = np.zeros([p, nclasses], dtype=Xf.dtype, order='F')

        if restart and np.any(self.w_ != 0):
            if verbose:
                print("Restart")
            if self.fit_intercept:
                w0[-1, ] = self.b_
                w0[0:-1, ] = self.w_
            else:
                w0 = self.w_

        if restart and (solver == 'auto' or solver == 'miso' or
                        solver == 'catalyst-miso' or solver == 'qning-miso'):
            n = X.shape[0]
            reset_dual = np.any(self.dual_variable is None)
            if not reset_dual and univariate:
                reset_dual = self.dual_variable.shape[0] != n
            if not reset_dual and not univariate:
                reset_dual = np.any(self.dual_variable.shape != [n, nclasses])
            if reset_dual and univariate:
                self.dual_variable = np.zeros(n, dtype=Xf.dtype)
            if reset_dual and not univariate:
                self.dual_variable = np.zeros([n, nclasses], dtype=Xf.dtype)

        w = np.copy(w0)
        cyanure_lib.erm_(
            Xf, yf, w0, w, dual_variable=self.dual_variable, loss=self.loss,
            penalty=self.penalty, solver=self.solver, lambd=float(lambd),
            lambd2=float(lambd2), lambd3=float(lambd3),
            intercept=bool(self.fit_intercept), tol=float(self.tol), it0=int(it0),
            nepochs=int(max_epochs), l_qning=int(l_qning),
            f_restart=int(f_restart), verbose=bool(verbose),
            univariate=bool(univariate), nthreads=int(nthreads), seed=int(self.random_state)
        )
        if self.fit_intercept:
            self.b_ = w[-1, ]
            self.w_ = w[0:-1, ]
        else:
            self.w_ = w

        self.n_features_in_ = self.w_.shape[0]

        return self

    @abstractmethod
    def predict(self, X):
        """
        predict the labels given an input matrix X (same format as fit)
        """
        return

    def get_weights(self):
        """
        get the model parameters (either w or the tuple (w,b))
        """
        return (self.w_, self.b_) if self.fit_intercept else self.w_

    def eval(self, X, y, lambd=0, lambd2=0, lambd3=0):
        """
        get the value of the objective function and computes a relative
        duality gap, see function fit for the format of parameters.
        """
        return self.fit(X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                        max_epochs=0, verbose=False, restart=True)

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    @classmethod
    def _get_param_names(cls):
        import inspect
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError()
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def set_params(self, **params):
        from collections import defaultdict
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

class Classifier(ERM):

    @abstractmethod
    def predict_proba(self, X):
        pass

class BinaryClassifier(Classifier):
    """
    The binary classification class, which derives from ERM. The goal is to
    minimize the following objective

        .. math::
            \\min_{w,b} \\frac{1}{n} \\sum_{i=1}^n
            L\\left( y_i, w^\\top x_i + b\\right) + \\psi(w),

        where :math:`L` is a classification loss, :math:`\\psi` is a
        regularization function (or constraint), :math:`w` is a p-dimensional
        vector representing model parameters, and b is an optional
        unregularized intercept. We expect binary labels in {-1,+1}.

        Parameters
        ----------
        loss: string, default='square'
            Loss function to be used. Possible choices are

            - 'square' =>  :math:`L(y,z) = \\frac{1}{2} ( y-z)^2`
            - 'logistic' => :math:`L(y,z) = \\log(1 + e^{-y z} )`
            - 'sqhinge' or 'squared_hinge' => :math:`L(y,z) = \\frac{1}{2} \\max( 0, 1- y z)^2`
            - 'safe-logistic' => :math:`L(y,z) = e^{ yz - 1 } - y z ~\\text{if}~ yz \\leq 1~~\\text{and}~~0` otherwise

        penalty: string, default='l2'
            Regularization function psi. Possible choices are

            - 'none' => :math:`\\psi(w) = 0`
            - 'l2' =>  :math:`\\psi(w) = \\frac{\\lambda}{2} \\|w\\|_2^2`
            - 'l1' =>  :math:`\\psi(w) = \\lambda \\|w\\|_1`
            - 'elastic-net' =>  :math:`\\psi(w) = \\lambda \\|w\\|_1 + \\frac{\\lambda_2}{2}\\|w\\|_2^2`
            - 'fused-lasso' => :math:`\\psi(w) = \\lambda \\sum_{i=2}^p |w[i]-w[i-1]| + \\lambda_2\\|w\\|_1 + \\frac{\\lambda_3}{2}\\|w\\|_2^2`
            - 'l1-ball'     => encodes the constraint :math:`\\|w\\|_1 \\leq \\lambda`
            - 'l2-ball'     => encodes the constraint :math:`\\|w\\|_2 \\leq \\lambda`

        fit_intercept: boolean, default='False'
            learns an unregularized intercept b

    """

    def _more_tags(self):
        return {"binary_only": True}

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, random_state=0):
        """
        The fitting function (the one that does the job)

        Parameters
        ----------

        X: numpy array, or scipy sparse CSR matrix
            input n x p numpy matrix; the samples are on the rows

        y: labels, numpy array.
            - vector of size n with {-1,+1} labels for binary classification,
              which will be automatically converted if labels in {0,1} are
              provided

        lambd: float, default=0
            first regularization parameter :math:`\\lambda`

        lambd2: float, default=0
            second regularization parameter :math:`\\lambda_2`, if needed

        lambd3: float, default=0
            third regularization parameter :math:`\\lambda_3`, if needed

        solver: string, default='auto'
            Optimization solver. Possible choices are

            - 'ista'
            - 'fista'
            - 'catalyst-ista'
            - 'qning-ista'  (proximal quasi-Newton method)
            - 'svrg'
            - 'catalyst-svrg' (accelerated SVRG with Catalyst)
            - 'qning-svrg'  (quasi-Newton SVRG)
            - 'acc-svrg'    (SVRG with direct acceleration)
            - 'miso'
            - 'catalyst-miso' (accelerated MISO with Catalyst)
            - 'qning-miso'  (quasi-Newton MISO)
            - 'auto'
            see the Latex documentation for more details.
            If you are unsure, use 'auto'

         tol: float, default='1e-3'
            Tolerance parameter. For almost all combinations of loss and
            penalty functions, this parameter is based on a duality gap.
            Assuming the (non-negative) objective function is :math:`f` and its
            optimal value is :math:`f^*`, the algorithm stops with the
            guarantee

            .. math::
                f(x_t) - f^*  \\leq  tol f(x_t)

         max_epochs: int, default=500
            Maximum number of iteration of the algorithm in terms of passes
            over the data

         it0: int, default=10
            Frequency of duality-gap computation

         verbose: boolean, default=True
            Display information or not

         nthreads: int, default=-1
            maximum number of cores the method may use (-1 = all cores).
            Note that more cores is not always better.

         seed: int, default=0
            random seed

         restart: boolean, default=False
            use a restart strategy (useful for computing regularization path)

         univariate: boolean, default=True
            univariate or multivariate problems

         l_qning: int, default=20
            memory parameter for the qning method

         f_restart: int, default=50
            restart strategy for fista


        Returns
        -------
        numpy array
            information about the optimization process (number of iterations,
            objective function values, duality gap) will be documented in the
            future if people ask me.
        """

        y = np.squeeze(y)
        uniq = np.unique(y)
        nb_classes = len(uniq)
        y_binary = y

        
        if nb_classes != 2:
            raise ValueError(
                "{} classes detected; use MulticlassClassifier instead".format(nb_classes))
        
        if len(uniq) == 0:
            raise ValueError("Empty training set")

        if not np.array_equal(uniq, [-1, 1]):
            print("You have called BinaryClassifier, labels should be either "
                  "-1 or 1, but they are")
            print(uniq)
            print("Automatic conversion to [-1,1]")
            neg = y == uniq[0]
            y_binary = np.zeros(y.shape, dtype=y.dtype)
            y_binary[neg] = -1
            y_binary[np.logical_not(neg)] = 1

        return super().fit(X, y_binary, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                           solver=solver, tol=tol, it0=it0,
                           max_epochs=max_epochs, l_qning=l_qning,
                           f_restart=f_restart, verbose=verbose,
                           restart=restart, univariate=True, nthreads=nthreads,
                           random_state=self.random_state)


    def decision_function(self, X):
        check_is_fitted(self)

        if X.ndim == 1:
            raise ValueError("Reshape your data")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], self.n_features_in_))

        if self.fit_intercept:
            scores = safe_sparse_dot(X, self.w_, dense_output=True) + self.b_
        else:
            scores = safe_sparse_dot(X, self.w_, dense_output=True) 
        return scores.ravel()

    def predict(self, X):
        check_is_fitted(self)

        if not scipy.sparse.issparse(X) and (X.dtype != "float32" or X.dtype != "float64"):
             X = np.asfortranarray(X, dtype="float64")

        if not scipy.sparse.issparse(X):
            if False in np.isfinite(X):
                raise ValueError("NaN of inf values in the training array(s)")
        
        if X.ndim == 1:
            raise ValueError("Reshape your data")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], self.n_features_in_))

        pred = X.dot(self.w_)
        if self.fit_intercept:
            pred += self.b_
        return np.sign(pred)

    def predict_proba(self, X):
        check_is_fitted(self)

        if not scipy.sparse.issparse(X):
            if False in np.isfinite(X):
                raise ValueError("NaN of inf values in the training array(s)")

        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for multi_class="multinomial" and binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d, copy=False)

    def score(self, X, y):
        """Compute classification accuracy of the model for new test data (X,y)
        """
        y = np.squeeze(y)
        uniq = np.unique(y)
        if not np.all(uniq == [-1, 1]):
            print("You have called BinaryClassifier, labels should be either "
                  "-1 or 1, but they are")
            print(np.unique(y))
            print("Automatic conversion to [-1,1]")
            neg = y == uniq[0]
            y[neg] = -1
            y[np.logical_not(neg)] = 1
        pred = np.squeeze(self.predict(X))
        return np.sum(np.squeeze(y) == pred) / pred.shape[0]


class Regression(ERM):
    """
    The regression class. The objective is the same as for the BinaryClassifier
    class, but we use a regression loss only (see below), and the targets will
    be real values.

    Parameters
    ----------
    loss: string, default='square'
        Only the square loss is implemented at this point

        - 'square' =>  :math:`L(y,z) = \\frac{1}{2} ( y-z)^2`

    penalty: string, default='l2'
        same as for the class BinaryClassifier

    fit_intercept: boolean, default='False'
        learns an unregularized intercept b

    """
    _estimator_type = "regressor"

    def __init__(self, loss='square', penalty='l2', fit_intercept=False):
        if loss != 'square':
            raise ValueError("square loss should be used")
        super().__init__(loss=loss, penalty=penalty,
                         fit_intercept=fit_intercept)

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, random_state=0):
        """
        The fitting function is the same as for the class BinaryClassifier,
        except that we do not necessarily expect binary labels in y.
        """
        X, y, _ = check_input(X, y, self)

        return super().fit(X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                           solver=solver, tol=tol, it0=it0,
                           max_epochs=max_epochs, l_qning=l_qning,
                           f_restart=f_restart, verbose=verbose,
                           restart=restart, univariate=True, nthreads=nthreads,
                           random_state=self.random_state)

    def predict(self, X):
        check_is_fitted(self)

        if not scipy.sparse.issparse(X):
            X = np.array(X)
            if (X.dtype != "float32" or X.dtype != "float64"):
                X = np.asfortranarray(X, dtype="float64")
            
            if False in np.isfinite(X):
                raise ValueError("NaN of inf values in the training array(s)")
            
        
        if X.ndim == 1:
            raise ValueError("Reshape your data")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], self.n_features_in_))

        pred = X.dot(self.w_)
        if self.fit_intercept:
            pred += self.b_
        return pred

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

class MultiClassifier(Classifier):
    r"""The multi-class classification class. The goal is to minimize the
    following objective

    .. math::
        \\min_{W,b} \\frac{1}{n} \\sum_{i=1}^n
        L\\left( y_i, W^\\top x_i + b\\right)   + \\psi(W),

    where :math:`L` is a classification loss, :math:`\\psi` is a regularization
    function (or constraint), :math:`W=[w_1,\\ldots,w_k]` is a (p x k) matrix
    that carries the k predictors, where k is the number of classes, and
    :math:`y_i` is a label in :math:`\\{1,\\ldots,k\\}`.
    b is a k-dimensional vector representing an unregularized intercept
    (which is optional).

    Parameters
    ----------
    loss: string, default='square'
        Loss function to be used. Possible choices are

        - any loss function compatible with the class BinaryClassifier e.g.
            ('square', 'logistic', 'sqhinge', 'safe-logistic'). In such a case,
            the loss function encodes a one vs. all strategy based on the
            chosen binary-classification loss.
        - 'multiclass-logistic', which is also called multinomial or
            softmax logistic:

        .. math::
            L(y, W^\\top x + b) = \\sum_{j=1}^k
            \\log\\left(e^{w_j^\\top + b_j} - e^{w_y^\\top + b_y} \\right)

    penalty: string, default='l2'
        Regularization function psi. Possible choices are

        - any penalty function compatible with the class BinaryClassifier such
          as ('none', 'l2', 'l1', 'elastic-net', 'fused-lasso', 'l1-ball',
          'l2-ball'). In such a case,. the penalty is applied on each predictor
           :math:`w_j` individually:

        .. math::
            \\psi(W) = \\sum_{j=1}^k \\psi(w_j).
        - 'l1l2', which is the multi-task group Lasso regularization

        .. math::
            \\psi(W) = \\lambda \\sum_{j=1}^p \\|W^j\\|_2~~~~
            \\text{where}~W^j~\\text{is the j-th row of}~W.
        - 'l1linf'

        .. math::
            \\psi(W) = \\lambda \\sum_{j=1}^p \\|W^j\\|_\\infty.

        - 'l1l2+l1', which is the multi-task group Lasso regularization + l1

        .. math::
            \\psi(W) = \\sum_{j=1}^p \\lambda \\|W^j\\|_2 + \\lambda_2 \|W^j\|_1 ~~~~
            \\text{where}~W^j~\\text{is the j-th row of}~W.


    fit_intercept: boolean, default='False'
        learns an unregularized intercept b, which is a k-dimensional vector
    """
    _estimator_type="classifier"

    def _more_tags(self):
        return {"_xfail_checks": {"check_classifiers_train": ("Different design"), }}

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, random_state=0):
        """Same as BinaryClassifier, but y should be a vector a n-dimensional
        vector of integers
        """
        print(y)
        X, y, le = check_input(X, y, self)

        self.le_ = le

        y = np.squeeze(y)
        if y.squeeze().ndim != 1 or np.any(y != y.astype(int)):
            raise ValueError("y should be a n-dimensional vector of integers")

        nclasses = np.max(y) + 1
        uniqu = np.unique(y)

        self.classes_ = uniqu

        if nclasses == 2:
            warnings.warn("Two classes detected, use BinaryClassifier instead")

        if X.shape[1] < 2:
            raise ValueError("There is only one 1 feature(s) in the training array!")    

        if (nclasses != uniqu.shape[0] or
                not all(np.unique(y) == np.arange(nclasses))):
            print("Class labels should be of the form")
            print(np.arange(nclasses))
            print("but they are")
            print(uniqu)
            print(X.shape)
            #TODO label shape
            #raise ValueError("Wrong label shape")

        return super().fit(
            X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3, solver=solver,
            tol=tol, it0=it0, max_epochs=max_epochs, l_qning=l_qning,
            f_restart=f_restart, verbose=verbose, restart=restart,
            univariate=False, nthreads=nthreads, random_state=self.random_state)

    def predict(self, X):
        """Predicts the class label"""
        check_is_fitted(self)

        if not scipy.sparse.issparse(X):
            X = np.array(X)

        if not scipy.sparse.issparse(X) and (X.dtype != "float32" or X.dtype != "float64"):
             X = np.asfortranarray(X, dtype="float64")

        if not scipy.sparse.issparse(X):
            if False in np.isfinite(X):
                raise ValueError("NaN of inf values in the training array(s)")
        
        if X.ndim == 1:
            raise ValueError("Reshape your data")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], self.n_features_in_))

        pred = X.dot(self.w_)
        if self.fit_intercept:
            pred += self.b_[np.newaxis, :]

        return np.argmax(pred, axis=1)

    def score(self, X, y):
        """Gives a classification score on new test data"""
        check_is_fitted(self)

        pred = np.squeeze(self.predict(X))
        return np.sum(np.squeeze(y) == pred) / pred.shape[0]

    def decision_function(self, X):
        check_is_fitted(self)

        if X.ndim == 1:
            raise ValueError("Reshape your data")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], self.n_features_in_))

        if self.fit_intercept:
            scores = safe_sparse_dot(X, self.w_, dense_output=True) + self.b_
        else:
            scores = safe_sparse_dot(X, self.w_, dense_output=True) 
        return scores.ravel() if scores.shape[1] == 1 else scores


    def predict_proba(self, X):
        check_is_fitted(self)

        if not scipy.sparse.issparse(X):
            if False in np.isfinite(X):
                raise ValueError("NaN of inf values in the training array(s)")

        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for multi_class="multinomial" and binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision = np.c_[-decision, decision]
        else:
            decision = decision
        return softmax(decision, copy=False)


class MultiVariateRegression(ERM):
    """
    The multivariate regression class. The objective is the same as for the
    MultiClassifier class, but we use a regression loss only (see below), and
    the targets :math:`y_i` are k-dimensional vectors.

        Parameters
        ----------
        loss: string, default='square'
            Only the square loss is implemented at this point. Given two
            k-dimensional vectors y,z:

            - 'square' =>  :math:`L(y,z) = \\frac{1}{2} \\|y-z\\|^2`

        penalty: string, default='l2'
            same as for the class MultiClassifier

        fit_intercept: boolean, default='False'
            learns an unregularized intercept b
    """

    def __init__(self, loss='square', penalty='l2', fit_intercept=False):
        if loss != 'square':
            raise ValueError("square loss should be used")
        super().__init__(loss=loss, penalty=penalty,
                         fit_intercept=fit_intercept)

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, random_state=0):
        """Same as ERM.fit, but y should be n x k, where k is size of the
        target for each data point
        """
        if y.squeeze().ndim <= 1:
            raise ValueError("y should be n x k, where k is size of the target "
                             "for each data point")
        return super().fit(
            X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3, solver=solver,
            tol=tol, it0=it0, max_epochs=max_epochs, l_qning=l_qning,
            f_restart=f_restart, verbose=verbose, restart=restart,
            univariate=False, nthreads=nthreads, random_state=self.random_state)

    def predict(self, X):
        
        """Predicts the targets"""
        pred = X.dot(self.w_)
        if self.fit_intercept:
            pred += self.b[np.newaxis, :]
        if self.le_ is None:                  
            return pred
        else:
            return self.le_.inverse_transform(pred)


class SKLearnClassifier(ERM):
    _estimator_type="classifier"


    def __init__(self, verbose=False, solver='auto', tol=1e-3, random_state=0):
        super(SKLearnClassifier, self).__init__(
            solver=solver, tol=tol, random_state=random_state)
        if verbose is not None:
            self.verbose = verbose

    def fit(self, X, y, C=None, lambd2=0, lambd3=0,
            solver='auto', tol=1e-3, it0=10, max_iter=None, l_qning=20,
            f_restart=50, restart=False, nthreads=-1, random_state=0):
        """Compatible with both binary and multi-classification. Here the parameter C replaces lambd,
        and max_iter replaces max_epochs.
        """
        X, y, le = check_input(X, y, self)
        self.le_ = le
        
        if len(X.shape) > 1 and X.shape[1] == 0:
            raise ValueError("0 feature(s) (shape=(" + str(X.shape[0]) + ", 0)) while a minimum of " + str(
                X.shape[0]) + " is required.")

        n = X.shape[0]
        if C is not None:
            self.C = C
        if max_iter is not None:
            self.max_iter = max_iter
        
        if self.le_ is not None:
            self.classes_ = self.le_.inverse_transform(np.unique(y))
        else:
            self.classes_ = np.unique(y)
        nb_classes = len(self.classes_)

        if not (type(self.C) == int or type(self.C) == float):
            raise ValueError("Penalty term must be positive")

        if self.loss == 'sqhinge':
            lambd = 1. / (2. * n * self.C)
        elif self.loss == 'logistic':
            lambd = 1. / (n * self.C)
        else:
            lambd = 1. / (2. * n * self.C)
        if nb_classes == 2:
            univariate = True
            if not np.all(self.classes_ == [-1, 1]):
                if self.le_ is not None:
                    neg = y == self.le_.transform(self.classes_)[0]
                else:
                    neg = y == self.classes_[0]
                y = y.copy()
                y[neg] = -1
                y[np.logical_not(neg)] = 1
        else:
            univariate = False
        super().fit(X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                    solver=solver, tol=tol, it0=it0,
                    max_epochs=self.max_iter, l_qning=l_qning,
                    f_restart=f_restart, verbose=self.verbose,
                    restart=restart, univariate=univariate, nthreads=nthreads,
                    random_state=self.random_state)

        self.w_ = self.w_.reshape(self.w_.shape[0], -1)
        if self.fit_intercept:
            self.b_ = self.b_.reshape(1, -1)
        else:
            self.b_ = 0.

        self.n_features_in_ = self.w_.shape[0]

        return self

    def decision_function(self, X):
        check_is_fitted(self)

        if X.ndim == 1:
            raise ValueError("Reshape your data")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], self.n_features_in_))

        scores = safe_sparse_dot(X, self.w_, dense_output=True) + self.b_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        check_is_fitted(self)

        X = X if scipy.sparse.issparse(
            X) else np.asfortranarray(X, dtype="float64")

        if not scipy.sparse.issparse(X):
            if False in np.isfinite(X):
                raise ValueError("NaN of inf values in the training array(s)")

        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int64)
        else:
            indices = scores.argmax(axis=1)
        if self.le_ is None:                  
            return self.classes_[indices]
        else:
            return self.le_.inverse_transform(indices)

    def predict_proba(self, X):
        check_is_fitted(self)

        X = X if scipy.sparse.issparse(
            X) else np.asfortranarray(X, dtype="float32")

        if not scipy.sparse.issparse(X):
            if False in np.isfinite(X):
                raise ValueError("NaN of inf values in the training array(s)")

        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for multi_class="multinomial" and binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision_2d = np.c_[-decision, decision]
        else:
            decision_2d = decision
        return softmax(decision_2d, copy=False)

    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)
        
        return np.average(y == self.predict(X), weights=sample_weight)



class LinearSVC(SKLearnClassifier):
    """
    A compatibility class for scikit-learn user, but only for square hinge loss
    It is perfectly equivalent to the BinaryClassifier class, but the
    regularization parameter (here "C") is provided during the class
    initialization. Note that :math:`C= \\frac{1}{2n \\lambda}`

    Parameters
    ----------

    loss: should be 'sqhinge' or 'squared_hinge'

    penalty: same as BinaryClassifier

    fit_intercept: same as BinaryClassifier

    C: regularization parameter

    max_iter: maximum number of iterations for the optimization solver
    """

    def __init__(self, loss='sqhinge', penalty='l2', fit_intercept=True, C=1,
                 max_iter=500):
        if loss != 'sqhinge' and loss != 'squared_hinge':
            print("LinearSVC is only compatible with squared hinge loss at "
                  "the moment")
        super(SKLearnClassifier, self).__init__(
            loss='sqhinge', penalty=penalty,
            fit_intercept=fit_intercept)
        self.C = C
        self.max_iter = max_iter


class LogisticRegression(SKLearnClassifier):
    """
    A compatibility class for scikit-learn user, but only for square hinge loss
    It is perfectly equivalent to the BinaryClassifier class, but the
    regularization parameter (here "C") is provided during the class
    initialization. Note that :math:`C= \\frac{1}{n \\lambda}`

    Parameters
    ----------

    loss: should be 'sqhinge' or 'squared_hinge'

    penalty: same as BinaryClassifier

    fit_intercept: same as BinaryClassifier

    C: regularization parameter

    max_iter: maximum number of iterations for the optimization solver
    """

    _estimator_type = "classifier"

    def __init__(self, penalty='l2', fit_intercept=True, C=1, max_iter=500, solver="auto", tol=1e-3, random_state=0):
        super(LogisticRegression, self).__init__(
            solver=solver, tol=tol, random_state=random_state)
        self.C = C
        self.max_iter = max_iter
        self.loss = 'logistic'
        self.penalty = penalty
        self.fit_intercept = fit_intercept


class Lasso(Regression):
    def __init__(self, fit_intercept=False):
        super().__init__(loss='square', penalty='l1',
                         fit_intercept=fit_intercept, random_state=0)
        self.aux = Regression(loss='square', penalty='l1',
                              fit_intercept=fit_intercept, random_state=random_state)

    def fit(self, X, y, lambd=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, random_state=0):
        n, p = X.shape
        if p <= 1000:
            # no active set
            super().fit(X, y, lambd=lambd, solver=solver, tol=tol, it0=it0, max_epochs=max_epochs, l_qning=l_qning,
                        f_restart=f_restart, verbose=verbose, restart=restart, nthreads=nthreads, random_state=random_state)
        else:
            scaling = 4.0
            init = min(100, p)
            restart = True
            num_as = math.ceil(math.log10(p / init) / math.log10(scaling))
            active_set = []
            n_active = 0
            self.w_ = np.zeros(p, dtype=X.dtype)
            if self.fit_intercept:
                self.b_ = 0

            for ii in range(num_as):
                if n_active == 0:
                    R = y
                else:
                    pred = self.aux.predict(X[:, active_set])
                    R = y.ravel() - pred.ravel()
                corr = np.abs(X.transpose().dot(R).ravel()) / n
                if n_active > 0:
                    corr[active_set] = -10e10
                n_new_as = max(
                    min(init * math.ceil(scaling ** ii), p) - n_active, 0)
                new_as = corr.argsort()[-n_new_as:]
                if len(new_as) == 0 or max(corr[new_as]) <= lambd * (1 + tol):
                    break
                if len(active_set) > 0:
                    neww = np.zeros(n_active + n_new_as, dtype=X.dtype)
                    neww[0:n_active] = self.aux.w
                    self.aux.w = neww
                    active_set = np.concatenate((active_set, new_as))
                else:
                    active_set = new_as
                    self.aux.w = np.zeros(len(active_set), dtype=X.dtype)
                n_active = len(active_set)
                if (verbose):
                    print("Size of the active set: " + str(n_active))
                self.aux.fit(X[:, active_set], y, lambd, tol=tol, it0=5, max_epochs=max_epochs, restart=restart,
                             solver=solver, verbose=verbose)
                self.w_[active_set] = self.aux.w
                if self.fit_intercept:
                    self.b_ = self.aux.b


# TODO: remove code duplication with Lasso
class L1Logistic(BinaryClassifier):
    def __init__(self, fit_intercept=False):
        super().__init__(loss='logistic', penalty='l1',
                         fit_intercept=fit_intercept)
        self.aux = BinaryClassifier(
            loss='logistic', penalty='l1', fit_intercept=fit_intercept)

    def fit(self, X, y, lambd=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, random_state=0):
        n, p = X.shape
        if p <= 1000:
            # no active set
            super().fit(X, y, lambd=lambd, solver=solver, tol=tol, it0=it0, max_epochs=max_epochs, l_qning=l_qning,
                        f_restart=f_restart, verbose=verbose, restart=restart, nthreads=nthreads, random_state=random_state)
        else:
            scaling = 4.0
            init = min(100, p)
            restart = True
            num_as = math.ceil(math.log10(p / init) / math.log10(scaling))
            active_set = []
            n_active = 0
            self.w_ = np.zeros(p, dtype=X.dtype)
            if self.fit_intercept:
                self.b_ = 0

            for ii in range(num_as):
                #  log(1 + exp(-y_i pred_i))
                # abs_grad =    sum - y_i/(1 + exp(y_i pred_i)) x_i
                if n_active == 0:
                    R = -0.5 * y.ravel()
                else:
                    pred = X[:, active_set].dot(self.aux.w)
                    if self.fit_intercept:
                        pred += self.aux.b
                    R = -y.ravel() / (1.0 + np.exp(y.ravel() * pred.ravel()))
                corr = np.abs(X.transpose().dot(R).ravel()) / X.shape[0]
                if n_active > 0:
                    corr[active_set] = -10e10
                n_new_as = max(
                    min(init * math.ceil(scaling ** (ii)), p) - n_active, 0)
                new_as = corr.argsort()[-n_new_as:]
                if len(new_as) == 0 or max(corr[new_as]) <= lambd * (1 + tol):
                    break
                if len(active_set) > 0:
                    neww = np.zeros(n_active + n_new_as, dtype=X.dtype)
                    neww[0:n_active] = self.aux.w
                    self.aux.w = neww
                    active_set = np.concatenate((active_set, new_as))
                else:
                    active_set = new_as
                    self.aux.w = np.zeros(len(active_set), dtype=X.dtype)
                n_active = len(active_set)
                if verbose:
                    print("Size of the active set: " + str(n_active))
                self.aux.fit(X[:, active_set], y, lambd, tol=tol, it0=5, max_epochs=max_epochs, restart=restart,
                             solver=solver, verbose=verbose)
                self.w_[active_set] = self.aux.w
                if self.fit_intercept:
                    self.b_ = self.aux.b
