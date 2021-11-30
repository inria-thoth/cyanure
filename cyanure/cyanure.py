# Author: Julien Mairal <julien.mairal@inria.fr>
#
# License: BSD 3 clause

from abc import abstractmethod, ABC

import math

import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot, softmax

import cyanure_lib

from cyanure.data_processing import check_input

from cyanure.logger import setup_custom_logger

logger = setup_custom_logger('project', "INFO")


class ERM(BaseEstimator, ABC):
    """The generic class for empirical risk minimization problems.
    For univariates problems, minimizes

        min_{w,b} (1/n) sum_{i=1}^n L( y_i, <w, x_i> + b)   + psi(w)

    """

    def _more_tags(self):
        return {"requires_y": True}

    def __init__(self, loss='square', penalty='l2', fit_intercept=True, dual=None, tol=1e-3, solver="auto",
                 random_state=0, max_iter=500, fista_restart=50,
                 verbose=True, restart=False, limited_memory_qning=20, _binary_problem=True,
                 lambd=0, lambd2=0, lambd3=0, duality_gap_interval=5, n_threads=-1):
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
            Note that for binary classification, we assume the y to be of
            the form {-1,+1}

        penalty: string, default='none'
            Regularization function psi. Possible choices are

            For binary_problem problems
            - 'none' => psi(w) = 0
            - 'l2' =>  psi{w) = (lambd/2) ||w||_2^2
            - 'l1' =>  psi{w) = lambd ||w||_1
            - 'elastic-net' =>  psi{w) = lambd ||w||_1 + (lambd2/2)||w||_2^2
            - 'fused-lasso' => psi(w) = lambd3 sum_{i=2}^p |w[i]-w[i-1]|
                                      + lambd||w||_1 + (lambd2/2)||w||_2^2
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
        self.dual = dual
        self.solver = solver
        self.tol = tol
        self.random_state = random_state
        self.max_iter = max_iter
        # TODO change regularization names
        self.lambd = lambd
        self.lambd2 = lambd2
        self.lambd3 = lambd3
        self.limited_memory_qning = limited_memory_qning
        self.fista_restart = fista_restart
        self.verbose = verbose
        self.restart = restart
        self._binary_problem = _binary_problem
        self.duality_gap_interval = duality_gap_interval
        self.n_threads = n_threads

    def fit(self, X, y, le_parameter=None):
        """
        The fitting function (the one that does the job)

        Parameters
        ----------

        X : numpy array, or scipy sparse CSR matrix
            input n X p numpy matrix; the samples are on the rows

        y : y, numpy array.
            - vector of size n with real values for regression
            - vector of size n with {-1,+1} y for binary classification,
              which will be automatically converted if y in {0,1} are
              provided
            - matrix of size n X k for multivariate regression
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

         max_iter: int, default=500
            Maximum number of iteration of the algorithm in terms of passes
            over the data

         duality_gap_interval: int, default=10
            Frequency of duality-gap computation

         verbose: boolean, default=True
            Display information or not

         n_threads: int, default=-1
            maximum number of cores the method may use (-1 = all cores).
            Note that more cores is not always better.

         seed: int, default=0
            random seed

         restart: boolean, default=False
            use a restart strategy (useful for computing regularization path)

         binary_problem: boolean, default=True
            binary_problem or multivariate problems

         limited_memory_qning: int, default=20
            memory paramter for the qning method

         fista_restart: int, default=50
            restart strategy for fista


        Returns
        -------

        test returns a numpy array carrying information about the optimization
        process (number of iterations, objective function values, duality gap)
        will be documented in the future if people ask me,
        """

        X, y, le = check_input(X, y, self)
        if le_parameter is not None:
            self.le_ = le_parameter
        else:
            self.le_ = le

        training_data_fortran = X.T if scipy.sparse.issparse(
            X) else np.asfortranarray(X.T)
        p = X.shape[1] + \
            1 if self.fit_intercept else X.shape[1]
        y = np.squeeze(y)

        if self._binary_problem:
            w0 = np.zeros(p, dtype=training_data_fortran.dtype)
            yf = np.squeeze(y.astype(training_data_fortran.dtype))
        else:
            if y.squeeze().ndim > 1:
                nclasses = y.squeeze().shape[1]
                yf = np.asfortranarray(y.T)
            else:
                nclasses = int(np.max(y) + 1)
                yf = np.squeeze(np.int32(y))
            w0 = np.zeros(
                [p, nclasses], dtype=training_data_fortran.dtype, order='F')

        if self.restart and np.any(self.w_ != 0):
            if self.verbose:
                logger.info("Restart")
            if self.fit_intercept:
                w0[-1, ] = self.b_
                w0[0:-1, ] = self.w_
            else:
                w0 = self.w_

        if self.restart and (self.solver == 'auto' or self.solver == 'miso' or
                             self.solver == 'catalyst-miso' or self.solver == 'qning-miso'):
            n = X.shape[0]
            reset_dual = np.any(self.dual is None)
            if not reset_dual and self._binary_problem:
                reset_dual = self.dual.shape[0] != n
            if not reset_dual and not self._binary_problem:
                reset_dual = np.any(self.dual.shape != [n, nclasses])
            if reset_dual and self._binary_problem:
                self.dual = np.zeros(n, dtype=training_data_fortran.dtype)
            if reset_dual and not self._binary_problem:
                self.dual = np.zeros(
                    [n, nclasses], dtype=training_data_fortran.dtype)

        w = np.copy(w0)
        optimization_info = cyanure_lib.erm_(
            training_data_fortran, yf, w0, w, dual_variable=self.dual, loss=self.loss,
            penalty=self.penalty, solver=self.solver, lambd=float(self.lambd),
            lambd2=float(self.lambd2), lambd3=float(self.lambd3),
            intercept=bool(self.fit_intercept), tol=float(self.tol), duality_gap_interval=int(self.duality_gap_interval),
            max_iter=int(self.max_iter), limited_memory_qning=int(self.limited_memory_qning),
            fista_restart=int(self.fista_restart), verbose=bool(self.verbose),
            univariate=bool(self._binary_problem), n_threads=int(self.n_threads), seed=int(self.random_state)
        )

        # TODO fix onevsall bug in c++ (optim_info.add(optim_info_col)) + remove ternary
        self.n_iter_ = optimization_info[0][-1] if optimization_info[0][-1] >= 1 else 1

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
        predict the y given an input matrix X (same format as fit)
        """

    def get_weights(self):
        """
        get the model parameters (either w or the tuple (w,b))
        """
        return (self.w_, self.b_) if self.fit_intercept else self.w_

    def eval(self, X, y):
        """
        get the value of the objective function and computes a relative
        duality gap, see function fit for the format of parameters.
        """
        return self.fit(X, y)

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
    def _get_param_namesrestart(cls):
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

        # Grouped by prefix
        nested_params = defaultdict(dict)
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

    def _more_tags(self):
        return {"multioutput": True}

    def __init__(self, loss='square', penalty='l2', fit_intercept=True, random_state=0,
                 lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
                 duality_gap_interval=10, max_iter=500, limited_memory_qning=20, fista_restart=50, verbose=True,
                 restart=False, n_threads=-1):
        if loss != 'square':
            raise ValueError("square loss should be used")
        super().__init__(loss=loss, penalty=penalty,
                         fit_intercept=fit_intercept, random_state=random_state, lambd=lambd,
                         lambd2=lambd2, lambd3=lambd3, solver=solver, tol=tol,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning, fista_restart=fista_restart, verbose=verbose,
                         restart=restart, n_threads=n_threads)

    def fit(self, X, y):
        """
        The fitting function is the same as for the class BinaryClassifier,
        except that we do not necessarily expect binary labels in y.
        """
        X, y, _ = check_input(X, y, self)

        if y.squeeze().ndim <= 1:
            self._binary_problem = True
        else:
            self._binary_problem = False

        return super().fit(X, y)

    def predict(self, X):
        check_is_fitted(self)

        if not scipy.sparse.issparse(X):
            X = np.array(X)
            if X.dtype != "float32" or X.dtype != "float64":
                X = np.asfortranarray(
                    X, dtype="float64")

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
            softmself.le_ = leax logistic:

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
    _estimator_type = "classifier"

    def __init__(self, loss='square', penalty='l2', fit_intercept=True, tol=0.001, solver="auto",
                 random_state=0, max_iter=500, fista_restart=50, verbose=True, restart=False,
                 limited_memory_qning=20, lambd=0, lambd2=0, lambd3=0, duality_gap_interval=5, n_threads=-1):
        super().__init__(loss=loss, penalty=penalty, fit_intercept=fit_intercept, tol=tol, solver=solver,
                         random_state=random_state, max_iter=max_iter, fista_restart=fista_restart,
                         verbose=verbose, restart=restart, limited_memory_qning=limited_memory_qning,
                         lambd=lambd, lambd2=lambd2, lambd3=lambd3, duality_gap_interval=duality_gap_interval,
                         n_threads=n_threads)

    def fit(self, X, y, le_parameter=None):
        """Same as BinaryClassifier, but y should be a vector a n-dimensional
        vector of integers
        """
        X, y, le = check_input(X, y, self)
        if le_parameter is not None:
            self.le_ = le_parameter
        else:
            self.le_ = le

        y = np.squeeze(y)
        unique = np.unique(y)
        nb_classes = len(unique)

        if self.le_ is not None:
            self.classes_ = self.le_.classes_
        else:
            self.classes_ = unique

        if (nb_classes != unique.shape[0] or
                not all(np.unique(y) == np.arange(nb_classes))):
            logger.info("Class y should be of the form")
            logger.info(np.arange(nb_classes))
            logger.info("but they are")
            logger.info(unique)
            if nb_classes != 2:
                raise Warning("Wrong label format for a multiclass problem!")
            else:
                logger.info(
                    "The y have been converted to respect the expected format.")

        if nb_classes == 2:
            self._binary_problem = True
            if self.le_ is not None:
                neg = y == self.le_.transform(self.classes_)[0]
            else:
                neg = y == self.classes_[0]
            y = y.copy()
            y[neg] = -1
            y[np.logical_not(neg)] = 1
        else:
            self._binary_problem = False

        return super().fit(
            X, y, le_parameter=self.le_)

    def predict(self, X):
        """Predicts the class label"""
        check_is_fitted(self)

        if not scipy.sparse.issparse(X):
            X = np.array(X)

        if not scipy.sparse.issparse(X) and (X.dtype != "float32" or X.dtype != "float64"):
            X = np.asfortranarray(X, dtype="float64")

        if not scipy.sparse.issparse(X) and False in np.isfinite(X):
            raise ValueError("NaN of inf values in the training array(s)")

        if X.ndim == 1:
            raise ValueError("Reshape your data")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], self.n_features_in_))

        pred = self.decision_function(X)

        output = None
        if len(self.classes_) == 2:
            if self.le_ is None:
                output = np.sign(pred)
                output[output == -1.0] = self.classes_[0]
                output = output.astype(np.int32)
            else:
                output = np.sign(pred)
                output[output == -1.0] = 0
                output = output.astype(np.int32)
                output = self.le_.inverse_transform(output)
        else:
            if self.le_ is None:
                output = np.argmax(pred, axis=1)
            else:
                output = self.le_.inverse_transform(np.argmax(pred, axis=1))

        return output

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
            scores = safe_sparse_dot(
                X, self.w_, dense_output=True) + self.b_
        else:
            scores = safe_sparse_dot(X, self.w_, dense_output=True)

        output = None
        if len(self.classes_) == 2:
            output = scores.ravel()
        else:
            output = scores.ravel() if scores.shape[1] == 1 else scores

        return output

    def predict_proba(self, X):
        check_is_fitted(self)

        if not scipy.sparse.issparse(X) and False in np.isfinite(X):

            raise ValueError("NaN of inf values in the training array(s)")

        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision = np.c_[-decision, decision]
        return softmax(decision, copy=False)


class SKLearnClassifier(ERM):
    _estimator_type = "classifier"

    def __init__(self, loss, penalty, fit_intercept=True,
                 verbose=False, lambd=0, lambd2=0, lambd3=0,
                 solver='auto', tol=1e-3, duality_gap_interval=10, max_iter=None, limited_memory_qning=20,
                 fista_restart=50, restart=False, n_threads=-1, random_state=0):
        super().__init__(
            loss=loss, penalty=penalty, fit_intercept=fit_intercept,
            solver=solver, tol=tol, random_state=random_state, verbose=verbose,
            lambd=lambd, lambd2=lambd2, lambd3=lambd3,
            duality_gap_interval=duality_gap_interval, max_iter=max_iter, limited_memory_qning=limited_memory_qning,
            fista_restart=fista_restart, restart=restart, n_threads=n_threads)

    def fit(self, X, y, le_parameter=None):
        """Compatible with both binary and multi-classification. Here the parameter C replaces lambd,
        and max_iter replaces max_iter.
        """
        X, y, le = check_input(X, y, self)
        if le_parameter is not None:
            self.le_ = le_parameter
        else:
            self.le_ = le

        if len(X.shape) > 1 and X.shape[1] == 0:
            raise ValueError("0 feature(s) (shape=(" + str(X.shape[0]) + ", 0)) while a minimum of " + str(
                X.shape[0]) + " is required.")

        if self.le_ is not None:
            self.classes_ = self.le_.inverse_transform(np.unique(y))
        else:
            self.classes_ = np.unique(y)
        nb_classes = len(self.classes_)

        if nb_classes == 2:
            self._binary_problem = True
            if not np.all(self.classes_ == [-1, 1]):
                if self.le_ is not None:
                    neg = y == self.le_.transform(self.classes_)[0]
                else:
                    neg = y == self.classes_[0]
                y = y.copy()
                y[neg] = -1
                y[np.logical_not(neg)] = 1
        else:
            self._binary_problem = False
        super().fit(X, y, le_parameter=self.le_)

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

        scores = safe_sparse_dot(
            X, self.w_, dense_output=True) + self.b_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        check_is_fitted(self)

        X = X if scipy.sparse.issparse(
            X) else np.asfortranarray(X, dtype="float64")

        if not scipy.sparse.issparse(X) and False in np.isfinite(X):
            raise ValueError("NaN of inf values in the training array(s)")

        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int64)
        else:
            indices = scores.argmax(axis=1)

        output = None
        if self.le_ is None:
            output = self.classes_[indices]
        else:
            output = self.le_.inverse_transform(indices)

        return output

    def predict_proba(self, X):
        check_is_fitted(self)

        X = X if scipy.sparse.issparse(
            X) else np.asfortranarray(X, dtype="float32")

        if not scipy.sparse.issparse(X) and False in np.isfinite(X):
            raise ValueError("NaN of inf values in the training array(s)")

        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for binary_problem="multinomial" and binary outcomes
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

    def __init__(self, loss='sqhinge', penalty='l2', fit_intercept=True,
                 verbose=False, lambd=0.1, lambd2=0, lambd3=0,
                 solver='auto', tol=1e-3, duality_gap_interval=10, max_iter=500, limited_memory_qning=20,
                 fista_restart=50, restart=False, n_threads=-1, random_state=0):
        if loss not in ['squared_hinge', 'sqhinge']:
            logger.error("LinearSVC is only compatible with squared hinge loss at "
                         "the moment")
        super().__init__(
            loss=loss, penalty=penalty, fit_intercept=fit_intercept,
            solver=solver, tol=tol, random_state=random_state, verbose=verbose,
            lambd=lambd, lambd2=lambd2, lambd3=lambd3,
            duality_gap_interval=duality_gap_interval, max_iter=max_iter,
            limited_memory_qning=limited_memory_qning,
            fista_restart=fista_restart, restart=restart, n_threads=n_threads)
        self.lambd = lambd
        self.max_iter = max_iter
        self.verbose = False


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

    def __init__(self, penalty='l2', loss='logistic', fit_intercept=True,
                 verbose=False, lambd=0, lambd2=0, lambd3=0,
                 solver='auto', tol=1e-3, duality_gap_interval=10, max_iter=500, limited_memory_qning=20,
                 fista_restart=50, restart=False, n_threads=-1, random_state=0):
        super().__init__(loss=loss, penalty=penalty, fit_intercept=fit_intercept,
                         solver=solver, tol=tol, random_state=random_state, verbose=verbose,
                         lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning,
                         fista_restart=fista_restart, restart=restart, n_threads=n_threads)


def compute_r(estimator_name, aux, X, y, active_set, n_active):
    R = None

    pred = aux.predict(X[:, active_set])
    if estimator_name == "Lasso":
        if n_active == 0:
            R = y
        else:
            R = y.ravel() - pred.ravel()
    elif estimator_name == "L1Logistic":
        if n_active == 0:
            R = -0.5 * y.ravel()
        else:
            R = -y.ravel() / (1.0 + np.exp(y.ravel() * pred.ravel()))

    return R


def fit_large_feature_number(estimator, aux, X, y):
    n, p = X.shape

    scaling = 4.0
    init = min(100, p)
    estimator.restart = True
    num_as = math.ceil(math.log10(p / init) / math.log10(scaling))
    active_set = []
    n_active = 0
    estimator.w_ = np.zeros(p, dtype=X.dtype)
    if estimator.fit_intercept:
        estimator.b_ = 0

    for ii in range(num_as):
        R = compute_r(estimator.__name__, aux, X, y, active_set, n_active)

        corr = np.abs(X.transpose().dot(R).ravel()) / n
        if n_active > 0:
            corr[active_set] = -10e10
        n_new_as = max(
            min(init * math.ceil(scaling ** ii), p) - n_active, 0)
        new_as = corr.argsort()[-n_new_as:]
        if len(new_as) == 0 or max(corr[new_as]) <= estimator.lambd * (1 + estimator.tol):
            break
        if len(active_set) > 0:
            neww = np.zeros(n_active + n_new_as,
                            dtype=X.dtype)
            neww[0:n_active] = aux.w_
            aux.w_ = neww
            active_set = np.concatenate((active_set, new_as))
        else:
            active_set = new_as
            aux.w_ = np.zeros(
                len(active_set), dtype=X.dtype)
        n_active = len(active_set)
        if estimator.verbose:
            logger.info("Size of the active set: {%d}", n_active)
        aux.fit(X[:, active_set], y)
        estimator.w_[active_set] = aux.w_
        if estimator.fit_intercept:
            estimator.b_ = aux.b_


class Lasso(Regression):
    def __init__(self, lambd=0, solver='auto', tol=1e-3,
                 duality_gap_interval=10, max_iter=500, limited_memory_qning=20, fista_restart=50, verbose=True,
                 restart=False, n_threads=-1, random_state=0, fit_intercept=True):
        super().__init__(loss='square', penalty='l1', lambd=lambd, solver=solver, tol=tol,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning, fista_restart=fista_restart,
                         verbose=verbose, restart=restart, n_threads=n_threads,
                         random_state=random_state, fit_intercept=fit_intercept)

    def fit(self, X, y):

        X, y, _ = check_input(X, y, self)

        _, p = X.shape
        if p <= 1000:
            # no active set
            super().fit(X, y)
        else:
            aux = Regression(loss='square', penalty='l1',
                             fit_intercept=self.fit_intercept, random_state=self.random_state)

            fit_large_feature_number(self, aux, X, y)

        return self


class L1Logistic(MultiClassifier):

    _estimator_type = "classifier"

    def _more_tags(self):
        return {"requires_y": True}

    def __init__(self, lambd=0, solver='auto', tol=1e-3,
                 duality_gap_interval=10, max_iter=500, limited_memory_qning=20, fista_restart=50, verbose=True,
                 restart=False, n_threads=-1, random_state=0, fit_intercept=True):
        super().__init__(loss='logistic', penalty='l1', lambd=lambd, solver=solver, tol=tol,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning,
                         fista_restart=fista_restart, verbose=verbose,
                         restart=restart, n_threads=n_threads, random_state=random_state,
                         fit_intercept=fit_intercept)

    def fit(self, X, y):

        X, y, le = check_input(X, y, self)
        self.le_ = le

        _, p = X.shape
        if p <= 1000:
            # no active set
            super().fit(X, y, le_parameter=self.le_)
        else:
            aux = MultiClassifier(
                loss='logistic', penalty='l1', fit_intercept=self.fit_intercept)

            fit_large_feature_number(self, aux, X, y)

        return self
