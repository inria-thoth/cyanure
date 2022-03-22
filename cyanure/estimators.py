# Author: Julien Mairal <julien.mairal@inria.fr>
#
# License: BSD 3 clause

from abc import abstractmethod, ABC

import math
import warnings

import numpy as np
import scipy.sparse

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot, softmax
from sklearn.exceptions import ConvergenceWarning

import cyanure_lib

from cyanure.data_processing import check_input_fit, check_input_inference

from cyanure.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class ERM(BaseEstimator, ABC):
    """The generic class for empirical risk minimization problems.
    For univariates problems, minimizes

        min_{w,b} (1/n) sum_{i=1}^n L( y_i, <w, x_i> + b)   + psi(w)

    """

    def _more_tags(self):
        return {"requires_y": True}

    def __init__(self, loss='square', penalty='l2', fit_intercept=False, dual=None, tol=1e-3, solver="auto",
                 random_state=0, max_iter=2000, fista_restart=60,
                 verbose=True, warm_start=False, limited_memory_qning=50, multi_class="auto",
                 lambda_1=0, lambda_2=0, lambda_3=0, duality_gap_interval=5, n_threads=-1):
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
            - 'l2' =>  psi{w) = (lambda_1/2) ||w||_2^2
            - 'l1' =>  psi{w) = lambda_1 ||w||_1
            - 'elasticnet' =>  psi{w) = lambda_1 ||w||_1 + (lambda_2/2)||w||_2^2
            - 'fused-lasso' => psi(w) = lambda_3 sum_{i=2}^p |w[i]-w[i-1]|
                                      + lambda_1||w||_1 + (lambda_2/2)||w||_2^2
            - 'l1-ball'     => encodes the constraint ||w||_1 <= lambda_1
            - 'l2-ball'     => encodes the constraint ||w||_2 <= lambda_1

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
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.limited_memory_qning = limited_memory_qning
        self.fista_restart = fista_restart
        self.verbose = verbose
        self.warm_start = warm_start
        self.multi_class = multi_class
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

        lambda_1: float, default=0
            first regularization parameter

        lambda_2: float, default=0
            second regularization parameter, if needed

        lambda_3: float, default=0
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
        loss = None

        X, y, le = check_input_fit(X, y, self)
        if le_parameter is not None:
            self.le_ = le_parameter
        else:
            self.le_ = le

        if (self.multi_class == "multinomial" or (self.multi_class == "auto" and not self._binary_problem)) and self.loss=="logistic":
            if (self.multi_class == "multinomial"):
                if len(np.unique(y)) != 2:
                    self._binary_problem=False
                else:
                    nclasses = len(np.unique(y))
            loss = "multiclass-logistic"
            logger.info("Loss has been set to multiclass-logistic because the multiclass parameter is set to multinomial!")

        if loss is None:
            loss = self.loss

        training_data_fortran = X.T if scipy.sparse.issparse(
            X) else np.asfortranarray(X.T)
        p = X.shape[1] + \
            1 if self.fit_intercept else X.shape[1]
        y = np.squeeze(y)

        if self._binary_problem:
            w0 = np.zeros((p), dtype=training_data_fortran.dtype)
            yf = np.squeeze(y.astype(training_data_fortran.dtype))
        else:
            if y.squeeze().ndim > 1:
                nclasses = y.squeeze().shape[1]
                yf = np.asfortranarray(y.T)
            else:
                nclasses = int(np.max(y) + 1)
                yf = np.squeeze(np.intc(np.float64(y)))
            w0 = np.zeros(
                [p, nclasses], dtype=training_data_fortran.dtype, order='F')

        if self.warm_start and hasattr(self, "coef_"):
            if self.verbose:
                logger.info("Restart")
            if self.fit_intercept:
                w0[-1, ] = self.intercept_
                w0[0:-1, ] = np.squeeze(self.coef_)
            else:
                w0 = np.squeeze(self.coef_)
                    
        if self.warm_start and (self.solver == 'auto' or self.solver == 'miso' or
                             self.solver == 'catalyst-miso' or self.solver == 'qning-miso'):
            n = X.shape[0]
            # TODO Ecrire test pour dual surtout défensif
            reset_dual = np.any(self.dual is None)
            if not reset_dual and self._binary_problem:
                reset_dual = self.dual.shape[0] != n
            if not reset_dual and not self._binary_problem:
                reset_dual = np.any(self.dual.shape != [n, nclasses])
            if reset_dual and self._binary_problem:
                self.dual = np.zeros(n, dtype=training_data_fortran.dtype, order='F')
            if reset_dual and not self._binary_problem:
                self.dual = np.zeros(
                    [n, nclasses], dtype=training_data_fortran.dtype, order='F')

        w = np.copy(w0)
        self.optimization_info_ = cyanure_lib.erm_(
            training_data_fortran, yf, w0, w, dual_variable=self.dual, loss=loss,
            penalty=self.penalty, solver=self.solver, lambda_1=float(self.lambda_1),
            lambda_2=float(self.lambda_2), lambda_3=float(self.lambda_3),
            intercept=bool(self.fit_intercept), tol=float(self.tol), duality_gap_interval=int(self.duality_gap_interval),
            max_iter=int(self.max_iter), limited_memory_qning=int(self.limited_memory_qning),
            fista_restart=int(self.fista_restart), verbose=bool(self.verbose),
            univariate=bool(self._binary_problem), n_threads=int(self.n_threads), seed=int(self.random_state)
        )

        if ((self.multi_class == "multinomial" or (self.multi_class == "auto" and not self._binary_problem)) and self.loss=="logistic") and self.optimization_info_.shape[0] == 1:
            self.optimization_info_ = np.repeat(self.optimization_info_, nclasses, axis=0)

        self.n_iter_ = np.array([self.optimization_info_[class_index][0][-1] for class_index in range(self.optimization_info_.shape[0])])

        # TODO Vérifier avec Julien
        for index in range(self.n_iter_.shape[0]):
            if self.n_iter_[index] == self.max_iter:
                warnings.warn("The max_iter was reached which means the coef_ did not converge", ConvergenceWarning)

        if self.fit_intercept:
            self.intercept_ = w[-1, ]
            self.coef_ = w[0:-1, ]
        else:
            self.coef_ = w

        self.n_features_in_ = self.coef_.shape[0]

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
        return (self.coef_, self.intercept_) if self.fit_intercept else self.coef_

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
    
    def densify(self):
        """
        Convert coefficient matrix to dense array format.
        Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
        default format of ``coef_`` and is required for fitting, so calling
        this method is only required on models that have previously been
        sparsified; otherwise, it is a no-op.

        Returns
        -------
        self
            Fitted estimator.

        """
        msg = "Estimator, %(name)s, must be fitted before densifying."
        check_is_fitted(self, msg=msg)
        if scipy.sparse.issparse(self.coef_):
            self.coef_ = self.coef_.toarray()
        return self

    def sparsify(self):
        """
        Convert coefficient matrix to sparse format.
        Converts the ``coef_`` member to a scipy.sparse matrix, which for
        L1-regularized models can be much more memory- and storage-efficient
        than the usual numpy.ndarray representation.
        The ``intercept_`` member is not converted.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        For non-sparse models, i.e. when there are not many zeros in ``coef_``,
        this may actually *increase* memory usage, so use this method with
        care. A rule of thumb is that the number of zero elements, which can
        be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
        to provide significant benefits.
        After calling this method, further fitting with the partial_fit
        method (if any) will not work until you call densify.

        """
        msg = "Estimator, %(name)s, must be fitted before sparsifying."
        check_is_fitted(self, msg=msg)
        self.coef_ = scipy.sparse.csr_matrix(self.coef_)
        if self.coef_.shape[0] == 1:
            self.coef_ = self.coef_.T
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
                 lambda_1=0, lambda_2=0, lambda_3=0, solver='auto', tol=1e-3,
                 duality_gap_interval=10, max_iter=500, limited_memory_qning=20, fista_restart=50, verbose=True,
                 warm_start=False, n_threads=-1, dual=None):
        if loss != 'square':
            raise ValueError("square loss should be used")
        super().__init__(loss=loss, penalty=penalty,
                         fit_intercept=fit_intercept, random_state=random_state, lambda_1=lambda_1,
                         lambda_2=lambda_2, lambda_3=lambda_3, solver=solver, tol=tol,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning, fista_restart=fista_restart, verbose=verbose,
                         warm_start=warm_start, n_threads=n_threads, dual=dual)

    def fit(self, X, y):
        """
        The fitting function is the same as for the class BinaryClassifier,
        except that we do not necessarily expect binary labels in y.
        """
        X, y, _ = check_input_fit(X, y, self)

        if y.squeeze().ndim <= 1:
            self._binary_problem = True
        else:
            self._binary_problem = False

        return super().fit(X, y)

    def predict(self, X):
        check_is_fitted(self)

        X = check_input_inference(X, self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        pred = safe_sparse_dot(
                X, self.coef_, dense_output=False) + self.intercept_
        
        return pred.squeeze()

    def score(self, X, y, sample_weight=None):
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)



class MultiClassifier(Classifier):
    """The multi-class classification class. The goal is to minimize the
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
          as ('none', 'l2', 'l1', 'elasticnet', 'fused-lasso', 'l1-ball',
          'l2-ball'). In such a case,. the penalty is applied on each predictor
          :math:`w_j` individually:

        .. math::
           \\psi(W) = \\sum_{j=1}^k \\psi(w_j).

        - 'l1l2', which is the multi-task group Lasso regularization

        .. math::
           \\psi(W) = \\lambda_1 \\sum_{j=1}^p \\|W^j\\|_2~~~~
           \\text{where}~W^j~\\text{is the j-th row of}~W.

        - 'l1linf'

        .. math::
            \\psi(W) = \\lambda_1 \\sum_{j=1}^p \\|W^j\\|_\\infty.

        - 'l1l2+l1', which is the multi-task group Lasso regularization + l1

        .. math::
            \\psi(W) = \\sum_{j=1}^p \\lambda_1 \\|W^j\\|_2 + \\lambda_2 \|W^j\|_1 ~~~~
            \\text{where}~W^j~\\text{is the j-th row of}~W.


    fit_intercept: boolean, default='False'
      learns an unregularized intercept b, which is a k-dimensional vector
        
    """
    _estimator_type = "classifier"

    def __init__(self, loss='square', penalty='l2', fit_intercept=True, tol=1e-3, solver="auto",
                 random_state=0, max_iter=500, fista_restart=50, verbose=True, warm_start=False, multi_class="auto",
                 limited_memory_qning=20, lambda_1=0, lambda_2=0, lambda_3=0, duality_gap_interval=5, n_threads=-1, dual=None):
        super().__init__(loss=loss, penalty=penalty, fit_intercept=fit_intercept, tol=tol, solver=solver,
                         random_state=random_state, max_iter=max_iter, fista_restart=fista_restart,
                         verbose=verbose, warm_start=warm_start, limited_memory_qning=limited_memory_qning,
                         lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, duality_gap_interval=duality_gap_interval,
                         n_threads=n_threads, multi_class= multi_class, dual=dual)

    def fit(self, X, y, le_parameter=None):
        """Same as BinaryClassifier, but y should be a vector a n-dimensional
        vector of integers
        """
        X, y, le = check_input_fit(X, y, self)
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

        if nb_classes != 2 and (nb_classes != unique.shape[0] or
                not all(np.unique(y) == np.arange(nb_classes))):
            logger.info("Class y should be of the form")
            logger.info(np.arange(nb_classes))
            logger.info("but they are")
            logger.info(unique)
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
            min_value = min(y)
            if min_value != 0:
                y = y - min_value
            self._binary_problem = False

        return super().fit(
            X, y, le_parameter=self.le_)

    def predict(self, X):
        """Predicts the class label"""
        check_is_fitted(self)

        X = check_input_inference(X, self)

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

        X = check_input_inference(X, self)

        pred = np.squeeze(self.predict(X))
        return np.sum(np.squeeze(y) == pred) / pred.shape[0]

    def decision_function(self, X):
        check_is_fitted(self)

        X = check_input_inference(X, self)

        if self.fit_intercept:
            scores = safe_sparse_dot(
                X, self.coef_, dense_output=False) + self.intercept_
        else:
            scores = safe_sparse_dot(X, self.coef_, dense_output=False)

        output = None
        if len(self.classes_) == 2:
            output = scores.ravel()
        else:
            output = scores.ravel() if scores.shape[1] == 1 else scores

        return output

    def predict_proba(self, X):
        check_is_fitted(self)

        X = check_input_inference(X, self)

        decision = self.decision_function(X)
        if decision.ndim == 1:
            # Workaround for binary outcomes
            # which requires softmax prediction with only a 1D decision.
            decision = np.c_[-decision, decision]
        return softmax(decision, copy=False)


class SKLearnClassifier(ERM):
    _estimator_type = "classifier"

    def __init__(self, loss, penalty, fit_intercept=True,
                 verbose=False, lambda_1=0, lambda_2=0, lambda_3=0,
                 solver='auto', tol=1e-3, duality_gap_interval=10, max_iter=None, limited_memory_qning=20,
                 fista_restart=50, warm_start=False, n_threads=-1, random_state=0 ,multi_class="auto", dual=None):
        super().__init__(
            loss=loss, penalty=penalty, fit_intercept=fit_intercept,
            solver=solver, tol=tol, random_state=random_state, verbose=verbose,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, multi_class= multi_class,
            duality_gap_interval=duality_gap_interval, max_iter=max_iter, limited_memory_qning=limited_memory_qning,
            fista_restart=fista_restart, warm_start=warm_start, n_threads=n_threads, dual=dual)

    def fit(self, X, y, le_parameter=None):
        """Compatible with both binary and multi-classification. Here the parameter C replaces lambda_1,
        and max_iter replaces max_iter.
        """
        X, y, le = check_input_fit(X, y, self)
        if le_parameter is not None:
            self.le_ = le_parameter
        else:
            self.le_ = le

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
                y = y.astype(int)
                y[neg] = -1
                y[np.logical_not(neg)] = 1
        else:
            min_value = min(y)
            if min_value != 0:
                y = y - min_value
            self._binary_problem = False
        super().fit(X, y, le_parameter=self.le_)

        self.coef_ = self.coef_.reshape(self.coef_.shape[0], -1)
        if self.fit_intercept:
            self.intercept_ = self.intercept_.reshape(1, -1)
        else:
            self.intercept_ = 0.

        self.n_features_in_ = self.coef_.shape[0]

        return self

    def decision_function(self, X):
        check_is_fitted(self)

        X = check_input_inference(X, self)

        scores = safe_sparse_dot(
            X, self.coef_, dense_output=False) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        check_is_fitted(self)

        X = check_input_inference(X, self)

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

        X = check_input_inference(X, self)

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
    initialization. Note that :math:`C= \\frac{1}{2n \\lambda_1}`

    Parameters
    ----------

    loss: should be 'sqhinge' or 'squared_hinge'

    penalty: same as BinaryClassifier

    fit_intercept: same as BinaryClassifier

    C: regularization parameter

    max_iter: maximum number of iterations for the optimization solver
    """

    def __init__(self, loss='sqhinge', penalty='l2', fit_intercept=True,
                 verbose=False, lambda_1=0.1, lambda_2=0, lambda_3=0,
                 solver='auto', tol=1e-3, duality_gap_interval=10, max_iter=500, limited_memory_qning=20,
                 fista_restart=50, warm_start=False, n_threads=-1, random_state=0, dual=None):
        if loss not in ['squared_hinge', 'sqhinge']:
            logger.error("LinearSVC is only compatible with squared hinge loss at "
                         "the moment")
        super().__init__(
            loss=loss, penalty=penalty, fit_intercept=fit_intercept,
            solver=solver, tol=tol, random_state=random_state, verbose=verbose,
            lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3,
            duality_gap_interval=duality_gap_interval, max_iter=max_iter,
            limited_memory_qning=limited_memory_qning,
            fista_restart=fista_restart, warm_start=warm_start, n_threads=n_threads, dual=dual)
        self.lambda_1 = lambda_1
        self.max_iter = max_iter
        self.verbose = False


class LogisticRegression(SKLearnClassifier):
    """
    A compatibility class for scikit-learn user, but only for square hinge loss
    It is perfectly equivalent to the BinaryClassifier class, but the
    regularization parameter (here "C") is provided during the class
    initialization. Note that :math:`C= \\frac{1}{n \\lambda_1}`

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
                 verbose=False, lambda_1=0, lambda_2=0, lambda_3=0,
                 solver='auto', tol=1e-3, duality_gap_interval=10, max_iter=500, limited_memory_qning=20,
                 fista_restart=50, warm_start=False, n_threads=-1, random_state=0, multi_class="auto", dual=None):
        super().__init__(loss=loss, penalty=penalty, fit_intercept=fit_intercept,
                         solver=solver, tol=tol, random_state=random_state, verbose=verbose,
                         lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning, multi_class= multi_class,
                         fista_restart=fista_restart, warm_start=warm_start, n_threads=n_threads, dual=dual)


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
    estimator.coef_ = np.zeros(p, dtype=X.dtype)
    if estimator.fit_intercept:
        estimator.intercept_ = 0

    for ii in range(num_as):
        R = compute_r(estimator.__name__, aux, X, y, active_set, n_active)

        corr = np.abs(X.transpose().dot(R).ravel()) / n
        if n_active > 0:
            corr[active_set] = -10e10
        n_new_as = max(
            min(init * math.ceil(scaling ** ii), p) - n_active, 0)
        new_as = corr.argsort()[-n_new_as:]
        if len(new_as) == 0 or max(corr[new_as]) <= estimator.lambda_1 * (1 + estimator.tol):
            break
        if len(active_set) > 0:
            neww = np.zeros(n_active + n_new_as,
                            dtype=X.dtype)
            neww[0:n_active] = aux.coef_
            aux.coef_ = neww
            active_set = np.concatenate((active_set, new_as))
        else:
            active_set = new_as
            aux.coef_ = np.zeros(
                len(active_set), dtype=X.dtype)
        n_active = len(active_set)
        if estimator.verbose:
            logger.info("Size of the active set: {%d}", n_active)
        aux.fit(X[:, active_set], y)
        estimator.coef_[active_set] = aux.coef_
        if estimator.fit_intercept:
            estimator.intercept_ = aux.intercept_


class Lasso(Regression):
    def __init__(self, lambda_1=0, solver='auto', tol=1e-3,
                 duality_gap_interval=10, max_iter=500, limited_memory_qning=20, fista_restart=50, verbose=True,
                 warm_start=False, n_threads=-1, random_state=0, fit_intercept=True, dual=None):
        super().__init__(loss='square', penalty='l1', lambda_1=lambda_1, solver=solver, tol=tol,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning, fista_restart=fista_restart,
                         verbose=verbose, warm_start=warm_start, n_threads=n_threads,
                         random_state=random_state, fit_intercept=fit_intercept, dual=dual)

    def fit(self, X, y):

        X, y, _ = check_input_fit(X, y, self)

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
        return {"requires_y": True,  "_xfail_checks": {
                "check_non_transformer_estimators_n_iter": (
                    "We have a different implementation of _n_iter in the multinomial case."
                ),
            }}

    def __init__(self, lambda_1=0, solver='auto', tol=1e-3,
                 duality_gap_interval=10, max_iter=500, limited_memory_qning=20, fista_restart=50, verbose=True,
                 warm_start=False, n_threads=-1, random_state=0, fit_intercept=True, multi_class="auto", dual=None):
        super().__init__(loss='logistic', penalty='l1', lambda_1=lambda_1, solver=solver, tol=tol,
                         duality_gap_interval=duality_gap_interval, max_iter=max_iter,
                         limited_memory_qning=limited_memory_qning,
                         fista_restart=fista_restart, verbose=verbose,
                         warm_start=warm_start, n_threads=n_threads, random_state=random_state,
                         fit_intercept=fit_intercept, multi_class= multi_class, dual=dual)

        if multi_class == "multinomial":
            self.loss = "multiclass-logistic"

    def fit(self, X, y):

        X, y, le = check_input_fit(X, y, self)
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
