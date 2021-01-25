# Author: Julien Mairal <julien.mairal@inria.fr>
#
# License: BSD 3 clause

from abc import abstractmethod
import numpy as np
import scipy.sparse
import cyanure_wrap
import math


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
    return cyanure_wrap.preprocess_(Xf, centering, normalize, not columns)


class ERM:
    """The generic class for empirical risk minimization problems.
    For univariates problems, minimizes

        min_{w,b} (1/n) sum_{i=1}^n L( y_i, <w, x_i> + b)   + psi(w)

    """

    def __init__(self, loss='square', penalty='l2', fit_intercept=False):
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
        if (loss == 'squared_hinge'):
            self.loss = 'sqhinge'
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.w = 0
        self.b = 0
        self.dual_variable = None

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, univariate=True, nthreads=-1, seed=0):
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

        if y.ndim == 0:
            print("Please provide label vector")
            return
        Xf = X.T if scipy.sparse.issparse(X) else np.asfortranarray(X.T)
        p = X.shape[1]+1 if self.fit_intercept else X.shape[1]
        y = np.squeeze(y)
        if y.shape[0] != X.shape[0]:
            print("X and y should have the same number of observations")
            return
        if univariate:
            w0 = np.zeros(p, dtype=Xf.dtype)
            yf = np.squeeze(y.astype(Xf.dtype))
        else:
            if y.squeeze().ndim > 1:
                nclasses = y.squeeze().shape[1]
                yf = np.asfortranarray(y.T)
            else:
                nclasses = int(np.max(y)+1)
                yf = np.squeeze(np.int32(y))
            w0 = np.zeros([p, nclasses], dtype=Xf.dtype, order='F')

        if restart and np.any(self.w != 0):
            if verbose:
                print("Restart")
            if self.fit_intercept:
                w0[-1, ] = self.b
                w0[0:-1, ] = self.w
            else:
                w0 = self.w

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
        optim_info = cyanure_wrap.erm_(
            Xf, yf, w0, w, dual_variable=self.dual_variable, loss=self.loss,
            penalty=self.penalty, solver=solver, lambd=float(lambd),
            lambd2=float(lambd2), lambd3=float(lambd3),
            intercept=bool(self.fit_intercept), tol=float(tol), it0=int(it0),
            nepochs=int(max_epochs), l_qning=int(l_qning),
            f_restart=int(f_restart), verbose=bool(verbose),
            univariate=bool(univariate), nthreads=int(nthreads), seed=int(seed)
        )
        if self.fit_intercept:
            self.b = w[-1, ]
            self.w = w[0:-1, ]
        else:
            self.w = w
        return optim_info

    @abstractmethod
    def predict(self, X):
        """
        predict the labels given an input matrix X (same format as fit)
        """
        pass

    def get_weights(self):
        """
        get the model parameters (either w or the tuple (w,b))
        """
        return (self.w, self.b) if self.fit_intercept else self.w

    def eval(self, X, y, lambd=0, lambd2=0, lambd3=0):
        """
        get the value of the objective function and computes a relative
        duality gap, see function fit for the format of parameters.
        """
        return self.fit(X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                        max_epochs=0, verbose=False, restart=True)


class BinaryClassifier(ERM):
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

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, seed=0):
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
        if nb_classes != 2:
            raise ValueError("{} classes detected; use MulticlassClassifier instead".format(nb_classes))
        
        if not np.all(uniq == [-1, 1]):
            print("You have called BinaryClassifier, labels should be either "
                  "-1 or 1, but they are")
            print(np.unique(y))
            print("Automatic conversion to [-1,1]")
            neg = y == uniq[0]
            y[neg] = -1
            y[np.logical_not(neg)] = 1
        return super().fit(X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                           solver=solver, tol=tol, it0=it0,
                           max_epochs=max_epochs, l_qning=l_qning,
                           f_restart=f_restart, verbose=verbose,
                           restart=restart, univariate=True, nthreads=nthreads,
                           seed=seed)

    def predict(self, X):
        pred = X.dot(self.w)
        if self.fit_intercept:
            pred += self.b
        return np.sign(pred)

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
        return np.sum(np.squeeze(y) == pred)/pred.shape[0]


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

    def __init__(self, loss='square', penalty='l2', fit_intercept=False):
        if loss != 'square':
            print("square loss should be used")
            return
        super().__init__(loss=loss, penalty=penalty,
                         fit_intercept=fit_intercept)

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, seed=0):
        """
        The fitting function is the same as for the class BinaryClassifier,
        except that we do not necessarily expect binary labels in y.
        """

        return super().fit(X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                           solver=solver, tol=tol, it0=it0,
                           max_epochs=max_epochs, l_qning=l_qning,
                           f_restart=f_restart, verbose=verbose,
                           restart=restart, univariate=True, nthreads=nthreads,
                           seed=seed)

    def predict(self, X):
        pred = X.dot(self.w)
        if self.fit_intercept:
            pred += self.b
        return pred


class MultiClassifier(ERM):
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

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, seed=0):
        """Same as BinaryClassifier, but y should be a vector a n-dimensional
        vector of integers
        """
        y = np.squeeze(y)
        if y.squeeze().ndim != 1 or np.any(y != y.astype(int)):
            print("y should be a n-dimensional vector of integers")
            return
        nclasses = np.max(y)+1
        uniqu = np.unique(y)
        if nclasses == 2:
            print("Two classes detected, use BinaryClassifier instead")
            return
        if (nclasses != uniqu.shape[0] or
                not all(np.unique(y) == np.arange(nclasses))):
            print("Class labels should be of the form")
            print(np.arange(nclasses))
            print("but they are")
            print(uniqu)
            return
        return super().fit(
            X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3, solver=solver,
            tol=tol, it0=it0, max_epochs=max_epochs, l_qning=l_qning,
            f_restart=f_restart, verbose=verbose, restart=restart,
            univariate=False, nthreads=nthreads, seed=seed)

    def predict(self, X):
        """Predicts the class label"""
        pred = X.dot(self.w)
        if self.fit_intercept:
            pred += self.b[np.newaxis, :]
        return np.argmax(pred, 1)

    def score(self, X, y):
        """Gives a classification score on new test data"""
        pred = np.squeeze(self.predict(X))
        return np.sum(np.squeeze(y) == pred)/pred.shape[0]


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
            print("square loss should be used")
            return
        super().__init__(loss=loss, penalty=penalty,
                         fit_intercept=fit_intercept)

    def fit(self, X, y, lambd=0, lambd2=0, lambd3=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, seed=0):
        """Same as ERM.fit, but y should be n x k, where k is size of the
        target for each data point
        """
        if y.squeeze().ndim <= 1:
            print("y should be n x k, where k is size of the target "
                  "for each data point")
            return
        return super().fit(
            X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3, solver=solver,
            tol=tol, it0=it0, max_epochs=max_epochs, l_qning=l_qning,
            f_restart=f_restart, verbose=verbose, restart=restart,
            univariate=False, nthreads=nthreads, seed=seed)

    def predict(self, X):
        """Predicts the targets"""
        pred = X.dot(self.w)
        if self.fit_intercept:
            pred += self.b[np.newaxis, :]
        return pred

# from sklearn.base import BaseEstimator
class SKLearnClassifier(ERM):
    def fit(self, X, y, C=None, verbose=False, lambd2=0, lambd3=0,
            solver='auto', tol=1e-3, it0=10, max_iter=None, l_qning=20,
            f_restart=50, restart=False, nthreads=-1, seed=0):
        """Compatible with both binary and multi-classification. Here the parameter C replaces lambd,
        and max_iter replaces max_epochs.
        """
        n = X.shape[0]
        if C is not None:
            self.C = C
        if verbose is not None:
            self.verbose = verbose
        if max_iter is not None:
            self.max_iter = max_iter
        self.classes_ = np.unique(y)
        nb_classes = len(self.classes_)
        if self.loss == 'sqhinge':
            lambd = 1. / (2. * n * self.C)
        elif self.loss == 'logistic':
            lambd = 1. / (n * self.C)
        else:
            lambd = 1. / (2. * n * self.C)
        if nb_classes == 2:
            univariate = True
            if not np.all(self.classes_ == [-1, 1]):
                neg = y == self.classes_[0]
                y = y.copy()
                y[neg] = -1
                y[np.logical_not(neg)] = 1
        else:
            univariate = False
        optim_info = super().fit(X, y, lambd=lambd, lambd2=lambd2, lambd3=lambd3,
                    solver=solver, tol=tol, it0=it0,
                    max_epochs=self.max_iter, l_qning=l_qning,
                    f_restart=f_restart, verbose=verbose,
                    restart=restart, univariate=univariate, nthreads=nthreads,
                    seed=seed)
        self.w = self.w.reshape(self.w.shape[0], -1)
        if self.fit_intercept:
            self.b = self.b.reshape(1, -1)
        else:
            self.b = 0.
        return optim_info

    def decision_function(self, X):
        n_features = self.w.shape[0]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        scores = X.dot(self.w) + self.b
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def score(self, X, y, sample_weight=None):
        return np.average(y == self.predict(X), weights=sample_weight)

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

    def __init__(self, penalty='l2', fit_intercept=True, C=1, max_iter=500):
        super(SKLearnClassifier, self).__init__(
            loss='logistic', penalty=penalty,
            fit_intercept=fit_intercept)
        self.C = C
        self.max_iter = max_iter


class Lasso(Regression):
    def __init__(self, fit_intercept=False):
        super().__init__(loss='square', penalty='l1',
                         fit_intercept=fit_intercept)
        self.aux = Regression(loss='square',penalty='l1',fit_intercept=fit_intercept)

    def fit(self, X, y, lambd=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, seed=0):
        n,p = X.shape
        if p <= 1000:
            # no active set
            super().fit(X,y,lambd=lambd,solver=solver,tol=tol,it0=it0,max_epochs=max_epochs,l_qning=l_qning,f_restart=f_restart,verbose=verbose,restart=restart,nthreads=nthreads,seed=seed)
        else:
            scaling = 4.0
            init = min(100,p)
            restart=True
            num_as = math.ceil(math.log10(p/init)/math.log10(scaling))
            active_set = []
            n_active = 0
            self.w = np.zeros(p, dtype=X.dtype)
            if self.fit_intercept:
                self.b=0

            for ii in range(num_as):
                if n_active == 0:
                    R = y
                else:
                    pred = self.aux.predict(X[:,active_set])
                    R = y.ravel() - pred.ravel()
                corr = np.abs(X.transpose().dot(R).ravel())/n
                if n_active > 0:
                    corr[active_set] = -10e10
                n_new_as = max(min(init*math.ceil(scaling ** (ii)),p) - n_active,0)
                new_as = corr.argsort()[-n_new_as:]
                if (len(new_as) == 0 or max(corr[new_as]) <= lambd*(1+tol)):
                    break;
                if len(active_set) > 0:
                    neww = np.zeros(n_active+n_new_as,dtype=X.dtype)
                    neww[0:n_active]=self.aux.w
                    self.aux.w=neww
                    active_set = np.concatenate((active_set, new_as))
                else:
                    active_set = new_as
                    self.aux.w = np.zeros(len(active_set),dtype=X.dtype)
                n_active = len(active_set)
                if (verbose):
                    print("Size of the active set: " + str(n_active))
                self.aux.fit(X[:,active_set],y,lambd,tol=tol,it0=5,max_epochs=max_epochs,restart=restart,solver=solver,verbose=verbose)
                self.w[active_set] = self.aux.w
                if self.fit_intercept:
                    self.b=self.aux.b
    
#TODO: remove code duplication with Lasso
class L1Logistic(BinaryClassifier):
    def __init__(self, fit_intercept=False):
        super().__init__(loss='logistic', penalty='l1',
                         fit_intercept=fit_intercept)
        self.aux = BinaryClassifier(loss='logistic',penalty='l1',fit_intercept=fit_intercept)

    def fit(self, X, y, lambd=0, solver='auto', tol=1e-3,
            it0=10, max_epochs=500, l_qning=20, f_restart=50, verbose=True,
            restart=False, nthreads=-1, seed=0):
        n,p = X.shape
        if p <= 1000:
            # no active set
            super().fit(X,y,lambd=lambd,solver=solver,tol=tol,it0=it0,max_epochs=max_epochs,l_qning=l_qning,f_restart=f_restart,verbose=verbose,restart=restart,nthreads=nthreads,seed=seed)
        else:
            scaling = 4.0
            init = min(100,p)
            restart=True
            num_as = math.ceil(math.log10(p/init)/math.log10(scaling))
            active_set = []
            n_active = 0
            self.w = np.zeros(p, dtype=X.dtype)
            if self.fit_intercept:
                self.b=0

            for ii in range(num_as):
                #  log(1 + exp(-y_i pred_i))
                # abs_grad =    sum - y_i/(1 + exp(y_i pred_i)) x_i
                if n_active == 0:
                    R = -0.5* y.ravel()
                else:
                    pred = X[:,active_set].dot(self.aux.w)
                    if self.fit_intercept:
                        pred += self.aux.b
                    R = -y.ravel() / (1.0 + np.exp( y.ravel() * pred.ravel()))
                corr = np.abs(X.transpose().dot(R).ravel())/X.shape[0]
                if n_active > 0:
                    corr[active_set] = -10e10
                n_new_as = max(min(init*math.ceil(scaling ** (ii)),p) - n_active,0)
                new_as = corr.argsort()[-n_new_as:]
                if (len(new_as) == 0 or max(corr[new_as]) <= lambd*(1+tol)):
                    break;
                if len(active_set) > 0:
                    neww = np.zeros(n_active+n_new_as,dtype=X.dtype)
                    neww[0:n_active]=self.aux.w
                    self.aux.w=neww
                    active_set = np.concatenate((active_set, new_as))
                else:
                    active_set = new_as
                    self.aux.w = np.zeros(len(active_set),dtype=X.dtype)
                n_active = len(active_set)
                if (verbose):
                    print("Size of the active set: " + str(n_active))
                self.aux.fit(X[:,active_set],y,lambd,tol=tol,it0=5,max_epochs=max_epochs,restart=restart,solver=solver,verbose=verbose)
                self.w[active_set] = self.aux.w
                if self.fit_intercept:
                    self.b=self.aux.b
 


