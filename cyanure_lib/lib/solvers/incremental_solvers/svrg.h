#ifndef SVRG_H
#define SVRG_H

#include "../incremental_solver.h"

#define USING_SVRG_SOLVER                  \
    USING_INCREMENTAL_SOLVER;              \
    using SVRG_Solver<loss_type>::_xtilde; \
    using SVRG_Solver<loss_type>::_gtilde;

#define USING_ACC_SVRG_SOLVER                                         \
    using Acc_SVRG_Solver<loss_type, allow_acc>::_y;                  \
    using Acc_SVRG_Solver<loss_type, allow_acc>::_etak;               \
    using Acc_SVRG_Solver<loss_type, allow_acc>::_gammak;             \
    using Acc_SVRG_Solver<loss_type, allow_acc>::_mu;                 \
    using Acc_SVRG_Solver<loss_type, allow_acc>::_deltak;             \
    using Acc_SVRG_Solver<loss_type, allow_acc>::_thetak;             \
    using Acc_SVRG_Solver<loss_type, allow_acc>::_accelerated_solver; \
    USING_SVRG_SOLVER;

template <typename loss_type>
class SVRG_Solver : public IncrementalSolver<loss_type>
{
public:
    USING_INCREMENTAL_SOLVER;
    SVRG_Solver(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param, const Vector<FeatureType>* Li = NULL) : IncrementalSolver<loss_type>(loss, regul, param, Li) {
    };

protected:
    virtual void solver_init(const D& x0)
    {
        // Rename x0 and x with w0 and w
        IncrementalSolver<loss_type>::solver_init(x0);
        _xtilde.copy(x0);
        _loss.grad(_xtilde, _gtilde);
    };

    virtual void solver_aux(D& x)
    {
        const int nn = _n / _minibatch;
        const FeatureType eta = FeatureType(1.0) / (3 * _L);
        D tmp;
        for (int ii = 0; ii < nn; ++ii)
        {
            tmp.copy(x);
            tmp.add(_gtilde, -eta);
            for (int jj = 0; jj < _minibatch; ++jj)
            {
                const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
                const FeatureType scal = _non_uniform_sampling ? FeatureType(1.0) / (_minibatch * _qi[ind] * _n) : FeatureType(1.0) / _minibatch;
                _loss.double_add_grad(x, _xtilde, ind, tmp, -scal * eta, scal * eta, jj == 0 ? FeatureType(_minibatch) : 0);
            }
            _regul.prox(tmp, x, eta);
            if (random() % nn == 0)
            {
                _xtilde.copy(x);
                _loss.grad(_xtilde, _gtilde);
            }
        }
    };
    
    void print() const
    {
        logging(logINFO) << "SVRG Solver";
        IncrementalSolver<loss_type>::print();
    };
    D _xtilde, _gtilde;
};


template <typename loss_type, bool allow_acc = true>
class Acc_SVRG_Solver : public SVRG_Solver<loss_type>
{
public:
    USING_SVRG_SOLVER;
    Acc_SVRG_Solver(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param, const Vector<FeatureType>* Li = NULL) : SVRG_Solver<loss_type>(loss, regul, param, Li)
    {
        _accelerated_solver = allow_acc;
        printf("%d \n", param.minibatch);
    };

    virtual void solver_init(const D& x0)
    {
        IncrementalSolver<loss_type>::solver_init(x0);
        _mu = _regul.strong_convexity();
        printf("%d \n", _minibatch);
        _nn = _n / _minibatch;
        _accelerated_solver = allow_acc && (FeatureType(20) * this->_oldL / _nn > _mu);
        if (_accelerated_solver)
        {
            _gammak = MAX(_L / (_nn), _mu);
            update_acceleration_parameters();
            _xtilde.copy(x0);
            _y.copy(x0);
            _loss.grad(_xtilde, _gtilde);
        }
        else
        {
            if (allow_acc) {
                logging(logWARNING) << "Problem is well conditioned, switching to regular solver";
            }
            SVRG_Solver<loss_type>::solver_init(x0);
        }
    };

    virtual void solver_aux(D& x)
    {
        if (_accelerated_solver)
        {
            for (int ii = 0; ii < _nn; ++ii)
            {
                
                x.copy(_y);
                x.add(_gtilde, -_etak);
                for (int jj = 0; jj < _minibatch; ++jj)
                {
                    const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
                    const FeatureType scal = _non_uniform_sampling ? FeatureType(1.0) / (_qi[ind] * _n * _minibatch) : FeatureType(1.0) / _minibatch;
                    _loss.double_add_grad(_y, _xtilde, ind, x, -scal * _etak, scal * _etak);
                }
                _regul.prox(x, x, _etak);

                const FeatureType alphak = _mu * _deltak / _gammak;
                const FeatureType betak = _deltak / (_gammak * _etak);
                const FeatureType a = (FeatureType(1.0) - alphak) / _thetak + alphak;
                update_acceleration_parameters();
                if (random() % _nn == 0)
                {
                    _y.add_scal(_xtilde, (FeatureType(1.0) - a) * _thetak, _thetak * (a - betak));
                    _y.add(x, betak * _thetak + FeatureType(1.0) - _thetak);
                    _xtilde.copy(x);
                    _loss.grad(_xtilde, _gtilde);
                }
                else
                {
                    _y.add_scal(_xtilde, FeatureType(1.0) - _thetak * a, _thetak * (a - betak));
                    _y.add(x, betak * _thetak);
                }
            };
        }
        else
        {
            SVRG_Solver<loss_type>::solver_aux(x);
        }
    };

protected:
    void print() const
    {
        logging(logINFO) << "Accelerated SVRG Solver";
        if (!_accelerated_solver) {
            logging(logWARNING) << "Problem is well conditioned, switching to regular solver";
        }
        IncrementalSolver<loss_type>::print();
    };

    bool _accelerated_solver;
    FeatureType _gammak, _mu, _deltak, _etak, _thetak;
    D _y;
    int _nn;

    void update_acceleration_parameters()
    {
        _deltak = MIN(solve_binomial(FeatureType(9.0) * _nn * _L / FeatureType(5.0), _gammak - _mu, -_gammak), FeatureType(1.0) / (3 * _nn));
        _gammak = (FeatureType(1.0) - _deltak) * _gammak + _mu * _deltak;
        _etak = MIN(FeatureType(1.0) / (3 * _L), FeatureType(1.0) / (15 * _gammak * _nn));
        _thetak = (3 * _nn * _deltak - 5 * _mu * _etak) / (3 - 5 * _mu * _etak);
    };
};

template <typename loss_type, bool allow_acc = true>
class SVRG_Solver_FastRidge : public Acc_SVRG_Solver<loss_type, allow_acc>
{
public:
    USING_ACC_SVRG_SOLVER;

    SVRG_Solver_FastRidge(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param, const Vector<FeatureType>* Li = NULL) : Acc_SVRG_Solver<loss_type, allow_acc>(loss, regul, param, Li),
        _is_lazy(loss_type::is_sparse())
    {
        if (param.minibatch > 1)
        {
            logging(logWARNING) << "Minibatch is not compatible with lazy updates. The minibatch parameter has been set to 1.";
        }
        _minibatch = 1;
    };
    virtual void solver_init(const D& x0)
    {
        Acc_SVRG_Solver<loss_type, allow_acc>::solver_init(x0);
        if (_loss.id() == PPA)
        {
            const FeatureType kappa = _loss.kappa();
            _gtilde.add(_xtilde, -kappa); // now gtilde has the right value
        }
    };

    /// define auxiliary solver ?
    virtual void solver_aux(D& x)
    {
        if (_accelerated_solver)
        {
            const FeatureType lambda_1 = _regul.lambda_1();
            DoubleLazyVector<FeatureType, PointerType>* lazyy = NULL;
            Vector<PointerType> indices;
            if (_is_lazy)
            {
                lazyy = new DoubleLazyVector<FeatureType, PointerType>(_y, _xtilde, _gtilde, _n);
            }
            for (int ii = 0; ii < _n; ++ii)
            {
                const FeatureType alphak = _mu * _deltak / _gammak;
                const FeatureType betak = _deltak / (_gammak * _etak);
                const FeatureType a = (FeatureType(1.0) - alphak) / _thetak + alphak;
                const FeatureType scalprox = FeatureType(1.0) / (FeatureType(1.0) + lambda_1 * _etak);
                const FeatureType eta = _etak;
                const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
                const FeatureType scaleta = _non_uniform_sampling ? eta / (_qi[ind] * _n) : eta;
                this->update_acceleration_parameters();
                const bool update_xtilde = random() % _n == 0;
                const FeatureType coeffy = _thetak * (a - betak);
                const FeatureType coeffx = update_xtilde ? betak * _thetak + FeatureType(1.0) - _thetak : betak * _thetak;
                const FeatureType coeffxtilde = update_xtilde ? (FeatureType(1.0) - a) * _thetak : FeatureType(1.0) - _thetak * a;

                if (update_xtilde || ii == _n - 1)
                {
                    if (_is_lazy)
                        lazyy->update();
                    x.copy(_y);
                    _loss.double_add_grad(_y, _xtilde, ind, x, -scaleta, scaleta);
                    x.add_scal(_gtilde, -scalprox * eta, scalprox);
                    _y.add_scal(_xtilde, coeffxtilde, coeffy);
                    _y.add(x, coeffx);
                }
                else
                {
                    const FeatureType coeff_add_grad = scaleta * scalprox * coeffx / (coeffy + scalprox * coeffx);
                    if (_is_lazy)
                    {
                        _loss.get_coordinates(ind, indices);
                        lazyy->update(indices);
                    }
                    _loss.double_add_grad(_y, _xtilde, ind, _y, -coeff_add_grad, coeff_add_grad);
                    if (_is_lazy)
                    {
                        lazyy->add_scal(coeffxtilde, -scalprox * eta * coeffx, coeffy + scalprox * coeffx);
                    }
                    else
                    {
                        _y.add_scal(_gtilde, -scalprox * eta * coeffx, coeffy + scalprox * coeffx);
                        _y.add(_xtilde, coeffxtilde);
                    }
                }
                if (update_xtilde)
                {
                    _xtilde.copy(x);
                    _loss.grad(_xtilde, _gtilde);
                }
            };
            if (_is_lazy)
                delete (lazyy);
        }
        else if (_loss.id() == PPA)
        {
            /// we will optimize implicitly  f(xtilde) - kappa <x , z> + (mu+kappa)/2|x|^2
            /// meaning, we want gtilde to be  Df(xtilde) - kappa z
            LazyVector<FeatureType, PointerType>* lazyx = NULL;
            Vector<PointerType> indices;
            if (_is_lazy)
            {
                // indices.resize(x.n());
                lazyx = new LazyVector<FeatureType, PointerType>(x, _gtilde, _n);
            }
            const FeatureType eta = FeatureType(1.0) / (3 * (_L - _loss.kappa()));
            const FeatureType lambda_1 = _regul.lambda_1() + _loss.kappa(); // take care of 0.5(mu+kappa)|x|^2,
            for (int ii = 0; ii < _n; ++ii)
            {
                const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
                const FeatureType scal = _non_uniform_sampling ? FeatureType(1.0) / (_qi[ind] * _n) : FeatureType(1.0);
                if (_is_lazy)
                {
                    _loss.get_coordinates(ind, indices);
                    lazyx->update(indices);
                    _loss.double_add_grad(x, _xtilde, ind, x, -scal * eta, scal * eta, 0);
                    lazyx->add_scal(-eta, FeatureType(1.0) / (FeatureType(1.0) + eta * lambda_1));
                }
                else
                {
                    _loss.double_add_grad(x, _xtilde, ind, x, -scal * eta, scal * eta, 0); // x <- x - s( D_i f(x) - D_i f(xtilde))
                    x.add_scal(_gtilde, -eta / (FeatureType(1.0) + eta * lambda_1), FeatureType(1.0) / (FeatureType(1.0) + eta * lambda_1));
                }

                if (random() % _n == 0)
                {
                    if (_is_lazy)
                        lazyx->update();
                    _xtilde.copy(x);
                    _loss.grad(_xtilde, _gtilde);         // gtilde will be equal to Df(xtilde) + kappa (xtilde- z)
                    _gtilde.add(_xtilde, -_loss.kappa()); // now gtilde has the right value
                }
            }
            if (_is_lazy)
            {
                lazyx->update();
                delete (lazyx);
            }
        }
        else
        {
            LazyVector<FeatureType, PointerType>* lazyx = NULL;
            Vector<PointerType> indices;
            if (_is_lazy)
            {
                // indices.resize(x.n());
                lazyx = new LazyVector<FeatureType, PointerType>(x, _gtilde, _n);
            }
            const FeatureType eta = FeatureType(1.0) / (3 * _L);
            const FeatureType lambda_1 = _regul.lambda_1(); // replace by lazyprox ?
            for (int ii = 0; ii < _n; ++ii)
            {
                const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
                const FeatureType scal = _non_uniform_sampling ? FeatureType(1.0) / (_qi[ind] * _n) : FeatureType(1.0);
                if (_is_lazy)
                {
                    _loss.get_coordinates(ind, indices);
                    lazyx->update(indices);
                    _loss.double_add_grad(x, _xtilde, ind, x, -scal * eta, scal * eta);
                    lazyx->add_scal(-eta, FeatureType(1.0) / (FeatureType(1.0) + eta * lambda_1)); // replace by lazyprox ?
                }
                else
                {
                    _loss.double_add_grad(x, _xtilde, ind, x, -scal * eta, scal * eta);
                    x.add_scal(_gtilde, -eta / (FeatureType(1.0) + eta * lambda_1), FeatureType(1.0) / (FeatureType(1.0) + eta * lambda_1));
                }
                if (random() % _n == 0)
                {
                    if (_is_lazy)
                        lazyx->update();
                    _xtilde.copy(x);
                    _loss.grad(_xtilde, _gtilde);
                }
            }
            if (_is_lazy)
            {
                lazyx->update();
                delete (lazyx);
            }
        }
    };

protected:
    void print() const
    {
        if (_accelerated_solver)
        {
            logging(logINFO) << "Accelerated SVRG Solver, ";
        }
        else
        {
            logging(logINFO) << "SVRG Solver, ";
        }
        if (_is_lazy)
        {
            logging(logINFO) << "specialized for sparse matrices and L2 regularization";
        }
        else
        {
            logging(logINFO) << "specialized for L2 regularization";
        }
        IncrementalSolver<loss_type>::print();
    };

private:
    const bool _is_lazy;
};

#endif