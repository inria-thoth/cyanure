#ifndef ISTA_H
#define ISTA_H

#include "solver.h"

template <typename loss_type>
class ISTA_Solver : public Solver<loss_type>
{
public:
    USING_SOLVER;
    ISTA_Solver(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param, const Vector<FeatureType>* Li = NULL) : Solver<loss_type>(loss, regul, param)
    {
        _L = 0;
        if (Li)
        {
            _Li.copy(*Li);
            _L = _Li.maxval() / 100;
        }
    };

protected:
    virtual void solver_init(const D& x0)
    {
        if (_L == 0)
        {
            _loss.lipschitz(_Li);
            _L = _Li.maxval() / 100;
        }
    };
    
    virtual void solver_aux(D& x)
    {
        int iter = 1;
        const FeatureType fx = _loss.eval(x);
        D grad, tmp, tmp2;
        _loss.grad(x, grad);
        while (iter < _max_iter_backtracking)
        {
            tmp2.copy(x);
            tmp2.add(grad, -FeatureType(1.0) / _L);
            _regul.prox(tmp2, tmp, FeatureType(1.0) / _L);
            const FeatureType fprox = _loss.eval(tmp);
            tmp2.copy(tmp);
            tmp2.sub(x);

            if (fprox <= fx + grad.dot(tmp2) + FeatureType(0.5) * _L * tmp2.nrm2sq() + EPSILON)
            {
                x.copy(tmp);
                break;
            }
            _L *= FeatureType(1.5);
            if (_verbose) {logging(logINFO) << "new value for L: " << _L;}
            ++iter;
            if (iter == _max_iter_backtracking) {logging(logINFO) << "Warning: maximum number of backtracking iterations has been reached";}
        }
    };
    
    void print() const
    {
        logging(logINFO) << "ISTA Solver";
    };
    
    FeatureType init_kappa_acceleration(const D& x0)
    {
        ISTA_Solver<loss_type>::solver_init(x0);
        return _L;
    };
};

template <typename loss_type>
class FISTA_Solver final : public ISTA_Solver<loss_type>
{
public:
    USING_SOLVER;
    FISTA_Solver(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param) : ISTA_Solver<loss_type>(loss, regul, param) {};

protected:
    virtual void solver_init(const D& x0)
    {
        ISTA_Solver<loss_type>::solver_init(x0);
        _t = FeatureType(1.0);
        _y.copy(x0);
    };
    
    virtual void solver_aux(D& x)
    {
        ISTA_Solver<loss_type>::solver_aux(_y);
        D diff;
        diff.copy(x);
        x.copy(_y);
        diff.sub(x);
        const FeatureType old_t = _t;
        _t = (1.0 + sqrt(1 + 4 * _t * _t)) / 2;
        _y.add(diff, (FeatureType(1.0) - old_t) / _t);
    };
    
    virtual void print() const
    {
        logging(logINFO) << "FISTA Solver";
    };

    FeatureType _t;
    D _y;
};


#endif