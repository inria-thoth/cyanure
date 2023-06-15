#ifndef INCREMENTAL_SOLVER_H
#define INCREMENTAL_SOLVER_H

#include "solver.h"


#define USING_INCREMENTAL_SOLVER                               \
    USING_SOLVER;                                              \
    using IncrementalSolver<loss_type>::_non_uniform_sampling; \
    using IncrementalSolver<loss_type>::_n;                    \
    using IncrementalSolver<loss_type>::_qi;

template <typename loss_type>
class IncrementalSolver : public Solver<loss_type>
{
public:
    USING_SOLVER;
    IncrementalSolver(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param, const Vector<FeatureType>* Li = NULL) : Solver<loss_type>(loss, regul, param)
    {
        _non_uniform_sampling = param.non_uniform_sampling;
        if (Li)
            _Li.copy(*Li);
    };

protected:
    virtual void solver_init(const D& x0)
    {
        if (_Li.n() == 0)
            _loss.lipschitz(_Li);
        _n = _Li.n();
        if (_L == 0)
        {
            _qi.copy(_Li);
            _qi.scal(FeatureType(1.0) / _qi.sum());
            const FeatureType Lmean = _Li.mean();
            const FeatureType Lmax = _Li.maxval();
            _non_uniform_sampling = (_non_uniform_sampling && Lmean <= FeatureType(0.9) * Lmax);
            _L = _non_uniform_sampling ? Lmean : Lmax;
            if (Solver<loss_type>::_minibatch > 1)
                heuristic_L(x0);
            _oldL = _L;
            if (_non_uniform_sampling)
                init_nonu_sampling();
        }
        this->check_mkl(x0);
    };
    
    void print() const
    {
        logging(logINFO) << "Incremental Solver ";
        if (_non_uniform_sampling)
        {
            logging(logINFO) << "with non uniform sampling";
        }
        else
        {
            logging(logINFO) << "with uniform sampling";
        }
        logging(logINFO) << "Lipschitz constant: " << _L;
    };

    bool _non_uniform_sampling;
    
    int _n;
    Vector<FeatureType> _qi;
    Vector<double> _Ui;
    Vector<int> _Ki;
    FeatureType _oldL;

    void init_nonu_sampling()
    {
        _Ui.resize(_n);
        for (int ii = 0; ii < _n; ++ii)
            _Ui[ii] = static_cast<double>(_qi[ii]);
        _Ui.scal(_n / _Ui.asum());
        _Ki.resize(_n);
        _Ki.set(0);
        List<int> overfull;
        List<int> underfull;
        for (int ii = 0; ii < _n; ++ii)
        {
            if (_Ui[ii] < double(1.0))
            {
                underfull.push_back(ii);
            }
            else if (_Ui[ii] > double(1.0))
            {
                overfull.push_back(ii);
            }
        }
        while (underfull.size() > 0 && overfull.size() > 0)
        {
            const int indj = underfull.front();
            underfull.pop_front();
            const int indi = overfull.front();
            overfull.pop_front();
            _Ki[indj] = indi;
            _Ui[indi] = _Ui[indi] + _Ui[indj] - double(1.0);
            if (_Ui[indi] < double(1.0))
            {
                underfull.push_back(indi);
            }
            else if (_Ui[indi] > double(1.0))
            {
                overfull.push_back(indi);
            }
        }
    };
    
    int nonu_sampling()
    {
        const double x = static_cast<double>(random() - 1) / INT_MAX;
        const int ind = static_cast<int>(floor(_n * x)) + 1;
        const double y = _n * x + 1 - ind;
        if (y < _Ui[ind - 1])
            return ind - 1;
        return _Ki[ind - 1];
    };
    
    virtual int minibatch() const { 
        return Solver<loss_type>::_minibatch; 
    };
    
    FeatureType init_kappa_acceleration(const D& x0)
    {
        IncrementalSolver<loss_type>::solver_init(x0);
        const FeatureType mu = _regul.strong_convexity();
        return ((this->_oldL / (_n)-mu));
    };

    void check_mkl(const Vector<FeatureType>& x0) const {};
    void check_mkl(const Matrix<FeatureType>& x0) const
    {
        if (x0.m() <= 15 || x0.n() <= 15)
        {
            set_mkl_sequential(); // TODO should be local --> demander des précisions à Julien
        }
    };

private:
    void heuristic_L(const D& x)
    {
        if (_verbose) {
            logging(logINFO) << "Heuristic: Initial L=" << _L;
        }
        const FeatureType Lmax = _L;
        _L /= Solver<loss_type>::_minibatch;
        int iter = 0;
        D tmp, tmp2, grad;
        while (iter <= log(Solver<loss_type>::_minibatch) / log(2.0) && _L < Lmax)
        {
            tmp.copy(x);
            const FeatureType fx = _loss.eval_random_minibatch(tmp, Solver<loss_type>::_minibatch);
            _loss.grad_random_minibatch(tmp, grad, Solver<loss_type>::_minibatch); // should do non uniform
            // compute grad and fx
            tmp.add(grad, -FeatureType(1.0) / _L);
            const FeatureType ftmp = _loss.eval_random_minibatch(tmp, Solver<loss_type>::_minibatch);
            tmp2.copy(tmp);
            tmp2.sub(x);
            const FeatureType s1 = fx + grad.dot(tmp2);
            const FeatureType s2 = FeatureType(0.5) * tmp2.nrm2sq();
            if (ftmp > s1 + _L * s2)
                _L = MIN(MAX(2.0 * _L, (ftmp - s1) / s2), Lmax);
            ++iter;
        }
        if (_verbose) {
            logging(logINFO) << ", Final L=" << _L;
        }
    }
};

#endif