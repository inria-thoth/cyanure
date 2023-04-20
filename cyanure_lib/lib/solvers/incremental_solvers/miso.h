#ifndef MISO_H
#define MISO_H

#include "../incremental_solver.h"

template <typename loss_type>
class MISO_Solver : public IncrementalSolver<loss_type>
{
public:
    USING_INCREMENTAL_SOLVER;
    MISO_Solver(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param, const Vector<FeatureType>* Li = NULL) : IncrementalSolver<loss_type>(loss, regul, param, Li)
    {
        _minibatch = 1;
        _mu = _regul.id() == L2 || _regul.id() == ELASTICNET ? _regul.strong_convexity() : 0;
        _kappa = _loss.kappa();
        if (_loss.id() == PPA){
            _mu += _kappa;
        }
        _isprox = (_regul.id() != L2 || _regul.intercept()) && _regul.id() != NONE;
        _is_lazy = _isprox && _regul.is_lazy() && _loss.is_sparse();
        _extern_zis = false;
        _count = 0;
    };

    virtual void set_dual_variable(const D& dual0)
    {
        _extern_zis = true;
        _zis.copyRef(dual0);
    };

    virtual void solve(const D& y, D& x)
    {
        if (_count > 0 && (_count % 10) != 0)
        {
            D& ref_barz = _isprox ? _barz : x;
            ref_barz.add(_oldy, -_kappa / _mu); // necessary to have PPA loss here
            ref_barz.add(y, _kappa / _mu);
            const bool is_lazy = _isprox && _regul.is_lazy() && _loss.is_sparse();
            if (_isprox && !is_lazy)
                _regul.prox(ref_barz, x, FeatureType(1.0) / _mu);
        }
        else if (_count == 0)
        {
            x.copy(y); // just to have the right size
        }
        if (_loss.id() == PPA)
            _loss.get_anchor_point(_oldy);
        Solver<loss_type>::solve(x, x);
    };

    virtual void save_state()
    {
        _count2 = _count;
        _barz2.copy(_barz);
        _zis2.copy(_zis);
        _oldy2.copy(_oldy);
    };
    
    virtual void restore_state()
    {
        _count = _count2;
        _barz.copy(_barz2);
        _zis.copy(_zis2);
        _oldy.copy(_oldy2);
    };

protected:
    virtual void solver_init(const D& x0)
    {
        // initial point will be in fact _z of PPA
        if (_count == 0)
        {
            IncrementalSolver<loss_type>::solver_init(x0);
            _delta = MIN(FeatureType(1.0), _n * _mu / (2 * _L));
            if (_non_uniform_sampling)
            {
                const FeatureType beta = FeatureType(0.5) * _mu * _n;
                if (this->_Li.maxval() <= beta)
                {
                    _non_uniform_sampling = false;
                    _delta = FeatureType(1.0);
                }
                else if (this->_Li.minval() >= beta)
                {
                    // no change
                }
                else
                {
                    _qi.copy(this->_Li);
                    _qi.thrsmax(beta);
                    _qi.scal(FeatureType(1.0) / _qi.sum());
                    Vector<FeatureType> tmp;
                    tmp.copy(_qi);
                    tmp.inv();
                    tmp.mult(tmp, this->_Li);
                    _L = tmp.maxval() / _n;
                    this->init_nonu_sampling();
                    _delta = MIN(_n * _qi.minval(), _n * _mu / (2 * _L));
                }
            }
            if (_non_uniform_sampling)
                _delta = MIN(_delta, _n * _qi.minval());
            if (_isprox)
                _barz.copy(x0); // if PPA, x0 should be the anchor point and _barz = X*_Zis + x0
            init_dual_variables(x0);
        }
    };
    
    virtual void solver_aux(D& x)
    {
        D& ref_barz = _isprox ? _barz : x;
        if (_count++ % 10 == 0)
        {
            if (_loss.id() == PPA)
            {
                _loss.get_anchor_point(ref_barz);
                ref_barz.scal(_kappa / _mu);
            }
            else
            {
                ref_barz.setZeros();
            }
            if (_count > 1 || _extern_zis)
                _loss.add_feature(_zis, ref_barz, FeatureType(1.0) / (_n * _mu));
            if (_isprox && !_is_lazy)
                _regul.prox(ref_barz, x, FeatureType(1.0) / _mu);
        }
        Vector<typename loss_type::index_type> indices;
        for (int ii = 0; ii < _n; ++ii)
        {
            const int ind = _non_uniform_sampling ? this->nonu_sampling() : random_r() % _n;
            const FeatureType scal = _non_uniform_sampling ? FeatureType(1.0) / (_qi[ind] * _n) : FeatureType(1.0);
            const FeatureType deltas = scal * _delta;
            if (_is_lazy)
            {
                _loss.get_coordinates(ind, indices);
                _regul.lazy_prox(ref_barz, x, indices, FeatureType(1.0) / _mu);
            }
            solver_aux_aux(x, ref_barz, ind, deltas);

            if (_isprox && (!_is_lazy || ii == _n - 1))
                _regul.prox(ref_barz, x, FeatureType(1.0) / _mu);
        }
    };
    
    void print() const
    {
        logging(logINFO) << "MISO Solver";
        IncrementalSolver<loss_type>::print();
    };

private:
    D _zis, _zis2;
    D _barz, _barz2;
    D _oldy, _oldy2;
    FeatureType _mu;
    FeatureType _kappa;
    FeatureType _delta;
    int _count, _count2;
    bool _perform_update_barz;
    bool _isprox, _is_lazy, _extern_zis;

    void inline init_dual_variables(const Vector<FeatureType>& x0)
    {
        if (_zis.n() != _n)
        {
            _zis.resize(_n);
            _zis.setZeros();
        }
    }
    
    void inline init_dual_variables(const Matrix<FeatureType>& x0)
    {
        const int nclasses = _loss.transpose() ? x0.m() : x0.n();
        if (_zis.n() != _n || _zis.m() != nclasses)
        {
            _zis.resize(nclasses, _n);
            _zis.setZeros();
        }
    }
    
    void inline solver_aux_aux(const Vector<FeatureType>& x, Vector<FeatureType>& ref_barz, const int ind, const FeatureType deltas)
    {
        const FeatureType oldzi = _zis[ind];
        _zis[ind] = (FeatureType(1.0) - deltas) * _zis[ind] + deltas * (-_loss.scal_grad(x, ind));
        _loss.add_feature(ref_barz, ind, (_zis[ind] - oldzi) / (_n * _mu));
    };
    
    void inline solver_aux_aux(const Matrix<FeatureType>& x, Matrix<FeatureType>& ref_barz, const int ind, const FeatureType deltas)
    {
        Vector<FeatureType> oldzi, newzi;
        _zis.copyCol(ind, oldzi);
        _zis.refCol(ind, newzi);
        _loss.scal_grad(x, ind, newzi);
        newzi.add_scal(oldzi, FeatureType(1.0) - deltas, -deltas);
        oldzi.sub(newzi);
        oldzi.scal(-FeatureType(1.0) / (_n * _mu));
        _loss.add_feature(ref_barz, ind, oldzi);
    };
};

#endif