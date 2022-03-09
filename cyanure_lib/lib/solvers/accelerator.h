#ifndef ACCELERATOR_H
#define ACCELERATOR_H


template <typename SolverType>
class Catalyst : public SolverType
{
public:
    typedef typename SolverType::LT loss_type;
    USING_SOLVER
        Catalyst(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param) : SolverType(loss, regul, param)
    {
        _auxiliary_solver = NULL;
        _loss_ppa = NULL;
        _accelerated_solver = true;
        _freq_restart = regul.strong_convexity() > 0 ? param.max_iter + 2 : param.freq_restart;
    };
    ~Catalyst()
    {
        if (_auxiliary_solver)
            delete (_auxiliary_solver);
        if (_loss_ppa)
            delete (_loss_ppa);
    };
    
    virtual void set_dual_variable(const D& dual0)
    {
        _dual_var.copyRef(dual0);
    };

protected:
    virtual void solver_init(const D& x0)
    {
        _kappa = this->init_kappa_acceleration(x0);
        _mu = _regul.strong_convexity();
        _count = 0;
        _accelerated_solver = _kappa > 0; // this->_oldL/(_n) >= _mu;
        if (_accelerated_solver)
        {
            ParamSolver<FeatureType> param2;
            param2.max_iter = 1;
            param2.duality_gap_interval = 2;
            param2.verbose = false;
            param2.minibatch = this->minibatch();
            this->_Li.add(_kappa);
            _loss_ppa = new ProximalPointLoss<loss_type>(_loss, x0, _kappa);
            _auxiliary_solver = new SolverType(*_loss_ppa, _regul, param2, &this->_Li);
            if (_dual_var.size() > 0)
                _auxiliary_solver->set_dual_variable(_dual_var);
            _y.copy(x0);
            _alpha = FeatureType(1.0);
        }
        else
        {
            if (_verbose) {logging(logINFO) << "Switching to regular solver, problem is well conditioned";}
            SolverType::solver_init(x0);
        }
    };
    
    virtual void solver_aux(D& x)
    {
        if (_accelerated_solver)
        {
            const FeatureType q = _mu / (_mu + _kappa);
            D xold;
            xold.copy(x);
            _auxiliary_solver->solve(_y, x);
            const FeatureType alphaold = _alpha;
            _alpha = solve_binomial(FeatureType(1.0), _alpha * _alpha - q, -_alpha * _alpha);
            FeatureType beta = alphaold * (FeatureType(1.0) - alphaold) / (alphaold * alphaold + _alpha);
            if (++_count % _freq_restart == 0)
            {
                beta = 0;
                _alpha = FeatureType(1.0);
            }
            _y.copy(xold);
            _y.add_scal(x, FeatureType(1.0) + beta, -beta);
            _loss_ppa->set_anchor_point(_y);
        }
        else
        {
            SolverType::solver_aux(x);
        }
    };
    
    void print() const
    {
        logging(logINFO) << "Catalyst Accelerator";
        SolverType::print();
    };

    int _count, _freq_restart;
    FeatureType _kappa, _alpha, _mu;
    D _y, _dual_var;
    bool _accelerated_solver;
    SolverType* _auxiliary_solver;
    ProximalPointLoss<loss_type>* _loss_ppa;
};

template <typename SolverType>
class QNing final : public Catalyst<SolverType>
{
public:
    typedef typename SolverType::LT loss_type;
    USING_SOLVER
        using Catalyst<SolverType>::_kappa;
    using Catalyst<SolverType>::_accelerated_solver;
    using Catalyst<SolverType>::_auxiliary_solver;
    using Catalyst<SolverType>::_loss_ppa;
    using Catalyst<SolverType>::_y;
    QNing(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param) : Catalyst<SolverType>(loss, regul, param),
        _l_memory(param.l_memory)
    {
        _skipping_steps = 0;
        _line_search_steps = 0;
    };

    virtual void solve(const D& x0, D& x)
    {
        Solver<loss_type>::solve(x0, x);
        if (_verbose)
            {
            logging(logINFO) << "Total additional line search steps: " << _line_search_steps;
            logging(logINFO) << "Total skipping l-bfgs steps: " << _skipping_steps;
            }
    };

protected:
    virtual void solver_init(const D& x0)
    {
        Catalyst<SolverType>::solver_init(x0);
        if (_accelerated_solver)
        {
            _h0 = FeatureType(1.0) / _kappa;
            _m = 0;
            if (_verbose) {logging(logINFO) << "Memory parameter: " << _l_memory;}
            _ys.resize(x0.size(), _l_memory);
            _ss.resize(x0.size(), _l_memory);
            _rhos.resize(_l_memory);
            _etak = FeatureType(1.0);
            _skipping_steps = 0;
            _line_search_steps = 0;
        }
    };

    virtual void solver_aux(D& x)
    {
        if (_accelerated_solver)
        {
            if (_gk.size() == 0)
                get_gradient(x);

            // update variable _y and test
            D oldyk;
            oldyk.copy(_y);
            D oldxk;
            oldxk.copy(x);
            FeatureType oldFk = _Fk;
            D oldgk;
            oldgk.copy(_gk);
            D g;
            get_lbfgs_direction(g);

            const int max_iter = 5;
            _auxiliary_solver->save_state();
            for (int ii = 1; ii <= max_iter; ++ii)
            {
                _y.copy(oldyk);
                _y.add(g, -_etak);
                _y.add(oldgk, -(FeatureType(1.0) - _etak) / _kappa);
                get_gradient(x); // _gk = kappa(x-y)
                if (_etak == 0 || _Fk <= oldFk - (FeatureType(0.25) / _kappa) * oldgk.nrm2sq())
                    break;
                if (_Fk > 1.05 * oldFk)
                {
                    _auxiliary_solver->restore_state();
                    x.copy(oldxk);
                }
                _etak /= 2;
                _line_search_steps++;
                if (ii == max_iter - 1 || _etak < FeatureType(0.1))
                {
                    _etak = 0;
                }
            }
            if (_Fk > 1.05 * oldFk)
            {
                _auxiliary_solver->restore_state();
                x.copy(oldxk);
                reset_lbfgs();
            }
            else
            {
                oldyk.add_scal(_y, FeatureType(1.0), -FeatureType(1.0));
                oldgk.add_scal(_gk, FeatureType(1.0), -FeatureType(1.0));
                update_lbfgs_matrix(oldyk, oldgk);
            }
            _etak = MAX(MIN(FeatureType(1.0), _etak * FeatureType(1.2)), FeatureType(0.1));
        }
        else
        {
            SolverType::solver_aux(x);
        }
    };
    
    void print() const
    {
        logging(logINFO) << "QNing Accelerator";
        SolverType::print();
    };

private:

    inline void get_lbfgs_direction(Vector<FeatureType>& g) const
    {
        g.copy(_gk);
        get_lbfgs_direction_aux(g);
    };

    inline void get_lbfgs_direction(Matrix<FeatureType>& g) const
    {
        g.copy(_gk);
        Vector<FeatureType> gg;
        g.toVect(gg);
        get_lbfgs_direction_aux(gg);
    };

    inline void get_lbfgs_direction_aux(Vector<FeatureType>& g) const
    {
        // two-loop recursion algorithm
        Vector<FeatureType> alphas(_l_memory);
        Vector<FeatureType> cols, coly;
        FeatureType gamma = FeatureType(1.0) / _kappa;
        for (int ii = _m - 1; ii >= MAX(_m - _l_memory, 0); --ii)
        {
            const int ind = ii % _l_memory;
            _ss.refCol(ind, cols);
            _ys.refCol(ind, coly);
            if (ii == _m - 1)
                gamma = cols.dot(coly) / coly.nrm2sq();
            alphas[ind] = _rhos[ind] * cols.dot(g);
            g.add(coly, -alphas[ind]);
        }
        g.scal(gamma);
        for (int ii = MAX(_m - _l_memory, 0); ii <= _m - 1; ++ii)
        {
            const int ind = ii % _l_memory;
            _ss.refCol(ind, cols);
            _ys.refCol(ind, coly);
            const FeatureType beta = _rhos[ind] * coly.dot(g);
            g.add(cols, alphas[ind] - beta);
        }
    };

    inline void update_lbfgs_matrix(const Matrix<FeatureType>& sk, const Matrix<FeatureType>& yk)
    {
        Vector<FeatureType> skk, ykk;
        sk.toVect(skk);
        yk.toVect(ykk);
        update_lbfgs_matrix(skk, ykk);
    };

    inline void update_lbfgs_matrix(const Vector<FeatureType>& sk, const Vector<FeatureType>& yk)
    {
        const FeatureType theta = sk.dot(yk);
        if (theta > FeatureType(1e-12))
        {
            Vector<FeatureType> coly, cols;
            const int ind = _m % _l_memory;
            _ys.refCol(ind, coly);
            coly.copy(yk);
            _ss.refCol(ind, cols);
            cols.copy(sk);
            _rhos[ind] = FeatureType(1.0) / theta;
            _m++;
        }
        else
        {
            _skipping_steps++;
            // if (_skipping_steps % 10 == 0)
            //    reset_lbfgs();
        }
    };

    void reset_lbfgs()
    {
        _m = 0;
    };
    
    void get_gradient(D& x)
    {
        _loss_ppa->set_anchor_point(_y);
        _auxiliary_solver->solve(_y, x);
        _gk.copy(_y);
        _gk.add_scal(x, -_kappa, _kappa);
        _Fk = _loss_ppa->eval(x) + _regul.eval(x);
    };

    FeatureType _h0;
    int _l_memory;
    INTM _m;
    Matrix<FeatureType> _ys, _ss;
    Vector<FeatureType> _rhos;
    D _gk, _xk;
    FeatureType _Fk;
    FeatureType _etak;
    int _skipping_steps, _line_search_steps;
};


#endif