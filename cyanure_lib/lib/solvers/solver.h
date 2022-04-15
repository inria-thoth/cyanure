#ifndef SOLVERS_H
#define SOLVERS_H

#include "../error_management/exception.h"
#include "../losses/loss.h"
#include "../regul/regularizer.h"
#include "../data_structure/list.h"
#include "../logging/logger.h"
#include  "../BLAS/configure_blas.h"
#include  "../timer.h"

#define USING_SOLVER                             \
    typedef typename loss_type::variable_type D; \
    typedef typename loss_type::value_type FeatureType;    \
    typedef typename loss_type::index_type PointerType;    \
    typedef loss_type LT;                        \
    using Solver<loss_type>::_L;                 \
    using Solver<loss_type>::_loss;              \
    using Solver<loss_type>::_regul;             \
    using Solver<loss_type>::_Li;                \
    using Solver<loss_type>::_verbose;           \
    using Solver<loss_type>::_max_iter_backtracking;

static const int NUMBER_OPTIM_PROCESS_INFO = 6;

loglevel_e loglevel = logDEBUG4;

enum solver_t
{
    ISTA,
    CATALYST_ISTA,
    QNING_ISTA,
    FISTA,
    SAGA,
    SVRG,
    SVRG_UNIFORM,
    CATALYST_SVRG,
    ACC_SVRG,
    MISO,
    CATALYST_MISO,
    QNING_SVRG,
    QNING_MISO,
    AUTO,
    INCORRECT_SOLVER
};

solver_t solver_from_string(char* regul)
{
    if (strcmp(regul, "ista") == 0)
        return ISTA;
    if (strcmp(regul, "catalyst-ista") == 0)
        return CATALYST_ISTA;
    if (strcmp(regul, "qning-ista") == 0)
        return QNING_ISTA;
    if (strcmp(regul, "fista") == 0)
        return FISTA;
    if (strcmp(regul, "saga") == 0)
        return SAGA;
    if (strcmp(regul, "svrg") == 0)
        return SVRG;
    if (strcmp(regul, "catalyst-svrg") == 0)
        return CATALYST_SVRG;
    if (strcmp(regul, "qning-svrg") == 0)
        return QNING_SVRG;
    if (strcmp(regul, "qning-miso") == 0)
        return QNING_MISO;
    if (strcmp(regul, "acc-svrg") == 0)
        return ACC_SVRG;
    if (strcmp(regul, "miso") == 0)
        return MISO;
    if (strcmp(regul, "catalyst-miso") == 0)
        return CATALYST_MISO;
    if (strcmp(regul, "svrg-uniform") == 0)
        return SVRG_UNIFORM;
    if (strcmp(regul, "auto") == 0)
        return AUTO;
    return INCORRECT_SOLVER;
}

template <typename T>
struct ParamSolver
{
    ParamSolver()
    {
        max_iter = 100;
        duality_gap_interval = 10;
        tol = T(1e-3);
        verbose = false;
        solver = FISTA;
        max_iter_backtracking = 500;
        minibatch = 1;
        threads = -1;
        non_uniform_sampling = true;
        l_memory = 20;
        freq_restart = 50;
    };
    int max_iter;
    T tol;
    int duality_gap_interval;
    bool verbose;
    solver_t solver;
    int max_iter_backtracking;
    int minibatch;
    int threads;
    bool non_uniform_sampling;
    int l_memory;
    int freq_restart;
};

template <typename loss_type>
class Solver
{
public:
    typedef typename loss_type::variable_type D;
    typedef typename loss_type::value_type T;
    typedef typename loss_type::index_type I;

    Solver(const loss_type& loss, const Regularizer<D, I>& regul, const ParamSolver<T>& param) : _loss(loss), _regul(regul)
    {
        _verbose = param.verbose;
        _it0 = MAX(param.duality_gap_interval, 1);
        _tol = param.tol;
        _nepochs = param.max_iter;
        _max_iter_backtracking = param.max_iter_backtracking;
        _best_dual = -INFINITY;
        _best_primal = INFINITY;
        _duality = _loss.provides_fenchel() && regul.provides_fenchel();
        _optim_info.resize(NUMBER_OPTIM_PROCESS_INFO, MAX(param.max_iter / _it0, 1));
        _L = 0;
    };
    virtual ~Solver() {};

    virtual void solve(const D& x0, D& x)
    {
        _time.start();
        x.copy(x0);
        if (!_duality && _nepochs > 1)
            _xold.copy(x0);
        solver_init(x0);
        if (_verbose)
        {
            logging(logINFO) << "*********************************";
            print();
            _loss.print();
            _regul.print();
        }

        for (int it = 1; it <= _nepochs; ++it)
        {
            if ((it % _it0) == 0)
                if (test_stopping_criterion(x, it))
                    break;
            solver_aux(x);
        }
        _time.stop();
        if (_verbose) {
            _time.printElapsed();
        }
        if (_best_primal != INFINITY)
            x.copy(_bestx);
    }
    
    void get_optim_info(OptimInfo<T>& optim) const
    {
        int count = 0;
        for (int ii = 0; ii < _optim_info.n(); ++ii)
            if (_optim_info(0, ii) != 0)
                ++count;
        if (count > 0)
        {
            optim.resize(1, NUMBER_OPTIM_PROCESS_INFO, count);
        }
        for (int ii = 0; ii < count; ++ii)
            for (int jj = 0; jj < NUMBER_OPTIM_PROCESS_INFO; ++jj){
                optim(0, jj, ii) = _optim_info(jj, ii);
            }
    };

    void eval(const D& x)
    {
        test_stopping_criterion(x, 1);
        _optim_info(5, 0) = 0;
    };

    virtual void set_dual_variable(const D& dual0) {};

    virtual void save_state() {};
    virtual void restore_state() {};

private:
    inline T get_dual(const D& x) const
    {
        if (!_regul.provides_fenchel() || !_loss.provides_fenchel())
        {
            logging(logERROR) << "Error: no duality gap available";
            return -INFINITY;
        }
        D grad1, grad2;
        _loss.get_dual_variable(x, grad1, grad2);
        const T dual = -_regul.fenchel(grad1, grad2);
        return dual - _loss.fenchel(grad1);
    };

    inline bool test_stopping_criterion(const D& x, const int it)
    {
        const T primal = _loss.eval(x) + _regul.eval(x);
        _best_primal = MIN(_best_primal, primal);
        const int ii = MAX(it / _it0 - 1, 0);
        const double sec = _time.getElapsed();
        Vector<T> optim;
        _optim_info.refCol(ii, optim);
        if (_best_primal == primal)
            _bestx.copy(x);
        if (_verbose)
        {
            if (primal == _best_primal)
            {
                logging(logINFO) << "Epoch: " << it << ", primal objective: " << primal << ", time: " << sec;
            }
            else
            {
                logging(logINFO) << "Epoch: " << it << ", primal objective: " << primal << ", best primal: " << _best_primal << ", time: " << sec;
            }
        }
        optim[0] = it;
        optim[1] = primal;
        optim[5] = sec;
        if (_duality)
        {
            const T dual = get_dual(x);
            _best_dual = MAX(_best_dual, dual);
            const T duality_gap = (_best_primal - _best_dual) / abs<T>(_best_primal);
            bool stop = false;            
            if ((it / _it0) >= 4){
                if (_optim_info(3, (it / _it0) - 4) == duality_gap){
                    stop = true;
                    logging(logWARNING) << "Your problem is prone to numerical instability. It would be safer to use double.";
                }
            }
            if (_verbose) {
                logging(logINFO) << "Best relative duality gap: " << duality_gap;
            }
            optim[2] = _best_dual;
            optim[3] = duality_gap;
            if(duality_gap < _tol){
                stop = true;
            }
            else if(duality_gap <= 0 ){
                logging(logWARNING) << "Your problem is prone to numerical instability. It would be safer to use double.";
                stop = true;
            }
            return stop;
        }
        else
        {
            _xold.sub(x);
            const T diff = sqrt(_xold.nrm2sq() / MAX(EPSILON, x.nrm2sq()));
            _xold.copy(x);
            optim[4] = diff;
            return diff < _tol;
        }
    }

protected:
    virtual void solver_init(const D& x0) = 0;
    virtual void solver_aux(D& x) = 0;
    virtual void print() const = 0;
    virtual int minibatch() const { return 1; };
    bool _verbose;
    int _it0;
    int _nepochs;
    int _max_iter_backtracking;
    int _restart_frequency;
    T _tol;
    const loss_type& _loss;
    const Regularizer<D, I>& _regul;
    Timer _time;
    T _best_dual;
    T _best_primal;
    Matrix<T> _optim_info;
    bool _duality;
    D _xold;
    T _L;
    D _bestx;
    Vector<T> _Li;
};

#endif
