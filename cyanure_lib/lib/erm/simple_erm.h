#ifndef SIMPLE_ERM_H
#define SIMPLE_ERM_H

template <typename M, typename loss_type>
class SIMPLE_ERM : ERM<M, loss_type> {

    using ERM<M, loss_type>::model;
    using ERM<M, loss_type>::optim_info;
    using ERM<M, loss_type>::param;
public:


    SIMPLE_ERM(const Vector<typename M::value_type>& w0, Vector<typename M::value_type>& w, Vector<typename M::value_type>& dual_variable, OptimInfo<typename M::value_type>& optim_info, const ParamSolver<typename M::value_type>& param, const ParamModel<typename M::value_type>& model) :ERM<M, loss_type>(optim_info, param, model), W0(w0), W(w), dual_variable(dual_variable) {
    }

    void solve_problem(const M& X, const Vector<typename M::value_type>& y) {
        init_omp(ERM<M, loss_type>::param.threads);
        typedef typename M::index_type I;
        typedef typename M::value_type T;
        typedef Vector<T> D;

        DataLinear<M> data(X, ERM<M, loss_type>::model.intercept);
        if (ERM<M, loss_type>::param.verbose)
            data.print();

        verify_input(X);
        LinearLossVec<M>* loss = get_loss(data, y);
        Regularizer<D, I>* regul = get_regul();


        Solver<loss_type>* solver;
        if (ERM<M, loss_type>::param.max_iter == 0)
        {
            ParamSolver<typename loss_type::value_type> param2 = ERM<M, loss_type>::param;
            param2.verbose = false;
            solver = new ISTA_Solver<loss_type>(*loss, *regul, param2);
            solver->eval(SIMPLE_ERM<M, loss_type>::W0);
            W.copy(SIMPLE_ERM<M, loss_type>::W0);
        }
        else
        {
            if (ERM<M, loss_type>::param.solver == SVRG && model.regul == L2 && !model.intercept)
            {
                solver = new SVRG_Solver_FastRidge<loss_type, false>(*loss, *regul, ERM<M, loss_type>::param);
            }
            else if (ERM<M, loss_type>::param.solver == ACC_SVRG && model.regul == L2 && !model.intercept)
            {
                solver = new SVRG_Solver_FastRidge<loss_type, true>(*loss, *regul, ERM<M, loss_type>::param);
            }
            else if (ERM<M, loss_type>::param.solver == CATALYST_SVRG && model.regul == L2 && !model.intercept)
            {
                solver = new Catalyst<SVRG_Solver_FastRidge<loss_type, false>>(*loss, *regul, ERM<M, loss_type>::param);
            }
            else if (ERM<M, loss_type>::param.solver == QNING_SVRG && model.regul == L2 && !model.intercept)
            {
                solver = new QNing<SVRG_Solver_FastRidge<loss_type, false>>(*loss, *regul, ERM<M, loss_type>::param);
            }
            else
            {
                regul->strong_convexity();
                solver = get_solver(*loss, *regul, ERM<M, loss_type>::param);

            }
            if (!solver)
            {
                W.copy(W0);
                delete (loss);
                delete (regul);
                return;
            }
            D new_w0;
            if (model.intercept)
            {
                data.set_intercept(W0, new_w0);
            }
            else
            {
                new_w0.copyRef(W0);
            }
            if (dual_variable.n() != 0)
                solver->set_dual_variable(dual_variable);
            solver->solve(new_w0, W);
            if (model.intercept)
            {
                data.reverse_intercept(W);
            }
        }
        if (model.regul == L1)
            for (int ii = 0; ii < W.n(); ++ii)
                if (abs<T>(W[ii]) < EPSILON)
                    W[ii] = 0;

        solver->get_optim_info(optim_info);
        delete (solver);
        delete (loss);
        delete (regul);
    }

private:

    const Vector<typename M::value_type>& W0;
    Vector<typename M::value_type>& W;
    const Vector<typename M::value_type>& dual_variable;


    inline void verify_input(const M& X) {
        if (model.intercept)
        {
            if (X.m() + 1 != W0.n())
            {
                cerr << "Dimension of initial point is not consistent. With intercept, if X is m x n, w0 should be (n+1)-dimensional.";
                return;
            }
        }
        else
        {
            if (X.m() != W0.n())
            {
                cerr << "Dimension of initial point is not consistent. If X is m x n, w0 should be n-dimensional.";
                return;
            }
        }

        if (param.max_iter < 0)
        {
            throw ValueError("Maximum number of iteration must be positive");
        }
        if (model.lambda_1 < 0)
        {
            throw ValueError("Penalty term must be positive");
        }
        if (param.tol < 0)
        {
            throw ValueError("Tolerance for stopping criteria must be positive");
        }
    };


    Regularizer<Vector<typename M::value_type>, typename M::index_type>* get_regul() {
        typedef Vector<typename M::value_type> D;
        typedef typename M::index_type I;
        Regularizer<D, I>* regul;


        switch (ERM<M, loss_type>::model.regul)
        {
        case L2:
            regul = new Ridge<D, I>(ERM<M, loss_type>::model);
            break;
        case L1:
            regul = new Lasso<D, I>(ERM<M, loss_type>::model);
            break;
        case L1BALL:
            regul = new L1Ball<D, I>(ERM<M, loss_type>::model);
            break;
        case L2BALL:
            regul = new L2Ball<D, I>(ERM<M, loss_type>::model);
            break;
        case FUSEDLASSO:
            regul = new FusedLasso<D, I>(ERM<M, loss_type>::model);
            break;
        case ELASTICNET:
            regul = new ElasticNet<D, I>(ERM<M, loss_type>::model);
            break;
        case NONE:
            regul = new None<D, I>(ERM<M, loss_type>::model);
            break;
        default:
            cerr << "Not implemented, no regularization is chosen";
            regul = new None<D, I>(ERM<M, loss_type>::model);
        }
        return regul;
    }

    Solver<loss_type>* get_solver(const loss_type& loss, const Regularizer<typename loss_type::variable_type, typename loss_type::index_type>& regul_tmp, const ParamSolver<typename loss_type::value_type>& param)
    {
        typedef typename loss_type::value_type T;
        Solver<loss_type>* solver;
        solver_t solver_type = param.solver;
        Regularizer<typename loss_type::variable_type, typename loss_type::index_type>* regul = get_regul();

        if (solver_type == AUTO)
        {
            const T L = loss.lipschitz();
            const int n = loss.n();
            const T lambda_1 = regul->strong_convexity();
            if (n < 1000)
            {
                solver_type = QNING_ISTA;
            }
            else if (lambda_1 < L / (100 * n))
            {
                solver_type = QNING_MISO;
            }
            else
            {
                solver_type = CATALYST_MISO;
            }
        }
        switch (solver_type)
        {
        case ISTA:
            solver = new ISTA_Solver<loss_type>(loss, *regul, param);
            break;
        case QNING_ISTA:
            solver = new QNing<ISTA_Solver<loss_type>>(loss, *regul, param);
            break;
        case CATALYST_ISTA:
            solver = new Catalyst<ISTA_Solver<loss_type>>(loss, *regul, param);
            break;
        case FISTA:
            solver = new FISTA_Solver<loss_type>(loss, *regul, param);
            break;
        case SVRG:
            solver = new SVRG_Solver<loss_type>(loss, *regul, param);
            break;
        case MISO:
            solver = regul->strong_convexity() > 0 ? new MISO_Solver<loss_type>(loss, *regul, param) : new Catalyst<MISO_Solver<loss_type>>(loss, *regul, param);
            break;
        case SVRG_UNIFORM:
        {
            ParamSolver<typename loss_type::value_type> param2 = param;
            param2.non_uniform_sampling = false;
            solver = new SVRG_Solver<loss_type>(loss, *regul, param2);
            break;
        }
        case CATALYST_SVRG:
            solver = new Catalyst<SVRG_Solver<loss_type>>(loss, *regul, param);
            break;
        case QNING_SVRG:
            solver = new QNing<SVRG_Solver<loss_type>>(loss, *regul, param);
            break;
        case CATALYST_MISO:
            solver = new Catalyst<MISO_Solver<loss_type>>(loss, *regul, param);
            break;
        case QNING_MISO:
            solver = new QNing<MISO_Solver<loss_type>>(loss, *regul, param);
            break;
        case ACC_SVRG:
            solver = new Acc_SVRG_Solver<loss_type>(loss, *regul, param);
            break;
        default:
            throw NotImplementedException("This solver is not implemented!");
            solver = NULL;
        }
        return solver;
    };


    LinearLossVec<M>* get_loss(DataLinear<M>& data, const Vector<typename M::value_type>& y) {
        LinearLossVec<M>* loss;
        switch (ERM<M, loss_type>::model.loss)
        {
        case SQUARE:
            loss = new SquareLoss<M>(data, y);
            break;
        case LOGISTIC:
            loss = new LogisticLoss<M>(data, y);
            break;
        case SQHINGE:
            loss = new SquaredHingeLoss<M>(data, y);
            break;
            // case HINGE: loss = new HingeLoss<M>(data,y); break;
        case SAFE_LOGISTIC:
            loss = new SafeLogisticLoss<M>(data, y);
            break;
        default:
            cerr << "Not implemented, square loss is chosen by default";
            loss = new SquareLoss<M>(data, y);
        }
        return loss;
    }
};

#endif