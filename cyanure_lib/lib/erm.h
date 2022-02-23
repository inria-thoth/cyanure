#ifndef ERM_H 
#define ERM_H

#include "data_structure/linalg.h"
#include "solvers/solvers.h"


// X is p x n
// y is nclasses x n
// W0 is p x nclasses if no intercept (or p+1 x nclasses with intercept)
// prediction model is   W0^T X  gives  nclasses x n
template <typename M>
void multivariate_erm(const M& X, const Matrix<typename M::value_type>& y, const Matrix<typename M::value_type>& W0, Matrix<typename M::value_type>& W, Matrix<typename M::value_type>& dual_variable, OptimInfo<typename M::value_type>& optim_info, const ParamSolver<typename M::value_type>& param, const ParamModel<typename M::value_type>& model)
{
    typedef typename M::value_type T;
    typedef typename M::index_type I;
    if ((model.intercept && X.m() + 1 != W0.m()) || (!model.intercept && X.m() != W0.m()))
    {
        cerr << "Dimension of initial point is not consistent.";
        return;
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

    init_omp(param.threads);
    typedef Matrix<T> D;
     
    if (is_loss_for_matrices(model.loss) || is_regul_for_matrices(model.regul))
    {
        DataMatrixLinear<M> data(X, model.intercept);
        if (param.verbose)
            data.print();
        LinearLossMat<M, Matrix<T>>* loss;
        switch (model.loss)
        {
        case SQUARE:
            loss = new SquareLossMat<M>(data, y);
            break;
        case LOGISTIC:
            loss = new LossMat<LogisticLoss<M>>(data, y);
            break;
        case SQHINGE:
            loss = new LossMat<SquaredHingeLoss<M>>(data, y);
            break;
            // case HINGE:  loss = new LossMat< HingeLoss<M> >(data,y); break;
        case SAFE_LOGISTIC:
            loss = new LossMat<SafeLogisticLoss<M>>(data, y);
            break;
        default:
            cerr << "Not implemented, square loss is chosen by default";
            loss = new SquareLossMat<M>(data, y);
        }
        const int nclass = W0.n();
       
        Regularizer<D, I>* regul = get_regul_mat<T, I>(model, nclass, loss->transpose());
        solve_mat<LinearLossMat<M, Matrix<T>>>(*loss, *regul, param, W0, W, dual_variable, optim_info);
        delete (regul);
        delete (loss);
    }
    else
    {
        W.copy(W0);
        const int nclass = W0.n();
        const int duality_gap_interval = MAX(param.duality_gap_interval, 1);
        optim_info.resize(nclass, NUMBER_OPTIM_PROCESS_INFO, MAX(param.max_iter / duality_gap_interval, 1));
        optim_info.setZeros();
        ParamSolver<T> param2 = param;
        param2.verbose = false;
        if (param.verbose)
        {
            DataMatrixLinear<M> data(X, model.intercept);
            data.print();
        }
        Timer global_all;
        global_all.start();
#pragma omp parallel for
        for (int ii = 0; ii < nclass; ++ii)
        {
            Vector<T> w0col, wcol, ycol, dualcol;
            OptimInfo<T> optim_info_col;
            W0.refCol(ii, w0col);
            W.refCol(ii, wcol);
            y.copyRow(ii, ycol);
            if (dual_variable.m() == nclass)
                dual_variable.copyRow(ii, dualcol);
            simple_erm(X, ycol, w0col, wcol, dualcol, optim_info_col, param2, model);
            if (dual_variable.m() == nclass)
                dual_variable.copyToRow(ii, dualcol);
#pragma omp critical
            {
                optim_info.add(optim_info_col, ii);
                if (param.verbose)
                {
                    const int noptim = optim_info_col.n() - 1;
                    logging(logINFO) << "Solver " << ii << " has terminated after " << optim_info_col(0, 0, noptim) << " epochs in " << optim_info_col(0, 5, noptim) << " seconds";
                    if (optim_info_col(0, 4, noptim) == 0)
                    {
                        logging(logINFO) << "   Primal objective: " << optim_info_col(0, 1, noptim) << ", relative duality gap: " << optim_info_col(0, 3, noptim);
                    }
                    else
                    {
                        logging(logINFO) << "   Primal objective: " << optim_info_col(0, 1, noptim) << ", tol: " << optim_info_col(0, 4, noptim);
                    }
                }
            }
        }
        global_all.stop();
        if (param.verbose)
        {
            logging(logINFO) << "Time for the one-vs-all strategy";
            global_all.printElapsed();
        }
    }
};

template <typename M>
void multivariate_erm(const M& X, const Vector<int>& y, const Matrix<typename M::value_type>& W0, Matrix<typename M::value_type>& W, Matrix<typename M::value_type>& dual_variable, OptimInfo<typename M::value_type>& optim_info, const ParamSolver<typename M::value_type>& param, const ParamModel<typename M::value_type>& model)
{
    typedef typename M::value_type T;
    typedef typename M::index_type I;
    if ((model.intercept && X.m() + 1 != W0.m()) || (!model.intercept && X.m() != W0.m()))
    {
        cerr << "Dimension of initial point is not consistent.";
        return;
    }
    const int nclass = y.maxval() + 1;
    if ((is_regression_loss(model.loss) || !is_loss_for_matrices(model.loss)))
    {
        const int n = y.n();
        Matrix<typename M::value_type> labels(nclass, n);
        labels.set(-(1.0));
        for (int ii = 0; ii < n; ++ii)
            labels(y[ii], ii) = (1.0);
        return multivariate_erm(X, labels, W0, W, dual_variable, optim_info, param, model);
    }
    init_omp(param.threads);
    typedef Matrix<T> D;
    DataMatrixLinear<M> data(X, model.intercept);
    if (param.verbose)
        data.print();
    LinearLossMat<M, Vector<int>>* loss;
    switch (model.loss)
    {
    case MULTI_LOGISTIC:
        loss = new MultiClassLogisticLoss<M>(data, y);
        break;
    default:
        cerr << "Not implemented, multilog loss is chosen by default";
        loss = new MultiClassLogisticLoss<M>(data, y);
    }
    Regularizer<D, I>* regul = get_regul_mat<T, I>(model, nclass, loss->transpose());
    solve_mat<LinearLossMat<M, Vector<int>>>(*loss, *regul, param, W0, W, dual_variable, optim_info);
    delete (regul);
    delete (loss);
};

template <typename M>
void simple_erm(const M& X, const Vector<typename M::value_type>& y, const Vector<typename M::value_type>& w0, Vector<typename M::value_type>& w, Vector<typename M::value_type>& dual_variable, OptimInfo<typename M::value_type>& optim_info, const ParamSolver<typename M::value_type>& param, const ParamModel<typename M::value_type>& model)
{
    init_omp(param.threads);
    typedef typename M::value_type T;
    typedef typename M::index_type I;
    typedef Vector<T> D;
    typedef LinearLossVec<M> loss_type;

    if (model.intercept)
    {
        if (X.m() + 1 != w0.n())
        {
            cerr << "Dimension of initial point is not consistent. With intercept, if X is m x n, w0 should be (n+1)-dimensional.";
            return;
        }
    }
    else
    {
        if (X.m() != w0.n())
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

    DataLinear<M> data(X, model.intercept);
    if (param.verbose)
        data.print();
    LinearLossVec<M>* loss;
    switch (model.loss)
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
    Regularizer<D, I>* regul;
    switch (model.regul)
    {
    case L2:
        regul = new Ridge<D, I>(model);
        break;
    case L1:
        regul = new Lasso<D, I>(model);
        break;
    case L1BALL:
        regul = new L1Ball<D, I>(model);
        break;
    case L2BALL:
        regul = new L2Ball<D, I>(model);
        break;
    case FUSEDLASSO:
        regul = new FusedLasso<D, I>(model);
        break;
    case ELASTICNET:
        regul = new ElasticNet<D, I>(model);
        break;
    case NONE:
        regul = new None<D, I>(model);
        break;
    default:
        cerr << "Not implemented, no regularization is chosen";
        regul = new None<D, I>(model);
    }
    Solver<loss_type>* solver;
    if (param.max_iter == 0)
    {
        ParamSolver<typename D::value_type> param2 = param;
        param2.verbose = false;
        solver = new ISTA_Solver<loss_type>(*loss, *regul, param2);
        solver->eval(w0);
        w.copy(w0);
    }
    else
    {
        if (param.solver == SVRG && model.regul == L2 && !model.intercept)
        {
            solver = new SVRG_Solver_FastRidge<loss_type, false>(*loss, *regul, param);
        }
        else if (param.solver == ACC_SVRG && model.regul == L2 && !model.intercept)
        {
            solver = new SVRG_Solver_FastRidge<loss_type, true>(*loss, *regul, param);
        }
        else if (param.solver == CATALYST_SVRG && model.regul == L2 && !model.intercept)
        {
            solver = new Catalyst<SVRG_Solver_FastRidge<loss_type, false>>(*loss, *regul, param);
        }
        else if (param.solver == QNING_SVRG && model.regul == L2 && !model.intercept)
        {
            solver = new QNing<SVRG_Solver_FastRidge<loss_type, false>>(*loss, *regul, param);
        }
        else
        {
            solver = get_solver<loss_type>(*loss, *regul, param);
            if (!solver)
            {
                w.copy(w0);
                delete (loss);
                delete (regul);
                return;
            }
        }
        D new_w0;
        if (model.intercept)
        {
            data.set_intercept(w0, new_w0);
        }
        else
        {
            new_w0.copyRef(w0);
        }
        if (dual_variable.n() != 0)
            solver->set_dual_variable(dual_variable);
        solver->solve(new_w0, w);
        if (model.intercept)
        {
            data.reverse_intercept(w);
        }
    }
    if (model.regul == L1)
        for (int ii = 0; ii < w.n(); ++ii)
            if (abs<T>(w[ii]) < EPSILON)
                w[ii] = 0;
   
    solver->get_optim_info(optim_info);
    delete (solver);
    delete (loss);
    delete (regul);
};

#endif