#ifndef MULTI_ERM_H
#define MULTI_ERM_H

template <typename InputMatrixType, typename LossType>
class MULTI_ERM : public ERM<InputMatrixType> {

public:
    MULTI_ERM(const Matrix<typename InputMatrixType::value_type>& w0, Matrix<typename InputMatrixType::value_type>& w, Matrix<typename InputMatrixType::value_type>& dual_variable, OptimInfo<typename InputMatrixType::value_type>& optim_info, const ParamSolver<typename InputMatrixType::value_type>& param, const ParamModel<typename InputMatrixType::value_type>& model) :ERM<InputMatrixType>(optim_info, param, model), W0(w0), W(w), dual_variable(dual_variable) {
    }

    // X is p x n
   // y is nclasses x n
   // W0 is p x nclasses if no intercept (or p+1 x nclasses with intercept)
   // prediction model is   W0^FeatureType X  gives  nclasses x n
    void solve_problem_vector(const InputMatrixType& X, const Vector<int>& y) {
        verify_input(X);

        const int nclass = y.maxval() + 1;
        if ((super::is_regression_loss(super::model.loss) || !super::is_loss_for_matrices(super::model.loss)))
        {
            const int n = y.n();
            Matrix<typename InputMatrixType::value_type> labels(nclass, n);

            labels.set(-(1.0));
            for (int ii = 0; ii < n; ++ii)
                labels(y[ii], ii) = (1.0);
            MULTI_ERM<InputMatrixType, LinearLossMat<InputMatrixType, Matrix<typename InputMatrixType::value_type>>> problem_configuration(W0, W, dual_variable, super::optim_info, super::param, super::model);
            return problem_configuration.solve_problem_matrix(X, labels);
        }

        init_omp(super::param.threads);

        typedef Matrix<FeatureType> D;
        DataMatrixLinear<InputMatrixType> data(X, super::model.intercept);

        if (super::param.verbose)
            data.print();

        LinearLossMat<InputMatrixType, Vector<int>>* loss = new MultiClassLogisticLoss<InputMatrixType>(data, y);;
        if (super::model.loss != MULTI_LOGISTIC) {
            logging(logERROR) << "Multilog loss is the only multi class implemented loss!";
            logging(logINFO) << "Multilog loss is used!";
        }
        const bool transpose = loss->transpose();

        Regularizer<D, PointerType>* regul = get_regul_mat(nclass, transpose);

        solve_mat(*loss, *regul);

        delete (regul);
        delete (loss);
    };


    // X is p x n
    // y is nclasses x n
    // W0 is p x nclasses if no intercept (or p+1 x nclasses with intercept)
    // prediction model is   W0^FeatureType X  gives  nclasses x n
    void solve_problem_matrix(const InputMatrixType& X, const Matrix<typename InputMatrixType::value_type>& y) {
        verify_input(X);

        init_omp(super::param.threads);
        typedef Matrix<FeatureType> D;


        if (super::is_loss_for_matrices(super::model.loss) || super::is_regul_for_matrices(super::model.regul))
        {
            DataMatrixLinear<InputMatrixType> data(X, super::model.intercept);
            if (super::param.verbose)
                data.print();

            LinearLossMat<InputMatrixType, Matrix<FeatureType>>* loss = get_loss_matrix(data, y);

            const int nclass = W0.n();

            Regularizer<D, PointerType>* regul = get_regul_mat(nclass, loss->transpose());
            solve_mat(*loss, *regul);
            delete (regul);
            delete (loss);
        }
        else
        {
            W.copy(W0);
            const int nclass = W0.n();
            const int duality_gap_interval = MAX(super::param.duality_gap_interval, 1);
            super::optim_info.resize(nclass, NUMBER_OPTIM_PROCESS_INFO, MAX(super::param.max_iter / duality_gap_interval, 1));
            super::optim_info.setZeros();
            ParamSolver<FeatureType> param2 = super::param;
            param2.verbose = false;
            if (super::param.verbose)
            {
                DataMatrixLinear<InputMatrixType> data(X, super::model.intercept);
                data.print();
            }
            Timer global_all;
            global_all.start();
#pragma omp parallel for
            for (int ii = 0; ii < nclass; ++ii)
            {
                Vector<FeatureType> w0col, wcol, ycol, dualcol;
                OptimInfo<FeatureType> optim_info_col;
                W0.refCol(ii, w0col);
                W.refCol(ii, wcol);
                y.copyRow(ii, ycol);
                if (dual_variable.m() == nclass)
                {
                    dual_variable.copyRow(ii, dualcol);
                }
                SIMPLE_ERM<InputMatrixType, LinearLossVec<InputMatrixType>> problem_configuration = SIMPLE_ERM<InputMatrixType, LinearLossVec<InputMatrixType>>(w0col, wcol, dualcol, optim_info_col, param2, super::model);
                problem_configuration.solve_problem(X, ycol);
                if (dual_variable.m() == nclass)
                    dual_variable.copyToRow(ii, dualcol);
#pragma omp critical
                {
                    super::optim_info.add(optim_info_col, ii);
                    if (super::param.verbose)
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
            if (super::param.verbose)
            {
                logging(logINFO) << "Time for the one-vs-all strategy";
                global_all.printElapsed();
            }
        }
    }

private:
    typedef ERM<InputMatrixType> super;

    typedef typename InputMatrixType::value_type FeatureType;
    typedef typename InputMatrixType::index_type PointerType;
    const Matrix<FeatureType>& W0;
    Matrix<FeatureType>& W;
    Matrix<FeatureType>& dual_variable;

    inline void verify_input(const InputMatrixType& X) {
        if (super::model.intercept)
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

        if (super::param.max_iter < 0)
        {
            throw ValueError("Maximum number of iteration must be positive");
        }
        if (super::model.lambda_1 < 0)
        {
            throw ValueError("Penalty term must be positive");
        }
        if (super::param.tol < 0)
        {
            throw ValueError("Tolerance for stopping criteria must be positive");
        }
    };

    LinearLossMat<InputMatrixType, Matrix<typename InputMatrixType::value_type>>* get_loss_matrix(DataMatrixLinear<InputMatrixType>& data, const Matrix<typename InputMatrixType::value_type>& y) {
        typedef typename InputMatrixType::value_type FeatureType;
        LinearLossMat<InputMatrixType, Matrix<FeatureType>>* loss;
        switch (super::model.loss)
        {
        case SQUARE:
            loss = new SquareLossMat<InputMatrixType>(data, y);
            break;
        case LOGISTIC:
            loss = new LossMat<LogisticLoss<InputMatrixType>>(data, y);
            break;
        case SQHINGE:
            loss = new LossMat<SquaredHingeLoss<InputMatrixType>>(data, y);
            break;
        case SAFE_LOGISTIC:
            loss = new LossMat<SafeLogisticLoss<InputMatrixType>>(data, y);
            break;
        default:
            cerr << "Not implemented, square loss is chosen by default";
            loss = new SquareLossMat<InputMatrixType>(data, y);
        }
        return loss;
    }

    inline void solve_mat(LossType& loss, const Regularizer<typename LossType::variable_type, typename LossType::index_type>& regul)
    {
        typedef typename LossType::value_type value_type;
        typedef typename LossType::variable_type variable_type;
        Solver<LossType>* solver;
        if (super::param.max_iter == 0)
        {
            ParamSolver<value_type> param2 = super::param;
            param2.verbose = false;
            solver = new ISTA_Solver<LossType>(loss, regul, param2);
            if (loss.transpose())
            {
                Matrix<value_type> W0T, WT;
                W0.transpose(W0T);
                solver->eval(W0T);
            }
            else
            {
                solver->eval(W0);
            }
            W.copy(W0);
        }
        else
        {
            solver = get_solver(loss, regul, super::param);
            if (!solver)
            {
                W.copy(W0);
                return;
            }
            variable_type new_W0;
            if (loss.intercept())
            {
                loss.set_intercept(W0, new_W0);
            }
            else
            {
                new_W0.copyRef(W0);
            }
            if (dual_variable.n() != 0)
                solver->set_dual_variable(dual_variable);
            if (loss.transpose())
            {
                Matrix<value_type> W0T, WT;
                new_W0.transpose(W0T);
                solver->solve(W0T, WT);
                WT.transpose(W);
            }
            else
            {
                solver->solve(new_W0, W);
            }
            if (loss.intercept())
            {
                loss.reverse_intercept(W);
            }
        }
        if (regul.id() == L1)
            for (INTM ii = 0; ii < W.n(); ++ii)
                for (INTM jj = 0; jj < W.m(); ++jj)
                    if (abs<value_type>(W(jj, ii)) < EPSILON)
                        W(jj, ii) = 0;

        solver->get_optim_info(super::optim_info);
        delete (solver);
    };

    Solver<LossType>* get_solver(const LossType& loss, const Regularizer<typename LossType::variable_type, typename LossType::index_type>& regul, const ParamSolver<typename LossType::value_type>& param)
    {
        Solver<LossType>* solver;
        solver_t solver_type = param.solver;

        if (solver_type == AUTO)
        {
            const FeatureType L = loss.lipschitz();
            const int n = loss.n();
            const FeatureType lambda_1 = regul.strong_convexity();
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
            solver = new ISTA_Solver<LossType>(loss, regul, param);
            break;
        case QNING_ISTA:
            solver = new QNing<ISTA_Solver<LossType>>(loss, regul, param);
            break;
        case CATALYST_ISTA:
            solver = new Catalyst<ISTA_Solver<LossType>>(loss, regul, param);
            break;
        case FISTA:
            solver = new FISTA_Solver<LossType>(loss, regul, param);
            break;
        case SVRG:
            solver = new SVRG_Solver<LossType>(loss, regul, param);
            break;
        case MISO:
            solver = regul.strong_convexity() > 0 ? new MISO_Solver<LossType>(loss, regul, param) : new Catalyst<MISO_Solver<LossType>>(loss, regul, param);
            break;
        case SVRG_UNIFORM:
        {
            ParamSolver<typename LossType::value_type> param2 = param;
            param2.non_uniform_sampling = false;
            solver = new SVRG_Solver<LossType>(loss, regul, param2);
            break;
        }
        case CATALYST_SVRG:
            solver = new Catalyst<SVRG_Solver<LossType>>(loss, regul, param);
            break;
        case QNING_SVRG:
            solver = new QNing<SVRG_Solver<LossType>>(loss, regul, param);
            break;
        case CATALYST_MISO:
            solver = new Catalyst<MISO_Solver<LossType>>(loss, regul, param);
            break;
        case QNING_MISO:
            solver = new QNing<MISO_Solver<LossType>>(loss, regul, param);
            break;
        case ACC_SVRG:
            solver = new Acc_SVRG_Solver<LossType>(loss, regul, param);
            break;
        default:
            throw NotImplementedException("This solver is not implemented!");
            solver = NULL;
        }
        return solver;
    };


    Regularizer<Matrix<typename InputMatrixType::value_type>, typename InputMatrixType::index_type>* get_regul_mat(const int nclass, const bool transpose)
    {
        typedef Matrix<FeatureType> D;
        typedef Vector<FeatureType> V;
        Regularizer<D, PointerType>* regul;
        switch (super::model.regul)
        {
        case L2:
            regul = transpose ? static_cast<Regularizer<D, PointerType> *>(new RegVecToMat<Ridge<V, PointerType>>(super::model))
                : new RegMat<Ridge<V, PointerType>>(super::model, nclass, transpose);
            break;
        case L1:
            regul = transpose ? static_cast<Regularizer<D, PointerType> *>(new RegVecToMat<Lasso<V, PointerType>>(super::model))
                : new RegMat<Lasso<V, PointerType>>(super::model, nclass, transpose);
            break;
        case ELASTICNET:
            regul = transpose ? static_cast<Regularizer<D, PointerType> *>(new RegVecToMat<ElasticNet<V, PointerType>>(super::model))
                : new RegMat<ElasticNet<V, PointerType>>(super::model, nclass, transpose);
            break;
        case L1BALL:
            regul = new RegMat<L1Ball<V, PointerType>>(super::model, nclass, transpose);
            break;
        case L2BALL:
            regul = new RegMat<L2Ball<V, PointerType>>(super::model, nclass, transpose);
            break;
        case L1L2:
            regul = new MixedL1L2<FeatureType, PointerType>(super::model, nclass, transpose);
            break;
        case L1L2_L1:
            regul = new MixedL1L2_L1<FeatureType, PointerType>(super::model, nclass, transpose);
            break;
        case L1LINF:
            regul = new MixedL1Linf<FeatureType, PointerType>(super::model, nclass, transpose);
            break;
        case FUSEDLASSO:
            regul = new RegMat<FusedLasso<V, PointerType>>(super::model, nclass, transpose);
            break;
        case NONE:
            regul = new None<D, PointerType>(super::model);
            break;
        default:
            cerr << "Not implemented, no regularization is chosen";
            regul = new None<D, PointerType>(super::model);
        }
        return regul;
    };

};

#endif