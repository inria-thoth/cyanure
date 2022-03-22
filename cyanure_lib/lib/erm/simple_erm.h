#ifndef SIMPLE_ERM_H
#define SIMPLE_ERM_H

#include "../solvers/ista.h"
#include "../solvers/accelerator.h"
#include "../solvers/incremental_solvers/svrg.h"
#include "../solvers/incremental_solvers/miso.h"
#include "../regul/regularizers.h"

template <typename InputMatrixType, typename LossType>
class SIMPLE_ERM : ERM<InputMatrixType> {
public:


    SIMPLE_ERM(const Vector<typename InputMatrixType::value_type>& w0, Vector<typename InputMatrixType::value_type>& w, Vector<typename InputMatrixType::value_type>& dual_variable, OptimInfo<typename InputMatrixType::value_type>& optim_info, const ParamSolver<typename InputMatrixType::value_type>& param, const ParamModel<typename InputMatrixType::value_type>& model) :ERM<InputMatrixType>(optim_info, param, model), W0(w0), W(w), dual_variable(dual_variable) {
    }

    void solve_problem(const InputMatrixType& X, const Vector<typename InputMatrixType::value_type>& y) {
        init_omp(super::param.threads);

        DataLinear<InputMatrixType> data(X, super::model.intercept);
        if (super::param.verbose)
            data.print();

        verify_input(X);
        LinearLossVec<InputMatrixType>* loss = get_loss(data, y);
        Regularizer<LabelsType, PointerType>* regul = get_regul();


        Solver<LossType>* solver;
        if (super::param.max_iter == 0)
        {
            ParamSolver<typename LossType::value_type> param2 = super::param;
            param2.verbose = false;
            solver = new ISTA_Solver<LossType>(*loss, *regul, param2);
            solver->eval(SIMPLE_ERM<InputMatrixType, LossType>::W0);
            W.copy(SIMPLE_ERM<InputMatrixType, LossType>::W0);
        }
        else
        {
            if (super::param.solver == SVRG && super::model.regul == L2 && !super::model.intercept)
            {
                solver = new SVRG_Solver_FastRidge<LossType, false>(*loss, *regul, super::param);
            }
            else if (super::param.solver == ACC_SVRG && super::model.regul == L2 && !super::model.intercept)
            {
                solver = new SVRG_Solver_FastRidge<LossType, true>(*loss, *regul, super::param);
            }
            else if (super::param.solver == CATALYST_SVRG && super::model.regul == L2 && !super::model.intercept)
            {
                solver = new Catalyst<SVRG_Solver_FastRidge<LossType, false>>(*loss, *regul, super::param);
            }
            else if (super::param.solver == QNING_SVRG && super::model.regul == L2 && !super::model.intercept)
            {
                solver = new QNing<SVRG_Solver_FastRidge<LossType, false>>(*loss, *regul, super::param);
            }
            else
            {
                regul->strong_convexity();
                solver = get_solver(*loss, *regul, super::param);

            }
            if (!solver)
            {
                W.copy(W0);
                delete (loss);
                delete (regul);
                return;
            }
            LabelsType new_w0;
            if (super::model.intercept)
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
            if (super::model.intercept)
            {
                data.reverse_intercept(W);
            }
        }
        if (super::model.regul == L1)
            for (int ii = 0; ii < W.n(); ++ii)
                if (abs<FeatureType>(W[ii]) < EPSILON)
                    W[ii] = 0;

        solver->get_optim_info(super::optim_info);
        delete (solver);
        delete (loss);
        delete (regul);
    }

private:
    typedef ERM<InputMatrixType> super;
    typedef typename InputMatrixType::index_type PointerType;
    typedef typename InputMatrixType::value_type FeatureType;
    typedef Vector<FeatureType> LabelsType;

    const Vector<FeatureType>& W0;
    Vector<FeatureType>& W;
    const Vector<FeatureType>& dual_variable;


    inline void verify_input(const InputMatrixType& X) {
        if (super::model.intercept)
        {
            if (X.m() + 1 != W0.n())
            {
                logging(logERROR) << "Dimension of initial point is not consistent. With intercept, if X is m x n, w0 should be (n+1)-dimensional.";
                return;
            }
        }
        else
        {
            if (X.m() != W0.n())
            {
                logging(logERROR) << "Dimension of initial point is not consistent. If X is m x n, w0 should be n-dimensional.";
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


    Regularizer<Vector<typename InputMatrixType::value_type>, typename InputMatrixType::index_type>* get_regul() {
        Regularizer<LabelsType, PointerType>* regul;


        switch (super::model.regul)
        {
        case L2:
            regul = new Ridge<LabelsType, PointerType>(super::model);
            break;
        case L1:
            regul = new Lasso<LabelsType, PointerType>(super::model);
            break;
        case L1BALL:
            regul = new L1Ball<LabelsType, PointerType>(super::model);
            break;
        case L2BALL:
            regul = new L2Ball<LabelsType, PointerType>(super::model);
            break;
        case FUSEDLASSO:
            regul = new FusedLasso<LabelsType, PointerType>(super::model);
            break;
        case ELASTICNET:
            regul = new ElasticNet<LabelsType, PointerType>(super::model);
            break;
        case NONE:
            regul = new None<LabelsType, PointerType>(super::model);
            break;
        default:
            logging(logERROR) << "Not implemented, no regularization is chosen";
            regul = new None<LabelsType, PointerType>(super::model);
        }
        return regul;
    }

    Solver<LossType>* get_solver(const LossType& loss, const Regularizer<typename LossType::variable_type, typename LossType::index_type>& regul_tmp, const ParamSolver<typename LossType::value_type>& param)
    {
        Solver<LossType>* solver;
        solver_t solver_type = param.solver;
        Regularizer<typename LossType::variable_type, typename LossType::index_type>* regul = get_regul();

        if (solver_type == AUTO)
        {
            const FeatureType L = loss.lipschitz();
            const int n = loss.n();
            const FeatureType lambda_1 = regul->strong_convexity();
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
            solver = new ISTA_Solver<LossType>(loss, *regul, param);
            break;
        case QNING_ISTA:
            solver = new QNing<ISTA_Solver<LossType>>(loss, *regul, param);
            break;
        case CATALYST_ISTA:
            solver = new Catalyst<ISTA_Solver<LossType>>(loss, *regul, param);
            break;
        case FISTA:
            solver = new FISTA_Solver<LossType>(loss, *regul, param);
            break;
        case SVRG:
            solver = new SVRG_Solver<LossType>(loss, *regul, param);
            break;
        case MISO:
            solver = regul->strong_convexity() > 0 ? new MISO_Solver<LossType>(loss, *regul, param) : new Catalyst<MISO_Solver<LossType>>(loss, *regul, param);
            break;
        case SVRG_UNIFORM:
        {
            ParamSolver<typename LossType::value_type> param2 = param;
            param2.non_uniform_sampling = false;
            solver = new SVRG_Solver<LossType>(loss, *regul, param2);
            break;
        }
        case CATALYST_SVRG:
            solver = new Catalyst<SVRG_Solver<LossType>>(loss, *regul, param);
            break;
        case QNING_SVRG:
            solver = new QNing<SVRG_Solver<LossType>>(loss, *regul, param);
            break;
        case CATALYST_MISO:
            solver = new Catalyst<MISO_Solver<LossType>>(loss, *regul, param);
            break;
        case QNING_MISO:
            solver = new QNing<MISO_Solver<LossType>>(loss, *regul, param);
            break;
        case ACC_SVRG:
            solver = new Acc_SVRG_Solver<LossType>(loss, *regul, param);
            break;
        default:
            throw NotImplementedException("This solver is not implemented!");
            solver = NULL;
        }
        return solver;
    };


    LinearLossVec<InputMatrixType>* get_loss(DataLinear<InputMatrixType>& data, const Vector<typename InputMatrixType::value_type>& y) {
        LinearLossVec<InputMatrixType>* loss;
        switch (super::model.loss)
        {
        case SQUARE:
            loss = new SquareLoss<InputMatrixType>(data, y);
            break;
        case LOGISTIC:
            loss = new LogisticLoss<InputMatrixType>(data, y);
            break;
        case SQHINGE:
            loss = new SquaredHingeLoss<InputMatrixType>(data, y);
            break;
            // case HINGE: loss = new HingeLoss<InputMatrixType>(data,y); break;
        case SAFE_LOGISTIC:
            loss = new SafeLogisticLoss<InputMatrixType>(data, y);
            break;
        default:
            logging(logERROR) << "Not implemented, square loss is chosen by default";
            loss = new SquareLoss<InputMatrixType>(data, y);
        }
        return loss;
    }
};

#endif