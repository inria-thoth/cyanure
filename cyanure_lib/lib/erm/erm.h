#ifndef ERM_H 
#define ERM_H


#include "../macro.h"
#include "../data_structure/structures/vector.h"
#include "../data_structure/structures/matrix.h"
#include "../data_structure/structures/sp_matrix.h"
#include "../data_structure/structures/sp_vector.h"
#include "../solvers/solver.h"
#include "../losses/loss_vec/loss_vec.h"
#include "../losses/loss_mat/loss_mat.h"
#include "../losses/loss_mat/loss/square_loss.h"
#include "../losses/loss_vec/loss/square_loss.h"
#include "../losses/loss_vec/loss/square_hinge_loss.h"
#include "../losses/loss_vec/loss/logistic_loss.h"
#include "../losses/loss_vec/loss/safe_logistic_loss.h"
#include "../losses/loss_mat/loss/multi_logistic_loss.h"
#include "../data_structure/data.h"


template <typename InputMatrixType>
class ERM {
public:

    ERM(OptimInfo<typename InputMatrixType::value_type>& optim_info, const ParamSolver<typename InputMatrixType::value_type>& param, const ParamModel<typename InputMatrixType::value_type>& model) : optim_info(optim_info), param(param), model(model) {
    }

protected:

    OptimInfo<typename InputMatrixType::value_type>& optim_info;
    const ParamSolver<typename InputMatrixType::value_type>& param;
    const ParamModel<typename InputMatrixType::value_type>& model;

    inline bool is_loss_for_matrices(const loss_t& loss) {
        return loss == SQUARE || loss == MULTI_LOGISTIC;
    };

    inline bool is_regression_loss(const loss_t& loss) {
        return loss == SQUARE;
    };

    inline bool is_regul_for_matrices(const regul_t& reg)
    {
        return reg == L1L2 || reg == L1LINF;
    }
};



#endif