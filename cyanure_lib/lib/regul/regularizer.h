#ifndef REGUL_H
#define REGUL_H

#include "../data_structure/linalg.h"
#include "../timer.h"
Timer timer_global, timer_global2, timer_global3;

enum regul_t
{
    L2,
    L1,
    ELASTICNET,
    L1BALL,
    L2BALL,
    FUSEDLASSO,
    L1L2,
    L1LINF,
    NONE,
    L1L2_L1,
    INCORRECT_REG
};

template <typename T>
struct ParamModel
{
    ParamModel()
    {
        regul = NONE;
        lambda_1 = 0;
        lambda_2 = 0;
        lambda_3 = 0;
        intercept = false;
        loss = SQUARE;
    };
    loss_t loss;
    regul_t regul;
    T lambda_1;
    T lambda_2;
    T lambda_3;
    bool intercept;
};

template <typename T>
void clean_param_model(ParamModel<T>& param)
{
    if (param.regul == FUSEDLASSO && param.lambda_1 == 0)
    {
        param.regul = ELASTICNET;
        param.lambda_1 = param.lambda_2;
        param.lambda_2 = param.lambda_3;
    };
    if (param.regul == ELASTICNET)
    {
        if (param.lambda_1 == 0)
        {
            param.regul = L2;
            param.lambda_1 = param.lambda_2;
        };
        if (param.lambda_2 == 0)
            param.regul = L1;
        if (param.lambda_1 == 0 && param.lambda_2 == 0)
            param.regul = NONE;
    }
    else
    {
        if (param.lambda_1 == 0)
            param.regul = NONE;
    }
}

static regul_t regul_from_string(char* regul)
{
    if (strcmp(regul, "l1") == 0)
        return L1;
    if (strcmp(regul, "l1-ball") == 0)
        return L1BALL;
    if (strcmp(regul, "fused-lasso") == 0)
        return FUSEDLASSO;
    if (strcmp(regul, "l2") == 0)
        return L2;
    if (strcmp(regul, "l2-ball") == 0)
        return L2BALL;
    if (strcmp(regul, "elasticnet") == 0)
        return ELASTICNET;
    if (strcmp(regul, "l1l2") == 0)
        return L1L2;
    if (strcmp(regul, "l1l2+l1") == 0)
        return L1L2_L1;
    if (strcmp(regul, "l1linf") == 0)
        return L1LINF;
    if (strcmp(regul, "none") == 0)
        return NONE;
    return INCORRECT_REG;
}

template <typename D, typename I>
class Regularizer
{
public:
    typedef typename D::value_type T;
    typedef I index_type;

    Regularizer(const ParamModel<T>& model) : _intercept(model.intercept), _id(model.regul) {};
    virtual ~Regularizer() {};

    virtual void prox(const D& input, D& output, const T eta) const = 0; // should be able to do inplace with output=input
    virtual T eval(const D& input) const = 0;
    virtual T fenchel(D& grad1, D& grad2) const = 0;
    virtual void print() const = 0;

    virtual bool is_lazy() const { return false; };
    virtual void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const {};

    virtual bool provides_fenchel() const { return true; };
    virtual regul_t id() const { return _id; };
    virtual bool intercept() const { return _intercept; };
    virtual T strong_convexity() const { return 0; };
    virtual T lambda_1() const { return 0; };
    virtual std::string getName() { return _name; };

protected:
    const bool _intercept;

private:
    explicit Regularizer<D, I>(const Regularizer<D, I>& reg);
    Regularizer<D, I>& operator=(const Regularizer<D, I>& reg);
    const regul_t _id;
    inline static const std::string _name = "Regulizer";
};

#endif
