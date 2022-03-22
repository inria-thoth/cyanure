#ifndef NORMS_H
#define NORMS_H

template <typename T>
struct normL2
{
public:
    typedef T value_type;
    normL2(const ParamModel<T> &model) : _lambda(model.lambda_1){};

    inline void prox(Vector<T> &x, const T thrs) const
    {
        const T nrm = x.nrm2();
        const T thrs2 = thrs * _lambda;
        if (nrm > thrs2)
        {
            x.scal((nrm - thrs2) / nrm);
        }
        else
        {
            x.setZeros();
        }
    };
    inline T eval(const Vector<T> &x) const
    {
        return _lambda * x.nrm2();
    };
    static inline void print()
    {
        logging(logINFO) << "L2";
    };
    inline T eval_dual(const Vector<T> &x) const
    {
        return x.nrm2() / _lambda;
    };
    static inline std::string getName() {return _name;};

private:
    const T _lambda;
    inline static const std::string _name = "L2";
};

template <typename T>
struct normLinf
{
public:
    typedef T value_type;
    normLinf(const ParamModel<T> &model) : _lambda(model.lambda_1){};

    inline void prox(Vector<T> &x, const T thrs) const
    {
        Vector<T> z;
        x.l1project(z, thrs * _lambda);
        x.sub(z);
    };
    inline T eval(const Vector<T> &x) const
    {
        return _lambda * x.fmaxval();
    };
    static inline void print()
    {
        logging(logINFO) << "LInf";
    }
    inline T eval_dual(const Vector<T> &x) const
    {
        return x.asum() / _lambda;
    };
    static inline std::string getName() {return _name;};

private:
    const T _lambda;
    inline static const std::string _name = "LInf";
};

template <typename T>
struct normL2_L1
{
public:
    typedef T value_type;
    normL2_L1(const ParamModel<T> &model) : _lambda(model.lambda_1), _lambda2(model.lambda_2){};

    inline void prox(Vector<T> &x, const T thrs) const
    {
        x.fastSoftThrshold(x, thrs * _lambda2);
        const T nrm = x.nrm2();
        const T thrs2 = thrs * _lambda;
        if (nrm > thrs2)
        {
            x.scal((nrm - thrs2) / nrm);
        }
        else
        {
            x.setZeros();
        }
    };
    inline T eval(const Vector<T> &x) const
    {
        return _lambda * x.nrm2() + _lambda2 * x.asum();
    };
    static inline void print()
    {
        logging(logINFO) << "L2+L1";
    };
    inline T eval_dual(const Vector<T> &x) const
    {
        Vector<T> sorted_x;
        sorted_x.copy(x);
        sorted_x.abs_vec();
        sorted_x.sort(false);
        const int n = x.n();
        T lambda_gamma = 0;
        T sum_sq = 0;
        T sum_lin = 0;
        for (int ii = 0; ii < n; ++ii)
        {
            lambda_gamma = sorted_x[ii];
            sum_lin += lambda_gamma;
            sum_sq += lambda_gamma * lambda_gamma;
            const T lambda_mu = _lambda * lambda_gamma / (_lambda2);
            if (sum_sq - 2 * lambda_gamma * sum_lin + (ii + 1) * lambda_gamma * lambda_gamma >= lambda_mu * lambda_mu)
            {
                sum_lin -= lambda_gamma;
                sum_sq -= lambda_gamma * lambda_gamma;
                const T dual = solve_binomial2((ii)*_lambda2 * _lambda2 - _lambda * _lambda, -2 * sum_lin * _lambda2, sum_sq);
                return dual;
            }
        }
        return 0;
    };
    static inline std::string getName() {return _name;};

private:
    const T _lambda;
    const T _lambda2;
    inline static const std::string _name = "L2+L1";
};

#endif