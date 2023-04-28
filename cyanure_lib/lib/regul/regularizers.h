#ifndef REGULARIZERS_H
#define REGULARIZERS_H

template <typename D, typename I>
class None final : public Regularizer<D, I>
{
public:
    typedef typename D::value_type T;

    None(const ParamModel<T> &model) : Regularizer<D, I>(model){};
    virtual void prox(const D &input, D &output, const T eta) const
    {
        output.copy(input);
    };
    inline T eval(const D &input) const { return 0; };
    inline T fenchel(D &grad1, D &grad2) const { return 0; };
    bool provides_fenchel() const { return false; };
    void print() const
    {
        logging(logINFO) << "No regularization";
    }
};

template <typename D, typename I>
class Ridge final : public Regularizer<D, I>
{
public:
    typedef typename D::value_type T;

    Ridge(const ParamModel<T> &model) : Regularizer<D, I>(model), _lambda(model.lambda_1){};

    inline void prox(const D &input, D &output, const T eta) const
    {
        output.copy(input);
        output.scal(T(1.0 / (1.0 + _lambda * eta)));
        if (this->_intercept)
        {
            const int n = input.n();
            output[n - 1] = input[n - 1];
        }
    };
    inline T eval(const D &input) const
    {
        const int n = input.n();
        const T res = input.nrm2sq();
        return (this->_intercept ? T(0.5) * _lambda * (res - input[n - 1] * input[n - 1]) : T(0.5) * _lambda * res);
    };
    inline T fenchel(D &grad1, D &grad2) const
    {
        return (this->_intercept & (abs<T>(grad2[grad2.n() - 1]) >
                                    1e-6))
                   ? INFINITY
                   : this->eval(grad2) / (_lambda * _lambda);
    };
    void print() const
    {
        logging(logINFO) << getName();
    }
    virtual T strong_convexity() const { return this->_intercept ? 0 : _lambda; };
    virtual T lambda_1() const { return _lambda; };
    inline void lazy_prox(const D &input, D &output, const Vector<I> &indices, const T eta) const
    {
        const T scal = T(1.0) / (T(1.0) + _lambda * eta);
        const int p = input.n();
        const int r = indices.n();
        for (int jj = 0; jj < r; ++jj)
            output[indices[jj]] = scal * input[indices[jj]];
        if (this->_intercept)
            output[p - 1] = input[p - 1];
    };
    virtual bool is_lazy() const { return true; };
    static inline std::string getName() { return "L2 regularization"; };

private:
    const T _lambda;
};

template <typename D, typename I>
class Lasso final : public Regularizer<D, I>
{
public:
    typedef typename D::value_type T;

    Lasso(const ParamModel<T> &model) : Regularizer<D, I>(model), _lambda(model.lambda_1){};
    inline void prox(const D &input, D &output, const T eta) const
    {
        input.fastSoftThrshold(output, eta * _lambda);
        if (this->_intercept)
        {
            const int n = input.n();
            output[n - 1] = input[n - 1];
        }
    };
    inline T eval(const D &input) const
    {
        const int n = input.n();
        const T res = input.asum();
        return (this->_intercept ? _lambda * (res - abs<T>(input[n - 1])) : _lambda * res);
    };
    inline T fenchel(D &grad1, D &grad2) const
    {
        const T mm = grad2.fmaxval();
        if (mm > _lambda)
            grad1.scal(_lambda / mm);
        return (this->_intercept & (abs<T>(grad2[grad2.n() - 1]) >
                                    1e-6))
                   ? INFINITY
                   : 0;
    };
    void print() const
    {
        logging(logINFO) << getName();
    }
    virtual T lambda_1() const { return _lambda; };
    inline void lazy_prox(const D &input, D &output, const Vector<I> &indices, const T eta) const
    {
        const int p = input.n();
        const int r = indices.n();
        const T thrs = _lambda * eta;
        for (int jj = 0; jj < r; ++jj)
            output[indices[jj]] = fastSoftThrs(input[indices[jj]], thrs);
        ;
        if (this->_intercept)
            output[p - 1] = input[p - 1];
    };
    virtual bool is_lazy() const { return true; };
    static inline std::string getName() { return "L1 regularization"; };

private:
    const T _lambda;
};

template <typename D, typename I>
class ElasticNet final : public Regularizer<D, I>
{
public:
    typedef typename D::value_type T;

    // min_x 0.5|y-x|^2 + lambda_1 |x|  + 0.5 lambda_2 x^2
    // min_x - y x + 0.5 x^2  + lambda_1 |x|  + 0.5 lambda_2 x^2
    // min_x - y x + 0.5 (1+lambda_2) x^2  + lambda_1 |x|
    // min_x - y/(1+lambda_2) x + 0.5  x^2  + lambda_1/(1+lambda_2) |x|
    ElasticNet(const ParamModel<T> &model) : Regularizer<D, I>(model), _lambda(model.lambda_1), _lambda2(model.lambda_2){};
    inline void prox(const D &input, D &output, const T eta) const
    {
        output.copy(input);
        output.fastSoftThrshold(_lambda * eta);
        output.scal(T(1.0) / (1 + _lambda2 * eta));
        if (this->_intercept)
        {
            const int n = input.n();
            output[n - 1] = input[n - 1];
        }
    };
    inline T eval(const D &input) const
    {
        const int n = input.n();
        const T res = _lambda * input.asum() + T(0.5) * _lambda2 * input.nrm2sq();
        return (this->_intercept ? res - _lambda * abs<T>(input[n - 1]) - T(0.5) * _lambda2 * input[n - 1] * input[n - 1] : res);
    };
    // max_x xy - lambda_1 |x| - 0.5 lambda_2 x^2
    // - min_x - xy + lambda_1 |x| + 0.5 lambda_2 x^2
    // -(1/lambda_2) min_x - xy/lambda_2  + lambda_1/lambda_2 |x| + 0.5 x^2
    // x^* = prox_(l1 lambda_1/lambda_2) [ y/lambda_2]
    // x^* = prox_(l1 lambda_1) [ y] /_lambda2
    inline T fenchel(D &grad1, D &grad2) const
    {
        D tmp;
        tmp.copy(grad2);
        grad2.fastSoftThrshold(_lambda);
        const int n = grad2.n();
        T res0 = _lambda * grad2.asum() / _lambda2 + T(0.5) * grad2.nrm2sq() / _lambda2;
        if (this->_intercept)
            res0 -= _lambda * abs<T>(grad2[n - 1]) / _lambda2 - T(0.5) * grad2[n - 1] * grad2[n - 1] / _lambda2;
        const T res = tmp.dot(grad2) / _lambda2 - res0;
        return (this->_intercept & (abs<T>(tmp[tmp.n() - 1]) >
                                    1e-6))
                   ? INFINITY
                   : res;
    };
    void print() const
    {
        logging(logINFO) << getName();
    }
    virtual T strong_convexity() const { return this->_intercept ? 0 : _lambda2; };
    virtual T lambda_1() const { return _lambda; };
    inline void lazy_prox(const D &input, D &output, const Vector<I> &indices, const T eta) const
    {
        const int p = input.n();
        const int r = indices.n();
        const T thrs = _lambda * eta;
        const T scal = T(1.0) / (T(1.0) + _lambda2 * eta);
        for (int jj = 0; jj < r; ++jj)
            output[indices[jj]] = scal * fastSoftThrs(input[indices[jj]], thrs);
        ;
        if (this->_intercept)
            output[p - 1] = input[p - 1];
    };
    virtual bool is_lazy() const { return true; };
    static inline std::string getName() { return "Elastic Net regularization"; };

private:
    const T _lambda;
    const T _lambda2;
};

template <typename D, typename I>
class L1Ball final : public Regularizer<D, I>
{
public:
    typedef typename D::value_type T;

    L1Ball(const ParamModel<T> &model) : Regularizer<D, I>(model), _lambda(model.lambda_1){};
    inline void prox(const D &input, D &output, const T eta) const
    {
        D tmp;
        tmp.copy(input);
        if (this->_intercept)
        {
            tmp[tmp.n() - 1] = 0;
            tmp.sparseProject(output, _lambda, 1, 0, 0, 0, false);
            output[output.n() - 1] = input[output.n() - 1];
        }
        else
        {
            tmp.sparseProject(output, _lambda, 1, 0, 0, 0, false);
        }
    };
    inline T eval(const D &input) const { return 0; };
    inline T fenchel(D &grad1, D &grad2) const
    {
        Vector<T> output;
        output.copy(grad2);
        if (this->_intercept)
            output[output.n() - 1] = 0;
        return _lambda * (output.fmaxval());
    };
    void print() const
    {
        logging(logINFO) << getName();
    }
    virtual T lambda_1() const { return _lambda; };
    static inline std::string getName() { return "L1 ball regularization"; };

private:
    const T _lambda;
};

template <typename D, typename I>
class L2Ball final : public Regularizer<D, I>
{
public:
    typedef typename D::value_type T;

    L2Ball(const ParamModel<T> &model) : Regularizer<D, I>(model), _lambda(model.lambda_1){};
    inline void prox(const D &input, D &output, const T eta) const
    {
        D tmp;
        tmp.copy(input);
        if (this->_intercept)
        {
            tmp[tmp.n() - 1] = 0;
            const T nrm = tmp.nrm2();
            if (nrm > _lambda)
                tmp.scal(_lambda / nrm);
            output[output.n() - 1] = input[output.n() - 1];
        }
        else
        {
            const T nrm = tmp.nrm2();
            if (nrm > _lambda)
                tmp.scal(_lambda / nrm);
        }
    };
    inline T eval(const D &input) const { return 0; };
    inline T fenchel(D &grad1, D &grad2) const
    {
        Vector<T> output;
        output.copy(grad2);
        if (this->_intercept)
            output[output.n() - 1] = 0;
        return _lambda * (output.nrm2());
    };
    void print() const
    {
        logging(logINFO) << getName();
    }
    virtual T lambda_1() const { return _lambda; };
    static inline std::string getName() { return "L2 ball regularization"; };

private:
    const T _lambda;
};

template <typename D, typename I>
class FusedLasso final : public Regularizer<D, I>
{
public:
    typedef typename D::value_type T;

    FusedLasso(const ParamModel<T> &model) : Regularizer<D, I>(model), _lambda(model.lambda_1), _lambda2(model.lambda_2), _lambda3(model.lambda_3){};
    inline void prox(const D &x, D &output, const T eta) const
    {
        output.resize(x.n());
        Vector<T> copyx;
        copyx.copy(x);
        copyx.fusedProjectHomotopy(output, _lambda2, _lambda, _lambda3, true);
    };
    inline T eval(const D &x) const
    {
        T sum = T();
        const int maxn = this->_intercept ? x.n() - 1 : x.n();
        for (int i = 0; i < maxn - 1; ++i)
            sum += _lambda3 * abs(x[i + 1] - x[i]) + _lambda * abs(x[i]) + T(0.5) * _lambda2 * x[i] * x[i];
        sum += _lambda2 * abs(x[maxn - 1]) + 0.5 * _lambda3 * x[maxn - 1] * x[maxn - 1];
        return sum;
    };
    inline T fenchel(D &grad1, D &grad2) const { return 0; };
    void print() const
    {
        logging(logINFO) << getName();
    }
    bool provides_fenchel() const { return false; };
    virtual T strong_convexity() const { return this->_intercept ? 0 : _lambda3; };
    virtual T lambda_1() const { return _lambda; };
    static inline std::string getName() { return "Fused Lasso regularization"; };

private:
    const T _lambda;
    const T _lambda2;
    const T _lambda3;
};

#endif
