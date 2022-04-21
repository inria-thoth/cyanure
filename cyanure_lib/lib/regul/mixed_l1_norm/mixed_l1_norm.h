#ifndef MIXED_L1_NORM_H
#define MIXED_L1_NORM_H

#include "norms.h"

template <typename N, typename I>
class MixedL1LN final : public Regularizer<Matrix<typename N::value_type>, I>
{
public:
    typedef typename N::value_type T;
    typedef Matrix<T> D;
    MixedL1LN(const ParamModel<T> &model, const int nclass, const bool transpose) : Regularizer<D, I>(model), _transpose(transpose), _lambda(model.lambda_1), _norm(model){};
    inline void prox(const D &x, D &y, const T eta) const
    {
        const int n = x.n();
        const int m = x.m();
        y.copy(x);
        if (_transpose)
        {
            const int nn = this->_intercept ? n - 1 : n;
#pragma omp parallel for
            for (int i = 0; i < nn; ++i)
            {
                Vector<T> col;
                y.refCol(i, col);
                _norm.prox(col, eta);
            }
        }
        else
        {
            const int nn = this->_intercept ? m - 1 : m;
#pragma omp parallel for
            for (int i = 0; i < nn; ++i)
            {
                Vector<T> row;
                y.copyRow(i, row);
                _norm.prox(row, eta);
                y.copyToRow(i, row);
            }
        }
    };
    T inline eval(const D &x) const
    {
        T sum = 0;
        const int n = x.n();
        const int m = x.m();
        if (_transpose)
        {
            const int nn = this->_intercept ? n - 1 : n;
#pragma omp parallel for reduction(+ \
                                   : sum)
            for (int i = 0; i < nn; ++i)
            {
                Vector<T> col;
                x.refCol(i, col);
                sum += _norm.eval(col);
            }
        }
        else
        {
            const int nn = this->_intercept ? m - 1 : m;
#pragma omp parallel for reduction(+ \
                                   : sum)
            for (int i = 0; i < nn; ++i)
            {
                Vector<T> row;
                x.copyRow(i, row);
                sum += _norm.eval(row);
            }
        }
        return sum;
    }
    // grad1 is nclasses * n
    inline T fenchel(D &grad1, D &grad2) const
    {
        const int n = grad2.n();
        const int m = grad2.m();
        T res = 0;
        T mm = 0;
        if (_transpose)
        {
            const int nn = this->_intercept ? n - 1 : n;
            for (int i = 0; i < nn; ++i)
            {
                Vector<T> col;
                grad2.refCol(i, col);
                mm = MAX(_norm.eval_dual(col), mm);
            }
            Vector<T> col;
            if (this->_intercept)
            {
                grad2.refCol(nn, col);
                if (col.nrm2sq() > T(1e-7))
                    res = INFINITY;
            }
        }
        else
        {
            const int nn = this->_intercept ? m - 1 : m;
            for (int i = 0; i < nn; ++i)
            {
                Vector<T> row;
                grad2.copyRow(i, row);
                mm = MAX(_norm.eval_dual(row), mm);
            }
            Vector<T> col;
            if (this->_intercept)
            {
                grad2.copyRow(nn, col);
                if (col.nrm2sq() > T(1e-7))
                    res = INFINITY;
            }
        }
        if (mm > T(1.0))
            grad1.scal(T(1.0) / mm);
        return res;
    };

    void print() const
    {
        logging(logINFO) << "Mixed L1-" << N::getName() << " norm regularization";
    }
    inline T lambda_1() const { return _lambda; };
    inline void lazy_prox(const D &input, D &output, const Vector<I> &indices, const T eta) const
    {
        output.resize(input.m(), input.n());
        const int r = indices.n();
        const int m = input.m();
        const int n = input.n();
        if (_transpose)
        {
#pragma omp parallel for
            for (int i = 0; i < r; ++i)
            {
                const int ind = indices[i];
                Vector<T> col, col1;
                input.refCol(ind, col1);
                output.refCol(ind, col);
                col.copy(col1);
                _norm.prox(col, eta);
            }
            if (this->_intercept)
            {
                Vector<T> col, col1;
                input.refCol(n - 1, col1);
                output.refCol(n - 1, col);
                col.copy(col1);
            }
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < r; ++i)
            {
                const int ind = indices[i];
                Vector<T> col;
                input.copyRow(ind, col);
                _norm.prox(col, eta);
                output.copyToRow(ind, col);
            }
            if (this->_intercept)
            {
                Vector<T> col;
                input.copyRow(m - 1, col);
                output.copyToRow(m - 1, col);
            }
        }
    };
    virtual bool is_lazy() const { return true; };

private:
    const bool _transpose;
    const T _lambda;
    N _norm;
};

template <typename T, typename I>
using MixedL1L2 = MixedL1LN<normL2<T>, I>;

template <typename T, typename I>
using MixedL1Linf = MixedL1LN<normLinf<T>, I>;

template <typename T, typename I>
using MixedL1L2_L1 = MixedL1LN<normL2_L1<T>, I>;

#endif