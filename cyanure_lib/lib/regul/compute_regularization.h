#ifndef COMPUTE_REGULARIZATION_H
#define COMPUTE_REGULARIZATION_H

template <typename Reg>
class RegMat final : public Regularizer<Matrix<typename Reg::T>, typename Reg::index_type>
{
public:
    typedef typename Reg::T T;
    typedef typename Reg::index_type I;
    RegMat(const ParamModel<T>& model, const int num_cols, const bool transpose) : Regularizer<Matrix<T>, I>(model), _N(num_cols), _transpose(transpose)
    {
        _regs = new Reg * [_N];
        for (int i = 0; i < _N; ++i)
            _regs[i] = new Reg(model);
    };
    virtual ~RegMat()
    {
        for (int i = 0; i < _N; ++i)
        {
            delete (_regs[i]);
            _regs[i] = NULL;
        }
        delete[](_regs);
    };
    void inline prox(const Matrix<T>& x, Matrix<T>& y, const T eta) const
    {
        y.copy(x);
        int i;
// #pragma omp parallel for private(i)
        for (i = 0; i < _N; ++i)
        {
            Vector<T> colx, coly;
            if (_transpose)
            {
                x.copyRow(i, colx);
                y.copyRow(i, coly);
            }
            else
            {
                x.refCol(i, colx);
                y.refCol(i, coly);
            }
            _regs[i]->prox(colx, coly, eta);
            if (_transpose)
                y.copyToRow(i, coly);
        }
    };
    T inline eval(const Matrix<T>& x) const
    {
        T sum = 0;
// #pragma omp parallel for reduction(+ \
                                   : sum)
        for (int i = 0; i < _N; ++i)
        {
            Vector<T> col;
            if (_transpose)
            {
                x.copyRow(i, col);
            }
            else
            {
                x.refCol(i, col);
            }
            const T val = _regs[i]->eval(col);
            sum += val;
        }
        return sum;
    };
    T inline fenchel(Matrix<T>& grad1, Matrix<T>& grad2) const
    {
        T sum = 0;
// #pragma omp parallel for reduction(+ \
                                   : sum)
        for (int i = 0; i < _N; ++i)
        {
            Vector<T> col1, col2;
            if (_transpose)
            {
                grad1.copyRow(i, col1);
                grad2.copyRow(i, col2);
            }
            else
            {
                grad1.refCol(i, col1);
                grad2.refCol(i, col2);
            }
            const T fench = _regs[i]->fenchel(col1, col2);
            sum += fench;
            if (_transpose)
            {
                grad1.copyToRow(i, col1);
                grad2.copyToRow(i, col2);
            }
        }
        return sum;
    };
    virtual bool provides_fenchel() const
    {
        bool ok = true;
        for (int i = 0; i < _N; ++i)
            ok = ok && _regs[i]->provides_fenchel();
        return ok;
    };
    void print() const
    {
        logging(logINFO) << "Regularization for matrices";
        _regs[0]->print();
    };
    virtual T lambda_1() const { return _regs[0]->lambda_1(); };
    inline void lazy_prox(const Matrix<T>& input, Matrix<T>& output, const Vector<I>& indices, const T eta) const
    {
// #pragma omp parallel for
        for (int i = 0; i < _N; ++i)
        {
            Vector<T> colx, coly;
            output.refCol(i, coly);
            if (_transpose)
            {
                input.copyRow(i, colx);
            }
            else
            {
                input.refCol(i, colx);
            }
            _regs[i]->lazy_prox(colx, coly, indices, eta);
        }
    };
    virtual bool is_lazy() const { return _regs[0]->is_lazy(); };

protected:
    int _N;
    Reg** _regs;
    bool _transpose;
};

template <typename Reg>
class RegVecToMat final : public Regularizer<Matrix<typename Reg::T>, typename Reg::index_type>
{
public:
    typedef typename Reg::T T;
    typedef typename Reg::index_type I;
    typedef Matrix<T> D;
    RegVecToMat(const ParamModel<T>& model) : Regularizer<D, I>(model), _intercept(model.intercept)
    {
        ParamModel<T> model2 = model;
        model2.intercept = false;
        _reg = new Reg(model2);
    };
    ~RegVecToMat() { delete (_reg); };

    inline void prox(const D& input, D& output, const T eta) const
    {
        Vector<T> w1, w2, b1, b2;
        output.resize(input.m(), input.n());
        get_wb(input, w1, b1);
        get_wb(output, w2, b2);
        _reg->prox(w1, w2, eta);
        if (_intercept)
            b2.copy(b1);
    };
    inline T eval(const D& input) const
    {
        Vector<T> w, b;
        get_wb(input, w, b);
        return _reg->eval(w);
    }
    inline T fenchel(D& grad1, D& grad2) const
    {
        Vector<T> g1;
        grad1.toVect(g1);
        Vector<T> w, b;
        get_wb(grad2, w, b);
        return (this->_intercept && ((b.nrm2sq()) > 1e-7) ? INFINITY : _reg->fenchel(g1, w));
    };
    void print() const
    {
        _reg->print();
    }
    virtual T strong_convexity() const
    {
        return _intercept ? 0 : _reg->strong_convexity();
    };
    virtual T lambda_1() const { return _reg->lambda_1(); };
    inline void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const
    {
        Vector<T> w1, w2, b1, b2;
        output.resize(input.m(), input.n());
        get_wb(input, w1, b1);
        get_wb(output, w2, b2);
        _reg->lazy_prox(w1, w2, indices, eta);
        if (_intercept)
            b2.copy(b1);
    };
    virtual bool is_lazy() const { return _reg->is_lazy(); };

private:
    inline void get_wb(const Matrix<T>& input, Vector<T>& w, Vector<T>& b) const
    {
        const int p = input.n();
        Matrix<T> W;
        if (_intercept)
        {
            input.refSubMat(0, p - 1, W);
            input.refCol(p - 1, b);
        }
        else
        {
            input.refSubMat(0, p, W);
        }
        W.toVect(w);
    };
    Reg* _reg;
    const bool _intercept;
};

#endif