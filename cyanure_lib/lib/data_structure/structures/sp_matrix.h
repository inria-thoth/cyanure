#ifndef SP_MATRIX_H
#define SP_MATRIX_H

#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif


#include "../declare_structures.h"


/// Sparse Matrix class, CSC format
template<typename floating_type, typename I> class SpMatrix {
    friend class Matrix<floating_type>;
    friend class SpVector<floating_type, I>;
public:
    typedef floating_type value_type;
    typedef SpVector<floating_type, I> col_type;
    typedef I index_type;
    /// Constructor, CSC format, existing data
    SpMatrix(floating_type* v, I* r, I* pB, I* pE, I m, I n, I nzmax);
    /// Constructor, new m x n matrix, with at most nzmax non-zeros values
    SpMatrix(I m, I n, I nzmax);
    /// Empty constructor
    SpMatrix();

    /// Destructor
    ~SpMatrix();

    /// Accessors
    /// reference the column i Io vec
    inline void refCol(I i, SpVector<floating_type, I>& vec) const;
    /// returns pB[i]
    inline I pB(const I i) const { return _pB[i]; };
    /// returns r[i]
    inline I r(const I i) const { return _r[i]; };
    /// returns v[i]
    inline floating_type v(const I i) const { return _v[i]; };
    /// returns the maximum number of non-zero elements
    inline I nzmax() const { return _nzmax; };
    /// returns the number of rows
    inline I n() const { return _n; };
    /// returns the number of columns
    inline I m() const { return _m; };
    /// returns the number of columns
    inline I V() const { return 1; };
    /// returns X[index]
    inline floating_type operator[](const I index) const;
    void getData(Vector<floating_type>& data, const I index) const;
    void setData(floating_type* v, I* r, I* pB, I* pE, I m, I n, I nzmax);

    /// print the sparse matrix
    inline void print(const std::string& name) const;
    /// compute the sum of the matrix elements
    inline floating_type asum() const;
    /// compute the sum of the matrix elements
    inline floating_type normFsq() const;
    /// Direct access to _pB
    inline I* pB() const { return _pB; };
    /// Direct access to _pE
    inline I* pE() const { return _pE; };
    /// Direct access to _r
    inline I* r() const { return _r; };
    /// Direct access to _v
    inline floating_type* v() const { return _v; };
    /// number of nonzeros elements
    inline I nnz() const { return _pB[_n]; };
    inline void add_direct(const SpMatrix<floating_type, I>& mat, const floating_type a);
    inline void copy_direct(const SpMatrix<floating_type, I>& mat);
    inline floating_type dot_direct(const SpMatrix<floating_type, I>& mat) const;

    /// Modifiers
    /// clear the matrix
    inline void clear();
    /// resize the matrix
    inline void resize(const I m, const I n, const I nzmax);
    /// scale the matrix by a
    inline void scal(const floating_type a) const;
    inline floating_type abs_mean() const;

    /// Algebraic operations
    /// aat <- A(:,indices)*A(:,indices)'
    inline void AAt(Matrix<floating_type>& aat, const Vector<I>& indices) const;
    /// XAt <- X*A'
    inline void XAt(const Matrix<floating_type>& X, Matrix<floating_type>& XAt) const;
    /// XAt <- X(:,indices)*A(:,indices)'
    inline void XAt(const Matrix<floating_type>& X, Matrix<floating_type>& XAt,
        const Vector<I>& indices) const;
    /// XAt <- sum_i w_i X(:,i)*A(:,i)'
    inline void wXAt(const Vector<floating_type>& w, const Matrix<floating_type>& X,
        Matrix<floating_type>& XAt, const int numthreads = -1) const;
    inline void XtX(Matrix<floating_type>& XtX) const;

    /// y <- A'*x
    inline void multTrans(const Vector<floating_type>& x, Vector<floating_type>& y,
        const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
    inline void multTrans(const SpVector<floating_type, I>& x, Vector<floating_type>& y,
        const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
    /// perform b = alpha*A*x + beta*b, when x is sparse
    inline void mult(const SpVector<floating_type, I>& x, Vector<floating_type>& b,
        const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
    /// perform b = alpha*A*x + beta*b, when x is sparse
    inline void mult(const Vector<floating_type>& x, Vector<floating_type>& b,
        const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
    /// perform C = a*A*B + b*C, possibly transposing A or B.
    inline void mult(const Matrix<floating_type>& B, Matrix<floating_type>& C,
        const bool transA = false, const bool transB = false,
        const floating_type a = 1.0, const floating_type b = 0.0) const;
    /// perform C = a*B*A + b*C, possibly transposing A or B.
    inline void multSwitch(const Matrix<floating_type>& B, Matrix<floating_type>& C,
        const bool transA = false, const bool transB = false,
        const floating_type a = 1.0, const floating_type b = 0.0) const;
    /// perform C = a*B*A + b*C, possibly transposing A or B.
    inline void mult(const SpMatrix<floating_type, I>& B, Matrix<floating_type>& C, const bool transA = false,
        const bool transB = false, const floating_type a = 1.0,
        const floating_type b = 0.0) const;
    /// make a copy of the matrix mat in the current matrix
    inline void copyTo(Matrix<floating_type>& mat) const { this->toFull(mat); };
    /// dot product;
    inline floating_type dot(const Matrix<floating_type>& x) const;
    inline void copyRow(const I i, Vector<floating_type>& x) const;
    inline void sum_cols(Vector<floating_type>& sum) const;
    inline void copy(const SpMatrix<floating_type, I>& mat);

    /// Conversions
    /// copy the sparse matrix into a dense matrix
    inline void toFull(Matrix<floating_type>& matrix) const;
    /// copy the sparse matrix into a dense transposed matrix
    inline void toFullTrans(Matrix<floating_type>& matrix) const;

    /// use the data from v, r for _v, _r
    inline void convert(const Matrix<floating_type>& v, const Matrix<I>& r,
        const I K);
    /// use the data from v, r for _v, _r
    inline void convert2(const Matrix<floating_type>& v, const Vector<I>& r,
        const I K);
    inline void normalize();
    inline void normalize_rows();
    /// returns the l2 norms ^2 of the columns
    inline void norm_2sq_cols(Vector<floating_type>& norms) const;
    /// returns the l0 norms of the columns
    inline void norm_0_cols(Vector<floating_type>& norms) const;
    /// returns the l1 norms of the columns
    inline void norm_1_cols(Vector<floating_type>& norms) const;
    inline void addVecToCols(const Vector<floating_type>& diag, const floating_type a = 1.0);
    inline void addVecToColsWeighted(const Vector<floating_type>& diag, const floating_type* weights, const floating_type a = 1.0);

    typedef SpVector<floating_type, I> col;
    static const bool is_sparse = true;

private:
    /// forbid copy constructor
    explicit SpMatrix(const SpMatrix<floating_type, I>& matrix);
    SpMatrix<floating_type, I>& operator=(const SpMatrix<floating_type, I>& matrix);

    /// if the data has been externally allocated
    bool _externAlloc;
    /// data
    floating_type* _v;
    /// row indices 
    I* _r;
    /// indices of the beginning of columns
    I* _pB;
    /// indices of the end of columns
    I* _pE;
    /// number of rows
    I _m;
    /// number of columns
    I _n;
    /// number of non-zero values
    I _nzmax;
};



/* ****************************
 * Implementation of SpMatrix
 * ****************************/


 /// Constructor, CSC format, existing data
template <typename floating_type, typename I> SpMatrix<floating_type, I>::SpMatrix(floating_type* v, I* r, I* pB, I* pE,
    I m, I n, I nzmax) :
    _externAlloc(true), _v(v), _r(r), _pB(pB), _pE(pE), _m(m), _n(n), _nzmax(nzmax)
{ };

/// Constructor, new m x n matrix, with at most nzmax non-zeros values
template <typename floating_type, typename I> SpMatrix<floating_type, I>::SpMatrix(I m, I n, I nzmax) :
    _externAlloc(false), _m(m), _n(n), _nzmax(nzmax) {
#pragma omp critical
        {
            _v = new floating_type[nzmax];
            _r = new I[nzmax];
            _pB = new I[_n + 1];
        }
        _pE = _pB + 1;
};

/// Empty constructor
template <typename floating_type, typename I> SpMatrix<floating_type, I>::SpMatrix() :
    _externAlloc(true), _v(NULL), _r(NULL), _pB(NULL), _pE(NULL),
    _m(0), _n(0), _nzmax(0) { };


template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::copy(const SpMatrix<floating_type, I>& mat) {
    this->resize(mat._m, mat._n, mat._nzmax);
    memcpy(_v, mat._v, _nzmax * sizeof(floating_type));
    memcpy(_r, mat._r, _nzmax * sizeof(I));
    memcpy(_pB, mat._pB, (_n + 1) * sizeof(I));
}


/// Destructor
template <typename floating_type, typename I> SpMatrix<floating_type, I>::~SpMatrix() {
    clear();
};

/// reference the column i Io vec
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::refCol(I i,
    SpVector<floating_type, I>& vec) const {
    if (vec._nzmax > 0) vec.clear();
    vec._v = _v + _pB[i];
    vec._r = _r + _pB[i];
    vec._externAlloc = true;
    vec._L = _pE[i] - _pB[i];
    vec._nzmax = vec._L;
};

/// print the sparse matrix
template<typename floating_type, typename I> inline void SpMatrix<floating_type, I>::print(const std::string& name) const {
    logging(logERROR) << name;
    logging(logERROR) << _m << " x " << _n << " , " << _nzmax;
    for (I i = 0; i < _n; ++i) {
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            logging(logERROR) << "(" << _r[j] << "," << i << ") = " << _v[j];
        }
    }
};

template<typename floating_type, typename I>
inline floating_type SpMatrix<floating_type, I>::operator[](const I index) const {
    const I num_col = (index / _m);
    const I num_row = index - num_col * _m;
    floating_type val = 0;
    for (I j = _pB[num_col]; j < _pB[num_col + 1]; ++j) {
        if (_r[j] == num_row) {
            val = _v[j];
            break;
        }
    }
    return val;
};
template<typename floating_type, typename I>
void SpMatrix<floating_type, I>::getData(Vector<floating_type>& data, const I index) const {
    data.resize(_m);
    data.setZeros();
    for (I i = _pB[index]; i < _pB[index + 1]; ++i)
        data[_r[i]] = _v[i];
};

template <typename floating_type, typename I>
void SpMatrix<floating_type, I>::setData(floating_type* v, I* r, I* pB, I* pE, I m, I n, I nzmax) {
    this->clear();
    _externAlloc = true;
    _v = v;
    _r = r;
    _pB = pB;
    _pE = pE;
    _m = m;
    _n = n;
    _nzmax = nzmax;
}

/// compute the sum of the matrix elements
template <typename floating_type, typename I> inline floating_type SpMatrix<floating_type, I>::asum() const {
    return cblas_asum<floating_type>(_pB[_n], _v, 1);
};

/// compute the sum of the matrix elements
template <typename floating_type, typename I> inline floating_type SpMatrix<floating_type, I>::normFsq() const {
    return cblas_dot<floating_type>(_pB[_n], _v, 1, _v, 1);
};

template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::add_direct(const SpMatrix<floating_type, I>& mat, const floating_type a) {
    Vector<floating_type> v2(mat._v, mat._nzmax);
    Vector<floating_type> v1(_v, _nzmax);
    v1.add(v2, a);
}

template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::copy_direct(const SpMatrix<floating_type, I>& mat) {
    Vector<floating_type> v2(mat._v, _pB[_n]);
    Vector<floating_type> v1(_v, _pB[_n]);
    v1.copy(v2);
}

template <typename floating_type, typename I>
inline floating_type SpMatrix<floating_type, I>::dot_direct(const SpMatrix<floating_type, I>& mat) const {
    Vector<floating_type> v2(mat._v, _pB[_n]);
    Vector<floating_type> v1(_v, _pB[_n]);
    return v1.dot(v2);
}

/// clear the matrix
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::clear() {
    if (!_externAlloc) {
        delete[](_r);
        delete[](_v);
        delete[](_pB);
    }
    _n = 0;
    _m = 0;
    _nzmax = 0;
    _v = NULL;
    _r = NULL;
    _pB = NULL;
    _pE = NULL;
    _externAlloc = true;
};

/// resize the matrix
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::resize(const I m,
    const I n, const I nzmax) {
    if (n == _n && m == _m && nzmax == _nzmax) return;
    this->clear();
    _n = n;
    _m = m;
    _nzmax = nzmax;
    _externAlloc = false;
#pragma omp critical
    {
        _v = new floating_type[nzmax];
        _r = new I[nzmax];
        _pB = new I[_n + 1];
    }
    _pE = _pB + 1;
    for (I i = 0; i <= _n; ++i) _pB[i] = 0;
};

/// resize the matrix
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::scal(const floating_type a) const {
    cblas_scal<floating_type>(_pB[_n], a, _v, 1);
};

///// resize the matrix
template <typename floating_type, typename I> inline floating_type SpMatrix<floating_type, I>::abs_mean() const {
    Vector<floating_type> vec(_v, _pB[_n]);
    return vec.abs_mean();
};


/// y <- A'*x
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::multTrans(const Vector<floating_type>& x, Vector<floating_type>& y,
    const floating_type alpha, const floating_type beta) const {
    y.resize(_n);
    if (beta) {
        y.scal(beta);
    }
    else {
        y.setZeros();
    }
    const floating_type* prX = x.rawX();
#pragma omp parallel for
    for (I i = 0; i < _n; ++i) {
        floating_type sum = floating_type();
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            sum += _v[j] * prX[_r[j]];
        }
        y[i] += alpha * sum;
    }
};

/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::multTrans(const SpVector<floating_type, I>& x, Vector<floating_type>& y,
    const floating_type alpha, const floating_type beta) const {
    y.resize(_n);
    if (beta) {
        y.scal(beta);
    }
    else {
        y.setZeros();
    }
    floating_type* prY = y.rawX();
    SpVector<floating_type, I> col;
    for (I i = 0; i < _n; ++i) {
        this->refCol(i, col);
        prY[i] += alpha * x.dot(col);
    }
};


/// y <- A*x
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::mult(const Vector<floating_type>& x, Vector<floating_type>& y,
    const floating_type alpha, const floating_type beta) const {
    y.resize(_m);
    if (beta) {
        y.scal(beta);
    }
    else {
        y.setZeros();
    }
    const floating_type* prX = x.rawX();
    for (I i = 0; i < _n; ++i) {
        floating_type sca = alpha * prX[i];
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            y[_r[j]] += sca * _v[j];
        }
    }
};


/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::mult(const SpVector<floating_type, I>& x, Vector<floating_type>& y,
    const floating_type alpha, const floating_type beta) const {
    y.resize(_m);
    if (beta) {
        y.scal(beta);
    }
    else {
        y.setZeros();
    }
    floating_type* prY = y.rawX();
    for (I i = 0; i < x.L(); ++i) {
        I ind = x.r(i);
        floating_type val = alpha * x.v(i);
        for (I j = _pB[ind]; j < _pE[ind]; ++j) {
            prY[_r[j]] += val * _v[j];
        }
    }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::mult(const Matrix<floating_type>& B, Matrix<floating_type>& C,
    const bool transA, const bool transB,
    const floating_type a, const floating_type b) const {
    if (transA) {
        if (transB) {
            C.resize(_n, B.m());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            SpVector<floating_type, I> tmp;
            Vector<floating_type> row(B.m());
            for (I i = 0; i < _n; ++i) {
                this->refCol(i, tmp);
                B.mult(tmp, row);
                C.addRow(i, row, a);
            }
        }
        else {
            C.resize(_n, B.n());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            SpVector<floating_type, I> tmp;
            Vector<floating_type> row(B.n());
            for (I i = 0; i < _n; ++i) {
                this->refCol(i, tmp);
                B.multTrans(tmp, row);
                C.addRow(i, row, a);
            }
        }
    }
    else {
        if (transB) {
            C.resize(_m, B.m());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            Vector<floating_type> row(B.n());
            Vector<floating_type> col;
            for (I i = 0; i < B.m(); ++i) {
                B.copyRow(i, row);
                C.refCol(i, col);
                this->mult(row, col, a, floating_type(1.0));
            }
        }
        else {
            C.resize(_m, B.n());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            Vector<floating_type> colB;
            Vector<floating_type> colC;
            for (I i = 0; i < B.n(); ++i) {
                B.refCol(i, colB);
                C.refCol(i, colC);
                this->mult(colB, colC, a, floating_type(1.0));
            }
        }
    }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::mult(const SpMatrix<floating_type, I>& B, Matrix<floating_type>& C,
    const bool transA, const bool transB,
    const floating_type a, const floating_type b) const {
    if (transA) {
        if (transB) {
            C.resize(_n, B.m());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            SpVector<floating_type, I> tmp;
            Vector<floating_type> row(B.m());
            for (I i = 0; i < _n; ++i) {
                this->refCol(i, tmp);
                B.mult(tmp, row);
                C.addRow(i, row, a);
            }
        }
        else {
            C.resize(_n, B.n());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            SpVector<floating_type, I> tmp;
            Vector<floating_type> row(B.n());
            for (I i = 0; i < _n; ++i) {
                this->refCol(i, tmp);
                B.multTrans(tmp, row);
                C.addRow(i, row, a);
            }
        }
    }
    else {
        if (transB) {
            C.resize(_m, B.m());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            SpVector<floating_type, I> colB;
            SpVector<floating_type, I> colA;
            for (I i = 0; i < _n; ++i) {
                this->refCol(i, colA);
                B.refCol(i, colB);
                C.rank1Update(colA, colB, a);
            }
        }
        else {
            C.resize(_m, B.n());
            if (b) {
                C.scal(b);
            }
            else {
                C.setZeros();
            }
            SpVector<floating_type, I> colB;
            Vector<floating_type> colC;
            for (I i = 0; i < B.n(); ++i) {
                B.refCol(i, colB);
                C.refCol(i, colC);
                this->mult(colB, colC, a);
            }
        }
    }
};

/// perform C = a*B*A + b*C, possibly transposing A or B.
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::multSwitch(const Matrix<floating_type>& B, Matrix<floating_type>& C,
    const bool transA, const bool transB,
    const floating_type a, const floating_type b) const {
    B.mult(*this, C, transB, transA, a, b);
};

template <typename floating_type, typename I>
inline floating_type SpMatrix<floating_type, I>::dot(const Matrix<floating_type>& x) const {
    floating_type sum = 0;
    for (I i = 0; i < _n; ++i)
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            sum += _v[j] * x(_r[j], j);
        }
    return sum;
};


template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::copyRow(const I ind, Vector<floating_type>& x) const {
    x.resize(_n);
    x.setZeros();
    for (I i = 0; i < _n; ++i) {
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            if (_r[j] == ind) {
                x[i] = _v[j];
            }
            else if (_r[j] > ind) {
                break;
            }
        }
    }
};

template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::addVecToCols(
    const Vector<floating_type>& vec, const floating_type a) {
    const floating_type* pr_vec = vec.rawX();
    if (isEqual(a, floating_type(1.0))) {
        for (I i = 0; i < _n; ++i)
            for (I j = _pB[i]; j < _pE[i]; ++j)
                _v[j] += pr_vec[_r[j]];
    }
    else {
        for (I i = 0; i < _n; ++i)
            for (I j = _pB[i]; j < _pE[i]; ++j)
                _v[j] += a * pr_vec[_r[j]];
    }
};

template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::addVecToColsWeighted(
    const Vector<floating_type>& vec, const floating_type* weights, const floating_type a) {
    const floating_type* pr_vec = vec.rawX();
    if (isEqual(a, floating_type(1.0))) {
        for (I i = 0; i < _n; ++i)
            for (I j = _pB[i]; j < _pE[i]; ++j)
                _v[j] += pr_vec[_r[j]] * weights[j - _pB[i]];
    }
    else {
        for (I i = 0; i < _n; ++i)
            for (I j = _pB[i]; j < _pE[i]; ++j)
                _v[j] += a * pr_vec[_r[j]] * weights[j - _pB[i]];
    }
};

template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::sum_cols(Vector<floating_type>& sum) const {
    sum.resize(_m);
    sum.setZeros();
    SpVector<floating_type, I> tmp;
    for (I i = 0; i < _n; ++i) {
        this->refCol(i, tmp);
        sum.add(tmp);
    }
};


template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::XtX(Matrix<floating_type>& XtX) const {
    XtX.resize(_n, _n);
    XtX.setZeros();
    SpVector<floating_type, I> col;
    Vector<floating_type> col_out;
    for (I i = 0; i < _n; ++i) {
        this->refCol(i, col);
        XtX.refCol(i, col_out);
        this->multTrans(col, col_out);
    }
};


/// aat <- A(:,indices)*A(:,indices)'
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::AAt(Matrix<floating_type>& aat,
    const Vector<I>& indices) const {
    I i, j, k;
    I K = _m;
    I M = indices.n();

    /* compute alpha alpha^floating_type */
    aat.resize(K, K);
    int NUM_THREADS = init_omp(MAX_THREADS);
    floating_type* aatT = new floating_type[NUM_THREADS * K * K];
    for (j = 0; j < NUM_THREADS * K * K; ++j) aatT[j] = floating_type();

#pragma omp parallel for private(i,j,k)
    for (i = 0; i < M; ++i) {
        I ii = indices[i];
#ifdef _OPENMP
        int numT = omp_get_thread_num();
#else
        int numT = 0;
#endif
        floating_type* write_area = aatT + numT * K * K;
        for (j = _pB[ii]; j < _pE[ii]; ++j) {
            for (k = _pB[ii]; k <= j; ++k) {
                write_area[_r[j] * K + _r[k]] += _v[j] * _v[k];
            }
        }
    }

    cblas_copy<floating_type>(K * K, aatT, 1, aat._X, 1);
    for (i = 1; i < NUM_THREADS; ++i)
        cblas_axpy<floating_type>(K * K, 1.0, aatT + K * K * i, 1, aat._X, 1);
    aat.fillSymmetric();
    delete[](aatT);
}


/// XAt <- X*A'
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::XAt(const Matrix<floating_type>& X,
    Matrix<floating_type>& XAt) const {
    I j, i;
    I n = X._m;
    I K = _m;
    I M = _n;

    XAt.resize(n, K);
    /* compute X alpha^floating_type */
 //   int NUM_THREADS=init_omp(MAX_THREADS);
    //floating_type* XatT=new floating_type[NUM_THREADS*n*K];
    //for (j = 0; j<NUM_THREADS*n*K; ++j) XatT[j]=floating_type();

 //#pragma omp parallel for private(i,j)
    for (i = 0; i < M; ++i) {
        //#ifdef _OPENMP
        //      int numT=omp_get_thread_num();
        //#else
        //      int numT=0;
        //#endif
        //      floating_type* write_area=XatT+numT*n*K;
        for (j = _pB[i]; j < _pE[i]; ++j) {
            cblas_axpy<floating_type>(n, _v[j], X._X + i * n, 1, XAt._X + _r[j] * n, 1);
        }
    }
    //  cblas_copy<floating_type>(n*K,XatT,1,XAt._X,1);
   //   for (i = 1; i<NUM_THREADS; ++i) 
   //      cblas_axpy<floating_type>(n*K,1.0,XatT+n*K*i,1,XAt._X,1);
   //   delete[](XatT);
};

/// XAt <- X(:,indices)*A(:,indices)'
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::XAt(const Matrix<floating_type>& X,
    Matrix<floating_type>& XAt, const Vector<I>& indices) const {
    I j, i;
    I n = X._m;
    I K = _m;
    I M = indices.n();

    XAt.resize(n, K);
    /* compute X alpha^floating_type */
    int NUM_THREADS = init_omp(MAX_THREADS);
    floating_type* XatT = new floating_type[NUM_THREADS * n * K];
    for (j = 0; j < NUM_THREADS * n * K; ++j) XatT[j] = floating_type();

#pragma omp parallel for private(i,j)
    for (i = 0; i < M; ++i) {
        I ii = indices[i];
#ifdef _OPENMP
        int numT = omp_get_thread_num();
#else
        int numT = 0;
#endif
        floating_type* write_area = XatT + numT * n * K;
        for (j = _pB[ii]; j < _pE[ii]; ++j) {
            cblas_axpy<floating_type>(n, _v[j], X._X + i * n, 1, write_area + _r[j] * n, 1);
        }
    }

    cblas_copy<floating_type>(n * K, XatT, 1, XAt._X, 1);
    for (i = 1; i < NUM_THREADS; ++i)
        cblas_axpy<floating_type>(n * K, 1.0, XatT + n * K * i, 1, XAt._X, 1);
    delete[](XatT);
};

/// XAt <- sum_i w_i X(:,i)*A(:,i)'
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::wXAt(const Vector<floating_type>& w,
    const Matrix<floating_type>& X, Matrix<floating_type>& XAt, const int numThreads) const {
    I j, l, i;
    I n = X._m;
    I K = _m;
    I M = _n;
    I Mx = X._n;
    I numRepX = M / Mx;
    assert(numRepX * Mx == M);
    XAt.resize(n, K);
    /* compute X alpha^floating_type */
    int NUM_THREADS = init_omp(numThreads);
    floating_type* XatT = new floating_type[NUM_THREADS * n * K];
    for (j = 0; j < NUM_THREADS * n * K; ++j) XatT[j] = floating_type();

#pragma omp parallel for private(i,j,l)
    for (i = 0; i < Mx; ++i) {
#ifdef _OPENMP
        int numT = omp_get_thread_num();
#else
        int numT = 0;
#endif
        floating_type* write_area = XatT + numT * n * K;
        for (l = 0; l < numRepX; ++l) {
            I ind = numRepX * i + l;
            if (w._X[ind] != 0)
                for (j = _pB[ind]; j < _pE[ind]; ++j) {
                    cblas_axpy<floating_type>(n, w._X[ind] * _v[j], X._X + i * n, 1, write_area + _r[j] * n, 1);
                }
        }
    }

    cblas_copy<floating_type>(n * K, XatT, 1, XAt._X, 1);
    for (i = 1; i < NUM_THREADS; ++i)
        cblas_axpy<floating_type>(n * K, 1.0, XatT + n * K * i, 1, XAt._X, 1);
    delete[](XatT);
};

/// copy the sparse matrix into a dense matrix
template<typename floating_type, typename I> inline void SpMatrix<floating_type, I>::toFull(Matrix<floating_type>& matrix) const {
    matrix.resize(_m, _n);
    matrix.setZeros();
    floating_type* out = matrix._X;
    for (I i = 0; i < _n; ++i) {
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            out[i * _m + _r[j]] = _v[j];
        }
    }
};

/// copy the sparse matrix into a full dense matrix
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::toFullTrans(
    Matrix<floating_type>& matrix) const {
    matrix.resize(_n, _m);
    matrix.setZeros();
    floating_type* out = matrix._X;
    for (I i = 0; i < _n; ++i) {
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            out[i + _r[j] * _n] = _v[j];
        }
    }
};


/// use the data from v, r for _v, _r
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::convert(const Matrix<floating_type>& vM,
    const Matrix<I>& rM, const I K) {
    const I M = rM.n();
    const I L = rM.m();
    const I* r = rM.X();
    const floating_type* v = vM.X();
    I count = 0;
    for (I i = 0; i < M * L; ++i) if (r[i] != -1) ++count;
    resize(K, M, count);
    count = 0;
    for (I i = 0; i < M; ++i) {
        _pB[i] = count;
        for (I j = 0; j < L; ++j) {
            if (r[i * L + j] == -1) break;
            _v[count] = v[i * L + j];
            _r[count++] = r[i * L + j];
        }
        _pE[i] = count;
    }
    for (I i = 0; i < M; ++i) sort(_r, _v, _pB[i], _pE[i] - 1);
};

/// use the data from v, r for _v, _r
template <typename floating_type, typename I> inline void SpMatrix<floating_type, I>::convert2(
    const Matrix<floating_type>& vM, const Vector<I>& rv, const I K) {
    const I M = vM.n();
    const I L = vM.m();
    I* r = rv.rawX();
    const floating_type* v = vM.X();
    I LL = 0;
    for (I i = 0; i < L; ++i) if (r[i] != -1) ++LL;
    this->resize(K, M, LL * M);
    I count = 0;
    for (I i = 0; i < M; ++i) {
        _pB[i] = count;
        for (I j = 0; j < LL; ++j) {
            _v[count] = v[i * L + j];
            _r[count++] = r[j];
        }
        _pE[i] = count;
    }
    for (I i = 0; i < M; ++i) sort(_r, _v, _pB[i], _pE[i] - 1);
};

/// returns the l2 norms ^2 of the columns
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::normalize() {
    SpVector<floating_type, I> col;
    for (I i = 0; i < _n; ++i) {
        this->refCol(i, col);
        const floating_type norm = col.nrm2sq();
        if (norm > 1e-10)
            col.scal(floating_type(1.0) / col.nrm2sq());
    }
};

/// returns the l2 norms ^2 of the columns
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::normalize_rows() {
    Vector<floating_type> norms(_m);
    norms.setZeros();
    for (I i = 0; i < _n; ++i) {
        for (I j = _pB[i]; j < _pE[i]; ++j) {
            norms[_r[j]] += _v[j] * _v[j];
        }
    }
    norms.Sqrt();
    for (I i = 0; i < _m; ++i)
        norms[i] = norms[i] < 1e-10 ? floating_type(1.0) : floating_type(1.0) / norms[i];
    for (I i = 0; i < _n; ++i)
        for (I j = _pB[i]; j < _pE[i]; ++j)
            _v[j] *= norms[_r[j]];
};




/// returns the l2 norms ^2 of the columns
template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::norm_2sq_cols(Vector<floating_type>& norms) const {
    norms.resize(_n);
    SpVector<floating_type, I> col;
    for (I i = 0; i < _n; ++i) {
        this->refCol(i, col);
        norms[i] = col.nrm2sq();
    }
};


template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::norm_0_cols(Vector<floating_type>& norms) const {
    norms.resize(_n);
    SpVector<floating_type, I> col;
    for (I i = 0; i < _n; ++i) {
        this->refCol(i, col);
        norms[i] = static_cast<floating_type>(col.length());
    }
};

template <typename floating_type, typename I>
inline void SpMatrix<floating_type, I>::norm_1_cols(Vector<floating_type>& norms) const {
    norms.resize(_n);
    SpVector<floating_type, I> col;
    for (I i = 0; i < _n; ++i) {
        this->refCol(i, col);
        norms[i] = col.asum();
    }
};


#endif