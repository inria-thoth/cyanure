#ifndef VECTOR_H
#define VECTOR_H

#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif

#undef max
#undef min


#include "../declare_structures.h"



/// Class for dense vector
template<typename floating_type> class Vector {
    friend class SpMatrix<floating_type>;
    friend class Matrix<floating_type>;
    friend class SpVector<floating_type>;
public:
    typedef floating_type value_type;
    typedef floating_type element;
    /// Empty constructor
    Vector();
    /// Constructor. Create a new vector of size n
    Vector(INTM n);
    /// Constructor with existing data
    Vector(floating_type* X, INTM n);
    /// Copy constructor
    explicit Vector<floating_type>(const Vector<floating_type>& vec);

    /// Destructor
    virtual ~Vector();

    /// Accessors
    /// Print the vector to std::cout
    inline void print(const char* name) const;
    inline void dump(const string& name) const;
    /// returns the index of the largest value
    inline INTM max() const;
    /// returns the index of the minimum value
    inline INTM min() const;
    /// returns the maximum value
    inline floating_type maxval() const;
    /// returns the minimum value
    inline floating_type minval() const;
    /// returns the index of the value with largest magnitude
    inline INTM fmax() const;
    /// returns the index of the value with smallest magnitude
    inline INTM fmin() const;
    /// returns the maximum magnitude
    inline floating_type fmaxval() const;
    /// returns the minimum magnitude
    inline floating_type fminval() const;
    /// returns a reference to X[index]
    inline floating_type& operator[](const INTM index);
    /// returns X[index]
    inline floating_type operator[](const INTM index) const;
    /// make a copy of x
    inline void copy(const Vector<floating_type>& x);
    inline void copyRef(const Vector<floating_type>& x);
    /// returns the size of the vector
    inline int n() const { return _n; };
    /// returns the size of the vector
    inline int size() const { return _n; };
    /// returns a modifiable reference of the data, DANGEROUS
    inline floating_type* rawX() const { return _X; };
    /// change artificially the size of the vector, DANGEROUS
    inline void fakeSize(const INTM n) { _n = n; };
    /// generate logarithmically spaced values
    inline void logspace(const INTM n, const floating_type a, const floating_type b);
    inline INTM nnz() const;

    /// Modifiers
    /// Set all values to zero
    inline void setZeros();
    /// resize the vector
    inline void resize(const INTM n, const bool set_zeros = true);
    /// change the data of the vector
    inline void setPointer(floating_type* X, const INTM n);
    inline void setData(floating_type* X, const INTM n) { this->setPointer(X, n); };
    inline void refData(const Vector<floating_type>& vec) { this->setPointer(vec.rawX(), vec.n()); };
    inline void refSubVec(INTM i, INTM n, Vector<floating_type>& mat) const { mat.setData(_X + i, n); };
    //inline void print(const char* name) const;
    inline void print(const string& name) const;

    /// put a random permutation of size n (for integral vectors)
    inline void randperm(int n);
    /// put a random permutation of size n (for integral vectors)
    inline void randi(int n);
    /// put random values in the vector (White Gaussian Noise)
    inline void setAleat();
    /// clear the vector
    inline void clear();
    /// performs soft-thresholding of the vector
    inline void softThrshold(const floating_type nu);
    inline void fastSoftThrshold(const floating_type nu);
    inline void fastSoftThrshold(Vector<floating_type>& out, const floating_type nu) const;
    inline void softThrsholdScal(Vector<floating_type>& out, const floating_type nu, const floating_type s);
    inline void hardThrshold(const floating_type nu);
    /// performs soft-thresholding of the vector
    inline void thrsmax(const floating_type nu);
    inline void thrsmin(const floating_type nu);
    inline void thrsabsmin(const floating_type nu);
    /// performs soft-thresholding of the vector
    inline void thrshold(const floating_type nu);
    /// performs soft-thresholding of the vector
    inline void thrsPos();
    /// set each value of the vector to val
    inline void set(const floating_type val);
    inline void setn(const INTM n) { _n = n; }; //DANGEROUS
    inline bool alltrue() const;
    inline bool allfalse() const;

    /// Algebric operations
    /// returns ||A||_2
    inline floating_type nrm2() const;
    /// returns ||A||_2^2
    inline floating_type nrm2sq() const;
    /// returns  A'x
    inline floating_type dot(const Vector<floating_type>& x) const;
    /// returns A'x, when x is sparse
    template <typename I>
    inline floating_type dot(const SpVector<floating_type, I>& x) const;
    /// A <- A + a*x
    inline void add(const Vector<floating_type>& x, const floating_type a = 1.0);
    /// A <- A + a*x
    template <typename I>
    inline void add(const SpVector<floating_type, I>& x, const floating_type a = 1.0);
    /// adds a to each value in the vector
    inline void add(const floating_type a);
    /// A <- b*A + a*x
    inline void add_scal(const Vector<floating_type>& x, const floating_type a = 1.0, const floating_type b = 0);
    /// A <- b*A + a*x
    template <typename I>
    inline void add_scal(const SpVector<floating_type, I>& x, const floating_type a = 1.0, const floating_type b = 0);
    /// A <- A - x
    inline void sub(const Vector<floating_type>& x);
    /// A <- A + a*x
    template <typename I>
    inline void sub(const SpVector<floating_type, I>& x);
    /// A <- A ./ x
    inline void div(const Vector<floating_type>& x);
    /// A <- x ./ y
    inline void div(const Vector<floating_type>& x, const Vector<floating_type>& y);
    /// A <- x .^ 2
    inline void sqr(const Vector<floating_type>& x);
    /// A <- 1 ./ sqrt(x) 
    inline void sqr();
    /// A <- 1 ./ sqrt(A) 
    inline void Sqrt(const Vector<floating_type>& x);
    /// A <- 1 ./ sqrt(x) 
    inline void Sqrt();
    /// A <- 1 ./ sqrt(x) 
    inline void Invsqrt(const Vector<floating_type>& x);
    /// A <- 1 ./ sqrt(A) 
    inline void Invsqrt();
    /// A <- 1./x
    inline void inv(const Vector<floating_type>& x);
    /// A <- 1./A
    inline void inv();
    /// A <- x .* y
    inline void mult(const Vector<floating_type>& x, const Vector<floating_type>& y);
    inline void mult_elementWise(const Vector<floating_type>& B, Vector<floating_type>& C) const { C.mult(*this, B); };
    /// normalize the vector
    inline void normalize();
    /// normalize the vector
    inline void normalize2(const floating_type thrs = 1.0);
    /// whiten
    inline void whiten(Vector<floating_type>& mean, const bool pattern = false);
    /// whiten
    inline void whiten(Vector<floating_type>& mean, const
        Vector<floating_type>& mask);
    /// whiten
    inline void whiten(const INTM V);
    /// whiten
    inline floating_type mean() const;
    inline floating_type abs_mean() const;
    inline floating_type mean_non_uniform(const Vector<floating_type>& qi) const;
    /// whiten
    inline floating_type std();
    /// compute the Kuhlback-Leiber divergence
    inline floating_type KL(const Vector<floating_type>& X);
    /// whiten
    inline void unwhiten(Vector<floating_type>& mean, const bool pattern = false);
    /// scale the vector by a
    inline void scal(const floating_type a);
    /// A <- -A
    inline void neg();
    /// replace each value by its exponential
    inline void exp();
    /// replace each value by its logarithm
    inline void log();
    /// replace each value by its absolute value
    inline void abs_vec();
    /// replace each value by its exponential
    inline void logexp();
    /// replace each value by its exponential
    inline floating_type softmax(const int y);
    inline floating_type logsumexp();
    /// computes the sum of the magnitudes of the vector
    inline floating_type asum() const;
    inline floating_type lzero() const;
    /// compute the sum of the differences
    inline floating_type afused() const;
    /// returns the sum of the vector
    inline floating_type sum() const;
    /// puts in signs, the sign of each point in the vector
    inline void sign(Vector<floating_type>& signs) const;
    /// projects the vector onto the l1 ball of radius thrs,
    /// returns true if the returned vector is null
    inline void l1project(Vector<floating_type>& out, const floating_type thrs, const bool simplex = false) const;
    inline void l1project_weighted(Vector<floating_type>& out, const Vector<floating_type>& weights, const floating_type thrs, const bool residual = false) const;
    inline void l1l2projectb(Vector<floating_type>& out, const floating_type thrs, const floating_type gamma, const bool pos = false,
        const int mode = 1);
    inline void sparseProject(Vector<floating_type>& out, const floating_type thrs, const int mode = 1, const floating_type lambda_1 = 0,
        const floating_type lambda_2 = 0, const floating_type lambda_3 = 0, const bool pos = false);
    inline void project_sft(const Vector<int>& labels, const int clas);
    inline void project_sft_binary(const Vector<floating_type>& labels);
    /// projects the vector onto the l1 ball of radius thrs,
    /// projects the vector onto the l1 ball of radius thrs,
    /// returns true if the returned vector is null
    inline void l1l2project(Vector<floating_type>& out, const floating_type thrs, const floating_type gamma, const bool pos = false) const;
    inline void fusedProject(Vector<floating_type>& out, const floating_type lambda_1, const floating_type lambda_2, const int itermax);
    inline void fusedProjectHomotopy(Vector<floating_type>& out, const floating_type lambda_1, const floating_type lambda_2, const floating_type lambda_3 = 0,
        const bool penalty = true);
    /// projects the vector onto the l1 ball of radius thrs,
    /// _sort the vector
    inline void sort(Vector<floating_type>& out, const bool mode) const;
    /// sort the vector
    inline void sort(const bool mode);
    //// sort the vector
    inline void sort2(Vector<floating_type>& out, Vector<INTM>& key, const bool mode) const;
    /// sort the vector
    inline void sort2(Vector<INTM>& key, const bool mode);
    /// sort the vector
    inline void applyBayerPattern(const int offset);


    /// Conversion
    /// make a sparse copy 
    inline void toSparse(SpVector<floating_type>& vec) const;
    /// extract the rows of a matrix corresponding to a binary mask
    inline void copyMask(Vector<floating_type>& out, Vector<bool>& mask) const;
    inline void getIndices(Vector<int>& ind) const { }; // irrelevant for dense vectors
    template <typename I>
    inline void refIndices(Vector<I>& ind) const { }; // irrelevant for dense vectors



private:
    /// = operator, 
    Vector<floating_type>& operator=(const Vector<floating_type>& vec);

    /// if the data has been externally allocated
    bool _externAlloc;
    /// data
    floating_type* _X;
    /// size of the vector
    INTM _n;
};


/* ***********************************
 * Implementation of the class Vector
 * ***********************************/


 /// Empty constructor
template <typename floating_type> Vector<floating_type>::Vector() :
    _externAlloc(true), _X(NULL), _n(0) {  };

/// Constructor. Create a new vector of size n
template <typename floating_type> Vector<floating_type>::Vector(INTM n) :
    _externAlloc(false), _n(n) {
#pragma omp critical
        {
            _X = new floating_type[_n];
        }
};

/// Constructor with existing data
template <typename floating_type> Vector<floating_type>::Vector(floating_type* X, INTM n) :
    _externAlloc(true), _X(X), _n(n) {  };

/// Copy constructor
template <typename floating_type> Vector<floating_type>::Vector(const Vector<floating_type>& vec) :
    _externAlloc(false), _n(vec._n) {
#pragma omp critical
        {
            _X = new floating_type[_n];
        }
        cblas_copy<floating_type>(_n, vec._X, 1, _X, 1);
};

/// Destructor
template <typename floating_type> Vector<floating_type>::~Vector() {
    clear();
};

/// Print the matrix to std::cout
template <typename floating_type> inline void Vector<floating_type>::print(const string& name) const {
    std::cerr << name << std::endl;
    std::cerr << _n << std::endl;
    for (INTM j = 0; j < _n; ++j) {
        fprintf(stderr, "%10.5g ", static_cast<double>(_X[j]));
    }
    fprintf(stderr, "\n ");
};

/// Print the matrix to std::cout
template <typename floating_type> inline void Vector<floating_type>::dump(const string& name) const {
    ofstream f;
    const char* cname = name.c_str();
    f.open(cname);
    f.precision(20);
    std::cerr << name << std::endl;
    f << _n << std::endl;
    for (INTM j = 0; j < _n; ++j) {
        f << static_cast<double>(_X[j]) << " ";
    }
    f << std::endl;
    f.close();
};




/// Print the vector to std::cout
template <> inline void Vector<double>::print(const char* name) const {
    printf("%s, %d\n", name, (int)_n);
    for (INTM i = 0; i < _n; ++i) {
        printf("%g ", _X[i]);
    }
    printf("\n");
};

/// Print the vector to std::cout
template <> inline void Vector<float>::print(const char* name) const {
    printf("%s, %d\n", name, (int)_n);
    for (INTM i = 0; i < _n; ++i) {
        printf("%g ", _X[i]);
    }
    printf("\n");
};

/// Print the vector to std::cout
template <> inline void Vector<int>::print(const char* name) const {
    printf("%s, %d\n", name, (int)_n);
    for (INTM i = 0; i < _n; ++i) {
        printf("%d ", _X[i]);
    }
    printf("\n");
};

/// Print the vector to std::cout
template <> inline void Vector<bool>::print(const char* name) const {
    printf("%s, %d\n", name, (int)_n);
    for (INTM i = 0; i < _n; ++i) {
        printf("%d ", _X[i] ? 1 : 0);
    }
    printf("\n");
};

/// returns the index of the largest value
template <typename floating_type> inline INTM Vector<floating_type>::max() const {
    INTM imax = 0;
    floating_type max = _X[0];
    for (INTM j = 1; j < _n; ++j) {
        floating_type cur = _X[j];
        if (cur > max) {
            imax = j;
            max = cur;
        }
    }
    return imax;
};

/// returns the index of the minimum value
template <typename floating_type> inline INTM Vector<floating_type>::min() const {
    INTM imin = 0;
    floating_type min = _X[0];
    for (INTM j = 1; j < _n; ++j) {
        floating_type cur = _X[j];
        if (cur < min) {
            imin = j;
            min = cur;
        }
    }
    return imin;
};

/// returns the maximum value
template <typename floating_type> inline floating_type Vector<floating_type>::maxval() const {
    return _X[this->max()];
};

/// returns the minimum value
template <typename floating_type> inline floating_type Vector<floating_type>::minval() const {
    return _X[this->min()];
};

/// returns the maximum magnitude
template <typename floating_type> inline floating_type Vector<floating_type>::fmaxval() const {
    return fabs(_X[this->fmax()]);
};

/// returns the minimum magnitude
template <typename floating_type> inline floating_type Vector<floating_type>::fminval() const {
    return fabs(_X[this->fmin()]);
};

template <typename floating_type>
inline void Vector<floating_type>::logspace(const INTM n, const floating_type a, const floating_type b) {
    floating_type first = log10(a);
    floating_type last = log10(b);
    floating_type step = (last - first) / (n - 1);
    this->resize(n);
    _X[0] = first;
    for (INTM i = 1; i < _n; ++i)
        _X[i] = _X[i - 1] + step;
    for (INTM i = 0; i < _n; ++i)
        _X[i] = pow(floating_type(10.0), _X[i]);
}

template <typename floating_type>
inline INTM Vector<floating_type>::nnz() const {
    INTM sum = 0;
    for (INTM i = 0; i < _n; ++i)
        if (_X[i] != floating_type()) ++sum;
    return sum;
};
/// generate logarithmically spaced values
template <>
inline void Vector<INTM>::logspace(const INTM n, const INTM a, const INTM b) {
    Vector<double> tmp(n);
    tmp.logspace(n, double(a), double(b));
    this->resize(n);
    _X[0] = a;
    _X[n - 1] = b;
    for (INTM i = 1; i < _n - 1; ++i) {
        INTM candidate = static_cast<INTM>(floor(static_cast<double>(tmp[i])));
        _X[i] = candidate > _X[i - 1] ? candidate : _X[i - 1] + 1;
    }
}

/// returns the index of the value with largest magnitude
template <typename floating_type> inline INTM Vector<floating_type>::fmax() const {
    return cblas_iamax<floating_type>(_n, _X, 1);
};

/// returns the index of the value with smallest magnitude
template <typename floating_type> inline INTM Vector<floating_type>::fmin() const {
    return cblas_iamin<floating_type>(_n, _X, 1);
};

/// returns a reference to X[index]
template <typename floating_type> inline floating_type& Vector<floating_type>::operator[] (const INTM i) {
    assert(i >= 0 && i < _n);
    return _X[i];
};

/// returns X[index]
template <typename floating_type> inline floating_type Vector<floating_type>::operator[] (const INTM i) const {
    assert(i >= 0 && i < _n);
    return _X[i];
};

/// make a copy of x
template <typename floating_type> inline void Vector<floating_type>::copy(const Vector<floating_type>& x) {
    if (_X != x._X) {
        this->resize(x.n());
        //cblas_copy<floating_type>(_n,x._X,1,_X,1);
        memcpy(_X, x._X, _n * sizeof(floating_type));
    }
};

/// make a copy of x
template <typename floating_type> inline void Vector<floating_type>::copyRef(const Vector<floating_type>& x) {
    this->setData(x.rawX(), x.n());
};


/// Set all values to zero
template <typename floating_type> inline void Vector<floating_type>::setZeros() {
    memset(_X, 0, _n * sizeof(floating_type));
};

/// resize the vector
template <typename floating_type> inline void Vector<floating_type>::resize(const INTM n, const bool set_zeros) {
    if (_n == n) return;
    clear();
#pragma omp critical
    {
        _X = new floating_type[n];
    }
    _n = n;
    _externAlloc = false;
    if (set_zeros)
        this->setZeros();
};

/// change the data of the vector
template <typename floating_type> inline void Vector<floating_type>::setPointer(floating_type* X, const INTM n) {
    clear();
    _externAlloc = true;
    _X = X;
    _n = n;
};

/// put a random permutation of size n (for integral vectors)
template <> inline void Vector<int>::randi(int n) {
    for (int i = 0; i < _n; ++i)
        _X[i] = static_cast<int>(random() % n);
};

/// put a random permutation of size n (for integral vectors)
template <> inline void Vector<int>::randperm(int n) {
    resize(n);
    Vector<int> table(n);
    for (int i = 0; i < n; ++i)
        table[i] = i;
    int size = n;
    for (int i = 0; i < n; ++i) {
        const int ind = random() % size;
        _X[i] = table[ind];
        table[ind] = table[size - 1];
        --size;
    }
};

/// put random values in the vector (white Gaussian Noise)
template <typename floating_type> inline void Vector<floating_type>::setAleat() {
    for (INTM i = 0; i < _n; ++i) _X[i] = normalDistrib<floating_type>();
};

/// clear the vector
template <typename floating_type> inline void Vector<floating_type>::clear() {
    if (!_externAlloc) delete[](_X);
    _n = 0;
    _X = NULL;
    _externAlloc = true;
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::softThrshold(const floating_type nu) {
    for (INTM i = 0; i < _n; ++i) {
        if (_X[i] > nu) {
            _X[i] -= nu;
        }
        else if (_X[i] < -nu) {
            _X[i] += nu;
        }
        else {
            _X[i] = 0;
        }
    }
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::fastSoftThrshold(const floating_type nu) {
    //#pragma omp parallel for
    for (INTM i = 0; i < _n; ++i)
    {
        _X[i] = fastSoftThrs(_X[i], nu);
    }
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::fastSoftThrshold(Vector<floating_type>& output, const floating_type nu) const {
    output.resize(_n, false);
    //#pragma omp parallel for
    for (INTM i = 0; i < _n; ++i)
        output[i] = fastSoftThrs(_X[i], nu);
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::softThrsholdScal(Vector<floating_type>& out, const floating_type nu, const floating_type s) {
    floating_type* Y = out.rawX();
    for (INTM i = 0; i < _n; ++i) {
        if (_X[i] > nu) {
            Y[i] = s * (_X[i] - nu);
        }
        else if (_X[i] < -nu) {
            Y[i] = s * (_X[i] + nu);
        }
        else {
            Y[i] = 0;
        }
    }
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::hardThrshold(const floating_type nu) {
    for (INTM i = 0; i < _n; ++i) {
        if (!(_X[i] > nu || _X[i] < -nu)) {
            _X[i] = 0;
        }
    }
};


/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsmax(const floating_type nu) {
    //#pragma omp parallel for private(i)
    for (INTM i = 0; i < _n; ++i)
        if (_X[i] < nu) _X[i] = nu;
}

/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsmin(const floating_type nu) {
    for (INTM i = 0; i < _n; ++i)
        _X[i] = MIN(_X[i], nu);
}

/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsabsmin(const floating_type nu) {
    for (INTM i = 0; i < _n; ++i)
        _X[i] = MAX(MIN(_X[i], nu), -nu);
}

/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrshold(const floating_type nu) {
    for (INTM i = 0; i < _n; ++i)
        if (abs<floating_type>(_X[i]) < nu)
            _X[i] = 0;
}
/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsPos() {
    for (INTM i = 0; i < _n; ++i) {
        if (_X[i] < 0) _X[i] = 0;
    }
};

template <>
inline bool Vector<bool>::alltrue() const {
    for (INTM i = 0; i < _n; ++i) {
        if (!_X[i]) return false;
    }
    return true;
};

template <>
inline bool Vector<bool>::allfalse() const {
    for (INTM i = 0; i < _n; ++i) {
        if (_X[i]) return false;
    }
    return true;
};

/// set each value of the vector to val
template <typename floating_type> inline void Vector<floating_type>::set(const floating_type val) {
    for (INTM i = 0; i < _n; ++i) _X[i] = val;
};

/// returns ||A||_2
template <typename floating_type> inline floating_type Vector<floating_type>::nrm2() const {
    return cblas_nrm2<floating_type>(_n, _X, 1);
};

/// returns ||A||_2^2
template <typename floating_type> inline floating_type Vector<floating_type>::nrm2sq() const {
    return cblas_dot<floating_type>(_n, _X, 1, _X, 1);
};

/// returns  A'x
template <typename floating_type> inline floating_type Vector<floating_type>::dot(const Vector<floating_type>& x) const {
    assert(_n == x._n);
    return cblas_dot<floating_type>(_n, _X, 1, x._X, 1);
};

/// returns A'x, when x is sparse
template <typename floating_type>
template <typename I>
inline floating_type Vector<floating_type>::dot(const SpVector<floating_type, I>& x) const {
    floating_type sum = 0;
    const I* r = x.rawR();
    const floating_type* v = x.rawX();
    for (INTT i = 0; i < x._L; ++i) {
        sum += _X[r[i]] * v[i];
    }
    return sum;
    //return cblas_doti<floating_type>(x._L,x._v,x._r,_X);
};

/// A <- A + a*x
template <typename floating_type> inline void Vector<floating_type>::add(const Vector<floating_type>& x, const floating_type a) {
    assert(_n == x._n);
    cblas_axpy<floating_type>(_n, a, x._X, 1, _X, 1);
};

template <typename floating_type> inline void Vector<floating_type>::add_scal(const Vector<floating_type>& x, const floating_type a, const floating_type b) {
    assert(_n == x._n);
    cblas_axpby<floating_type>(_n, a, x._X, 1, b, _X, 1);
};

/// A <- A + a*x
template <typename floating_type>
template <typename I>
inline void Vector<floating_type>::add(const SpVector<floating_type, I>& x,
    const floating_type a) {
    if (a == 1.0) {
        for (INTM i = 0; i < x._L; ++i)
            _X[x._r[i]] += x._v[i];
    }
    else {
        for (INTM i = 0; i < x._L; ++i)
            _X[x._r[i]] += a * x._v[i];
    }
};

/// A <- A + a*x
template <typename floating_type>
template <typename I>
inline void Vector<floating_type>::add_scal(const SpVector<floating_type, I>& x,
    const floating_type a, const floating_type b) {
    if (b != floating_type(1.0)) {
        if (b == 0) {
            this->setZeros();
        }
        else {
            this->scal(b);
        }
    }
    if (a == floating_type(1.0)) {
        for (I i = 0; i < x._L; ++i)
            _X[x._r[i]] += x._v[i];
    }
    else {
        for (I i = 0; i < x._L; ++i)
            _X[x._r[i]] += a * x._v[i];
    }
};



/// adds a to each value in the vector
template <typename floating_type> inline void Vector<floating_type>::add(const floating_type a) {
    for (INTM i = 0; i < _n; ++i) _X[i] += a;
};

/// A <- A - x
template <typename floating_type> inline void Vector<floating_type>::sub(const Vector<floating_type>& x) {
    assert(_n == x._n);
    vSub<floating_type>(_n, _X, x._X, _X);
};

/// A <- A + a*x
template <typename floating_type>
template <typename I>
inline void Vector<floating_type>::sub(const SpVector<floating_type, I>& x) {
    for (INTM i = 0; i < x._L; ++i)
        _X[x._r[i]] -= x._v[i];
};

/// A <- A ./ x
template <typename floating_type> inline void Vector<floating_type>::div(const Vector<floating_type>& x) {
    assert(_n == x._n);
    vDiv<floating_type>(_n, _X, x._X, _X);
};

/// A <- x ./ y
template <typename floating_type> inline void Vector<floating_type>::div(const Vector<floating_type>& x, const Vector<floating_type>& y) {
    assert(_n == x._n);
    vDiv<floating_type>(_n, x._X, y._X, _X);
};


/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::sqr(const Vector<floating_type>& x) {
    this->resize(x._n);
    vSqr<floating_type>(_n, x._X, _X);
}

/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::sqr() {
    vSqr<floating_type>(_n, _X, _X);
}

/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Invsqrt(const Vector<floating_type>& x) {
    this->resize(x._n);
    vInvSqrt<floating_type>(_n, x._X, _X);
}
/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Sqrt(const Vector<floating_type>& x) {
    this->resize(x._n);
    vSqrt<floating_type>(_n, x._X, _X);
}
/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Invsqrt() {
    vInvSqrt<floating_type>(_n, _X, _X);
}
/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Sqrt() {
    vSqrt<floating_type>(_n, _X, _X);
}


/// A <- 1./x
template <typename floating_type> inline void Vector<floating_type>::inv(const Vector<floating_type>& x) {
    this->resize(x.n());
    vInv<floating_type>(_n, x._X, _X);
};

/// A <- 1./A
template <typename floating_type> inline void Vector<floating_type>::inv() {
    vInv<floating_type>(_n, _X, _X);
};

/// A <- x .* y
template <typename floating_type> inline void Vector<floating_type>::mult(const Vector<floating_type>& x,
    const Vector<floating_type>& y) {
    this->resize(x.n());
    vMul<floating_type>(_n, x._X, y._X, _X);
};
;

/// normalize the vector
template <typename floating_type> inline void Vector<floating_type>::normalize() {
    floating_type norm = nrm2();
    if (norm > EPSILON) scal(1.0 / norm);
};

/// normalize the vector
template <typename floating_type> inline void Vector<floating_type>::normalize2(const floating_type thrs) {
    floating_type norm = nrm2();
    if (norm > thrs) scal(thrs / norm);
};

/// whiten
template <typename floating_type> inline void Vector<floating_type>::whiten(
    Vector<floating_type>& meanv, const bool pattern) {
    if (pattern) {
        const INTM n = static_cast<INTM>(sqrt(static_cast<floating_type>(_n)));
        INTM count[4];
        for (INTM i = 0; i < 4; ++i) count[i] = 0;
        INTM offsetx = 0;
        for (INTM j = 0; j < n; ++j) {
            offsetx = (offsetx + 1) % 2;
            INTM offsety = 0;
            for (INTM k = 0; k < n; ++k) {
                offsety = (offsety + 1) % 2;
                meanv[2 * offsetx + offsety] += _X[j * n + k];
                count[2 * offsetx + offsety]++;
            }
        }
        for (INTM i = 0; i < 4; ++i)
            meanv[i] /= count[i];
        offsetx = 0;
        for (INTM j = 0; j < n; ++j) {
            offsetx = (offsetx + 1) % 2;
            INTM offsety = 0;
            for (INTM k = 0; k < n; ++k) {
                offsety = (offsety + 1) % 2;
                _X[j * n + k] -= meanv[2 * offsetx + offsety];
            }
        }
    }
    else {
        const INTM V = meanv.n();
        const INTM sizePatch = _n / V;
        for (INTM j = 0; j < V; ++j) {
            floating_type mean = 0;
            for (INTM k = 0; k < sizePatch; ++k) {
                mean += _X[sizePatch * j + k];
            }
            mean /= sizePatch;
            for (INTM k = 0; k < sizePatch; ++k) {
                _X[sizePatch * j + k] -= mean;
            }
            meanv[j] = mean;
        }
    }
};

/// whiten
template <typename floating_type> inline void Vector<floating_type>::whiten(
    Vector<floating_type>& meanv, const Vector<floating_type>& mask) {
    const INTM V = meanv.n();
    const INTM sizePatch = _n / V;
    for (INTM j = 0; j < V; ++j) {
        floating_type mean = 0;
        for (INTM k = 0; k < sizePatch; ++k) {
            mean += _X[sizePatch * j + k];
        }
        mean /= cblas_asum(sizePatch, mask._X + j * sizePatch, 1);
        for (INTM k = 0; k < sizePatch; ++k) {
            if (mask[sizePatch * j + k])
                _X[sizePatch * j + k] -= mean;
        }
        meanv[j] = mean;
    }
};

/// whiten
template <typename floating_type> inline void Vector<floating_type>::whiten(const INTM V) {
    const INTM sizePatch = _n / V;
    for (INTM j = 0; j < V; ++j) {
        floating_type mean = 0;
        for (INTM k = 0; k < sizePatch; ++k) {
            mean += _X[sizePatch * j + k];
        }
        mean /= sizePatch;
        for (INTM k = 0; k < sizePatch; ++k) {
            _X[sizePatch * j + k] -= mean;
        }
    }
};

template <typename floating_type> inline floating_type Vector<floating_type>::KL(const Vector<floating_type>& Y) {
    floating_type sum = 0;
    floating_type* prY = Y.rawX();
    for (INTM i = 0; i < _n; ++i) {
        if (_X[i] > 1e-20) {
            if (prY[i] < 1e-60) {
                sum += 1e200;
            }
            else {
                sum += _X[i] * log_alt<floating_type>(_X[i] / prY[i]);
            }
            //sum += _X[i]*log_alt<floating_type>(_X[i]/(prY[i]+1e-100));
        }
    }
    sum += floating_type(-1.0) + Y.sum();
    return sum;
};

/// unwhiten
template <typename floating_type> inline void Vector<floating_type>::unwhiten(
    Vector<floating_type>& meanv, const bool pattern) {
    if (pattern) {
        const INTM n = static_cast<INTM>(sqrt(static_cast<floating_type>(_n)));
        INTM offsetx = 0;
        for (INTM j = 0; j < n; ++j) {
            offsetx = (offsetx + 1) % 2;
            INTM offsety = 0;
            for (INTM k = 0; k < n; ++k) {
                offsety = (offsety + 1) % 2;
                _X[j * n + k] += meanv[2 * offsetx + offsety];
            }
        }
    }
    else {
        const INTM V = meanv.n();
        const INTM sizePatch = _n / V;
        for (INTM j = 0; j < V; ++j) {
            floating_type mean = meanv[j];
            for (INTM k = 0; k < sizePatch; ++k) {
                _X[sizePatch * j + k] += mean;
            }
        }
    }
};


/// return the mean
template <typename floating_type> inline floating_type Vector<floating_type>::mean() const {
    return this->sum() / _n;
}

template <typename floating_type> inline floating_type Vector<floating_type>::abs_mean() const {
    return this->asum() / _n;
};

template <typename floating_type> inline floating_type Vector<floating_type>::mean_non_uniform(const Vector<floating_type>& qi) const {
    Vector<floating_type> tmp;
    tmp.copy(*this);
    tmp.mult(qi, tmp);
    return tmp.sum();
};

/// return the std
template <typename floating_type> inline floating_type Vector<floating_type>::std() {
    floating_type E = this->mean();
    floating_type std = 0;
    for (INTM i = 0; i < _n; ++i) {
        floating_type tmp = _X[i] - E;
        std += tmp * tmp;
    }
    std /= _n;
    return sqr_alt<floating_type>(std);
}

/// scale the vector by a
template <typename floating_type> inline void Vector<floating_type>::scal(const floating_type a) {
    return cblas_scal<floating_type>(_n, a, _X, 1);
};

/// A <- -A
template <typename floating_type> inline void Vector<floating_type>::neg() {
    for (INTM i = 0; i < _n; ++i) _X[i] = -_X[i];
};

/// replace each value by its exponential
template <typename floating_type> inline void Vector<floating_type>::exp() {
    vExp<floating_type>(_n, _X, _X);
};

/// replace each value by its absolute value
template <typename floating_type> inline void Vector<floating_type>::abs_vec() {
    vAbs<floating_type>(_n, _X, _X);
};

/// replace each value by its logarithm
template <typename floating_type> inline void Vector<floating_type>::log() {
    for (INTM i = 0; i < _n; ++i) _X[i] = alt_log<floating_type>(_X[i]);
};

/// replace each value by its exponential
template <typename floating_type> inline void Vector<floating_type>::logexp() {
    for (INTM i = 0; i < _n; ++i) {
        _X[i] = logexp2(_X[i]);
        /*if (_X[i] < -30) {
           _X[i]=0;
        } else if (_X[i] < 30) {
           _X[i]= alt_log<floating_type>( floating_type(1.0) + exp_alt<floating_type>( _X[i] ) );
        }*/
    }
};

template <typename floating_type> inline floating_type Vector<floating_type>::logsumexp() {
    floating_type mm = this->maxval();
    this->add(-mm);
    this->exp();
    return mm + alt_log<floating_type>(this->asum());
};

/// replace each value by its exponential
template <typename floating_type> inline floating_type Vector<floating_type>::softmax(const int y) {
    this->add(-_X[y]);
    _X[y] = -INFINITY;
    floating_type max = this->maxval();
    if (max > 30) {
        return max;
    }
    else if (max < -30) {
        return 0;
    }
    else {
        _X[y] = floating_type(0.0);
        this->exp();
        return alt_log<floating_type>(this->sum());
    }
};

/// computes the sum of the magnitudes of the vector
template <typename floating_type> inline floating_type Vector<floating_type>::asum() const {
    return cblas_asum<floating_type>(_n, _X, 1);
};

template <typename floating_type> inline floating_type Vector<floating_type>::lzero() const {
    INTM count = 0;
    for (INTM i = 0; i < _n; ++i)
        if (_X[i] != 0) ++count;
    return count;
};


template <typename floating_type> inline floating_type Vector<floating_type>::afused() const {
    floating_type sum = 0;
    for (INTM i = 1; i < _n; ++i) {
        sum += abs<floating_type>(_X[i] - _X[i - 1]);
    }
    return sum;
}
/// returns the sum of the vector
template <typename floating_type> inline floating_type Vector<floating_type>::sum() const {
    floating_type sum = floating_type();
    for (INTM i = 0; i < _n; ++i) sum += _X[i];
    return sum;
};

/// puts in signs, the sign of each poINTM in the vector
template <typename floating_type> inline void Vector<floating_type>::sign(Vector<floating_type>& signs) const {
    floating_type* prSign = signs.rawX();
    for (INTM i = 0; i < _n; ++i) {
        if (_X[i] == 0) {
            prSign[i] = 0.0;
        }
        else {
            prSign[i] = _X[i] > 0 ? 1.0 : -1.0;
        }
    }
};

/// projects the vector onto the l1 ball of radius thrs,
/// returns true if the returned vector is null
template <typename floating_type> inline void Vector<floating_type>::l1project(Vector<floating_type>& out,
    const floating_type thrs, const bool simplex) const {
    out.copy(*this);
    if (simplex) {
        out.thrsPos();
    }
    else {
        vAbs<floating_type>(_n, out._X, out._X);
    }
    floating_type norm1 = out.sum();
    if (norm1 <= thrs) {
        if (!simplex) out.copy(*this);
        return;
    }
    floating_type* prU = out._X;
    INTM sizeU = _n;

    floating_type sum = floating_type();
    INTM sum_card = 0;

    while (sizeU > 0) {
        // put the pivot in prU[0]
        swap(prU[0], prU[sizeU / 2]);
        floating_type pivot = prU[0];
        INTM sizeG = 1;
        floating_type sumG = pivot;

        for (INTM i = 1; i < sizeU; ++i) {
            if (prU[i] >= pivot) {
                sumG += prU[i];
                swap(prU[sizeG++], prU[i]);
            }
        }

        if (sum + sumG - pivot * (sum_card + sizeG) <= thrs) {
            sum_card += sizeG;
            sum += sumG;
            prU += sizeG;
            sizeU -= sizeG;
        }
        else {
            ++prU;
            sizeU = sizeG - 1;
        }
    }
    floating_type lambda_1 = (sum - thrs) / sum_card;
    out.copy(*this);
    if (simplex) {
        out.thrsPos();
    }
    out.softThrshold(lambda_1);
};

/// projects the vector onto the l1 ball of radius thrs,
/// returns true if the returned vector is null
template <typename floating_type> inline void Vector<floating_type>::l1project_weighted(Vector<floating_type>& out, const Vector<floating_type>& weights,
    const floating_type thrs, const bool residual) const {
    out.copy(*this);
    if (thrs == 0) {
        out.setZeros();
        return;
    }
    vAbs<floating_type>(_n, out._X, out._X);
    out.div(weights);
    Vector<INTM> keys(_n);
    for (INTM i = 0; i < _n; ++i) keys[i] = i;
    out.sort2(keys, false);
    floating_type sum1 = 0;
    floating_type sum2 = 0;
    floating_type lambda_1 = 0;
    for (INTM i = 0; i < _n; ++i) {
        const floating_type lambda_old = lambda_1;
        const floating_type fact = weights[keys[i]] * weights[keys[i]];
        lambda_1 = out[i];
        sum2 += fact;
        sum1 += fact * lambda_1;
        if (sum1 - lambda_1 * sum2 >= thrs) {
            sum2 -= fact;
            sum1 -= fact * lambda_1;
            lambda_1 = lambda_old;
            break;
        }
    }
    lambda_1 = MAX(0, (sum1 - thrs) / sum2);

    if (residual) {
        for (INTM i = 0; i < _n; ++i) {
            out._X[i] = _X[i] > 0 ? MIN(_X[i], lambda_1 * weights[i]) : MAX(_X[i], -lambda_1 * weights[i]);
        }
    }
    else {
        for (INTM i = 0; i < _n; ++i) {
            out._X[i] = _X[i] > 0 ? MAX(0, _X[i] - lambda_1 * weights[i]) : MIN(0, _X[i] + lambda_1 * weights[i]);
        }
    }
};


template <typename floating_type>
inline void Vector<floating_type>::project_sft_binary(const Vector<floating_type>& y) {
    floating_type mean = this->mean();
    Vector<floating_type> ztilde, xtilde;
    ztilde.resize(_n);
    int count = 0;
    if (mean > 0) {
        for (int ii = 0; ii < _n; ++ii)
            if (y[ii] > 0) {
                count++;
                ztilde[ii] = _X[ii] + floating_type(1.0);
            }
            else {
                ztilde[ii] = _X[ii];
            }
        ztilde.l1project(xtilde, floating_type(count));
        for (int ii = 0; ii < _n; ++ii)
            _X[ii] = y[ii] > 0 ? xtilde[ii] - floating_type(1.0) : xtilde[ii];
    }
    else {
        for (int ii = 0; ii < _n; ++ii)
            if (y[ii] > 0) {
                ztilde[ii] = -_X[ii];
            }
            else {
                count++;
                ztilde[ii] = -_X[ii] + floating_type(1.0);
            }
        ztilde.l1project(xtilde, floating_type(count));
        for (int ii = 0; ii < _n; ++ii)
            _X[ii] = y[ii] > 0 ? -xtilde[ii] : -xtilde[ii] + floating_type(1.0);
    }
};

template <typename floating_type>
inline void Vector<floating_type>::project_sft(const Vector<int>& labels, const int clas) {
    Vector<floating_type> y(_n);
    for (int ii = 0; ii < _n; ++ii) y[ii] = labels[ii] == clas ? floating_type(1.0) : -floating_type(1.0);
    this->project_sft_binary(y);
    /*   floating_type mean = this->mean();
       floating_type thrs=mean;

       while (abs(mean) > EPSILON) {
          INTM n_seuils=0;
          for (INTM i = 0; i< _n; ++i) {
             _X[i] = _X[i]-thrs;
             if (labels[i]==clas) {
                if (_X[i] < -1.0) {
                   _X[i]=-1.0;
                   ++n_seuils;
                }
             } else {
                if (_X[i] < 0) {
                   ++n_seuils;
                   _X[i]=0;
                }
             }
          }
          mean = this->mean();
          thrs= mean * _n/(_n-n_seuils);*/
          //}
};

template <typename floating_type>
inline void Vector<floating_type>::sparseProject(Vector<floating_type>& out, const floating_type thrs, const int mode, const floating_type lambda_1,
    const floating_type lambda_2, const floating_type lambda_3, const bool pos) {
    if (mode == 1) {
        /// min_u ||b-u||_2^2 / ||u||_1 <= thrs
        this->l1project(out, thrs, pos);
    }
    else if (mode == 2) {
        /// min_u ||b-u||_2^2 / ||u||_2^2 + lambda_1||u||_1 <= thrs
        if (lambda_1 > 1e-10) {
            this->scal(lambda_1);
            this->l1l2project(out, thrs, 2.0 / (lambda_1 * lambda_1), pos);
            this->scal(floating_type(1.0 / lambda_1));
            out.scal(floating_type(1.0 / lambda_1));
        }
        else {
            out.copy(*this);
            out.normalize2();
            out.scal(sqrt(thrs));
        }
    }
    else if (mode == 3) {
        /// min_u ||b-u||_2^2 / ||u||_1 + (lambda_1/2) ||u||_2^2 <= thrs
        this->l1l2project(out, thrs, lambda_1, pos);
    }
    else if (mode == 4) {
        /// min_u 0.5||b-u||_2^2  + lambda_1||u||_1 / ||u||_2^2 <= thrs
        out.copy(*this);
        if (pos)
            out.thrsPos();
        out.softThrshold(lambda_1);
        floating_type nrm = out.nrm2sq();
        if (nrm > thrs)
            out.scal(sqr_alt<floating_type>(thrs / nrm));
    }
    else if (mode == 5) {
        /// min_u 0.5||b-u||_2^2  + lambda_1||u||_1 +lambda_2 Fused(u) / ||u||_2^2 <= thrs
        //      this->fusedProject(out,lambda_1,lambda_2,100);
        //      floating_type nrm=out.nrm2sq();
        //      if (nrm > thrs)
        //         out.scal(sqr_alt<floating_type>(thrs/nrm));
        //  } else if (mode == 6) {
        /// min_u 0.5||b-u||_2^2  + lambda_1||u||_1 +lambda_2 Fused(u) +0.5lambda_3 ||u||_2^2 
        this->fusedProjectHomotopy(out, lambda_1, lambda_2, lambda_3, true);
    }
    else if (mode == 6) {
        /// min_u ||b-u||_2^2  /  lambda_1||u||_1 +lambda_2 Fused(u) + 0.5lambda3||u||_2^2 <= thrs
        this->fusedProjectHomotopy(out, lambda_1 / thrs, lambda_2 / thrs, lambda_3 / thrs, false);
    }
    else {
        /// min_u ||b-u||_2^2 / (1-lambda_1)*||u||_2^2 + lambda_1||u||_1 <= thrs
        if (lambda_1 < 1e-10) {
            out.copy(*this);
            if (pos)
                out.thrsPos();
            out.normalize2();
            out.scal(sqrt(thrs));
        }
        else if (lambda_1 > 0.999999) {
            this->l1project(out, thrs, pos);
        }
        else {
            this->sparseProject(out, thrs / (1.0 - lambda_1), 2, lambda_1 / (1 - lambda_1), 0, 0, pos);
        }
    }
};

/// returns true if the returned vector is null
template <typename floating_type>
inline void Vector<floating_type>::l1l2projectb(Vector<floating_type>& out, const floating_type thrs, const floating_type gamma, const bool pos,
    const int mode) {
    if (mode == 1) {
        /// min_u ||b-u||_2^2 / ||u||_2^2 + gamma ||u||_1 <= thrs
        this->scal(gamma);
        this->l1l2project(out, thrs, 2.0 / (gamma * gamma), pos);
        this->scal(floating_type(1.0 / gamma));
        out.scal(floating_type(1.0 / gamma));
    }
    else if (mode == 2) {
        /// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
        this->l1l2project(out, thrs, gamma, pos);
    }
    else if (mode == 3) {
        /// min_u 0.5||b-u||_2^2  + gamma||u||_1 / ||u||_2^2 <= thrs
        out.copy(*this);
        if (pos)
            out.thrsPos();
        out.softThrshold(gamma);
        floating_type nrm = out.nrm2();
        if (nrm > thrs)
            out.scal(thrs / nrm);
    }
}

/// returns true if the returned vector is null
/// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
template <typename floating_type>
inline void Vector<floating_type>::l1l2project(Vector<floating_type>& out, const floating_type thrs, const floating_type gamma, const bool pos) const {
    if (gamma == 0)
        return this->l1project(out, thrs, pos);
    out.copy(*this);
    if (pos) {
        out.thrsPos();
    }
    else {
        vAbs<floating_type>(_n, out._X, out._X);
    }
    floating_type norm = out.sum() + gamma * out.nrm2sq();
    if (norm <= thrs) {
        if (!pos) out.copy(*this);
        return;
    }

    /// BEGIN
    floating_type* prU = out._X;
    INTM sizeU = _n;

    floating_type sum = 0;
    INTM sum_card = 0;

    while (sizeU > 0) {
        // put the pivot in prU[0]
        swap(prU[0], prU[sizeU / 2]);
        floating_type pivot = prU[0];
        INTM sizeG = 1;
        floating_type sumG = pivot + 0.5 * gamma * pivot * pivot;

        for (INTM i = 1; i < sizeU; ++i) {
            if (prU[i] >= pivot) {
                sumG += prU[i] + 0.5 * gamma * prU[i] * prU[i];
                swap(prU[sizeG++], prU[i]);
            }
        }
        if (sum + sumG - pivot * (1 + 0.5 * gamma * pivot) * (sum_card + sizeG) <
            thrs * (1 + gamma * pivot) * (1 + gamma * pivot)) {
            sum_card += sizeG;
            sum += sumG;
            prU += sizeG;
            sizeU -= sizeG;
        }
        else {
            ++prU;
            sizeU = sizeG - 1;
        }
    }
    floating_type a = gamma * gamma * thrs + 0.5 * gamma * sum_card;
    floating_type b = 2 * gamma * thrs + sum_card;
    floating_type c = thrs - sum;
    floating_type delta = b * b - 4 * a * c;
    floating_type lambda_1 = (-b + sqrt(delta)) / (2 * a);

    out.copy(*this);
    if (pos) {
        out.thrsPos();
    }
    out.fastSoftThrshold(lambda_1);
    out.scal(floating_type(1.0 / (1 + lambda_1 * gamma)));
};

template <typename floating_type>
static inline floating_type fusedHomotopyAux(const bool& sign1,
    const bool& sign2,
    const bool& sign3,
    const floating_type& c1,
    const floating_type& c2) {
    if (sign1) {
        if (sign2) {
            return sign3 ? 0 : c2;
        }
        else {
            return sign3 ? -c2 - c1 : -c1;
        }
    }
    else {
        if (sign2) {
            return sign3 ? c1 : c1 + c2;
        }
        else {
            return sign3 ? -c2 : 0;
        }
    }
};

template <typename floating_type>
inline void Vector<floating_type>::fusedProjectHomotopy(Vector<floating_type>& alpha,
    const floating_type lambda_1, const floating_type lambda_2, const floating_type lambda_3,
    const bool penalty) {
    floating_type* pr_DtR = _X;
    const INTM K = _n;
    alpha.setZeros();
    Vector<floating_type> u(K); // regularization path for gamma
    Vector<floating_type> Du(K); // regularization path for alpha
    Vector<floating_type> DDu(K); // regularization path for alpha
    Vector<floating_type> gamma(K); // auxiliary variable
    Vector<floating_type> c(K); // auxiliary variables
    Vector<floating_type> scores(K); // auxiliary variables
    gamma.setZeros();
    floating_type* pr_gamma = gamma.rawX();
    floating_type* pr_u = u.rawX();
    floating_type* pr_Du = Du.rawX();
    floating_type* pr_DDu = DDu.rawX();
    floating_type* pr_c = c.rawX();
    floating_type* pr_scores = scores.rawX();
    Vector<INTM> ind(K + 1);
    Vector<bool> signs(K);
    ind.set(K);
    INTM* pr_ind = ind.rawX();
    bool* pr_signs = signs.rawX();

    /// Computation of DtR
    floating_type sumBeta = this->sum();

    /// first element is selected, gamma and alpha are updated
    pr_gamma[0] = sumBeta / K;
    /// update alpha
    alpha.set(pr_gamma[0]);
    /// update DtR
    this->sub(alpha);
    for (INTM j = K - 2; j >= 0; --j)
        pr_DtR[j] += pr_DtR[j + 1];

    pr_DtR[0] = 0;
    pr_ind[0] = 0;
    pr_signs[0] = pr_DtR[0] > 0;
    pr_c[0] = floating_type(1.0) / K;
    INTM currentInd = this->fmax();
    floating_type currentLambda = abs<floating_type>(pr_DtR[currentInd]);
    bool newAtom = true;

    /// Solve the Lasso using simplified LARS
    for (INTM i = 1; i < K; ++i) {
        /// exit if constraINTMs are satisfied
        /// min_u ||b-u||_2^2  +  lambda_1||u||_1 +lambda_2 Fused(u) + 0.5lambda3||u||_2^2 
        if (penalty && currentLambda <= lambda_2) break;
        if (!penalty) {
            /// min_u ||b-u||_2^2  /  lambda_1||u||_1 +lambda_2 Fused(u) + 0.5lambda3||u||_2^2 <= 1.0
            scores.copy(alpha);
            scores.softThrshold(lambda_1 * currentLambda / lambda_2);
            scores.scal(floating_type(1.0 / (1.0 + lambda_3 * currentLambda / lambda_2)));
            if (lambda_1 * scores.asum() + lambda_2 * scores.afused() + 0.5 *
                lambda_3 * scores.nrm2sq() >= floating_type(1.0)) break;
        }

        /// Update pr_ind and pr_c
        if (newAtom) {
            INTM j;
            for (j = 1; j < i; ++j)
                if (pr_ind[j] > currentInd) break;
            for (INTM k = i; k > j; --k) {
                pr_ind[k] = pr_ind[k - 1];
                pr_c[k] = pr_c[k - 1];
                pr_signs[k] = pr_signs[k - 1];
            }
            pr_ind[j] = currentInd;
            pr_signs[j] = pr_DtR[currentInd] > 0;
            pr_c[j - 1] = floating_type(1.0) / (pr_ind[j] - pr_ind[j - 1]);
            pr_c[j] = floating_type(1.0) / (pr_ind[j + 1] - pr_ind[j]);
        }

        // Compute u
        pr_u[0] = pr_signs[1] ? -pr_c[0] : pr_c[0];
        if (i == 1) {
            pr_u[1] = pr_signs[1] ? pr_c[0] + pr_c[1] : -pr_c[0] - pr_c[1];
        }
        else {
            pr_u[1] = pr_signs[1] ? pr_c[0] + pr_c[1] : -pr_c[0] - pr_c[1];
            pr_u[1] += pr_signs[2] ? -pr_c[1] : pr_c[1];
            for (INTM j = 2; j < i; ++j) {
                pr_u[j] = 2 * fusedHomotopyAux<floating_type>(pr_signs[j - 1],
                    pr_signs[j], pr_signs[j + 1], pr_c[j - 1], pr_c[j]);
            }
            pr_u[i] = pr_signs[i - 1] ? -pr_c[i - 1] : pr_c[i - 1];
            pr_u[i] += pr_signs[i] ? pr_c[i - 1] + pr_c[i] : -pr_c[i - 1] - pr_c[i];
        }

        // Compute Du 
        pr_Du[0] = pr_u[0];
        for (INTM k = 1; k < pr_ind[1]; ++k)
            pr_Du[k] = pr_Du[0];
        for (INTM j = 1; j <= i; ++j) {
            pr_Du[pr_ind[j]] = pr_Du[pr_ind[j] - 1] + pr_u[j];
            for (INTM k = pr_ind[j] + 1; k < pr_ind[j + 1]; ++k)
                pr_Du[k] = pr_Du[pr_ind[j]];
        }

        /// Compute DDu 
        DDu.copy(Du);
        for (INTM j = K - 2; j >= 0; --j)
            pr_DDu[j] += pr_DDu[j + 1];

        /// Check constraINTMs
        floating_type max_step1 = INFINITY;
        if (penalty) {
            max_step1 = currentLambda - lambda_2;
        }

        /// Check changes of sign
        floating_type max_step2 = INFINITY;
        INTM step_out = -1;
        for (INTM j = 1; j <= i; ++j) {
            floating_type ratio = -pr_gamma[pr_ind[j]] / pr_u[j];
            if (ratio > 0 && ratio <= max_step2) {
                max_step2 = ratio;
                step_out = j;
            }
        }
        floating_type max_step3 = INFINITY;
        /// Check new variables entering the active set
        for (INTM j = 1; j < K; ++j) {
            floating_type sc1 = (currentLambda - pr_DtR[j]) / (floating_type(1.0) - pr_DDu[j]);
            floating_type sc2 = (currentLambda + pr_DtR[j]) / (floating_type(1.0) + pr_DDu[j]);
            if (sc1 <= 1e-10) sc1 = INFINITY;
            if (sc2 <= 1e-10) sc2 = INFINITY;
            pr_scores[j] = MIN(sc1, sc2);
        }
        for (INTM j = 0; j <= i; ++j) {
            pr_scores[pr_ind[j]] = INFINITY;
        }
        currentInd = scores.fmin();
        max_step3 = pr_scores[currentInd];
        floating_type step = MIN(max_step1, MIN(max_step3, max_step2));
        if (step == 0 || step == INFINITY) break;

        /// Update gamma, alpha, DtR, currentLambda
        for (INTM j = 0; j <= i; ++j) {
            pr_gamma[pr_ind[j]] += step * pr_u[j];
        }
        alpha.add(Du, step);
        this->add(DDu, -step);
        currentLambda -= step;
        if (step == max_step2) {
            /// Update signs,pr_ind, pr_c
            for (INTM k = step_out; k <= i; ++k)
                pr_ind[k] = pr_ind[k + 1];
            pr_ind[i] = K;
            for (INTM k = step_out; k <= i; ++k)
                pr_signs[k] = pr_signs[k + 1];
            pr_c[step_out - 1] = floating_type(1.0) / (pr_ind[step_out] - pr_ind[step_out - 1]);
            pr_c[step_out] = floating_type(1.0) / (pr_ind[step_out + 1] - pr_ind[step_out]);
            i -= 2;
            newAtom = false;
        }
        else {
            newAtom = true;
        }
    }

    if (penalty) {
        alpha.softThrshold(lambda_1);
        alpha.scal(floating_type(1.0 / (1.0 + lambda_3)));
    }
    else {
        alpha.softThrshold(lambda_1 * currentLambda / lambda_2);
        alpha.scal(floating_type(1.0 / (1.0 + lambda_3 * currentLambda / lambda_2)));
    }
};

template <typename floating_type>
inline void Vector<floating_type>::fusedProject(Vector<floating_type>& alpha, const floating_type lambda_1, const floating_type lambda_2,
    const int itermax) {
    floating_type* pr_alpha = alpha.rawX();
    floating_type* pr_beta = _X;
    const INTM K = alpha.n();

    floating_type total_alpha = alpha.sum();
    /// Modification of beta
    for (INTM i = K - 2; i >= 0; --i)
        pr_beta[i] += pr_beta[i + 1];

    for (INTM i = 0; i < itermax; ++i) {
        floating_type sum_alpha = 0;
        floating_type sum_diff = 0;
        /// Update first coordinate
        floating_type gamma_old = pr_alpha[0];
        pr_alpha[0] = (K * gamma_old + pr_beta[0] -
            total_alpha) / K;
        floating_type diff = pr_alpha[0] - gamma_old;
        sum_diff += diff;
        sum_alpha += pr_alpha[0];
        total_alpha += K * diff;

        /// Update alpha_j
        for (INTM j = 1; j < K; ++j) {
            pr_alpha[j] += sum_diff;
            floating_type gamma_old = pr_alpha[j] - pr_alpha[j - 1];
            floating_type gamma_new = softThrs((K - j) * gamma_old + pr_beta[j] -
                (total_alpha - sum_alpha), lambda_2) / (K - j);
            pr_alpha[j] = pr_alpha[j - 1] + gamma_new;
            floating_type diff = gamma_new - gamma_old;
            sum_diff += diff;
            sum_alpha += pr_alpha[j];
            total_alpha += (K - j) * diff;
        }
    }
    alpha.softThrshold(lambda_1);

};

/// sort the vector
template <typename floating_type>
inline void Vector<floating_type>::sort(const bool mode) {
    if (mode) {
        lasrt<floating_type>(incr, _n, _X);
    }
    else {
        lasrt<floating_type>(decr, _n, _X);
    }
};


/// sort the vector
template <typename floating_type>
inline void Vector<floating_type>::sort(Vector<floating_type>& out, const bool mode) const {
    out.copy(*this);
    out.sort(mode);
};

template <typename floating_type>
inline void Vector<floating_type>::sort2(Vector<INTM>& key, const bool mode) {
    quick_sort(key.rawX(), _X, (INTM)0, _n - 1, mode);
};


template <typename floating_type>
inline void Vector<floating_type>::sort2(Vector<floating_type>& out, Vector<INTM>& key, const bool mode) const {
    out.copy(*this);
    out.sort2(key, mode);
}

template <typename floating_type>
inline void Vector<floating_type>::applyBayerPattern(const int offset) {
    INTM sizePatch = _n / 3;
    INTM n = static_cast<INTM>(sqrt(static_cast<floating_type>(sizePatch)));
    if (offset == 0) {
        // R
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 1 : 2;
            const INTM off = (i % 2) ? 0 : 1;
            for (INTM j = off; j < n; j += step) {
                _X[i * n + j] = 0;
            }
        }
        // G
        for (INTM i = 0; i < n; ++i) {
            const INTM step = 2;
            const INTM off = (i % 2) ? 1 : 0;
            for (INTM j = off; j < n; j += step) {
                _X[sizePatch + i * n + j] = 0;
            }
        }
        // B
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 2 : 1;
            const INTM off = 0;
            for (INTM j = off; j < n; j += step) {
                _X[2 * sizePatch + i * n + j] = 0;
            }
        }
    }
    else if (offset == 1) {
        // R
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 2 : 1;
            const INTM off = (i % 2) ? 1 : 0;
            for (INTM j = off; j < n; j += step) {
                _X[i * n + j] = 0;
            }
        }
        // G
        for (INTM i = 0; i < n; ++i) {
            const INTM step = 2;
            const INTM off = (i % 2) ? 0 : 1;
            for (INTM j = off; j < n; j += step) {
                _X[sizePatch + i * n + j] = 0;
            }
        }
        // B
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 1 : 2;
            const INTM off = 0;
            for (INTM j = off; j < n; j += step) {
                _X[2 * sizePatch + i * n + j] = 0;
            }
        }
    }
    else if (offset == 2) {
        // R
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 1 : 2;
            const INTM off = 0;
            for (INTM j = off; j < n; j += step) {
                _X[i * n + j] = 0;
            }
        }
        // G
        for (INTM i = 0; i < n; ++i) {
            const INTM step = 2;
            const INTM off = (i % 2) ? 0 : 1;
            for (INTM j = off; j < n; j += step) {
                _X[sizePatch + i * n + j] = 0;
            }
        }
        // B
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 2 : 1;
            const INTM off = (i % 2) ? 1 : 0;
            for (INTM j = off; j < n; j += step) {
                _X[2 * sizePatch + i * n + j] = 0;
            }
        }
    }
    else if (offset == 3) {
        // R
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 2 : 1;
            const INTM off = 0;
            for (INTM j = off; j < n; j += step) {
                _X[i * n + j] = 0;
            }
        }
        // G
        for (INTM i = 0; i < n; ++i) {
            const INTM step = 2;
            const INTM off = (i % 2) ? 1 : 0;
            for (INTM j = off; j < n; j += step) {
                _X[sizePatch + i * n + j] = 0;
            }
        }
        // B
        for (INTM i = 0; i < n; ++i) {
            const INTM step = (i % 2) ? 1 : 2;
            const INTM off = (i % 2) ? 0 : 1;
            for (INTM j = off; j < n; j += step) {
                _X[2 * sizePatch + i * n + j] = 0;
            }
        }
    }
};


/// make a sparse copy 
template <typename floating_type> inline void Vector<floating_type>::toSparse(
    SpVector<floating_type>& vec) const {
    INTM L = 0;
    floating_type* v = vec._v;
    INTM* r = vec._r;
    for (INTM i = 0; i < _n; ++i) {
        if (_X[i] != floating_type()) {
            v[L] = _X[i];
            r[L++] = i;
        }
    }
    vec._L = L;
};


template <typename floating_type>
inline void Vector<floating_type>::copyMask(Vector<floating_type>& out, Vector<bool>& mask) const {
    out.resize(_n);
    INTM pointer = 0;
    for (INTM i = 0; i < _n; ++i) {
        if (mask[i])
            out[pointer++] = _X[i];
    }
    out.setn(pointer);
};

/// Class for dense vector
template<typename floating_type, typename I> class LazyVector {
public:
    LazyVector(Vector<floating_type>& x, const Vector<floating_type>& z, const int n) : _x(x), _z(z), _n(n + 1), _p(x.n()) {
        _current_time = 0;
        _dates.resize(_p);
        _dates.setZeros();
        _stats1.resize(n + 1);
        _stats2.resize(n + 1);
        _stats1[0] = floating_type(1.0);
        _stats2[0] = 0;
    };
    void inline update() {
        for (int ii = 0; ii < _p; ++ii) {
            update(ii);
        }
        _current_time = 0;
        _dates.setZeros();
    };
    void inline update(const I ind) {
        const int last_time = _dates[ind];
        if (last_time != _current_time) {
            _x[ind] = (_stats1[_current_time] / _stats1[last_time]) * _x[ind] + _stats1[_current_time] * (_stats2[_current_time] - _stats2[last_time]) * _z[ind];
            _dates[ind] = _current_time;
        }
    };
    void inline update(const Vector<I>& indices) {
        const int p = indices.n();
        for (int ii = 0; ii < p; ++ii) {
            update(indices[ii]);
        }
    };
    void inline add_scal(const floating_type a, const floating_type b) { // performs x <- a(x - b z) 
        if (_current_time == _n)
            update();
        _current_time++;
        _stats2[_current_time] = _stats2[_current_time - 1] + a / _stats1[_current_time - 1];
        _stats1[_current_time] = _stats1[_current_time - 1] * b;
        if (_stats1[_current_time] < 1e-7)
            update(); // to prevent numerical stability problems
    };

private:
    Vector<floating_type>& _x;
    const Vector<floating_type>& _z;
    const int _n;
    const int _p;
    Vector<floating_type> _stats1, _stats2;
    Vector<int> _dates;
    int _current_time;
};

/// Class for dense vector
template<typename floating_type, typename I> class DoubleLazyVector {
public:
    DoubleLazyVector(Vector<floating_type>& x, const Vector<floating_type>& z1, const Vector<floating_type>& z2, const int n) : _x(x), _z1(z1), _z2(z2), _n(n + 1), _p(x.n()) {
        _current_time = 0;
        _dates.resize(_p);
        _dates.setZeros();
        _stats1.resize(n + 1);
        _stats2.resize(n + 1);
        _stats3.resize(n + 1);
        _stats1[0] = floating_type(1.0);
        _stats2[0] = 0;
        _stats3[0] = 0;
    };
    void inline update() {
        for (int ii = 0; ii < _p; ++ii) {
            update(ii);
        }
        _current_time = 0;
        _dates.setZeros();
    };
    void inline update(const I ind) {
        const int last_time = _dates[ind];
        if (last_time != _current_time) {
            _x[ind] = _stats1[_current_time] * (_x[ind] / _stats1[last_time] + (_stats2[_current_time] - _stats2[last_time]) * _z1[ind] + (_stats3[_current_time] - _stats3[last_time]) * _z2[ind]);
            _dates[ind] = _current_time;
        }
    };
    void inline update(const Vector<I>& indices) {
        const int p = indices.n();
        for (int ii = 0; ii < p; ++ii) {
            update(indices[ii]);
        }
    };
    void inline add_scal(const floating_type a, const floating_type b, const floating_type c) {
        if (_current_time == _n)
            update();
        _current_time++;
        _stats1[_current_time] = _stats1[_current_time - 1] * c;
        _stats2[_current_time] = _stats2[_current_time - 1] + a / _stats1[_current_time];
        _stats3[_current_time] = _stats3[_current_time - 1] + b / _stats1[_current_time];
        if (_stats1[_current_time] < 1e-6)
            update(); // to prevent numerical stability problems
    };

private:
    Vector<floating_type>& _x;
    const Vector<floating_type>& _z1;
    const Vector<floating_type>& _z2;
    const int _n;
    const int _p;
    Vector<floating_type> _stats1, _stats2, _stats3;
    Vector<int> _dates;
    int _current_time;
};


#endif