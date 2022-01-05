/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

/* \file
 *                toolbox Linalg 
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File linalg.h
 * \brief Contains Matrix, Vector classes */

#ifndef LINALG_H
#define LINALG_H

#include "misc.h"
#include "cblas_alt_template.h"
#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif
#include "utils.h"

#undef max
#undef min

/// Dense OptimInfo class
template<typename T> class OptimInfo;
/// Sparse OptimInfo class
template<typename T, typename I = INTM> class SpOptimInfo;
/// Dense Matrix class
template<typename T> class Matrix;
/// Sparse Matrix class
template<typename T, typename I = INTM> class SpMatrix;
/// Dense Vector class
template<typename T> class Vector;
/// Sparse Vector class
template<typename T, typename I = INTM> class SpVector;

template <typename T> 
static inline bool isZero(const T lambda_1) {
   return static_cast<double>(abs<T>(lambda_1)) < 1e-99;
}

template <typename T> 
static inline bool isEqual(const T lambda_1, const T lambda_2) {
   return static_cast<double>(abs<T>(lambda_1-lambda_2)) < 1e-99;
}


template <typename T>
static inline T softThrs(const T x, const T lambda_1) {
   if (x > lambda_1) {
      return x-lambda_1;
   } else if (x < -lambda_1) {
      return x+lambda_1;
   } else {
      return 0;
   }
};

template <typename T>
static inline T fastSoftThrs(const T x, const T lambda_1) {
   return x + T(0.5)*(abs<T>(x-lambda_1) - abs<T>(x+lambda_1));
};


template <typename T>
static inline T hardThrs(const T x, const T lambda_1) {
   return (x > lambda_1 || x < -lambda_1) ? x : 0;
};

template <typename T>
static inline T xlogx(const T x) {
   if (x < -1e-20) {
      return INFINITY;
   } else if (x < 1e-20) {
      return 0;
   } else {
      return x*alt_log<T>(x);
   }
}

template <typename T>
static inline T logexp(const T x) {
   if (x < -30) {
      return 0;
   } else if (x < 30) {
      return alt_log<T>( T(1.0) + exp_alt<T>( x ) );
   } else {
      return x;
   }
}

template <typename T>
static inline T logexp2(const T x) {
   return (x > 0) ? x + log_alt<T>(T(1.0)+ exp_alt<T>(-x)) :
      log( T(1.0) + exp_alt<T>( x ) );
}

template <typename T>
static T solve_binomial(const T a, const T b, const T c) {
   const T delta = b*b-4*a*c;
   return (-b + alt_sqrt<T>(delta))/(2*a); // returns single largest solution, assiming delta > 0;
};

template <typename T>
static T solve_binomial2(const T a, const T b, const T c) {
   const T delta = b*b-4*a*c;
   return (-b - alt_sqrt<T>(delta))/(2*a); // returns single largest solution, assiming delta > 0;
};

/// Class OptimInfo
template<typename T> class OptimInfo {
    public:
    typedef T value_type;
    typedef Vector<T> col_type;
    typedef INTM index_type;
    typedef Vector<T> element;

    /// Constructor with existing data X of an nclass x m x n matrix
    OptimInfo(T* X, INTM nclass, INTM m, INTM n);
    /// Constructor for a new m x n matrix
    OptimInfo(INTM nclass, INTM m, INTM n);
    /// Empty constructor
    OptimInfo();

    /// Destructor
    virtual ~OptimInfo();

    /// Accessors
    /// Number of class
    inline INTM nclass() const { return _nclass; };
    /// Number of rows
    inline INTM m() const { return _m; };
    /// Number of columns
    inline INTM n() const { return _n; };
    /// size
    inline INTM size() const { return _nclass*_n*_m; };
    /// Return a modifiable reference to X(i,j,k)
    inline T& operator()(const INTM i, const INTM j, const INTM k);
    /// Return the value X(i,j,k)
    inline T operator()(const INTM i, const INTM j, const INTM k) const;
    /// Return a modifiable reference to X(i) (1D indexing)
    inline T& operator[](const INTM index) { return _X[index]; };
    /// Return the value X(i) (1D indexing)
    inline T operator[](const INTM index) const { return _X[index]; };
    /// Copy the column i into x
    inline void copyCol(const INTM i, Vector<T>& x) const;
    /// make a copy of the OptimInfo optim in the current OptimInfo
   inline void copy(const OptimInfo<T>& optim);
   /// Set all the values to zero
   inline void setZeros();
   /// Resize the optiminfo
   inline void resize(INTM nclass,INTM m, INTM n, const bool set_zeros = true);
   /// add alpha*optimInfo to the current matrix
   inline void add(const OptimInfo<T>& mat, const int index, const T alpha = 1.0);

   /// Change the data in the optimInfo
   inline void setData(T* X, INTM nclass, INTM m, INTM n);

    /// Debugging function
   /// Print the matrix to std::cout
   inline void print(const string& name) const;
   inline void dump(const string& name) const;

    /// clear the vector
   inline void clear();

   typedef Vector<T> col;
   typedef Matrix<T> mat;
   static const bool is_sparse = false;

   protected:
   /// Forbid lazy copies
   explicit OptimInfo<T>(const OptimInfo<T>& matrix);
   /// Forbid lazy copies
   OptimInfo<T>& operator=(const OptimInfo<T>& matrix);

   /// is the data allocation external or not
   bool _externAlloc;
   /// pointer to the data
   T* _X;
   /// number of class
   INTM _nclass;
   /// number of rows
   INTM _m;
   /// number of columns
   INTM _n;

};

/// Class Matrix
template<typename T> class Matrix {
   friend class SpMatrix<T>;
   public:
   typedef T value_type;
   typedef Vector<T> col_type;
   typedef INTM index_type;
   typedef Vector<T> element;

   /// Constructor with existing data X of an m x n matrix
   Matrix(T* X, INTM m, INTM n);
   /// Constructor for a new m x n matrix
   Matrix(INTM m, INTM n);
   /// Empty constructor
   Matrix();

   /// Destructor
   virtual ~Matrix();

   /// Accessors
   /// Number of rows
   inline INTM m() const { return _m; };
   /// Number of columns
   inline INTM n() const { return _n; };
   /// size
   inline INTM size() const { return _n*_m; };
   /// Return a modifiable reference to X(i,j)
   inline T& operator()(const INTM i, const INTM j);
   /// Return the value X(i,j)
   inline T operator()(const INTM i, const INTM j) const;
   /// Return a modifiable reference to X(i) (1D indexing)
   inline T& operator[](const INTM index) { return _X[index]; };
   /// Return the value X(i) (1D indexing)
   inline T operator[](const INTM index) const { return _X[index]; };
   /// Copy the column i into x
   inline void copyCol(const INTM i, Vector<T>& x) const;
   /// Copy the column i into x
   inline void copyRow(const INTM i, Vector<T>& x) const;
   inline void scalRow(const INTM i, const T s) const;
   inline void copyToRow(const INTM i, const Vector<T>& x);
   /// Copy the column i into x
   inline void extract_rawCol(const INTM i, T* x) const;
   /// Copy the column i into x
   virtual void add_rawCol(const INTM i, T* DtXi, const T a) const;
   /// Copy the column i into x
   inline void getData(Vector<T>& data, const INTM i) const;
   /// Reference the column i into the vector x
   inline void refCol(INTM i, Vector<T>& x) const;
   /// Reference the column i to i+n into the Matrix mat
   inline void refSubMat(INTM i, INTM n, Matrix<T>& mat) const;
   /// extract a sub-matrix of a symmetric matrix
   inline void subMatrixSym(const Vector<INTM>& indices, 
         Matrix<T>& subMatrix) const;
   /// reference a modifiable reference to the data, DANGEROUS
   inline T* rawX() const { return _X; };
   /// return a non-modifiable reference to the data
   inline const T* X() const { return _X; };
   /// make a copy of the matrix mat in the current matrix
   inline void copy(const Matrix<T>& mat);
   /// make a copy of the matrix mat in the current matrix
   inline void copyTo(Matrix<T>& mat) const { mat.copy(*this); };
   /// make a copy of the matrix mat in the current matrix
   inline void copyRef(const Matrix<T>& mat);

   /// Debugging function
   /// Print the matrix to std::cout
   inline void print(const string& name) const;
   inline void dump(const string& name) const;


   /// Modifiers
   /// clean a dictionary matrix
   inline void clean();
   /// Resize the matrix
   inline void resize(INTM m, INTM n, const bool set_zeros = true);
   /// Change the data in the matrix
   inline void setData(T* X, INTM m, INTM n);
   /// Change the data in the matrix
   inline void refData(const Matrix<T>& mat) {
      this->setData(mat.rawX(),mat.m(),mat.n());
   };
   /// modify _m
   inline void setm(const INTM m) { _m = m; }; //DANGEROUS
   /// modify _n
   inline void setn(const INTM n) { _n = n; }; //DANGEROUS
   /// Set all the values to zero
   inline void setZeros();
   /// Set all the values to a scalar
   inline void set(const T a);
   /// Clear the matrix
   inline void clear();
   /// Put white Gaussian noise in the matrix 
   inline void setAleat();
   /// set the matrix to the identity;
   inline void eye();
   /// Normalize all columns to unit l2 norm
   inline void normalize();
   /// Normalize all columns which l2 norm is greater than one.
   inline void normalize2();
   /// center the columns of the matrix
   inline void center();
   /// center the columns of the matrix
   inline void center_rows();
   /// center the columns of the matrix
   inline void normalize_rows();
   /// center the columns of the matrix and keep the center values
   inline void center(Vector<T>& centers);
   /// scale the matrix by the a
   inline void scal(const T a);
   /// make the matrix symmetric by copying the upper-right part
   /// into the lower-left part
   inline void fillSymmetric();
   inline void fillSymmetric2();
   /// change artificially the size of the matrix, DANGEROUS
   inline void fakeSize(const INTM m, const INTM n) { _n = n; _m=m;};
   /// whiten
   inline void whiten(const INTM V);
   /// whiten
   inline void whiten(Vector<T>& mean, const bool pattern = false);
   /// whiten
   inline void whiten(Vector<T>& mean, const Vector<T>& mask);
   /// whiten
   inline void unwhiten(Vector<T>& mean, const bool pattern = false);
   /// whiten
   inline void sum_cols(Vector<T>& sum) const;

   /// Analysis functions
   /// Check wether the columns of the matrix are normalized or not
   inline bool isNormalized() const;
   /// return the 1D-index of the value of greatest magnitude
   inline INTM fmax() const;
   /// return the 1D-index of the value of greatest magnitude
   inline T fmaxval() const;
   /// return the 1D-index of the value of lowest magnitude
   inline INTM fmin() const;

   // Algebric operations
   /// Transpose the current matrix and put the result in the matrix
   /// trans
   inline void transpose(Matrix<T>& trans) const;
   /// A <- -A
   inline void neg();
   /// add one to the diagonal
   inline void incrDiag();
   inline void addDiag(const Vector<T>& diag);
   inline void addDiag(const T diag);
   inline void addToCols(const Vector<T>& diag);
   inline void addVecToCols(const Vector<T>& diag, const T a = 1.0);
   /// perform a rank one approximation uv' using the power method
   /// u0 is an initial guess for u (can be empty).
   inline void svdRankOne(const Vector<T>& u0,
         Vector<T>& u, Vector<T>& v) const;
   inline void singularValues(Vector<T>& u) const;
   inline void svd(Matrix<T>& U, Vector<T>& S, Matrix<T>&V) const;
   inline void svd2(Matrix<T>& U, Vector<T>& S, const int num = -1, const int method = 0) const;
   inline void SymEig(Matrix<T>& U, Vector<T>& S) const;
   inline void InvsqrtMat(Matrix<T>& out, const T lambda_1 = 0) const;
   inline void sqrtMat(Matrix<T>& out) const;
//   inline void Inv(Matrix<T>& out) const;

   /// find the eigenvector corresponding to the largest eigenvalue
   /// when the current matrix is symmetric. u0 is the initial guess.
   /// using two iterations of the power method
   inline void eigLargestSymApprox(const Vector<T>& u0,
         Vector<T>& u) const;
   /// find the eigenvector corresponding to the eivenvalue with the 
   /// largest magnitude when the current matrix is symmetric,
   /// using the power method. It 
   /// returns the eigenvalue. u0 is an initial guess for the 
   /// eigenvector.
   inline T eigLargestMagnSym(const Vector<T>& u0, 
         Vector<T>& u) const;
   /// returns the value of the eigenvalue with the largest magnitude
   /// using the power iteration.
   inline T eigLargestMagnSym() const;
   /// inverse the matrix when it is symmetric
   inline void invSym();
   inline void invSymPos();
   /// perform b = alpha*A'x + beta*b
   inline void multTrans(const Vector<T>& x, Vector<T>& b,
         const T alpha = 1.0, const T beta = 0.0) const;
   /// perform b = alpha*A'x + beta*b
   inline void multTrans(const Vector<T>& x, Vector<T>& b,
         const Vector<bool>& active) const;
   /// perform b = A'x, when x is sparse
   template <typename I>
   inline void multTrans(const SpVector<T,I>& x, Vector<T>& b, const T alpha =1.0, const T beta = 0.0) const;
   /// perform b = alpha*A*x+beta*b
   inline void mult(const Vector<T>& x, Vector<T>& b, 
         const T alpha = 1.0, const T beta = 0.0) const;
   inline void mult_loop(const Vector<T>& x, Vector<T>& b) const;

   /// perform b = alpha*A*x + beta*b, when x is sparse
   template <typename I>
   inline void mult(const SpVector<T,I>& x, Vector<T>& b, 
         const T alpha = 1.0, const T beta = 0.0) const;
   template <typename I>
   inline void mult_loop(const SpVector<T,I>& x, Vector<T>& b) const {
      this->mult(x,b);
   }
   /// perform C = a*A*B + b*C, possibly transposing A or B.
   inline void mult(const Matrix<T>& B, Matrix<T>& C, 
         const bool transA = false, const bool transB = false,
         const T a = 1.0, const T b = 0.0) const;
   /// perform C = a*B*A + b*C, possibly transposing A or B.
   inline void multSwitch(const Matrix<T>& B, Matrix<T>& C, 
         const bool transA = false, const bool transB = false,
         const T a = 1.0, const T b = 0.0) const;
   /// perform C = A*B, when B is sparse
   template <typename I>
   inline void mult(const SpMatrix<T,I>& B, Matrix<T>& C, const bool transA = false,
         const bool transB = false, const T a = 1.0,
         const T b = 0.0) const;
   /// mult by a diagonal matrix on the left
   inline void multDiagLeft(const Vector<T>& diag);
   /// mult by a diagonal matrix on the right
   inline void multDiagRight(const Vector<T>& diag);
   /// mult by a diagonal matrix on the right
   inline void AddMultDiagRight(const Vector<T>& diag, Matrix<T>& mat);
   /// C = A .* B, elementwise multiplication
   inline void mult_elementWise(const Matrix<T>& B, Matrix<T>& C) const;
   inline void div_elementWise(const Matrix<T>& B, Matrix<T>& C) const;
   /// XtX = A'*A
   inline void XtX(Matrix<T>& XtX) const;
   /// XXt = A*A'
   inline void XXt(Matrix<T>& XXt) const;
   /// XXt = A*A' where A is an upper triangular matrix
   inline void upperTriXXt(Matrix<T>& XXt, 
         const INTM L) const;
   /// extract the diagonal
   inline void diag(Vector<T>& d) const;
   /// set the diagonal
   inline void setDiag(const Vector<T>& d);
   /// set the diagonal
   inline void setDiag(const T val);
   /// each element of the matrix is replaced by its exponential
   inline void exp();
   /// each element of the matrix is replaced by its square root
   inline void pow(const T a);
   inline void Sqrt();
   inline void Invsqrt();
   inline void sqr();
   /// return vec1'*A*vec2, where vec2 is sparse
   template <typename I>
   inline T quad(const Vector<T>& vec1, const SpVector<T,I>& vec2) const;
   /// return vec1'*A*vec2, where vec2 is sparse
   template <typename I>
   inline void quad_mult(const Vector<T>& vec1, const SpVector<T,I>& vec2,
         Vector<T>& y, const T a = 1.0, const T b = 0.0) const;
   /// return vec'*A*vec when vec is sparse
   template <typename I>
   inline T quad(const SpVector<T,I>& vec) const;
   /// add alpha*mat to the current matrix
   inline void add(const Matrix<T>& mat, const T alpha = 1.0);
   /// add alpha*mat to the current matrix
   inline void add_scal(const Matrix<T>& mat, const T alpha = 1.0, const T beta = 1.0);
   /// add alpha to the current matrix
   inline void add(const T alpha);
   /// add alpha*mat to the current matrix
   inline T dot(const Matrix<T>& mat) const;
   /// substract the matrix mat to the current matrix
   inline void sub(const Matrix<T>& mat);
   /// inverse the elements of the matrix
   inline void inv_elem();
   /// inverse the elements of the matrix
   inline void inv() { this->inv_elem(); };
   /// return the trace of the matrix
   inline T trace() const;
   /// compute the sum of the magnitude of the matrix values
   inline T asum() const;
   /// compute the sum of the magnitude of the matrix values
   inline T sum() const;
   /// return ||A||_F
   inline T normF() const;
   /// whiten
   inline T mean() const;
   /// whiten
   inline T abs_mean() const;
   /// whiten
   /// return ||A||_F^2
   inline T normFsq() const;
   /// return ||A||_F^2
   inline T nrm2sq() const { return this->normFsq(); };
   /// return ||At||_{inf,2} (max of l2 norm of the columns)
   inline T norm_inf_2_col() const;
   /// return ||At||_{1,2} (max of l2 norm of the columns)
   inline T norm_1_2_col() const;
   /// returns the l2 norms of the columns
   inline void norm_2_cols(Vector<T>& norms) const;
   /// returns the l2 norms of the columns
   inline void norm_2_rows(Vector<T>& norms) const;
   /// returns the linf norms of the columns
   inline void norm_inf_cols(Vector<T>& norms) const;
   /// returns the linf norms of the columns
   inline void norm_inf_rows(Vector<T>& norms) const;
   /// returns the linf norms of the columns
   inline void norm_l1_rows(Vector<T>& norms) const;
   /// returns the linf norms of the columns
   inline void get_sum_cols(Vector<T>& sum) const;
   /// returns the linf norms of the columns
   inline void dot_col(const Matrix<T>& mat, Vector<T>& dots) const;
   /// returns the l2 norms ^2 of the columns
   inline void norm_2sq_cols(Vector<T>& norms) const;
   /// returns the l2 norms of the columns
   inline void norm_2sq_rows(Vector<T>& norms) const;
   inline void thrsmax(const T nu);
   inline void thrsmin(const T nu);
   inline void thrsabsmin(const T nu);
   /// perform soft-thresholding of the matrix, with the threshold nu
   inline void softThrshold(const T nu);
   inline void fastSoftThrshold(const T nu);
   inline void fastSoftThrshold(Matrix<T>& output, const T nu) const;
   inline void hardThrshold(const T nu);
   /// perform soft-thresholding of the matrix, with the threshold nu
   inline void thrsPos();
   /// perform A <- A + alpha*vec1*vec2'
   inline void rank1Update(const Vector<T>& vec1, const Vector<T>& vec2,
         const T alpha = 1.0);
   /// perform A <- A + alpha*vec1*vec2', when vec1 is sparse
   template <typename I>
   inline void rank1Update(const SpVector<T,I>& vec1, const Vector<T>& vec2,
         const T alpha = 1.0);
   /// perform A <- A + alpha*vec1*vec2', when vec2 is sparse
   template <typename I>
   inline void rank1Update(const Vector<T>& vec1, const SpVector<T,I>& vec2,
         const T alpha = 1.0);
   template <typename I>
   inline void rank1Update_mult(const Vector<T>& vec1, const Vector<T>& vec1b,
         const SpVector<T,I>& vec2,
         const T alpha = 1.0);
   /// perform A <- A + alpha*vec*vec', when vec2 is sparse
   template <typename I>
   inline void rank1Update(const SpVector<T,I>& vec,
         const T alpha = 1.0);
   /// perform A <- A + alpha*vec*vec', when vec2 is sparse
   template <typename I>
   inline void rank1Update(const SpVector<T,I>& vec, const SpVector<T,I>& vec2,
         const T alpha = 1.0);
   /// Compute the mean of the columns
   inline void meanCol(Vector<T>& mean) const;
   /// Compute the mean of the rows
   inline void meanRow(Vector<T>& mean) const;
   /// fill the matrix with the row given
   inline void fillRow(const Vector<T>& row);
   /// fill the matrix with the row given
   inline void extractRow(const INTM i, Vector<T>& row) const;
   inline void setRow(const INTM i, const Vector<T>& row);
   inline void addRow(const INTM i, const Vector<T>& row, const T a=1.0);
   /// compute x, such that b = Ax, WARNING this function needs to be u
   /// updated
   inline void conjugateGradient(const Vector<T>& b, Vector<T>& x,
         const T tol = 1e-4, const int = 4) const;
   /// compute x, such that b = Ax, WARNING this function needs to be u
   /// updated, the temporary vectors are given.
   inline void drop(char* fileName) const;
   /// compute a Nadaraya Watson estimator
   inline void NadarayaWatson(const Vector<INTM>& ind, const T sigma);
   /// performs soft-thresholding of the vector
   inline void blockThrshold(const T nu, const INTM sizeGroup);
   /// performs sparse projections of the columns 
   inline void sparseProject(Matrix<T>& out, const T thrs,   const int mode = 1, const T lambda_1 = 0,
         const T lambda_2 = 0, const T lambda_3 = 0, const bool pos = false, const int numThreads=-1);
   inline void transformFilter();

   /// Conversion
   /// make a sparse copy of the current matrix
   inline void toSparse(SpMatrix<T>& matrix) const;
   /// make a sparse copy of the current matrix
   inline void toSparseTrans(SpMatrix<T>& matrixTrans);
   /// make a reference of the matrix to a vector vec 
   inline void toVect(Vector<T>& vec) const;
   /// Accessor
   inline INTM V() const { return 1;};
   /// extract the rows of a matrix corresponding to a binary mask
   inline void copyMask(Matrix<T>& out, Vector<bool>& mask) const;

   typedef Vector<T> col;
   static const bool is_sparse = false;

   protected:
   /// Forbid lazy copies
   explicit Matrix<T>(const Matrix<T>& matrix);
   /// Forbid lazy copies
   Matrix<T>& operator=(const Matrix<T>& matrix);

   /// is the data allocation external or not
   bool _externAlloc;
   /// pointer to the data
   T* _X;
   /// number of rows
   INTM _m;
   /// number of columns
   INTM _n;

};



/// Class for dense vector
template<typename T> class Vector {
   friend class SpMatrix<T>;
   friend class Matrix<T>;
   friend class SpVector<T>;
   public:
   typedef T value_type;
   typedef T element;
   /// Empty constructor
   Vector();
   /// Constructor. Create a new vector of size n
   Vector(INTM n);
   /// Constructor with existing data
   Vector(T* X, INTM n);
   /// Copy constructor
   explicit Vector<T>(const Vector<T>& vec);

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
   inline T maxval() const;
   /// returns the minimum value
   inline T minval() const;
   /// returns the index of the value with largest magnitude
   inline INTM fmax() const;
   /// returns the index of the value with smallest magnitude
   inline INTM fmin() const;
   /// returns the maximum magnitude
   inline T fmaxval() const;
   /// returns the minimum magnitude
   inline T fminval() const;
   /// returns a reference to X[index]
   inline T& operator[](const INTM index);
   /// returns X[index]
   inline T operator[](const INTM index) const;
   /// make a copy of x
   inline void copy(const Vector<T>& x);
   inline void copyRef(const Vector<T>& x);
   /// returns the size of the vector
   inline int n() const { return _n; };
   /// returns the size of the vector
   inline int size() const { return _n; };
   /// returns a modifiable reference of the data, DANGEROUS
   inline T* rawX() const { return _X; };
   /// change artificially the size of the vector, DANGEROUS
   inline void fakeSize(const INTM n) { _n = n; };
   /// generate logarithmically spaced values
   inline void logspace(const INTM n, const T a, const T b);
   inline INTM nnz() const;

   /// Modifiers
   /// Set all values to zero
   inline void setZeros();
   /// resize the vector
   inline void resize(const INTM n, const bool set_zeros = true);
   /// change the data of the vector
   inline void setPointer(T* X, const INTM n);
   inline void setData(T* X, const INTM n) { this->setPointer(X,n); };
   inline void refData(const Vector<T>& vec) { this->setPointer(vec.rawX(),vec.n()); };
   inline void refSubVec(INTM i, INTM n, Vector<T>& mat) const { mat.setData(_X+i,n); };
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
   inline void softThrshold(const T nu);
   inline void fastSoftThrshold(const T nu);
   inline void fastSoftThrshold(Vector<T>& out, const T nu) const;
   inline void softThrsholdScal(Vector<T>& out, const T nu, const T s);
   inline void hardThrshold(const T nu);
   /// performs soft-thresholding of the vector
   inline void thrsmax(const T nu);
   inline void thrsmin(const T nu);
   inline void thrsabsmin(const T nu);
   /// performs soft-thresholding of the vector
   inline void thrshold(const T nu);
   /// performs soft-thresholding of the vector
   inline void thrsPos();
   /// set each value of the vector to val
   inline void set(const T val);
   inline void setn(const INTM n) { _n = n; }; //DANGEROUS
   inline bool alltrue() const;
   inline bool allfalse() const;

   /// Algebric operations
   /// returns ||A||_2
   inline T nrm2() const;
   /// returns ||A||_2^2
   inline T nrm2sq() const;
   /// returns  A'x
   inline T dot(const Vector<T>& x) const;
   /// returns A'x, when x is sparse
   template <typename I>
   inline T dot(const SpVector<T,I>& x) const;
   /// A <- A + a*x
   inline void add(const Vector<T>& x, const T a = 1.0);
   /// A <- A + a*x
   template <typename I>
   inline void add(const SpVector<T,I>& x, const T a = 1.0);
   /// adds a to each value in the vector
   inline void add(const T a);
   /// A <- b*A + a*x
   inline void add_scal(const Vector<T>& x, const T a = 1.0, const T b = 0);
   /// A <- b*A + a*x
   template <typename I>
      inline void add_scal(const SpVector<T,I>& x, const T a = 1.0, const T b = 0);
   /// A <- A - x
   inline void sub(const Vector<T>& x);
   /// A <- A + a*x
   template <typename I>
   inline void sub(const SpVector<T,I>& x);
   /// A <- A ./ x
   inline void div(const Vector<T>& x);
   /// A <- x ./ y
   inline void div(const Vector<T>& x, const Vector<T>& y);
   /// A <- x .^ 2
   inline void sqr(const Vector<T>& x);
   /// A <- 1 ./ sqrt(x) 
   inline void sqr();
   /// A <- 1 ./ sqrt(A) 
   inline void Sqrt(const Vector<T>& x);
   /// A <- 1 ./ sqrt(x) 
   inline void Sqrt();
   /// A <- 1 ./ sqrt(x) 
   inline void Invsqrt(const Vector<T>& x);
   /// A <- 1 ./ sqrt(A) 
   inline void Invsqrt();
   /// A <- 1./x
   inline void inv(const Vector<T>& x);
   /// A <- 1./A
   inline void inv();
   /// A <- x .* y
   inline void mult(const Vector<T>& x, const Vector<T>& y);
   inline void mult_elementWise(const Vector<T>& B, Vector<T>& C) const { C.mult(*this,B); };
   /// normalize the vector
   inline void normalize();
   /// normalize the vector
   inline void normalize2(const T thrs = 1.0);
   /// whiten
   inline void whiten(Vector<T>& mean, const bool pattern = false);
   /// whiten
   inline void whiten(Vector<T>& mean, const
         Vector<T>& mask);
   /// whiten
   inline void whiten(const INTM V);
   /// whiten
   inline T mean() const;
   inline T abs_mean() const;
   inline T mean_non_uniform(const Vector<T>& qi) const;
   /// whiten
   inline T std();
   /// compute the Kuhlback-Leiber divergence
   inline T KL(const Vector<T>& X);
   /// whiten
   inline void unwhiten(Vector<T>& mean, const bool pattern = false);
   /// scale the vector by a
   inline void scal(const T a);
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
   inline T softmax(const int y);
   inline T logsumexp();
   /// computes the sum of the magnitudes of the vector
   inline T asum() const;
   inline T lzero() const;
   /// compute the sum of the differences
   inline T afused() const;
   /// returns the sum of the vector
   inline T sum() const;
   /// puts in signs, the sign of each point in the vector
   inline void sign(Vector<T>& signs) const;
   /// projects the vector onto the l1 ball of radius thrs,
   /// returns true if the returned vector is null
   inline void l1project(Vector<T>& out, const T thrs, const bool simplex = false) const;
   inline void l1project_weighted(Vector<T>& out, const Vector<T>& weights, const T thrs, const bool residual = false) const;
   inline void l1l2projectb(Vector<T>& out, const T thrs, const T gamma, const bool pos = false,
         const int mode = 1);
   inline void sparseProject(Vector<T>& out, const T thrs,   const int mode = 1, const T lambda_1 = 0,
         const T lambda_2 = 0, const T lambda_3 = 0, const bool pos = false);
   inline void project_sft(const Vector<int>& labels, const int clas);
   inline void project_sft_binary(const Vector<T>& labels);
   /// projects the vector onto the l1 ball of radius thrs,
   /// projects the vector onto the l1 ball of radius thrs,
   /// returns true if the returned vector is null
   inline void l1l2project(Vector<T>& out, const T thrs, const T gamma, const bool pos = false) const;
   inline void fusedProject(Vector<T>& out, const T lambda_1, const T lambda_2, const int itermax);
   inline void fusedProjectHomotopy(Vector<T>& out, const T lambda_1,const T lambda_2,const T lambda_3 = 0,
         const bool penalty = true);
   /// projects the vector onto the l1 ball of radius thrs,
   /// _sort the vector
   inline void sort(Vector<T>& out, const bool mode) const;
   /// sort the vector
   inline void sort(const bool mode);
   //// sort the vector
   inline void sort2(Vector<T>& out, Vector<INTM>& key, const bool mode) const;
   /// sort the vector
   inline void sort2(Vector<INTM>& key, const bool mode);
   /// sort the vector
   inline void applyBayerPattern(const int offset);


   /// Conversion
   /// make a sparse copy 
   inline void toSparse(SpVector<T>& vec) const;
   /// extract the rows of a matrix corresponding to a binary mask
   inline void copyMask(Vector<T>& out, Vector<bool>& mask) const;
   inline void getIndices(Vector<int>& ind) const { }; // irrelevant for dense vectors
   template <typename I>
   inline void refIndices(Vector<I>& ind) const { }; // irrelevant for dense vectors



   private:
   /// = operator, 
   Vector<T>& operator=(const Vector<T>& vec);

   /// if the data has been externally allocated
   bool _externAlloc;
   /// data
   T* _X;
   /// size of the vector
   INTM _n;
};



/// Sparse Matrix class, CSC format
template<typename T, typename I> class SpMatrix {
   friend class Matrix<T>;
   friend class SpVector<T,I>;
   public:
   typedef T value_type;
   typedef SpVector<T,I> col_type;
   typedef I index_type;
   /// Constructor, CSC format, existing data
   SpMatrix(T* v, I* r, I* pB, I* pE, I m, I n, I nzmax);
   /// Constructor, new m x n matrix, with at most nzmax non-zeros values
   SpMatrix(I m, I n, I nzmax);
   /// Empty constructor
   SpMatrix();

   /// Destructor
   ~SpMatrix();

   /// Accessors
   /// reference the column i Io vec
   inline void refCol(I i, SpVector<T,I>& vec) const;
   /// returns pB[i]
   inline I pB(const I i) const { return _pB[i]; };
   /// returns r[i]
   inline I r(const I i) const { return _r[i]; };
   /// returns v[i]
   inline T v(const I i) const { return _v[i]; };
   /// returns the maximum number of non-zero elements
   inline I nzmax() const { return _nzmax; };
   /// returns the number of rows
   inline I n() const { return _n; };
   /// returns the number of columns
   inline I m() const { return _m; };
   /// returns the number of columns
   inline I V() const { return 1; };
   /// returns X[index]
   inline T operator[](const I index) const;
   void getData(Vector<T>& data, const I index) const;
   void setData(T* v, I* r, I* pB, I* pE, I m, I n, I nzmax);

   /// print the sparse matrix
   inline void print(const string& name) const;
   /// compute the sum of the matrix elements
   inline T asum() const;
   /// compute the sum of the matrix elements
   inline T normFsq() const;
   /// Direct access to _pB
   inline I* pB() const { return _pB; };
   /// Direct access to _pE
   inline I* pE() const { return _pE; };
   /// Direct access to _r
   inline I* r() const { return _r; };
   /// Direct access to _v
   inline T* v() const { return _v; };
   /// number of nonzeros elements
   inline I nnz() const { return _pB[_n]; };
   inline void add_direct(const SpMatrix<T,I>& mat, const T a);
   inline void copy_direct(const SpMatrix<T,I>& mat);
   inline T dot_direct(const SpMatrix<T,I>& mat) const;

   /// Modifiers
   /// clear the matrix
   inline void clear();
   /// resize the matrix
   inline void resize(const I m, const I n, const I nzmax);
   /// scale the matrix by a
   inline void scal(const T a) const;
   inline T abs_mean() const;

   /// Algebraic operations
   /// aat <- A*A'
   inline void AAt(Matrix<T>& aat) const;
   /// aat <- A(:,indices)*A(:,indices)'
   inline void AAt(Matrix<T>& aat, const Vector<I>& indices) const;
   /// aat <- sum_i w_i A(:,i)*A(:,i)'
   inline void wAAt(const Vector<T>& w, Matrix<T>& aat) const;
   /// XAt <- X*A'
   inline void XAt(const Matrix<T>& X, Matrix<T>& XAt) const;
   /// XAt <- X(:,indices)*A(:,indices)'
   inline void XAt(const Matrix<T>& X, Matrix<T>& XAt, 
         const Vector<I>& indices) const;
   /// XAt <- sum_i w_i X(:,i)*A(:,i)'
   inline void wXAt( const Vector<T>& w, const Matrix<T>& X, 
         Matrix<T>& XAt, const int numthreads=-1) const;
   inline void XtX(Matrix<T>& XtX) const;

   /// y <- A'*x
   inline void multTrans(const Vector<T>& x, Vector<T>& y,
         const T alpha = 1.0, const T beta = 0.0) const;
   inline void multTrans(const SpVector<T,I>& x, Vector<T>& y,
         const T alpha = 1.0, const T beta = 0.0) const;
   /// perform b = alpha*A*x + beta*b, when x is sparse
   inline void mult(const SpVector<T,I>& x, Vector<T>& b, 
         const T alpha = 1.0, const T beta = 0.0) const;
   /// perform b = alpha*A*x + beta*b, when x is sparse
   inline void mult(const Vector<T>& x, Vector<T>& b, 
         const T alpha = 1.0, const T beta = 0.0) const;
   /// perform C = a*A*B + b*C, possibly transposing A or B.
   inline void mult(const Matrix<T>& B, Matrix<T>& C, 
         const bool transA = false, const bool transB = false,
         const T a = 1.0, const T b = 0.0) const;
   /// perform C = a*B*A + b*C, possibly transposing A or B.
   inline void multSwitch(const Matrix<T>& B, Matrix<T>& C, 
         const bool transA = false, const bool transB = false,
         const T a = 1.0, const T b = 0.0) const;
   /// perform C = a*B*A + b*C, possibly transposing A or B.
   inline void mult(const SpMatrix<T,I>& B, Matrix<T>& C, const bool transA = false,
         const bool transB = false, const T a = 1.0,
         const T b = 0.0) const;
   /// make a copy of the matrix mat in the current matrix
   inline void copyTo(Matrix<T>& mat) const { this->toFull(mat); };
   /// dot product;
   inline T dot(const Matrix<T>& x) const;
   inline void copyRow(const I i, Vector<T>& x) const;
   inline void sum_cols(Vector<T>& sum) const;
   inline void copy(const SpMatrix<T,I>& mat);

   /// Conversions
   /// copy the sparse matrix into a dense matrix
   inline void toFull(Matrix<T>& matrix) const;
   /// copy the sparse matrix into a dense transposed matrix
   inline void toFullTrans(Matrix<T>& matrix) const;

   /// use the data from v, r for _v, _r
   inline void convert(const Matrix<T>&v, const Matrix<I>& r,
         const I K);
   /// use the data from v, r for _v, _r
   inline void convert2(const Matrix<T>&v, const Vector<I>& r,
         const I K);
   inline void normalize(); 
   inline void normalize_rows(); 
   /// returns the l2 norms ^2 of the columns
   inline void norm_2sq_cols(Vector<T>& norms) const;
   /// returns the l0 norms of the columns
   inline void norm_0_cols(Vector<T>& norms) const;
   /// returns the l1 norms of the columns
   inline void norm_1_cols(Vector<T>& norms) const;
   inline void addVecToCols(const Vector<T>& diag, const T a = 1.0);
   inline void addVecToColsWeighted(const Vector<T>& diag, const T* weights, const T a = 1.0);

   typedef SpVector<T,I> col;
   static const bool is_sparse = true;

   private:
   /// forbid copy constructor
   explicit SpMatrix(const SpMatrix<T,I>& matrix);
   SpMatrix<T,I>& operator=(const SpMatrix<T,I>& matrix);

   /// if the data has been externally allocated
   bool _externAlloc;
   /// data
   T* _v;
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

/// Sparse vector class
template <typename T, typename I> class SpVector {
   friend class Matrix<T>;
   friend class SpMatrix<T,I>;
   friend class Vector<T>;
   public:
   typedef T value_type;
   /// Constructor, of the sparse vector of size L.
   SpVector(T* v, I* r, I L, I nzmax);
   /// Constructor, allocates nzmax slots
   SpVector(I nzmax);
   /// Empty constructor
   SpVector();

   /// Destructor
   ~SpVector();

   /// Accessors
   /// returns the length of the vector
   inline T nzmax() const { return _nzmax; };
   /// returns the length of the vector
   inline T length() const { return _L; };
   /// computes the sum of the magnitude of the elements
   inline T asum() const;
   /// computes the l2 norm ^2 of the vector
   inline T nrm2sq() const;
   /// computes the l2 norm  of the vector
   inline T nrm2() const;
   /// computes the linf norm  of the vector
   inline T fmaxval() const;
   /// print the vector to std::cerr
   inline void print(const string& name) const;
   inline void refIndices(Vector<I>& indices) const;
   /// creates a reference on the vector val
   inline void refVal(Vector<T>& val) const;
   /// access table r
   inline I r(const I i) const { return _r[i]; };
   /// access table r
   inline T v(const I i) const { return _v[i]; };
   inline T* rawX() const { return _v; };
   inline I* rawR() const { return _r; };

   /// 
   inline I L() const { return _L; };
   /// 
   inline void setL(const I L) { _L=L; };
   /// a <- a.^2
   inline void sqr();
   /// dot product
   inline T dot(const SpVector<T,I>& vec) const;
   /// dot product
   inline T dot(const Vector<T>& vec) const;
   /// dot product
   inline void scal(const T a);

   /// Modifiers
   /// clears the vector
   inline void clear();
   /// resizes the vector
   inline void resize(const I nzmax);

   /// resize the vector as a sparse matrix
   void inline toSpMatrix(SpMatrix<T,I>& out,
         const I m, const I n) const;
  /// resize the vector as a sparse matrix
   void inline toFull(Vector<T>& out) const;
   inline void getIndices(Vector<int>& ind) const;

   private:
   /// forbids lazy copies
   explicit SpVector(const SpVector<T,I>& vector);
   SpVector<T,I>& operator=(const SpVector<T,I>& vector);

   /// external allocation 
   bool _externAlloc;
   /// data
   T* _v;
   /// indices
   I* _r;
   /// length
   I _L;
   /// maximum number of nonzeros elements
   I _nzmax;
};

/// Class for dense vector
template<typename T, typename I> class LazyVector {
   public:
      LazyVector(Vector<T>& x, const Vector<T>& z, const int n) : _x(x), _z(z), _n(n+1), _p(x.n()) { 
         _current_time=0;
         _dates.resize(_p);
         _dates.setZeros();
         _stats1.resize(n+1);
         _stats2.resize(n+1);
         _stats1[0]=T(1.0);
         _stats2[0]=0;
      };
      void inline update() {
         for (int ii=0; ii<_p; ++ii) {
            update(ii);
         }
         _current_time=0;
         _dates.setZeros();
      };
      void inline update(const I ind) {
         const int last_time=_dates[ind];
         if (last_time != _current_time) {
            _x[ind] = (_stats1[_current_time]/_stats1[last_time])*_x[ind] + _stats1[_current_time]*(_stats2[_current_time]-_stats2[last_time])*_z[ind];
            _dates[ind]=_current_time;
         }
      };
      void inline update(const Vector<I>& indices) {
         const int p = indices.n();
         for (int ii=0; ii<p; ++ii) {
            update(indices[ii]);
         }
      };
      void inline add_scal(const T a, const T b) { // performs x <- a(x - b z) 
         if (_current_time == _n)
            update();
         _current_time++;
         _stats2[_current_time]=_stats2[_current_time-1] + a/_stats1[_current_time-1];
         _stats1[_current_time]=_stats1[_current_time-1]*b;
         if (_stats1[_current_time] < 1e-7)
            update(); // to prevent numerical stability problems
      };

   private:
      Vector<T>& _x;
      const Vector<T>& _z;
      const int _n;
      const int _p;
      Vector<T> _stats1, _stats2;
      Vector<int> _dates;
      int _current_time;
};

/// Class for dense vector
template<typename T, typename I> class DoubleLazyVector {
   public:
      DoubleLazyVector(Vector<T>& x, const Vector<T>& z1, const Vector<T>& z2, const int n) : _x(x), _z1(z1), _z2(z2), _n(n+1), _p(x.n()) { 
         _current_time=0;
         _dates.resize(_p);
         _dates.setZeros();
         _stats1.resize(n+1);
         _stats2.resize(n+1);
         _stats3.resize(n+1);
         _stats1[0]=T(1.0);
         _stats2[0]=0;
         _stats3[0]=0;
      };
      void inline update() {
         for (int ii=0; ii<_p; ++ii) {
            update(ii);
         }
         _current_time=0;
         _dates.setZeros();
      };
      void inline update(const I ind) {
         const int last_time=_dates[ind];
         if (last_time != _current_time) {
            _x[ind] = _stats1[_current_time]* ( _x[ind]/_stats1[last_time] +  (_stats2[_current_time]-_stats2[last_time])*_z1[ind] + (_stats3[_current_time]-_stats3[last_time])*_z2[ind]);
            _dates[ind]=_current_time;
         }
      };
      void inline update(const Vector<I>& indices) {
         const int p = indices.n();
         for (int ii=0; ii<p; ++ii) {
            update(indices[ii]);
         }
      };
      void inline add_scal(const T a, const T b, const T c) {
         if (_current_time == _n)
            update();
         _current_time++;
         _stats1[_current_time]=_stats1[_current_time-1]*c;
         _stats2[_current_time]=_stats2[_current_time-1] + a/_stats1[_current_time];
         _stats3[_current_time]=_stats3[_current_time-1] + b/_stats1[_current_time];
         if (_stats1[_current_time] < 1e-6)
            update(); // to prevent numerical stability problems
      };

   private:
      Vector<T>& _x;
      const Vector<T>& _z1;
      const Vector<T>& _z2;
      const int _n;
      const int _p;
      Vector<T> _stats1, _stats2, _stats3;
      Vector<int> _dates;
      int _current_time;
};



/* ***********************************
 * Implementation of the class Vector
 * ***********************************/


/// Empty constructor
template <typename T> Vector<T>::Vector() :
   _externAlloc(true), _X(NULL),  _n(0) {  };

/// Constructor. Create a new vector of size n
template <typename T> Vector<T>::Vector(INTM n) :
   _externAlloc(false), _n(n) {
#pragma omp critical
      {
         _X=new T[_n];
      }
   };

/// Constructor with existing data
template <typename T> Vector<T>::Vector(T* X, INTM n) :
   _externAlloc(true), _X(X),  _n(n) {  };

/// Copy constructor
template <typename T> Vector<T>::Vector(const Vector<T>& vec) :
   _externAlloc(false), _n(vec._n) {
#pragma omp critical
      {
         _X=new T[_n];
      }
      cblas_copy<T>(_n,vec._X,1,_X,1);
   };

/// Destructor
template <typename T> Vector<T>::~Vector() {
   clear();
};

/// Print the matrix to std::cout
template <typename T> inline void Vector<T>::print(const string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _n << std::endl;
   for (INTM j = 0; j<_n; ++j) {
      printf("%10.5g ",static_cast<double>(_X[j]));
   }
   printf("\n ");
};

/// Print the matrix to std::cout
template <typename T> inline void Vector<T>::dump(const string& name) const {
   ofstream f; 
   const char * cname = name.c_str();
   f.open(cname);
   f.precision(20);
   std::cerr << name << std::endl;
   f <<  _n << std::endl;
   for (INTM j = 0; j<_n; ++j) {
      f << static_cast<double>(_X[j]) << " ";
   }
   f << std::endl;
   f.close();
};




/// Print the vector to std::cout
template <> inline void Vector<double>::print(const char* name) const {
   printf("%s, %d\n",name,(int)_n);
   for (INTM i = 0; i<_n; ++i) {
      printf("%g ",_X[i]);
   }
   printf("\n");
};

/// Print the vector to std::cout
template <> inline void Vector<float>::print(const char* name) const {
   printf("%s, %d\n",name,(int)_n);
   for (INTM i = 0; i<_n; ++i) {
      printf("%g ",_X[i]);
   }
   printf("\n");
};

/// Print the vector to std::cout
template <> inline void Vector<int>::print(const char* name) const {
   printf("%s, %d\n",name,(int)_n);
   for (INTM i = 0; i<_n; ++i) {
      printf("%d ",_X[i]);
   }
   printf("\n");
};

/// Print the vector to std::cout
template <> inline void Vector<bool>::print(const char* name) const {
   printf("%s, %d\n",name,(int)_n);
   for (INTM i = 0; i<_n; ++i) {
      printf("%d ",_X[i] ? 1 : 0);
   }
   printf("\n");
};

/// returns the index of the largest value
template <typename T> inline INTM Vector<T>::max() const {
   INTM imax=0;
   T max=_X[0];
   for (INTM j = 1; j<_n; ++j) {
      T cur = _X[j];
      if (cur > max) {
         imax=j;
         max = cur;
      }
   }
   return imax;
};

/// returns the index of the minimum value
template <typename T> inline INTM Vector<T>::min() const {
   INTM imin=0;
   T min=_X[0];
   for (INTM j = 1; j<_n; ++j) {
      T cur = _X[j];
      if (cur < min) {
         imin=j;
         min = cur;
      }
   }
   return imin;
};

/// returns the maximum value
template <typename T> inline T Vector<T>::maxval() const {
   return _X[this->max()];
};

/// returns the minimum value
template <typename T> inline T Vector<T>::minval() const {
   return _X[this->min()];
};

/// returns the maximum magnitude
template <typename T> inline T Vector<T>::fmaxval() const {
   return fabs(_X[this->fmax()]);
};

/// returns the minimum magnitude
template <typename T> inline T Vector<T>::fminval() const {
   return fabs(_X[this->fmin()]);
};

template <typename T>
inline void Vector<T>::logspace(const INTM n, const T a, const T b) {
   T first=log10(a);
   T last=log10(b);
   T step = (last-first)/(n-1);
   this->resize(n);
   _X[0]=first;
   for (INTM i = 1; i<_n; ++i)
      _X[i]=_X[i-1]+step;
   for (INTM i = 0; i<_n; ++i)
      _X[i]=pow(T(10.0),_X[i]);
}

template <typename T>
inline INTM Vector<T>::nnz() const {
   INTM sum=0;
   for (INTM i = 0; i<_n; ++i) 
      if (_X[i] != T()) ++sum;
   return sum;
};
/// generate logarithmically spaced values
template <>
inline void Vector<INTM>::logspace(const INTM n, const INTM a, const INTM b) {
   Vector<double> tmp(n);
   tmp.logspace(n,double(a),double(b));
   this->resize(n);
   _X[0]=a;
   _X[n-1]=b;
   for (INTM i = 1; i<_n-1; ++i) {
      INTM candidate=static_cast<INTM>(floor(static_cast<double>(tmp[i])));
      _X[i]= candidate > _X[i-1] ? candidate : _X[i-1]+1;
   }
}

/// returns the index of the value with largest magnitude
template <typename T> inline INTM Vector<T>::fmax() const {
   return cblas_iamax<T>(_n,_X,1);
};

/// returns the index of the value with smallest magnitude
template <typename T> inline INTM Vector<T>::fmin() const {
   return cblas_iamin<T>(_n,_X,1);
};

/// returns a reference to X[index]
template <typename T> inline T& Vector<T>::operator[] (const INTM i) {
   assert(i>=0 && i<_n);
   return _X[i];
};

/// returns X[index]
template <typename T> inline T Vector<T>::operator[] (const INTM i) const {
   assert(i>=0 && i<_n);
   return _X[i];
};

/// make a copy of x
template <typename T> inline void Vector<T>::copy(const Vector<T>& x) {
   if (_X != x._X) {
      this->resize(x.n());
      //cblas_copy<T>(_n,x._X,1,_X,1);
      memcpy(_X,x._X,_n*sizeof(T));
   }
};

/// make a copy of x
template <typename T> inline void Vector<T>::copyRef(const Vector<T>& x) {
   this->setData(x.rawX(),x.n());
};


/// Set all values to zero
template <typename T> inline void Vector<T>::setZeros() {
   memset(_X,0,_n*sizeof(T));
};

/// resize the vector
template <typename T> inline void Vector<T>::resize(const INTM n, const bool set_zeros) {
   if (_n == n) return;
   clear();
#pragma omp critical
   {
      _X=new T[n];
   }
   _n=n;
   _externAlloc=false;
   if (set_zeros)
      this->setZeros();
};

/// change the data of the vector
template <typename T> inline void Vector<T>::setPointer(T* X, const INTM n) {
   clear();
   _externAlloc=true;
   _X=X;
   _n=n;
};

/// put a random permutation of size n (for integral vectors)
template <> inline void Vector<int>::randi(int n) {
   for (int i = 0; i<_n; ++i)
      _X[i]=static_cast<int>(random() % n);
};

/// put a random permutation of size n (for integral vectors)
template <> inline void Vector<int>::randperm(int n) {
   resize(n);
   Vector<int> table(n);
   for (int i = 0; i<n; ++i)
      table[i]=i;
   int size=n;
   for (int i = 0; i<n; ++i) {
      const int ind=random() % size;
      _X[i]=table[ind];
      table[ind]=table[size-1];
      --size;
   }
};

/// put random values in the vector (white Gaussian Noise)
template <typename T> inline void Vector<T>::setAleat() {
   for (INTM i = 0; i<_n; ++i) _X[i]=normalDistrib<T>();
};

/// clear the vector
template <typename T> inline void Vector<T>::clear() {
   if (!_externAlloc) delete[](_X);
   _n=0;
   _X=NULL;
   _externAlloc=true;
};

/// performs soft-thresholding of the vector
template <typename T> inline void Vector<T>::softThrshold(const T nu) {
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] > nu) {
         _X[i] -= nu;
      } else if (_X[i] < -nu) {
         _X[i] += nu;
      } else {
         _X[i] = 0;
      }
   }
};

/// performs soft-thresholding of the vector
template <typename T> inline void Vector<T>::fastSoftThrshold(const T nu) {
//#pragma omp parallel for
   for (INTM i = 0; i<_n; ++i) 
      _X[i]=fastSoftThrs(_X[i],nu);
};

/// performs soft-thresholding of the vector
template <typename T> inline void Vector<T>::fastSoftThrshold(Vector<T>& output, const T nu) const {
   output.resize(_n,false);
//#pragma omp parallel for
   for (INTM i = 0; i<_n; ++i) 
      output[i]=fastSoftThrs(_X[i],nu);
};

/// performs soft-thresholding of the vector
template <typename T> inline void Vector<T>::softThrsholdScal(Vector<T>& out, const T nu, const T s) {
   T* Y = out.rawX();
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] > nu) {
         Y[i] = s*(_X[i]-nu);
      } else if (_X[i] < -nu) {
         Y[i] = s*(_X[i]+nu);
      } else {
         Y[i] = 0;
      }
   }
};

/// performs soft-thresholding of the vector
template <typename T> inline void Vector<T>::hardThrshold(const T nu) {
   for (INTM i = 0; i<_n; ++i) {
      if (!(_X[i] > nu || _X[i] < -nu)) {
         _X[i] = 0;
      }
   }
};


/// performs thresholding of the vector
template <typename T> inline void Vector<T>::thrsmax(const T nu) {
//#pragma omp parallel for private(i)
   for (INTM i = 0; i<_n; ++i) 
      if (_X[i] < nu) _X[i]=nu;
}

/// performs thresholding of the vector
template <typename T> inline void Vector<T>::thrsmin(const T nu) {
   for (INTM i = 0; i<_n; ++i) 
      _X[i]=MIN(_X[i],nu);
}

/// performs thresholding of the vector
template <typename T> inline void Vector<T>::thrsabsmin(const T nu) {
   for (INTM i = 0; i<_n; ++i) 
      _X[i]=MAX(MIN(_X[i],nu),-nu);
}

/// performs thresholding of the vector
template <typename T> inline void Vector<T>::thrshold(const T nu) {
   for (INTM i = 0; i<_n; ++i) 
      if (abs<T>(_X[i]) < nu) 
         _X[i]=0;
}
/// performs soft-thresholding of the vector
template <typename T> inline void Vector<T>::thrsPos() {
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] < 0) _X[i]=0;
   }
};

template <>
inline bool Vector<bool>::alltrue() const {
   for (INTM i = 0; i<_n; ++i) {
      if (!_X[i]) return false;
   }
   return true;
};

template <>
inline bool Vector<bool>::allfalse() const {
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i]) return false;
   }
   return true;
};

/// set each value of the vector to val
template <typename T> inline void Vector<T>::set(const T val) {
   for (INTM i = 0; i<_n; ++i) _X[i]=val;
};

/// returns ||A||_2
template <typename T> inline T Vector<T>::nrm2() const {
   return cblas_nrm2<T>(_n,_X,1);
};

/// returns ||A||_2^2
template <typename T> inline T Vector<T>::nrm2sq() const {
   return cblas_dot<T>(_n,_X,1,_X,1);
};

/// returns  A'x
template <typename T> inline T Vector<T>::dot(const Vector<T>& x) const {
   assert(_n == x._n);
   return cblas_dot<T>(_n,_X,1,x._X,1);
};

/// returns A'x, when x is sparse
template <typename T> 
template <typename I> 
inline T Vector<T>::dot(const SpVector<T,I>& x) const {
   T sum=0;
   const I* r = x.rawR();
   const T* v = x.rawX();
   for (INTT i = 0; i<x._L; ++i) {
      sum += _X[r[i]]*v[i];
   }
   return sum;
   //return cblas_doti<T>(x._L,x._v,x._r,_X);
};

/// A <- A + a*x
template <typename T> inline void Vector<T>::add(const Vector<T>& x, const T a) {
   assert(_n == x._n);
   cblas_axpy<T>(_n,a,x._X,1,_X,1);
};

template <typename T> inline void Vector<T>::add_scal(const Vector<T>& x, const T a, const T b) {
   assert(_n == x._n);
   cblas_axpby<T>(_n,a,x._X,1,b,_X,1);
};

/// A <- A + a*x
template <typename T> 
template <typename I> 
inline void Vector<T>::add(const SpVector<T,I>& x,
      const T a) {
   if (a == 1.0) {
      for (INTM i = 0; i<x._L; ++i)
         _X[x._r[i]]+=x._v[i];
   } else {
      for (INTM i = 0; i<x._L; ++i)
         _X[x._r[i]]+=a*x._v[i];
   }
};

/// A <- A + a*x
template <typename T> 
template <typename I>
inline void Vector<T>::add_scal(const SpVector<T,I>& x,
      const T a, const T b) {
   if (b != T(1.0)) {
      if (b==0) {
         this->setZeros();
      } else {
         this->scal(b);
      }   
   }
   if (a == T(1.0)) {
      for (I i = 0; i<x._L; ++i)
         _X[x._r[i]]+=x._v[i];
   } else {
      for (I i = 0; i<x._L; ++i)
         _X[x._r[i]]+=a*x._v[i];
   }
};



/// adds a to each value in the vector
template <typename T> inline void Vector<T>::add(const T a) {
   for (INTM i = 0; i<_n; ++i) _X[i]+=a;
};

/// A <- A - x
template <typename T> inline void Vector<T>::sub(const Vector<T>& x) {
   assert(_n == x._n);
   vSub<T>(_n,_X,x._X,_X);
};

/// A <- A + a*x
template <typename T> 
template <typename I> 
inline void Vector<T>::sub(const SpVector<T,I>& x) {
   for (INTM i = 0; i<x._L; ++i)
      _X[x._r[i]]-=x._v[i];
};

/// A <- A ./ x
template <typename T> inline void Vector<T>::div(const Vector<T>& x) {
   assert(_n == x._n);
   vDiv<T>(_n,_X,x._X,_X);
};

/// A <- x ./ y
template <typename T> inline void Vector<T>::div(const Vector<T>& x, const Vector<T>& y) {
   assert(_n == x._n);
   vDiv<T>(_n,x._X,y._X,_X);
};


/// A <- x .^ 2
template <typename T> inline void Vector<T>::sqr(const Vector<T>& x) {
   this->resize(x._n);
   vSqr<T>(_n,x._X,_X);
}

/// A <- x .^ 2
template <typename T> inline void Vector<T>::sqr() {
   vSqr<T>(_n,_X,_X);
}

/// A <- x .^ 2
template <typename T> inline void Vector<T>::Invsqrt(const Vector<T>& x) {
   this->resize(x._n);
   vInvSqrt<T>(_n,x._X,_X);
}
/// A <- x .^ 2
template <typename T> inline void Vector<T>::Sqrt(const Vector<T>& x) {
   this->resize(x._n);
   vSqrt<T>(_n,x._X,_X);
}
/// A <- x .^ 2
template <typename T> inline void Vector<T>::Invsqrt() {
   vInvSqrt<T>(_n,_X,_X);
}
/// A <- x .^ 2
template <typename T> inline void Vector<T>::Sqrt() {
   vSqrt<T>(_n,_X,_X);
}


/// A <- 1./x
template <typename T> inline void Vector<T>::inv(const Vector<T>& x) {
   this->resize(x.n());
   vInv<T>(_n,x._X,_X);
};

/// A <- 1./A
template <typename T> inline void Vector<T>::inv() {
   vInv<T>(_n,_X,_X);
};

/// A <- x .* y
template <typename T> inline void Vector<T>::mult(const Vector<T>& x,
      const Vector<T>& y) {
   this->resize(x.n());
   vMul<T>(_n,x._X,y._X,_X);
};
;

/// normalize the vector
template <typename T> inline void Vector<T>::normalize() {
   T norm=nrm2();
   if (norm > EPSILON) scal(1.0/norm);
};

/// normalize the vector
template <typename T> inline void Vector<T>::normalize2(const T thrs) {
   T norm=nrm2();
   if (norm > thrs) scal(thrs/norm);
};

/// whiten
template <typename T> inline void Vector<T>::whiten(
      Vector<T>& meanv, const bool pattern) {
   if (pattern) {
      const INTM n =static_cast<INTM>(sqrt(static_cast<T>(_n)));
      INTM count[4];
      for (INTM i = 0; i<4; ++i) count[i]=0;
      INTM offsetx=0;
      for (INTM j = 0; j<n; ++j) {
         offsetx= (offsetx+1) % 2;
         INTM offsety=0;
         for (INTM k = 0; k<n; ++k) {
            offsety= (offsety+1) % 2;
            meanv[2*offsetx+offsety]+=_X[j*n+k];
            count[2*offsetx+offsety]++;
         }
      }
      for (INTM i = 0; i<4; ++i)
         meanv[i] /= count[i];
      offsetx=0;
      for (INTM j = 0; j<n; ++j) {
         offsetx= (offsetx+1) % 2;
         INTM offsety=0;
         for (INTM k = 0; k<n; ++k) {
            offsety= (offsety+1) % 2;
            _X[j*n+k]-=meanv[2*offsetx+offsety];
         }
      }
   } else {
      const INTM V = meanv.n();
      const INTM sizePatch=_n/V;
      for (INTM j = 0; j<V; ++j) {
         T mean = 0;
         for (INTM k = 0; k<sizePatch; ++k) {
            mean+=_X[sizePatch*j+k];
         }
         mean /= sizePatch;
         for (INTM k = 0; k<sizePatch; ++k) {
            _X[sizePatch*j+k]-=mean;
         }
         meanv[j]=mean;
      }
   }
};

/// whiten
template <typename T> inline void Vector<T>::whiten(
      Vector<T>& meanv, const Vector<T>& mask) {
   const INTM V = meanv.n();
   const INTM sizePatch=_n/V;
   for (INTM j = 0; j<V; ++j) {
      T mean = 0;
      for (INTM k = 0; k<sizePatch; ++k) {
         mean+=_X[sizePatch*j+k];
      }
      mean /= cblas_asum(sizePatch,mask._X+j*sizePatch,1);
      for (INTM k = 0; k<sizePatch; ++k) {
         if (mask[sizePatch*j+k])
            _X[sizePatch*j+k]-=mean;
      }
      meanv[j]=mean;
   }
};

/// whiten
template <typename T> inline void Vector<T>::whiten(const INTM V) {
   const INTM sizePatch=_n/V;
   for (INTM j = 0; j<V; ++j) {
      T mean = 0;
      for (INTM k = 0; k<sizePatch; ++k) {
         mean+=_X[sizePatch*j+k];
      }
      mean /= sizePatch;
      for (INTM k = 0; k<sizePatch; ++k) {
         _X[sizePatch*j+k]-=mean;
      }
   }
};

template <typename T> inline T Vector<T>::KL(const Vector<T>& Y) {
   T sum = 0;
   T* prY = Y.rawX();
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] > 1e-20) {
         if (prY[i] < 1e-60) {
            sum += 1e200;
         } else {
            sum += _X[i]*log_alt<T>(_X[i]/prY[i]);
         }
         //sum += _X[i]*log_alt<T>(_X[i]/(prY[i]+1e-100));
      }
   }
   sum += T(-1.0) + Y.sum();
   return sum;
};

/// unwhiten
template <typename T> inline void Vector<T>::unwhiten(
      Vector<T>& meanv, const bool pattern) {
   if (pattern) {
      const INTM n =static_cast<INTM>(sqrt(static_cast<T>(_n)));
      INTM offsetx=0;
      for (INTM j = 0; j<n; ++j) {
         offsetx= (offsetx+1) % 2;
         INTM offsety=0;
         for (INTM k = 0; k<n; ++k) {
            offsety= (offsety+1) % 2;
            _X[j*n+k]+=meanv[2*offsetx+offsety];
         }
      }
   } else  {
      const INTM V = meanv.n();
      const INTM sizePatch=_n/V;
      for (INTM j = 0; j<V; ++j) {
         T mean = meanv[j];
         for (INTM k = 0; k<sizePatch; ++k) {
            _X[sizePatch*j+k]+=mean;
         }
      }
   }
};


/// return the mean
template <typename T> inline T Vector<T>::mean() const {
   return this->sum()/_n;
}

template <typename T> inline T Vector<T>::abs_mean() const {
   return this->asum()/_n;
};

template <typename T> inline T Vector<T>::mean_non_uniform(const Vector<T>& qi) const {
   Vector<T> tmp;
   tmp.copy(*this);
   tmp.mult(qi,tmp);
   return tmp.sum();
};

/// return the std
template <typename T> inline T Vector<T>::std() {
   T E = this->mean();
   T std=0;
   for (INTM i = 0; i<_n; ++i) {
      T tmp=_X[i]-E;
      std += tmp*tmp;
   }
   std /= _n;
   return sqr_alt<T>(std);
}

/// scale the vector by a
template <typename T> inline void Vector<T>::scal(const T a) {
   return cblas_scal<T>(_n,a,_X,1);
};

/// A <- -A
template <typename T> inline void Vector<T>::neg() {
   for (INTM i = 0; i<_n; ++i) _X[i]=-_X[i];
};

/// replace each value by its exponential
template <typename T> inline void Vector<T>::exp() {
   vExp<T>(_n,_X,_X);
};

/// replace each value by its absolute value
template <typename T> inline void Vector<T>::abs_vec() {
   vAbs<T>(_n,_X,_X);
};

/// replace each value by its logarithm
template <typename T> inline void Vector<T>::log() {
   for (INTM i=0; i<_n; ++i) _X[i]=alt_log<T>(_X[i]);
};

/// replace each value by its exponential
template <typename T> inline void Vector<T>::logexp() {
   for (INTM i = 0; i<_n; ++i) {
      _X[i]=logexp2(_X[i]);
      /*if (_X[i] < -30) {
         _X[i]=0;
      } else if (_X[i] < 30) {
         _X[i]= alt_log<T>( T(1.0) + exp_alt<T>( _X[i] ) );
      }*/
   }
};

template <typename T> inline T Vector<T>::logsumexp() {
   T mm=this->maxval();
   this->add(-mm);
   this->exp();
   return mm+alt_log<T>(this->asum());
};

/// replace each value by its exponential
template <typename T> inline T Vector<T>::softmax(const int y) {
   this->add(-_X[y]);
   _X[y]=-INFINITY;
   T max=this->maxval();
   if (max > 30) {
      return max;
   } else if (max < -30) {
      return 0;
   } else {
      _X[y]=T(0.0);
      this->exp();
      return alt_log<T>(this->sum());
   }
};

/// computes the sum of the magnitudes of the vector
template <typename T> inline T Vector<T>::asum() const {
   return cblas_asum<T>(_n,_X,1);
};

template <typename T> inline T Vector<T>::lzero() const {
   INTM count=0;
   for (INTM i = 0; i<_n; ++i) 
      if (_X[i] != 0) ++count;
   return count;
};


template <typename T> inline T Vector<T>::afused() const {
   T sum = 0;
   for (INTM i = 1; i<_n; ++i) {
      sum += abs<T>(_X[i]-_X[i-1]);
   }
   return sum;
}
/// returns the sum of the vector
template <typename T> inline T Vector<T>::sum() const {
   T sum=T();
   for (INTM i = 0; i<_n; ++i) sum +=_X[i]; 
   return sum;
};

/// puts in signs, the sign of each poINTM in the vector
template <typename T> inline void Vector<T>::sign(Vector<T>& signs) const {
   T* prSign=signs.rawX();
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] == 0) {
         prSign[i]=0.0; 
      } else {
         prSign[i] = _X[i] > 0 ? 1.0 : -1.0;
      }
   }
};

/// projects the vector onto the l1 ball of radius thrs,
/// returns true if the returned vector is null
template <typename T> inline void Vector<T>::l1project(Vector<T>& out,
      const T thrs, const bool simplex) const {
   out.copy(*this);
   if (simplex) {
      out.thrsPos();
   } else {
      vAbs<T>(_n,out._X,out._X);
   }
   T norm1 = out.sum();
   if (norm1 <= thrs) {
      if (!simplex) out.copy(*this);
      return;
   }
   T* prU = out._X;
   INTM sizeU = _n;

   T sum = T();
   INTM sum_card = 0;

   while (sizeU > 0) {
      // put the pivot in prU[0]
      swap(prU[0],prU[sizeU/2]);
      T pivot = prU[0];
      INTM sizeG=1;
      T sumG=pivot;

      for (INTM i = 1; i<sizeU; ++i) {
         if (prU[i] >= pivot) {
            sumG += prU[i];
            swap(prU[sizeG++],prU[i]);
         }
      }

      if (sum + sumG - pivot*(sum_card + sizeG) <= thrs) {
         sum_card += sizeG;
         sum += sumG;
         prU +=sizeG;
         sizeU -= sizeG;
      } else {
         ++prU;
         sizeU = sizeG-1;
      }
   }
   T lambda_1 = (sum-thrs)/sum_card;
   out.copy(*this);
   if (simplex) {
      out.thrsPos();
   }
   out.softThrshold(lambda_1);
};

/// projects the vector onto the l1 ball of radius thrs,
/// returns true if the returned vector is null
template <typename T> inline void Vector<T>::l1project_weighted(Vector<T>& out, const Vector<T>& weights,
      const T thrs, const bool residual) const {
   out.copy(*this);
   if (thrs==0) {
      out.setZeros();
      return;
   }
   vAbs<T>(_n,out._X,out._X);
   out.div(weights);
   Vector<INTM> keys(_n);
   for (INTM i = 0; i<_n; ++i) keys[i]=i;
   out.sort2(keys,false);
   T sum1=0;
   T sum2=0;
   T lambda_1=0;
   for (INTM i = 0; i<_n; ++i) {
      const T lambda_old=lambda_1;
      const T fact=weights[keys[i]]*weights[keys[i]];
      lambda_1=out[i];
      sum2 += fact;
      sum1 += fact*lambda_1;
      if (sum1 - lambda_1*sum2 >= thrs) {
         sum2-=fact;
         sum1-=fact*lambda_1;
         lambda_1=lambda_old;
         break;
      }
   }
   lambda_1=MAX(0,(sum1-thrs)/sum2);

   if (residual) {
      for (INTM i = 0; i<_n; ++i) {
         out._X[i]=_X[i] > 0 ? MIN(_X[i],lambda_1*weights[i]) : MAX(_X[i],-lambda_1*weights[i]);
      }
   } else {
      for (INTM i = 0; i<_n; ++i) {
         out._X[i]=_X[i] > 0 ? MAX(0,_X[i]-lambda_1*weights[i]) : MIN(0,_X[i]+lambda_1*weights[i]);
      }
   }
};


template <typename T>
inline void Vector<T>::project_sft_binary(const Vector<T>& y) {
   T mean = this->mean();
   Vector<T> ztilde, xtilde;
   ztilde.resize(_n);
   int count=0;
   if (mean > 0) {
      for (int ii=0; ii<_n; ++ii) 
         if (y[ii] > 0) {
            count++;
            ztilde[ii]=_X[ii]+T(1.0);
         } else {
            ztilde[ii]= _X[ii];
         }
      ztilde.l1project(xtilde,T(count));
      for (int ii=0; ii<_n; ++ii) 
         _X[ii] = y[ii] > 0 ? xtilde[ii]-T(1.0) : xtilde[ii];
   } else {
      for (int ii=0; ii<_n; ++ii) 
         if (y[ii] > 0) {
            ztilde[ii]=-_X[ii];
         } else {
            count++;
            ztilde[ii]=- _X[ii] + T(1.0);
         }
      ztilde.l1project(xtilde,T(count));
      for (int ii=0; ii<_n; ++ii) 
         _X[ii] = y[ii] > 0 ? -xtilde[ii] :  -xtilde[ii]+T(1.0);
   }
};

template <typename T>
inline void Vector<T>::project_sft(const Vector<int>& labels, const int clas) {
   Vector<T> y(_n);
   for (int ii=0; ii<_n; ++ii) y[ii] = labels[ii]==clas ? T(1.0) : -T(1.0);
   this->project_sft_binary(y);
/*   T mean = this->mean();
   T thrs=mean;

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

template <typename T>
inline void Vector<T>::sparseProject(Vector<T>& out, const T thrs, const int mode, const T lambda_1,
      const T lambda_2, const T lambda_3, const bool pos) {
   if (mode == 1) {
      /// min_u ||b-u||_2^2 / ||u||_1 <= thrs
      this->l1project(out,thrs,pos);
   } else if (mode == 2) {
      /// min_u ||b-u||_2^2 / ||u||_2^2 + lambda_1||u||_1 <= thrs
      if (lambda_1 > 1e-10) {
         this->scal(lambda_1);
         this->l1l2project(out,thrs,2.0/(lambda_1*lambda_1),pos);
         this->scal(T(1.0/lambda_1));
         out.scal(T(1.0/lambda_1));
      } else {
         out.copy(*this);
         out.normalize2();
         out.scal(sqrt(thrs));
      }
   } else if (mode == 3) {
      /// min_u ||b-u||_2^2 / ||u||_1 + (lambda_1/2) ||u||_2^2 <= thrs
      this->l1l2project(out,thrs,lambda_1,pos);
   } else if (mode == 4) {
      /// min_u 0.5||b-u||_2^2  + lambda_1||u||_1 / ||u||_2^2 <= thrs
      out.copy(*this);
      if (pos) 
         out.thrsPos();
      out.softThrshold(lambda_1);
      T nrm=out.nrm2sq();
      if (nrm > thrs)
         out.scal(sqr_alt<T>(thrs/nrm));
   } else if (mode == 5) {
      /// min_u 0.5||b-u||_2^2  + lambda_1||u||_1 +lambda_2 Fused(u) / ||u||_2^2 <= thrs
      //      this->fusedProject(out,lambda_1,lambda_2,100);
      //      T nrm=out.nrm2sq();
      //      if (nrm > thrs)
      //         out.scal(sqr_alt<T>(thrs/nrm));
      //  } else if (mode == 6) {
      /// min_u 0.5||b-u||_2^2  + lambda_1||u||_1 +lambda_2 Fused(u) +0.5lambda_3 ||u||_2^2 
      this->fusedProjectHomotopy(out,lambda_1,lambda_2,lambda_3,true);
} else if (mode==6) {
   /// min_u ||b-u||_2^2  /  lambda_1||u||_1 +lambda_2 Fused(u) + 0.5lambda3||u||_2^2 <= thrs
   this->fusedProjectHomotopy(out,lambda_1/thrs,lambda_2/thrs,lambda_3/thrs,false);
} else {
   /// min_u ||b-u||_2^2 / (1-lambda_1)*||u||_2^2 + lambda_1||u||_1 <= thrs
   if (lambda_1 < 1e-10) {
      out.copy(*this);
      if (pos) 
         out.thrsPos();
      out.normalize2();
      out.scal(sqrt(thrs));
   } else if (lambda_1 > 0.999999) {
      this->l1project(out,thrs,pos);
   } else {
      this->sparseProject(out,thrs/(1.0-lambda_1),2,lambda_1/(1-lambda_1),0,0,pos);
   }
}
};

/// returns true if the returned vector is null
template <typename T>
inline void Vector<T>::l1l2projectb(Vector<T>& out, const T thrs, const T gamma, const bool pos,
      const int mode) {
   if (mode == 1) {
      /// min_u ||b-u||_2^2 / ||u||_2^2 + gamma ||u||_1 <= thrs
      this->scal(gamma);
      this->l1l2project(out,thrs,2.0/(gamma*gamma),pos);
      this->scal(T(1.0/gamma));
      out.scal(T(1.0/gamma));
   } else if (mode == 2) {
      /// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
      this->l1l2project(out,thrs,gamma,pos);
   } else if (mode == 3) {
      /// min_u 0.5||b-u||_2^2  + gamma||u||_1 / ||u||_2^2 <= thrs
      out.copy(*this);
      if (pos) 
         out.thrsPos();
      out.softThrshold(gamma);
      T nrm=out.nrm2();
      if (nrm > thrs)
         out.scal(thrs/nrm);
   }
}

/// returns true if the returned vector is null
/// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
template <typename T>
   inline void Vector<T>::l1l2project(Vector<T>& out, const T thrs, const T gamma, const bool pos) const {
      if (gamma == 0) 
         return this->l1project(out,thrs,pos);
      out.copy(*this);
      if (pos) {
         out.thrsPos();
      } else {
         vAbs<T>(_n,out._X,out._X);
      }
      T norm = out.sum() + gamma*out.nrm2sq();
      if (norm <= thrs) {
         if (!pos) out.copy(*this);
         return;
      }

      /// BEGIN
      T* prU = out._X;
      INTM sizeU = _n;

      T sum = 0;
      INTM sum_card = 0;

      while (sizeU > 0) {
         // put the pivot in prU[0]
         swap(prU[0],prU[sizeU/2]);
         T pivot = prU[0];
         INTM sizeG=1;
         T sumG=pivot+0.5*gamma*pivot*pivot;

         for (INTM i = 1; i<sizeU; ++i) {
            if (prU[i] >= pivot) {
               sumG += prU[i]+0.5*gamma*prU[i]*prU[i];
               swap(prU[sizeG++],prU[i]);
            }
         }
         if (sum + sumG - pivot*(1+0.5*gamma*pivot)*(sum_card + sizeG) <
               thrs*(1+gamma*pivot)*(1+gamma*pivot)) {
            sum_card += sizeG;
            sum += sumG;
            prU +=sizeG;
            sizeU -= sizeG;
         } else {
            ++prU;
            sizeU = sizeG-1;
         }
      }
      T a = gamma*gamma*thrs+0.5*gamma*sum_card;
      T b = 2*gamma*thrs+sum_card;
      T c=thrs-sum;
      T delta = b*b-4*a*c;
      T lambda_1 = (-b+sqrt(delta))/(2*a);

      out.copy(*this);
      if (pos) {
         out.thrsPos();
      }
      out.fastSoftThrshold(lambda_1);
      out.scal(T(1.0/(1+lambda_1*gamma)));
   };

template <typename T>
static inline T fusedHomotopyAux(const bool& sign1,
      const bool& sign2,
      const bool& sign3,
      const T& c1,
      const T& c2) {
   if (sign1) {
      if (sign2) {
         return sign3 ? 0 : c2;
      } else {
         return sign3 ? -c2-c1 : -c1;
      }
   } else {
      if (sign2) {
         return sign3 ? c1 : c1+c2;
      } else {
         return sign3 ? -c2 : 0;
      }
   }
};

template <typename T>
inline void Vector<T>::fusedProjectHomotopy(Vector<T>& alpha, 
      const T lambda_1,const T lambda_2,const T lambda_3,
      const bool penalty) {
   T* pr_DtR=_X;
   const INTM K = _n;
   alpha.setZeros();
   Vector<T> u(K); // regularization path for gamma
   Vector<T> Du(K); // regularization path for alpha
   Vector<T> DDu(K); // regularization path for alpha
   Vector<T> gamma(K); // auxiliary variable
   Vector<T> c(K); // auxiliary variables
   Vector<T> scores(K); // auxiliary variables
   gamma.setZeros();
   T* pr_gamma = gamma.rawX();
   T* pr_u = u.rawX();
   T* pr_Du = Du.rawX();
   T* pr_DDu = DDu.rawX();
   T* pr_c = c.rawX();
   T* pr_scores = scores.rawX();
   Vector<INTM> ind(K+1);
   Vector<bool> signs(K);
   ind.set(K);
   INTM* pr_ind = ind.rawX();
   bool* pr_signs = signs.rawX();

   /// Computation of DtR
   T sumBeta = this->sum();

   /// first element is selected, gamma and alpha are updated
   pr_gamma[0]=sumBeta/K;
   /// update alpha
   alpha.set(pr_gamma[0]);
   /// update DtR
   this->sub(alpha);
   for (INTM j = K-2; j>=0; --j) 
      pr_DtR[j] += pr_DtR[j+1];

   pr_DtR[0]=0;
   pr_ind[0]=0;
   pr_signs[0] = pr_DtR[0] > 0;
   pr_c[0]=T(1.0)/K;
   INTM currentInd=this->fmax();
   T currentLambda=abs<T>(pr_DtR[currentInd]);
   bool newAtom = true;

   /// Solve the Lasso using simplified LARS
   for (INTM i = 1; i<K; ++i) {
      /// exit if constraINTMs are satisfied
      /// min_u ||b-u||_2^2  +  lambda_1||u||_1 +lambda_2 Fused(u) + 0.5lambda3||u||_2^2 
      if (penalty && currentLambda <= lambda_2) break;
      if (!penalty) {
         /// min_u ||b-u||_2^2  /  lambda_1||u||_1 +lambda_2 Fused(u) + 0.5lambda3||u||_2^2 <= 1.0
         scores.copy(alpha);
         scores.softThrshold(lambda_1*currentLambda/lambda_2);
         scores.scal(T(1.0/(1.0+lambda_3*currentLambda/lambda_2)));
         if (lambda_1*scores.asum()+lambda_2*scores.afused()+0.5*
               lambda_3*scores.nrm2sq() >= T(1.0)) break;
      }

      /// Update pr_ind and pr_c
      if (newAtom) {
         INTM j;
         for (j = 1; j<i; ++j) 
            if (pr_ind[j] > currentInd) break;
         for (INTM k = i; k>j; --k) {
            pr_ind[k]=pr_ind[k-1];
            pr_c[k]=pr_c[k-1];
            pr_signs[k]=pr_signs[k-1];
         }
         pr_ind[j]=currentInd;
         pr_signs[j]=pr_DtR[currentInd] > 0;
         pr_c[j-1]=T(1.0)/(pr_ind[j]-pr_ind[j-1]);
         pr_c[j]=T(1.0)/(pr_ind[j+1]-pr_ind[j]);
      }

      // Compute u
      pr_u[0]= pr_signs[1] ? -pr_c[0] : pr_c[0];
      if (i == 1) {
         pr_u[1]=pr_signs[1] ? pr_c[0]+pr_c[1] : -pr_c[0]-pr_c[1];
      } else {
         pr_u[1]=pr_signs[1] ? pr_c[0]+pr_c[1] : -pr_c[0]-pr_c[1];
         pr_u[1]+=pr_signs[2] ? -pr_c[1] : pr_c[1];
         for (INTM j = 2; j<i; ++j) {
            pr_u[j]=2*fusedHomotopyAux<T>(pr_signs[j-1],
                  pr_signs[j],pr_signs[j+1], pr_c[j-1],pr_c[j]);
         }
         pr_u[i] = pr_signs[i-1] ? -pr_c[i-1] : pr_c[i-1];
         pr_u[i] += pr_signs[i] ? pr_c[i-1]+pr_c[i] : -pr_c[i-1]-pr_c[i];
      } 

      // Compute Du 
      pr_Du[0]=pr_u[0];
      for (INTM k = 1; k<pr_ind[1]; ++k)
         pr_Du[k]=pr_Du[0];
      for (INTM j = 1; j<=i; ++j) {
         pr_Du[pr_ind[j]]=pr_Du[pr_ind[j]-1]+pr_u[j];
         for (INTM k = pr_ind[j]+1; k<pr_ind[j+1]; ++k)
            pr_Du[k]=pr_Du[pr_ind[j]];
      }

      /// Compute DDu 
      DDu.copy(Du);
      for (INTM j = K-2; j>=0; --j) 
         pr_DDu[j] += pr_DDu[j+1];

      /// Check constraINTMs
      T max_step1 = INFINITY;
      if (penalty) {
         max_step1 = currentLambda-lambda_2;
      } 

      /// Check changes of sign
      T max_step2 = INFINITY;
      INTM step_out = -1;
      for (INTM j = 1; j<=i; ++j) {
         T ratio = -pr_gamma[pr_ind[j]]/pr_u[j];
         if (ratio > 0 && ratio <= max_step2) {
            max_step2=ratio;
            step_out=j;
         }
      }
      T max_step3 = INFINITY;
      /// Check new variables entering the active set
      for (INTM j = 1; j<K; ++j) {
         T sc1 = (currentLambda-pr_DtR[j])/(T(1.0)-pr_DDu[j]);
         T sc2 = (currentLambda+pr_DtR[j])/(T(1.0)+pr_DDu[j]);
         if (sc1 <= 1e-10) sc1=INFINITY;
         if (sc2 <= 1e-10) sc2=INFINITY;
         pr_scores[j]= MIN(sc1,sc2);
      }
      for (INTM j = 0; j<=i; ++j) {
         pr_scores[pr_ind[j]]=INFINITY;
      }
      currentInd = scores.fmin();
      max_step3 = pr_scores[currentInd];
      T step = MIN(max_step1,MIN(max_step3,max_step2));
      if (step == 0 || step == INFINITY) break; 

      /// Update gamma, alpha, DtR, currentLambda
      for (INTM j = 0; j<=i; ++j) {
         pr_gamma[pr_ind[j]]+=step*pr_u[j];
      }
      alpha.add(Du,step);
      this->add(DDu,-step);
      currentLambda -= step;
      if (step == max_step2) {
         /// Update signs,pr_ind, pr_c
         for (INTM k = step_out; k<=i; ++k) 
            pr_ind[k]=pr_ind[k+1];
         pr_ind[i]=K;
         for (INTM k = step_out; k<=i; ++k) 
            pr_signs[k]=pr_signs[k+1];
         pr_c[step_out-1]=T(1.0)/(pr_ind[step_out]-pr_ind[step_out-1]);
         pr_c[step_out]=T(1.0)/(pr_ind[step_out+1]-pr_ind[step_out]);
         i-=2;
         newAtom=false;
      } else {
         newAtom=true;
      }
   }

   if (penalty) {
      alpha.softThrshold(lambda_1);
      alpha.scal(T(1.0/(1.0+lambda_3)));
   } else {
      alpha.softThrshold(lambda_1*currentLambda/lambda_2);
      alpha.scal(T(1.0/(1.0+lambda_3*currentLambda/lambda_2)));
   }
};

template <typename T>
inline void Vector<T>::fusedProject(Vector<T>& alpha, const T lambda_1, const T lambda_2,
      const int itermax) {
   T* pr_alpha= alpha.rawX();
   T* pr_beta=_X;
   const INTM K = alpha.n();

   T total_alpha =alpha.sum();
   /// Modification of beta
   for (INTM i = K-2; i>=0; --i) 
      pr_beta[i]+=pr_beta[i+1];

   for (INTM i = 0; i<itermax; ++i) {
      T sum_alpha=0;
      T sum_diff = 0;
      /// Update first coordinate
      T gamma_old=pr_alpha[0];
      pr_alpha[0]=(K*gamma_old+pr_beta[0]-
            total_alpha)/K;
      T diff = pr_alpha[0]-gamma_old;
      sum_diff += diff;
      sum_alpha += pr_alpha[0];
      total_alpha +=K*diff;

      /// Update alpha_j
      for (INTM j = 1; j<K; ++j) {
         pr_alpha[j]+=sum_diff;
         T gamma_old=pr_alpha[j]-pr_alpha[j-1];
         T gamma_new=softThrs((K-j)*gamma_old+pr_beta[j]-
               (total_alpha-sum_alpha),lambda_2)/(K-j);
         pr_alpha[j]=pr_alpha[j-1]+gamma_new;
         T diff = gamma_new-gamma_old;
         sum_diff += diff;
         sum_alpha+=pr_alpha[j];
         total_alpha +=(K-j)*diff;
      }
   }
   alpha.softThrshold(lambda_1);

};

/// sort the vector
template <typename T>
inline void Vector<T>::sort(const bool mode) {
   if (mode) {
      lasrt<T>(incr,_n,_X);
   } else {
      lasrt<T>(decr,_n,_X);
   }
};


/// sort the vector
template <typename T>
inline void Vector<T>::sort(Vector<T>& out, const bool mode) const {
   out.copy(*this);
   out.sort(mode);
};

template <typename T>
inline void Vector<T>::sort2(Vector<INTM>& key, const bool mode) {
   quick_sort(key.rawX(),_X,(INTM)0,_n-1,mode);
};


template <typename T>
inline void Vector<T>::sort2(Vector<T>& out, Vector<INTM>& key, const bool mode) const {
   out.copy(*this);
   out.sort2(key,mode);
}

template <typename T>
inline void Vector<T>::applyBayerPattern(const int offset) {
   INTM sizePatch=_n/3;
   INTM n = static_cast<INTM>(sqrt(static_cast<T>(sizePatch)));
   if (offset == 0) {
      // R
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 1 : 2;
         const INTM off = (i % 2) ? 0 : 1;
         for (INTM j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (INTM i = 0; i<n; ++i) {
         const INTM step = 2;
         const INTM off = (i % 2) ? 1 : 0;
         for (INTM j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 2 : 1;
         const INTM off = 0;
         for (INTM j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   } else if (offset == 1) {
      // R
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 2 : 1;
         const INTM off = (i % 2) ? 1 : 0;
         for (INTM j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (INTM i = 0; i<n; ++i) {
         const INTM step = 2;
         const INTM off = (i % 2) ? 0 : 1;
         for (INTM j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 1 : 2;
         const INTM off = 0;
         for (INTM j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   } else if (offset == 2) {
      // R
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 1 : 2;
         const INTM off = 0;
         for (INTM j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (INTM i = 0; i<n; ++i) {
         const INTM step = 2;
         const INTM off = (i % 2) ? 0 : 1;
         for (INTM j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 2 : 1;
         const INTM off = (i % 2) ? 1 : 0;
         for (INTM j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   } else if (offset == 3) {
      // R
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 2 : 1;
         const INTM off = 0;
         for (INTM j = off; j<n; j+=step) {
            _X[i*n+j]=0;
         }
      }
      // G
      for (INTM i = 0; i<n; ++i) {
         const INTM step = 2;
         const INTM off = (i % 2) ? 1 : 0;
         for (INTM j = off; j<n; j+=step) {
            _X[sizePatch+i*n+j]=0;
         }
      }
      // B
      for (INTM i = 0; i<n; ++i) {
         const INTM step = (i % 2) ? 1 : 2;
         const INTM off = (i % 2) ? 0 : 1;
         for (INTM j = off; j<n; j+=step) {
            _X[2*sizePatch+i*n+j]=0;
         }
      }
   }
};


/// make a sparse copy 
template <typename T> inline void Vector<T>::toSparse(
      SpVector<T>& vec) const {
   INTM L=0;
   T* v = vec._v;
   INTM* r = vec._r;
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] != T()) {
         v[L]=_X[i];
         r[L++]=i;
      }
   }
   vec._L=L;
};


template <typename T>
inline void Vector<T>::copyMask(Vector<T>& out, Vector<bool>& mask) const {
   out.resize(_n);
   INTM pointer=0;
   for (INTM i = 0; i<_n; ++i) {
      if (mask[i])
         out[pointer++]=_X[i];
   }
   out.setn(pointer);
};

template <typename T>
inline void Matrix<T>::copyMask(Matrix<T>& out, Vector<bool>& mask) const {
   out.resize(_m,_n);
   INTM count=0;
   for (INTM i = 0; i<mask.n(); ++i)
      if (mask[i])
         ++count;
   out.setm(count);
   for (INTM i = 0; i<_n; ++i) {
      INTM pointer=0;
      for (INTM j = 0; j<_m; ++j) {
         if (mask[j]) {
            out[i*count+pointer]=_X[i*_m+j];
            ++pointer;
         }
      }
   }
};



/* ****************************
 * Implementation of SpMatrix 
 * ****************************/


/// Constructor, CSC format, existing data
template <typename T, typename I> SpMatrix<T,I>::SpMatrix(T* v, I* r, I* pB, I* pE,
      I m, I n, I nzmax) :
   _externAlloc(true), _v(v), _r(r), _pB(pB), _pE(pE), _m(m), _n(n), _nzmax(nzmax)
{ };

/// Constructor, new m x n matrix, with at most nzmax non-zeros values
template <typename T, typename I> SpMatrix<T,I>::SpMatrix(I m, I n, I nzmax) :
   _externAlloc(false), _m(m), _n(n), _nzmax(nzmax) {
#pragma omp critical
      {
         _v=new T[nzmax];
         _r=new I[nzmax];
         _pB=new I[_n+1];
      }
      _pE=_pB+1;
   };

/// Empty constructor
template <typename T, typename I> SpMatrix<T,I>::SpMatrix() :
   _externAlloc(true), _v(NULL), _r(NULL), _pB(NULL), _pE(NULL),
   _m(0),_n(0),_nzmax(0) { };


template <typename T, typename I>
inline void SpMatrix<T,I>::copy(const SpMatrix<T,I>& mat) {
   this->resize(mat._m,mat._n,mat._nzmax);
   memcpy(_v,mat._v,_nzmax*sizeof(T));
   memcpy(_r,mat._r,_nzmax*sizeof(I));
   memcpy(_pB,mat._pB,(_n+1)*sizeof(I));
}


/// Destructor
template <typename T, typename I> SpMatrix<T,I>::~SpMatrix() {
   clear();
};

/// reference the column i Io vec
template <typename T, typename I> inline void SpMatrix<T,I>::refCol(I i, 
      SpVector<T,I>& vec) const {
   if (vec._nzmax > 0) vec.clear();
   vec._v=_v+_pB[i];
   vec._r=_r+_pB[i];
   vec._externAlloc=true;
   vec._L=_pE[i]-_pB[i];
   vec._nzmax=vec._L;
};

/// print the sparse matrix
template<typename T, typename I> inline void SpMatrix<T,I>::print(const string& name) const {
   cerr << name << endl;
   cerr << _m << " x " << _n << " , " << _nzmax << endl;
   for (I i = 0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         cerr << "(" <<_r[j] << "," << i << ") = " << _v[j] << endl;
      }
   }
};

template<typename T, typename I>
inline T SpMatrix<T,I>::operator[](const I index) const {
   const I num_col=(index/_m);
   const I num_row=index -num_col*_m;
   T val = 0;
   for (I j = _pB[num_col]; j<_pB[num_col+1]; ++j) {
      if (_r[j]==num_row) {
         val=_v[j];
         break;
      }
   }
   return val;
};
template<typename T, typename I>
void SpMatrix<T,I>::getData(Vector<T>& data, const I index) const {
   data.resize(_m);
   data.setZeros();
   for (I i = _pB[index]; i< _pB[index+1]; ++i) 
      data[_r[i]]=_v[i];
};

template <typename T, typename I>
void SpMatrix<T,I>::setData(T* v, I* r, I* pB, I* pE, I m, I n, I nzmax) {
   this->clear();
   _externAlloc =true;
    _v = v;
    _r=r;
    _pB=pB;
    _pE=pE;
    _m=m;
    _n=n;
    _nzmax=nzmax;
}

/// compute the sum of the matrix elements
template <typename T, typename I> inline T SpMatrix<T,I>::asum() const {
   return cblas_asum<T>(_pB[_n],_v,1);
};

/// compute the sum of the matrix elements
template <typename T, typename I> inline T SpMatrix<T,I>::normFsq() const {
   return cblas_dot<T>(_pB[_n],_v,1,_v,1);
};

template <typename T, typename I>
inline void SpMatrix<T,I>::add_direct(const SpMatrix<T,I>& mat, const T a) {
   Vector<T> v2(mat._v,mat._nzmax);
   Vector<T> v1(_v,_nzmax);
   v1.add(v2,a);
}

template <typename T, typename I>
inline void SpMatrix<T,I>::copy_direct(const SpMatrix<T,I>& mat) {
   Vector<T> v2(mat._v,_pB[_n]);
   Vector<T> v1(_v,_pB[_n]);
   v1.copy(v2);
}

template <typename T, typename I>
inline T SpMatrix<T,I>::dot_direct(const SpMatrix<T,I>& mat) const {
   Vector<T> v2(mat._v,_pB[_n]);
   Vector<T> v1(_v,_pB[_n]);
   return v1.dot(v2);
}

/// clear the matrix
template <typename T, typename I> inline void SpMatrix<T,I>::clear() {
   if (!_externAlloc) {
      delete[](_r);
      delete[](_v);
      delete[](_pB);
   }
   _n=0;
   _m=0;
   _nzmax=0;
   _v=NULL;
   _r=NULL;
   _pB=NULL;
   _pE=NULL;
   _externAlloc=true;
};

/// resize the matrix
template <typename T, typename I> inline void SpMatrix<T,I>::resize(const I m, 
      const I n, const I nzmax) {
   if (n == _n && m == _m && nzmax == _nzmax) return;
   this->clear();
   _n=n;
   _m=m;
   _nzmax=nzmax;
   _externAlloc=false;
#pragma omp critical
   {
      _v = new T[nzmax];
      _r = new I[nzmax];
      _pB = new I[_n+1];
   }
   _pE = _pB+1;
   for (I i = 0; i<=_n; ++i) _pB[i]=0;
};

/// resize the matrix
template <typename T, typename I> inline void SpMatrix<T,I>::scal(const T a) const {
   cblas_scal<T>(_pB[_n],a,_v,1);
};

///// resize the matrix
template <typename T, typename I> inline T SpMatrix<T,I>::abs_mean() const {
   Vector<T> vec(_v,_pB[_n]);
   return vec.abs_mean();
};


/// y <- A'*x
template <typename T, typename I>
inline void SpMatrix<T,I>::multTrans(const Vector<T>& x, Vector<T>& y,
      const T alpha, const T beta) const {
   y.resize(_n);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   const T* prX = x.rawX();
#pragma omp parallel for
   for (I i = 0; i<_n; ++i) {
      T sum=T();
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         sum+=_v[j]*prX[_r[j]];
      }
      y[i] += alpha*sum;
   }
};

/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename T, typename I>
inline void SpMatrix<T,I>::multTrans(const SpVector<T,I>& x, Vector<T>& y, 
      const T alpha, const T beta) const {
   y.resize(_n);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   T* prY = y.rawX();
   SpVector<T,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      prY[i] += alpha*x.dot(col);
   }
};


/// y <- A*x
template <typename T, typename I>
inline void SpMatrix<T,I>::mult(const Vector<T>& x, Vector<T>& y,
      const T alpha, const T beta) const {
   y.resize(_m);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   const T* prX = x.rawX();
   for (I i = 0; i<_n; ++i) {
      T sca=alpha* prX[i];
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         y[_r[j]] += sca*_v[j];
      }
   }
};


/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename T, typename I>
inline void SpMatrix<T,I>::mult(const SpVector<T,I>& x, Vector<T>& y, 
      const T alpha, const T beta) const {
   y.resize(_m);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   T* prY = y.rawX();
   for (I i = 0; i<x.L(); ++i) {
      I ind=x.r(i);
      T val = alpha * x.v(i);
      for (I j = _pB[ind]; j<_pE[ind]; ++j) {
         prY[_r[j]] += val *_v[j];
      }
   }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename T, typename I>
inline void SpMatrix<T,I>::mult(const Matrix<T>& B, Matrix<T>& C, 
      const bool transA, const bool transB,
      const T a, const T b) const {
   if (transA) {
      if (transB) {
         C.resize(_n,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<T,I> tmp;
         Vector<T> row(B.m());
         for (I i = 0; i<_n; ++i) {
            this->refCol(i,tmp);
            B.mult(tmp,row);
            C.addRow(i,row,a);
         }
      } else {
         C.resize(_n,B.n());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<T,I> tmp;
         Vector<T> row(B.n());
         for (I i = 0; i<_n; ++i) {
            this->refCol(i,tmp);
            B.multTrans(tmp,row);
            C.addRow(i,row,a);
         }
      }
   } else {
      if (transB) {
         C.resize(_m,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<T> row(B.n());
         Vector<T> col;
         for (I i = 0; i<B.m(); ++i) {
            B.copyRow(i,row);
            C.refCol(i,col);
            this->mult(row,col,a,T(1.0));
         }
      } else {
         C.resize(_m,B.n());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<T> colB;
         Vector<T> colC;
         for (I i = 0; i<B.n(); ++i) {
            B.refCol(i,colB);
            C.refCol(i,colC);
            this->mult(colB,colC,a,T(1.0));
         }
      }
   }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename T, typename I>
inline void SpMatrix<T,I>::mult(const SpMatrix<T,I>& B, Matrix<T>& C, 
      const bool transA, const bool transB,
      const T a, const T b) const {
   if (transA) {
      if (transB) {
         C.resize(_n,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<T,I> tmp;
         Vector<T> row(B.m());
         for (I i = 0; i<_n; ++i) {
            this->refCol(i,tmp);
            B.mult(tmp,row);
            C.addRow(i,row,a);
         }
      } else {
         C.resize(_n,B.n());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<T,I> tmp;
         Vector<T> row(B.n());
         for (I i = 0; i<_n; ++i) {
            this->refCol(i,tmp);
            B.multTrans(tmp,row);
            C.addRow(i,row,a);
         }
      }
   } else {
      if (transB) {
         C.resize(_m,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<T,I> colB;
         SpVector<T,I> colA;
         for (I i = 0; i<_n; ++i) {
            this->refCol(i,colA);
            B.refCol(i,colB);
            C.rank1Update(colA,colB,a);
         }
      } else {
         C.resize(_m,B.n());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<T,I> colB;
         Vector<T> colC;
         for (I i = 0; i<B.n(); ++i) {
            B.refCol(i,colB);
            C.refCol(i,colC);
            this->mult(colB,colC,a);
         }
      }
   }
};

/// perform C = a*B*A + b*C, possibly transposing A or B.
template <typename T, typename I>
inline void SpMatrix<T,I>::multSwitch(const Matrix<T>& B, Matrix<T>& C, 
      const bool transA, const bool transB,
      const T a, const T b) const {
   B.mult(*this,C,transB,transA,a,b);
};

template <typename T, typename I>
inline T SpMatrix<T,I>::dot(const Matrix<T>& x) const {
   T sum=0;
   for (I i = 0; i<_n; ++i)
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         sum+=_v[j]*x(_r[j],j);
      }
   return sum;
};


template <typename T, typename I>
inline void SpMatrix<T,I>::copyRow(const I ind, Vector<T>& x) const {
   x.resize(_n);
   x.setZeros();
   for (I i = 0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         if (_r[j]==ind) {
            x[i]=_v[j];
         } else if (_r[j] > ind) {
            break;
         }
      }
   }
};

template <typename T, typename I> 
inline void SpMatrix<T,I>::addVecToCols(
      const Vector<T>& vec, const T a) {
   const T* pr_vec = vec.rawX();
   if (isEqual(a,T(1.0))) {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += pr_vec[_r[j]];
   } else {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += a*pr_vec[_r[j]];
   }
};

template <typename T, typename I> 
inline void SpMatrix<T,I>::addVecToColsWeighted(
      const Vector<T>& vec, const T* weights, const T a) {
   const T* pr_vec = vec.rawX();
   if (isEqual(a,T(1.0))) {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += pr_vec[_r[j]]*weights[j-_pB[i]];
   } else {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += a*pr_vec[_r[j]]*weights[j-_pB[i]];
   }
};

template <typename T, typename I> 
inline void SpMatrix<T,I>::sum_cols(Vector<T>& sum) const {
   sum.resize(_m);
   sum.setZeros();
   SpVector<T,I> tmp;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,tmp);
      sum.add(tmp);
   }
};

/// aat <- A*A'
template <typename T, typename I> inline void SpMatrix<T,I>::AAt(Matrix<T>& aat) const {
   I i,j,k;
   I K=_m;
   I M=_n;

   /* compute alpha alpha^T */
   aat.resize(K,K);
   int NUM_THREADS=init_omp(MAX_THREADS);
   T* aatT=new T[NUM_THREADS*K*K];
   for (j = 0; j<NUM_THREADS*K*K; ++j) aatT[j]=T();

#pragma omp parallel for private(i,j,k)
   for (i = 0; i<M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      T* write_area=aatT+numT*K*K;
      for (j = _pB[i]; j<_pE[i]; ++j) {
         for (k = _pB[i]; k<=j; ++k) {
            write_area[_r[j]*K+_r[k]]+=_v[j]*_v[k];
         }
      }
   }

   cblas_copy<T>(K*K,aatT,1,aat._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<T>(K*K,1.0,aatT+K*K*i,1,aat._X,1);
   aat.fillSymmetric();
   delete[](aatT);
}

template <typename T, typename I>
inline void SpMatrix<T,I>::XtX(Matrix<T>& XtX) const {
   XtX.resize(_n,_n);
   XtX.setZeros();
   SpVector<T,I> col;
   Vector<T> col_out;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      XtX.refCol(i,col_out);
      this->multTrans(col,col_out);
   }
};


/// aat <- A(:,indices)*A(:,indices)'
template <typename T, typename I> inline void SpMatrix<T,I>::AAt(Matrix<T>& aat,
      const Vector<I>& indices) const {
   I i,j,k;
   I K=_m;
   I M=indices.n();

   /* compute alpha alpha^T */
   aat.resize(K,K);
   int NUM_THREADS=init_omp(MAX_THREADS);
   T* aatT=new T[NUM_THREADS*K*K];
   for (j = 0; j<NUM_THREADS*K*K; ++j) aatT[j]=T();

#pragma omp parallel for private(i,j,k)
   for (i = 0; i<M; ++i) {
      I ii = indices[i];
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      T* write_area=aatT+numT*K*K;
      for (j = _pB[ii]; j<_pE[ii]; ++j) {
         for (k = _pB[ii]; k<=j; ++k) {
            write_area[_r[j]*K+_r[k]]+=_v[j]*_v[k];
         }
      }
   }

   cblas_copy<T>(K*K,aatT,1,aat._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<T>(K*K,1.0,aatT+K*K*i,1,aat._X,1);
   aat.fillSymmetric();
   delete[](aatT);
}

/// aat <- sum_i w_i A(:,i)*A(:,i)'
template <typename T, typename I> inline void SpMatrix<T,I>::wAAt(const Vector<T>& w,
      Matrix<T>& aat) const {
   I i,j,k;
   I K=_m;
   I M=_n;

   /* compute alpha alpha^T */
   aat.resize(K,K);
   int NUM_THREADS=init_omp(MAX_THREADS);
   T* aatT=new T[NUM_THREADS*K*K];
   for (j = 0; j<NUM_THREADS*K*K; ++j) aatT[j]=T();

#pragma omp parallel for private(i,j,k)
   for (i = 0; i<M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      T* write_area=aatT+numT*K*K;
      for (j = _pB[i]; j<_pE[i]; ++j) {
         for (k = _pB[i]; k<=j; ++k) {
            write_area[_r[j]*K+_r[k]]+=w._X[i]*_v[j]*_v[k];
         }
      }
   }

   cblas_copy<T>(K*K,aatT,1,aat._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<T>(K*K,1.0,aatT+K*K*i,1,aat._X,1);
   aat.fillSymmetric();
   delete[](aatT);
}

/// XAt <- X*A'
template <typename T, typename I> inline void SpMatrix<T,I>::XAt(const Matrix<T>& X,
      Matrix<T>& XAt) const {
   I j,i;
   I n=X._m;
   I K=_m;
   I M=_n;

   XAt.resize(n,K);
   /* compute X alpha^T */
//   int NUM_THREADS=init_omp(MAX_THREADS);
   //T* XatT=new T[NUM_THREADS*n*K];
   //for (j = 0; j<NUM_THREADS*n*K; ++j) XatT[j]=T();

//#pragma omp parallel for private(i,j)
   for (i = 0; i<M; ++i) {
//#ifdef _OPENMP
//      int numT=omp_get_thread_num();
//#else
//      int numT=0;
//#endif
//      T* write_area=XatT+numT*n*K;
      for (j = _pB[i]; j<_pE[i]; ++j) {
         cblas_axpy<T>(n,_v[j],X._X+i*n,1,XAt._X+_r[j]*n,1);
      }
   }
 //  cblas_copy<T>(n*K,XatT,1,XAt._X,1);
//   for (i = 1; i<NUM_THREADS; ++i) 
//      cblas_axpy<T>(n*K,1.0,XatT+n*K*i,1,XAt._X,1);
//   delete[](XatT);
};

/// XAt <- X(:,indices)*A(:,indices)'
template <typename T, typename I> inline void SpMatrix<T,I>::XAt(const Matrix<T>& X,
      Matrix<T>& XAt, const Vector<I>& indices) const {
   I j,i;
   I n=X._m;
   I K=_m;
   I M=indices.n();

   XAt.resize(n,K);
   /* compute X alpha^T */
   int NUM_THREADS=init_omp(MAX_THREADS);
   T* XatT=new T[NUM_THREADS*n*K];
   for (j = 0; j<NUM_THREADS*n*K; ++j) XatT[j]=T();

#pragma omp parallel for private(i,j)
   for (i = 0; i<M; ++i) {
      I ii = indices[i];
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      T* write_area=XatT+numT*n*K;
      for (j = _pB[ii]; j<_pE[ii]; ++j) {
         cblas_axpy<T>(n,_v[j],X._X+i*n,1,write_area+_r[j]*n,1);
      }
   }

   cblas_copy<T>(n*K,XatT,1,XAt._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<T>(n*K,1.0,XatT+n*K*i,1,XAt._X,1);
   delete[](XatT);
};

/// XAt <- sum_i w_i X(:,i)*A(:,i)'
template <typename T, typename I> inline void SpMatrix<T,I>::wXAt(const Vector<T>& w,
      const Matrix<T>& X, Matrix<T>& XAt, const int numThreads) const {
   I j,l,i;
   I n=X._m;
   I K=_m;
   I M=_n;
   I Mx = X._n;
   I numRepX= M/Mx;
   assert(numRepX*Mx == M);
   XAt.resize(n,K);
   /* compute X alpha^T */
   int NUM_THREADS=init_omp(numThreads);
   T* XatT=new T[NUM_THREADS*n*K];
   for (j = 0; j<NUM_THREADS*n*K; ++j) XatT[j]=T();

#pragma omp parallel for private(i,j,l)
   for (i = 0; i<Mx; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      T * write_area=XatT+numT*n*K;
      for (l = 0; l<numRepX; ++l) {
         I ind=numRepX*i+l;
         if (w._X[ind] != 0)
            for (j = _pB[ind]; j<_pE[ind]; ++j) {
               cblas_axpy<T>(n,w._X[ind]*_v[j],X._X+i*n,1,write_area+_r[j]*n,1);
            }
      }
   }

   cblas_copy<T>(n*K,XatT,1,XAt._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<T>(n*K,1.0,XatT+n*K*i,1,XAt._X,1);
   delete[](XatT);
};

/// copy the sparse matrix into a dense matrix
template<typename T, typename I> inline void SpMatrix<T,I>::toFull(Matrix<T>& matrix) const {
   matrix.resize(_m,_n);
   matrix.setZeros();
   T* out = matrix._X;
   for (I i=0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         out[i*_m+_r[j]]=_v[j];
      }
   }
};

/// copy the sparse matrix into a full dense matrix
template <typename T, typename I> inline void SpMatrix<T,I>::toFullTrans(
      Matrix<T>& matrix) const {
   matrix.resize(_n,_m);
   matrix.setZeros();
   T* out = matrix._X;
   for (I i=0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         out[i+_r[j]*_n]=_v[j];
      }
   }
};


/// use the data from v, r for _v, _r
template <typename T, typename I> inline void SpMatrix<T,I>::convert(const Matrix<T>&vM, 
      const Matrix<I>& rM, const I K) {
   const I M = rM.n();
   const I L = rM.m();
   const I* r = rM.X();
   const T* v = vM.X();
   I count=0;
   for (I i = 0; i<M*L; ++i) if (r[i] != -1) ++count;
   resize(K,M,count);
   count=0;
   for (I i = 0; i<M; ++i) {
      _pB[i]=count;
      for (I j = 0; j<L; ++j) {
         if (r[i*L+j] == -1) break;
         _v[count]=v[i*L+j];
         _r[count++]=r[i*L+j];
      }
      _pE[i]=count;
   }
   for (I i = 0; i<M; ++i) sort(_r,_v,_pB[i],_pE[i]-1);
};

/// use the data from v, r for _v, _r
template <typename T, typename I> inline void SpMatrix<T,I>::convert2(
      const Matrix<T>&vM, const Vector<I>& rv, const I K) {
   const I M = vM.n();
   const I L = vM.m();
   I* r = rv.rawX();
   const T* v = vM.X();
   I LL=0;
   for (I i = 0; i<L; ++i) if (r[i] != -1) ++LL;
   this->resize(K,M,LL*M);
   I count=0;
   for (I i = 0; i<M; ++i) {
      _pB[i]=count;
      for (I j = 0; j<LL; ++j) {
         _v[count]=v[i*L+j];
         _r[count++]=r[j];
      }
      _pE[i]=count;
   }
   for (I i = 0; i<M; ++i) sort(_r,_v,_pB[i],_pE[i]-1);
};

/// returns the l2 norms ^2 of the columns
template <typename T, typename I>
inline void SpMatrix<T,I>::normalize() {
   SpVector<T,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      const T norm = col.nrm2sq();
      if (norm > 1e-10)
         col.scal(T(1.0)/col.nrm2sq());
   }
};

/// returns the l2 norms ^2 of the columns
template <typename T, typename I>
inline void SpMatrix<T,I>::normalize_rows() {
   Vector<T> norms(_m);
   norms.setZeros();
   for (I i = 0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         norms[_r[j]] += _v[j]*_v[j];
      }
   }
   norms.Sqrt();
   for (I i = 0; i<_m; ++i) 
      norms[i] = norms[i] < 1e-10 ? T(1.0) : T(1.0)/norms[i];
   for (I i = 0; i<_n; ++i) 
      for (I j = _pB[i]; j<_pE[i]; ++j) 
         _v[j] *= norms[_r[j]];
};




/// returns the l2 norms ^2 of the columns
template <typename T, typename I>
inline void SpMatrix<T,I>::norm_2sq_cols(Vector<T>& norms) const {
   norms.resize(_n);
   SpVector<T,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      norms[i] = col.nrm2sq();
   }
};


template <typename T, typename I>
inline void SpMatrix<T,I>::norm_0_cols(Vector<T>& norms) const {
   norms.resize(_n);
   SpVector<T,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      norms[i] = static_cast<T>(col.length());
   }
};

template <typename T, typename I>
inline void SpMatrix<T,I>::norm_1_cols(Vector<T>& norms) const {
   norms.resize(_n);
   SpVector<T,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      norms[i] =col.asum();
   }
};


/* ***************************
 * Implementation of SpVector 
 * ***************************/


/// Constructor, of the sparse vector of size L.
template <typename T, typename I> SpVector<T,I>::SpVector(T* v, I* r, I L, I nzmax) :
   _externAlloc(true), _v(v), _r(r), _L(L), _nzmax(nzmax)  { };

/// Constructor, allocates nzmax slots
template <typename T, typename I> SpVector<T,I>::SpVector(I nzmax) :
   _externAlloc(false), _L(0), _nzmax(nzmax) {
#pragma omp critical
      {
         _v = new T[nzmax];
         _r = new I[nzmax];
      }
   };

/// Empty constructor
template <typename T, typename I> SpVector<T,I>::SpVector() : _externAlloc(true), _v(NULL), _r(NULL), _L(0),
   _nzmax(0) { };


/// Destructor
template <typename T, typename I> SpVector<T,I>::~SpVector() { clear(); };


/// computes the sum of the magnitude of the elements
template <typename T, typename I> inline T SpVector<T,I>::asum() const {
   return cblas_asum<T>(_L,_v,1);
};

/// computes the l2 norm ^2 of the vector
template <typename T, typename I> inline T SpVector<T,I>::nrm2sq() const {
   return cblas_dot<T>(_L,_v,1,_v,1);
};

/// computes the l2 norm of the vector
template <typename T, typename I> inline T SpVector<T,I>::nrm2() const {
   return cblas_nrm2<T>(_L,_v,1);
};

/// computes the l2 norm of the vector
template <typename T, typename I> inline T SpVector<T,I>::fmaxval() const {
   Vector<T> tmp(_v,_L);
   return tmp.fmaxval();
};

/// print the vector to std::cerr
template <typename T, typename I> inline void SpVector<T,I>::print(const string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _nzmax << std::endl;
   for (I i = 0; i<_L; ++i)
      cerr << "(" <<_r[i] << ", " <<  _v[i] << ")" << endl;
};

/// create a reference on the vector r
template <typename T, typename I> inline void SpVector<T,I>::refIndices(
      Vector<I>& indices) const {
   indices.setPointer(_r,_L);   
};

template <typename T, typename I> inline void SpVector<T,I>::getIndices(Vector<int>& indices) const {
//   indices.resize(_L);
   indices.setn(_L);
   for (int ii=0; ii<_L; ++ii)
      indices[ii]=_r[ii]; 
};

/// creates a reference on the vector val
template <typename T, typename I> inline void SpVector<T,I>::refVal(
      Vector<T>& val) const {
   val.setPointer(_v,_L);   
};

/// a <- a.^2
template <typename T, typename I> inline void SpVector<T,I>::sqr() {
   vSqr<T>(_L,_v,_v);
};

template <typename T, typename I>
inline void SpVector<T,I>::scal(const T a) {
   cblas_scal<T>(_L,a,_v,1);
};

template <typename T, typename I>
inline T SpVector<T,I>::dot(const SpVector<T,I>& vec) const {
   T sum=T();
   I countI = 0;
   I countJ = 0;
   while (countI < _L && countJ < vec._L) {
      const I rI = _r[countI];
      const I rJ = vec._r[countJ];
      if (rI > rJ) {
         ++countJ;
      } else if (rJ > rI) {
         ++countI;
      } else {
         sum+=_v[countI]*vec._v[countJ];
         ++countI;
         ++countJ;
      }
   }
   return sum;
};

template <typename T, typename I>
inline T SpVector<T,I>::dot(const Vector<T>& vec) const {
   //return cblas_doti(_L,_v,_r,vec.rawX());
   T sum=T();
   for (int countI=0; countI < _L; ++countI)
      sum+=_v[countI]*vec[_r[countI]];
   return sum;
};

/// clears the vector
template <typename T, typename I> inline void SpVector<T,I>::clear() {
   if (!_externAlloc) {
      delete[](_v);
      delete[](_r);
   }
   _v=NULL;
   _r=NULL;
   _L=0;
   _nzmax=0;
   _externAlloc=true;
};

/// resizes the vector
template <typename T, typename I> inline void SpVector<T,I>::resize(const I nzmax) {
   if (_nzmax != nzmax) {
      clear();
      _nzmax=nzmax;
      _L=0;
      _externAlloc=false;
#pragma omp critical
      {
         _v=new T[nzmax];
         _r=new I[nzmax];
      }
   }
};

template <typename T, typename I> void inline SpVector<T,I>::toSpMatrix(
      SpMatrix<T,I>& out, const I m, const I n) const {
   out.resize(m,n,_L);
   cblas_copy<T>(_L,_v,1,out._v,1);
   I current_col=0;
   I* out_r=out._r;
   I* out_pB=out._pB;
   out_pB[0]=current_col;
   for (I i = 0; i<_L; ++i) {
      I col=_r[i]/m;
      if (col > current_col) {
         out_pB[current_col+1]=i;
         current_col++;
         i--;
      } else {
         out_r[i]=_r[i]-col*m;
      }
   }
   for (current_col++ ; current_col < n+1; ++current_col) 
      out_pB[current_col]=_L;
};

template <typename T, typename I> void inline SpVector<T,I>::toFull(Vector<T>& out)
   const {
      out.setZeros();
      T* X = out.rawX();
      for (I i = 0; i<_L; ++i)
         X[_r[i]]=_v[i];
   };


/* ************************************
 * Implementation of the class Matrix 
 * ************************************/

/// Constructor with existing data X of an m x n matrix
template <typename T> Matrix<T>::Matrix(T* X, INTM m, INTM n) :
   _externAlloc(true), _X(X), _m(m), _n(n) {  };


/// Constructor for a new m x n matrix
template <typename T> Matrix<T>::Matrix(INTM m, INTM n) :
   _externAlloc(false), _m(m), _n(n)  {
#pragma omp critical
      {
         _X= new T[_n*_m];
      }
   };

/// Empty constructor
template <typename T> Matrix<T>::Matrix() :
   _externAlloc(false), _X(NULL), _m(0), _n(0) { };

/// Destructor
template <typename T> Matrix<T>::~Matrix() {
   clear();
};

/// Return a modifiable reference to X(i,j)
template <typename T> inline T& Matrix<T>::operator()(const INTM i, const INTM j) {
   return _X[j*_m+i];
};

/// Return the value X(i,j)
template <typename T> inline T Matrix<T>::operator()(const INTM i, const INTM j) const {
   return _X[j*_m+i];
};

/// Print the matrix to std::cout
template <typename T> inline void Matrix<T>::print(const string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _m << " x " << _n << std::endl;
   for (INTM i = 0; i<_m; ++i) {
      for (INTM j = 0; j<_n; ++j) {
         printf("%10.5g ",static_cast<double>(_X[j*_m+i]));
      }
      printf("\n ");
   }
   printf("\n ");
};

/// Print the matrix to std::cout
template <typename T> inline void Matrix<T>::dump(const string& name) const {
   ofstream f; 
   const char * cname = name.c_str();
   f.open(cname);
   f.precision(20);
   std::cerr << name << std::endl;
   f << _m << " x " << _n << std::endl;
   for (INTM i = 0; i<_m; ++i) {
      for (INTM j = 0; j<_n; ++j) {
         f << static_cast<double>(_X[j*_m+i]) << " ";
      }
      f << std::endl;
   }
   f << std::endl;
   f.close();
};

/// Copy the column i INTMo x
template <typename T> inline void Matrix<T>::copyCol(const INTM i, Vector<T>& x) const {
   assert(i >= 0 && i<_n);
   x.resize(_m);
   cblas_copy<T>(_m,_X+i*_m,1,x._X,1);
};

/// Copy the column i INTMo x
template <typename T> inline void Matrix<T>::copyRow(const INTM i, Vector<T>& x) const {
   assert(i >= 0 && i<_m);
   x.resize(_n);
   cblas_copy<T>(_n,_X+i,_m,x._X,1);
};

/// Copy the column i INTMo x
template <typename T> inline void Matrix<T>::scalRow(const INTM i, const T s) const {
   assert(i >= 0 && i<_m);
   for (int ii=0; ii<_n; ++ii)
      _X[i+ii*_m] *= s;
};


/// Copy the column i INTMo x
template <typename T> inline void Matrix<T>::copyToRow(const INTM i, const Vector<T>& x) {
   assert(i >= 0 && i<_m);
   cblas_copy<T>(_n,x._X,1,_X+i,_m);
};

/// Copy the column i INTMo x
template <typename T> inline void Matrix<T>::extract_rawCol(const INTM i, T* x) const {
   assert(i >= 0 && i<_n);
   cblas_copy<T>(_m,_X+i*_m,1,x,1);
};

/// Copy the column i INTMo x
template <typename T> inline void Matrix<T>::add_rawCol(const INTM i, T* x, const T a) const {
   assert(i >= 0 && i<_n);
   cblas_axpy<T>(_m,a,_X+i*_m,1,x,1);
};

/// Copy the column i INTMo x
template <typename T> inline void Matrix<T>::getData(Vector<T>& x, const INTM i) const {
   this->copyCol(i,x);
};

/// Reference the column i into the vector x
template <typename T> inline void Matrix<T>::refCol(INTM i, Vector<T>& x) const {
   assert(i >= 0 && i<_n);
   x.clear();
   x._X=_X+i*_m;
   x._n=_m;
   x._externAlloc=true; 
};

/// Reference the column i to i+n INTMo the Matrix mat
template <typename T> inline void Matrix<T>::refSubMat(INTM i, INTM n, Matrix<T>& mat) const {
   mat.setData(_X+i*_m,_m,n);
}

/// Check wether the columns of the matrix are normalized or not
template <typename T> inline bool Matrix<T>::isNormalized() const {
   for (INTM i = 0; i<_n; ++i) {
      T norm=cblas_nrm2<T>(_m,_X+_m*i,1);
      if (fabs(norm - 1.0) > 1e-6) return false;
   }
   return true;
};

/// clean a dictionary matrix
template <typename T>
inline void Matrix<T>::clean() {
   this->normalize();
   Matrix<T> G;
   this->XtX(G);
   T* prG = G._X;
   /// remove the diagonal
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = i+1; j<_n; ++j) {
         if (prG[i*_n+j] > 0.99) {
            // remove nasty column j and put random values inside
            Vector<T> col;
            this->refCol(j,col);
            col.setAleat();
            col.normalize();
         }
      }
   }
};

/// return the 1D-index of the value of greatest magnitude
template <typename T> inline INTM Matrix<T>::fmax() const {
   return cblas_iamax<T>(_n*_m,_X,1);
};

/// return the value of greatest magnitude
template <typename T> inline T Matrix<T>::fmaxval() const {
   return _X[cblas_iamax<T>(_n*_m,_X,1)];
};


/// return the 1D-index of the value of lowest magnitude
template <typename T> inline INTM Matrix<T>::fmin() const {
   return cblas_iamin<T>(_n*_m,_X,1);
};

/// extract a sub-matrix of a symmetric matrix
template <typename T> inline void Matrix<T>::subMatrixSym(
      const Vector<INTM>& indices, Matrix<T>& subMatrix) const {
   INTM L = indices.n();
   subMatrix.resize(L,L);
   T* out = subMatrix._X;
   INTM* rawInd = indices.rawX();
   for (INTM i = 0; i<L; ++i)
      for (INTM j = 0; j<=i; ++j)
         out[i*L+j]=_X[rawInd[i]*_n+rawInd[j]];
   subMatrix.fillSymmetric();
};

/// Resize the matrix
template <typename T> inline void Matrix<T>::resize(INTM m, INTM n, const bool set_zeros) {
   if (_n==n && _m==m) return;
   clear();
   _n=n;
   _m=m;
   _externAlloc=false;
#pragma omp critical
   {
      _X=new T[_n*_m];
   }
   if (set_zeros)
      setZeros();
};

/// Change the data in the matrix
template <typename T> inline void Matrix<T>::setData(T* X, INTM m, INTM n) {
   clear();
   _X=X;
   _m=m;
   _n=n;
   _externAlloc=true;
};

/// Set all the values to zero
template <typename T> inline void Matrix<T>::setZeros() {
   memset(_X,0,_n*_m*sizeof(T));
};

/// Set all the values to a scalar
template <typename T> inline void Matrix<T>::set(const T a) {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]=a;
};

/// Clear the matrix
template <typename T> inline void Matrix<T>::clear() {
   if (!_externAlloc) delete[](_X);
   _n=0;
   _m=0;
   _X=NULL;
   _externAlloc=true;
};

/// Put white Gaussian noise in the matrix 
template <typename T> inline void Matrix<T>::setAleat() {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]=normalDistrib<T>();
};

/// set the matrix to the identity
template <typename T> inline void Matrix<T>::eye() {
   this->setZeros();
   for (INTM i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] = T(1.0);
};

/// Normalize all columns to unit l2 norm
template <typename T> inline void Matrix<T>::normalize() {
   //T constant = 1.0/sqrt(_m);
   for (INTM i = 0; i<_n; ++i) {
      T norm=cblas_nrm2<T>(_m,_X+_m*i,1);
      if (norm > 1e-10) {
         T invNorm=1.0/norm;
         cblas_scal<T>(_m,invNorm,_X+_m*i,1);
      }  else {
         // for (INTM j = 0; j<_m; ++j) _X[_m*i+j]=constant;
         Vector<T> d;
         this->refCol(i,d);
         d.setAleat();
         d.normalize();
      } 
   }
};

/// Normalize all columns which l2 norm is greater than one.
template <typename T> inline void Matrix<T>::normalize2() {
   for (INTM i = 0; i<_n; ++i) {
      T norm=cblas_nrm2<T>(_m,_X+_m*i,1);
      if (norm > 1.0) {
         T invNorm=1.0/norm;
         cblas_scal<T>(_m,invNorm,_X+_m*i,1);
      } 
   }
};

/// center the matrix
template <typename T> inline void Matrix<T>::center() {
   for (INTM i = 0; i<_n; ++i) {
      Vector<T> col;
      this->refCol(i,col);
      T sum = col.sum();
      col.add(-sum/static_cast<T>(_m));
   }
};

/// center the matrix
template <typename T> inline void Matrix<T>::center_rows() {
   Vector<T> mean_rows(_m);
   mean_rows.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         mean_rows[j] += _X[i*_m+j];
   mean_rows.scal(T(1.0)/_n);
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         _X[i*_m+j] -= mean_rows[j];
};

/// center the matrix
template <typename T> inline void Matrix<T>::normalize_rows() {
   Vector<T> norm_rows(_m);
   norm_rows.setZeros();
   for (INTM i = 0; i<_n; ++i)
      for (INTM j = 0; j<_m; ++j)
         norm_rows[j] += _X[i*_m+j]*_X[i*_m+j];
   for (INTM j = 0; j<_m; ++j)
      norm_rows[j]  = norm_rows[j] < T(1e-10) ? T(1e-10) : T(1.0)/sqrt(norm_rows[j]);
   this->multDiagLeft(norm_rows);
};

/// center the matrix and keep the center values
template <typename T> inline void Matrix<T>::center(Vector<T>& centers) {
   centers.resize(_n);
   for (INTM i = 0; i<_n; ++i) {
      Vector<T> col;
      this->refCol(i,col);
      T sum = col.sum()/static_cast<T>(_m);
      centers[i]=sum;
      col.add(-sum);
   }
};

/// scale the matrix by the a
template <typename T> inline void Matrix<T>::scal(const T a) {
   cblas_scal<T>(_n*_m,a,_X,1);
};

/// make a copy of the matrix mat in the current matrix
template <typename T> inline void Matrix<T>::copy(const Matrix<T>& mat) {
   if (_X != mat._X) {
      resize(mat._m,mat._n);
      //   cblas_copy<T>(_m*_n,mat._X,1,_X,1);
      memcpy(_X,mat._X,_m*_n*sizeof(T));
   }
};

/// make a copy of the matrix mat in the current matrix
template <typename T> inline void Matrix<T>::copyRef(const Matrix<T>& mat) {
   this->setData(mat.rawX(),mat.m(),mat.n());
};

/// make the matrix symmetric by copying the upper-right part
/// INTMo the lower-left part
template <typename T> inline void Matrix<T>::fillSymmetric() {
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j =0; j<i; ++j) {
         _X[j*_m+i]=_X[i*_m+j];
      }
   }
};
template <typename T> inline void Matrix<T>::fillSymmetric2() {
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j =0; j<i; ++j) {
         _X[i*_m+j]=_X[j*_m+i];
      }
   }
};


template <typename T> inline void Matrix<T>::whiten(const INTM V) {
   const INTM sizePatch=_m/V;
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = 0; j<V; ++j) {
         T mean = 0;
         for (INTM k = 0; k<sizePatch; ++k) {
            mean+=_X[i*_m+sizePatch*j+k];
         }
         mean /= sizePatch;
         for (INTM k = 0; k<sizePatch; ++k) {
            _X[i*_m+sizePatch*j+k]-=mean;
         }
      }
   }
};

template <typename T> inline void Matrix<T>::whiten(Vector<T>& mean, const bool pattern) {
   mean.setZeros();
   if (pattern) {
      const INTM n =static_cast<INTM>(sqrt(static_cast<T>(_m)));
      INTM count[4];
      for (INTM i = 0; i<4; ++i) count[i]=0;
      for (INTM i = 0; i<_n; ++i) {
         INTM offsetx=0;
         for (INTM j = 0; j<n; ++j) {
            offsetx= (offsetx+1) % 2;
            INTM offsety=0;
            for (INTM k = 0; k<n; ++k) {
               offsety= (offsety+1) % 2;
               mean[2*offsetx+offsety]+=_X[i*_m+j*n+k];
               count[2*offsetx+offsety]++;
            }
         }
      }
      for (INTM i = 0; i<4; ++i)
         mean[i] /= count[i];
      for (INTM i = 0; i<_n; ++i) {
         INTM offsetx=0;
         for (INTM j = 0; j<n; ++j) {
            offsetx= (offsetx+1) % 2;
            INTM offsety=0;
            for (INTM k = 0; k<n; ++k) {
               offsety= (offsety+1) % 2;
               _X[i*_m+j*n+k]-=mean[2*offsetx+offsety];
            }
         }
      }
   } else  {
      const INTM V = mean.n();
      const INTM sizePatch=_m/V;
      for (INTM i = 0; i<_n; ++i) {
         for (INTM j = 0; j<V; ++j) {
            for (INTM k = 0; k<sizePatch; ++k) {
               mean[j]+=_X[i*_m+sizePatch*j+k];
            }
         }
      }
      mean.scal(T(1.0)/(_n*sizePatch));
      for (INTM i = 0; i<_n; ++i) {
         for (INTM j = 0; j<V; ++j) {
            for (INTM k = 0; k<sizePatch; ++k) {
               _X[i*_m+sizePatch*j+k]-=mean[j];
            }
         }
      }
   }
};

template <typename T> inline void Matrix<T>::whiten(Vector<T>& mean, const
      Vector<T>& mask) {
   const INTM V = mean.n();
   const INTM sizePatch=_m/V;
   mean.setZeros();
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = 0; j<V; ++j) {
         for (INTM k = 0; k<sizePatch; ++k) {
            mean[j]+=_X[i*_m+sizePatch*j+k];
         }
      }
   }
   for (INTM i = 0; i<V; ++i)
      mean[i] /= _n*cblas_asum(sizePatch,mask._X+i*sizePatch,1);
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = 0; j<V; ++j) {
         for (INTM k = 0; k<sizePatch; ++k) {
            if (mask[sizePatch*j+k])
               _X[i*_m+sizePatch*j+k]-=mean[j];
         }
      }
   }
};


template <typename T> inline void Matrix<T>::unwhiten(Vector<T>& mean, const bool pattern) {
   if (pattern) {
      const INTM n =static_cast<INTM>(sqrt(static_cast<T>(_m)));
      for (INTM i = 0; i<_n; ++i) {
         INTM offsetx=0;
         for (INTM j = 0; j<n; ++j) {
            offsetx= (offsetx+1) % 2;
            INTM offsety=0;
            for (INTM k = 0; k<n; ++k) {
               offsety= (offsety+1) % 2;
               _X[i*_m+j*n+k]+=mean[2*offsetx+offsety];
            }
         }
      }
   } else {
      const INTM V = mean.n();
      const INTM sizePatch=_m/V;
      for (INTM i = 0; i<_n; ++i) {
         for (INTM j = 0; j<V; ++j) {
            for (INTM k = 0; k<sizePatch; ++k) {
               _X[i*_m+sizePatch*j+k]+=mean[j];
            }
         }
      }
   }
};

/// Transpose the current matrix and put the result in the matrix
/// trans
template <typename T> inline void Matrix<T>::transpose(Matrix<T>& trans) const {
   trans.resize(_n,_m);
   T* out = trans._X;
   for (INTM i = 0; i<_n; ++i)
      for (INTM j = 0; j<_m; ++j)
         out[j*_n+i] = _X[i*_m+j];
};

/// A <- -A
template <typename T> inline void Matrix<T>::neg() {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]=-_X[i];
};

template <typename T> inline void Matrix<T>::incrDiag() {
   for (INTM i = 0; i<MIN(_n,_m); ++i) ++_X[i*_m+i];
};

template <typename T> inline void Matrix<T>::addDiag(
      const Vector<T>& diag) {
   T* d= diag.rawX();
   for (INTM i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] += d[i];
};

template <typename T> inline void Matrix<T>::addDiag(
      const T diag) {
   for (INTM i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] += diag;
};

template <typename T> inline void Matrix<T>::addToCols(
      const Vector<T>& cent) {
   Vector<T> col;
   for (INTM i = 0; i<_n; ++i) {
      this->refCol(i,col);      
      col.add(cent[i]);
   }
};

template <typename T> inline void Matrix<T>::addVecToCols(
      const Vector<T>& vec, const T a) {
   Vector<T> col;
   for (INTM i = 0; i<_n; ++i) {
      this->refCol(i,col);      
      col.add(vec,a);
   }
};

/// perform a rank one approximation uv' using the power method
/// u0 is an initial guess for u (can be empty).
template <typename T> inline void Matrix<T>::svdRankOne(const Vector<T>& u0,
      Vector<T>& u, Vector<T>& v) const {
   int i;
   const int max_iter=MAX(_m,MAX(_n,200));
   const T eps=1e-10;
   u.resize(_m);
   v.resize(_n);
   T norm=u0.nrm2();
   Vector<T> up(u0);
   if (norm < EPSILON) up.setAleat();
   up.normalize();
   multTrans(up,v);
   for (i = 0; i<max_iter; ++i) {
      mult(v,u);
      norm=u.nrm2();
      u.scal(1.0/norm);
      multTrans(u,v);
      T theta=u.dot(up);
      if (i > 10 && (1 - fabs(theta)) < eps) break;
      up.copy(u);
   }
};

template <typename T> inline void Matrix<T>::svd2(Matrix<T>& U, Vector<T>& S, const int num, const int method) const {
   const INTM num_eig= (num == -1 || method <= 1) ? MIN(_m,_n) : MIN(MIN(_m,num),_n);
   S.resize(num_eig);
   U.resize(_m,num_eig);
   if (method==0) {
      // gesv
      T* vv = NULL;
      Matrix<T> copyX;
      copyX.copy(*this);
      gesvd<T>(reduced,no,_m,_n,copyX._X,_m,S.rawX(),U.rawX(),_m,vv,1);
   } else if (method==1) {
      // syev
      if (_m == num_eig) {
         this->XXt(U);
         syev<T>(allV,lower,_m,U.rawX(),_m,S.rawX());
      } else {
         Matrix<T> XXt(_m,_m);
         this->XXt(XXt); // in fact should do XtX, but will do that later
         Vector<T> ss(_m);
         syev<T>(allV,lower,_m,XXt.rawX(),_m,ss.rawX());
         memcpy(U.rawX(),XXt.rawX()+(_m-num_eig)*_m,_m*num_eig*sizeof(T));
         memcpy(S.rawX(),ss.rawX()+_m-num_eig,num_eig*sizeof(T));
      }
      S.thrsPos();
      S.Sqrt();
   } else if (method==2) {
      // syevr
      Matrix<T> XXt(_m,_m);
      this->XXt(XXt); // in fact should do XtX, but will do that later
      if (_m == num_eig) {
         syevr(allV,rangeAll,lower,_m,XXt.rawX(),_m,T(0),T(0),0,0,S.rawX(),U.rawX(),_m);
      } else {
         Vector<T> ss(_m);
         syevr(allV,range,lower,_m,XXt.rawX(),_m,T(0),T(0),_m-num_eig+1,_m,ss.rawX(),U.rawX(),_m);
         memcpy(S.rawX(),ss.rawX(),num_eig*sizeof(T));
      }
      S.thrsPos();
      for (int ii=0; ii<S.n(); ++ii)
         S[ii]=alt_sqrt<T>(S[ii]);
      //S.Sqrt();
   } 
   if (method==1 || method==2) {
      Vector<T> col, col2;
      Vector<T> tmpcol(_m);
      const int n=U.n();
      for (int ii=0; ii<n/2; ++ii) {
         T tmp=S[n-ii-1];
         S[n-ii-1]=S[ii];
         S[ii]=tmp;
         U.refCol(n-ii-1,col);
         U.refCol(ii,col2);
         tmpcol.copy(col);
         col.copy(col2);
         col2.copy(tmpcol);
      }
   }
}

template <typename T> inline void Matrix<T>::SymEig(Matrix<T>& U, Vector<T>& S) const {
   const int num_eig=_m;
   S.resize(_m);
   U.resize(_m,_m);
   syevr(allV,rangeAll,lower,_m,_X,_m,T(0),T(0),0,0,S.rawX(),U.rawX(),_m);
   S.thrsPos();
}

template <typename T> inline void Matrix<T>::InvsqrtMat(Matrix<T>& out, const T lambda_1) const {
   const int num_eig=_m;
   Vector<T> S;
   S.resize(_m);
   Matrix<T> U, U2;
   U.resize(_m,_m);
   syevr(allV,rangeAll,lower,_m,_X,_m,T(0),T(0),0,0,S.rawX(),U.rawX(),_m);
   S.thrsPos();
   //for (int ii=0; ii<_m; ++ii) S[ii]=sqrt(S[ii])/(S[ii]+lambda_1);
   //for (int ii=0; ii<_m; ++ii) S[ii]= S[ii] > 1e-6 ? T(1.0)/S[ii] : 0;
   for (int ii=0; ii<_m; ++ii) S[ii]= S[ii] > 1e-6 ? T(1.0)/sqrt(S[ii]+lambda_1) : 0;
   U2.copy(U);
   U2.multDiagRight(S);
   U2.mult(U,out,false,true);
}

template <typename T> inline void Matrix<T>::sqrtMat(Matrix<T>& out) const {
   const int num_eig=_m;
   Vector<T> S;
   S.resize(_m);
   Matrix<T> U, U2;
   U.resize(_m,_m);
   syevr(allV,rangeAll,lower,_m,_X,_m,T(0),T(0),0,0,S.rawX(),U.rawX(),_m);
   S.thrsPos();
   S.Sqrt();
   U2.copy(U);
   U2.multDiagRight(S);
   U2.mult(U,out,false,true);
}



template <typename T> inline void Matrix<T>::singularValues(Vector<T>& u) const {
   u.resize(MIN(_m,_n));
   if (_m > 10*_n) {
      Matrix<T> XtX;
      this->XtX(XtX);
      syev<T>(no,lower,_n,XtX.rawX(),_n,u.rawX());
      u.thrsPos();
      u.Sqrt();
   } else if (_n > 10*_m) { 
      Matrix<T> XXt;
      this->XXt(XXt);
      syev<T>(no,lower,_m,XXt.rawX(),_m,u.rawX());
      u.thrsPos();
      u.Sqrt();
   } else {
      T* vu = NULL;
      T* vv = NULL;
      Matrix<T> copyX;
      copyX.copy(*this);
      gesvd<T>(no,no,_m,_n,copyX._X,_m,u.rawX(),vu,1,vv,1);
   }
};

template <typename T> inline void Matrix<T>::svd(Matrix<T>& U, Vector<T>& S, Matrix<T>&V) const {
   const INTM num_eig=MIN(_m,_n);
   S.resize(num_eig);
   U.resize(_m,num_eig);
   V.resize(num_eig,_n);
   if (_m > 10*_n) {
      Matrix<T> Vt(_n,_n);
      this->XtX(Vt);
      syev<T>(allV,lower,_n,Vt.rawX(),_n,S.rawX());
      S.thrsPos();
      S.Sqrt();
      this->mult(Vt,U);
      Vt.transpose(V);
      Vector<T> inveigs;
      inveigs.copy(S);
      for (INTM i = 0; i<num_eig; ++i) 
         if (S[i] > 1e-10) {
            inveigs[i]=T(1.0)/S[i];
         } else {
            inveigs[i]=T(1.0);
         }
      U.multDiagRight(inveigs);
   } else if (_n > 10*_m) {
      this->XXt(U);
      syev<T>(allV,lower,_m,U.rawX(),_m,S.rawX());
      S.thrsPos();
      S.Sqrt();
      U.mult(*this,V,true,false);
      Vector<T> inveigs;
      inveigs.copy(S);
      for (INTM i = 0; i<num_eig; ++i) 
         if (S[i] > 1e-10) {
            inveigs[i]=T(1.0)/S[i];
         } else {
            inveigs[i]=T(1.0);
         }
      V.multDiagLeft(inveigs);
   } else {
      Matrix<T> copyX;
      copyX.copy(*this);
      gesvd<T>(reduced,reduced,_m,_n,copyX._X,_m,S.rawX(),U.rawX(),_m,V.rawX(),num_eig);
   }
};

/// find the eigenvector corresponding to the largest eigenvalue
/// when the current matrix is symmetric. u0 is the initial guess.
/// using two iterations of the power method
template <typename T> inline void Matrix<T>::eigLargestSymApprox(
      const Vector<T>& u0, Vector<T>& u) const {
   int i,j;
   const int max_iter=100;
   const T eps=10e-6;
   u.copy(u0);
   T norm = u.nrm2();
   T theta;
   u.scal(1.0/norm);
   Vector<T> up(u);
   Vector<T> uor(u);
   T lambda_1=T();

   for (j = 0; j<2;++j) {
      up.copy(u);
      for (i = 0; i<max_iter; ++i) {
         mult(up,u);
         norm = u.nrm2();
         u.scal(1.0/norm);
         theta=u.dot(up);
         if ((1 - fabs(theta)) < eps) break;
         up.copy(u);
      }
      lambda_1+=theta*norm;
      if (isnan(lambda_1)) {
         std::cerr << "eigLargestSymApprox failed" << std::endl;
         exit(1);
      }
      if (j == 1 && lambda_1 < eps) {
         u.copy(uor);
         break;
      }
      if (theta >= 0) break;
      u.copy(uor);
      for (i = 0; i<_m; ++i) _X[i*_m+i]-=lambda_1;
   }
};

/// find the eigenvector corresponding to the eivenvalue with the 
/// largest magnitude when the current matrix is symmetric,
/// using the power method. It 
/// returns the eigenvalue. u0 is an initial guess for the 
/// eigenvector.
template <typename T> inline T Matrix<T>::eigLargestMagnSym(
      const Vector<T>& u0, Vector<T>& u) const {
   const int max_iter=1000;
   const T eps=10e-6;
   u.copy(u0);
   T norm = u.nrm2();
   u.scal(1.0/norm);
   Vector<T> up(u);
   T lambda_1=T();

   for (int i = 0; i<max_iter; ++i) {
      mult(u,up);
      u.copy(up);
      norm=u.nrm2();
      if (norm > 0) u.scal(1.0/norm);
      if (norm == 0 || fabs(norm-lambda_1)/norm < eps) break;
      lambda_1=norm;
   }
   return norm;
};

/// returns the value of the eigenvalue with the largest magnitude
/// using the power iteration.
template <typename T> inline T Matrix<T>::eigLargestMagnSym() const {
   const int max_iter=1000;
   const T eps=10e-6;
   Vector<T> u(_m);
   u.setAleat();
   T norm = u.nrm2();
   u.scal(1.0/norm);
   Vector<T> up(u);
   T lambda_1=T();
   for (int i = 0; i<max_iter; ++i) {
      mult(u,up);
      u.copy(up);
      norm=u.nrm2();
      if (fabs(norm-lambda_1) < eps) break;
      lambda_1=norm;
      u.scal(1.0/norm);
   }
   return norm;
};

/// inverse the matrix when it is symmetric
template <typename T> inline void Matrix<T>::invSym() {
   sytri<T>(upper,_n,_X,_n);
   this->fillSymmetric();
};
template <typename T> inline void Matrix<T>::invSymPos() {
   potri<T>(upper,_n,_X,_n);
   this->fillSymmetric();
};

/// perform b = alpha*A'x + beta*b
template <typename T> inline void Matrix<T>::multTrans(const Vector<T>& x, 
      Vector<T>& b, const T a, const T c) const {
   b.resize(_n);
   //   assert(x._n == _m && b._n == _n);
   cblas_gemv<T>(CblasColMajor,CblasTrans,_m,_n,a,_X,_m,x._X,1,c,b._X,1);
};

/// perform b = A'x, when x is sparse
template <typename T> 
template <typename I> 
inline void Matrix<T>::multTrans(const SpVector<T,I>& x, 
      Vector<T>& b, const T alpha, const T beta) const {
   b.resize(_n);
   Vector<T> col;
   if (beta) {
      for (INTM i = 0; i<_n; ++i) {
         refCol(i,col);
         b._X[i] = alpha*col.dot(x);
      }
   } else {

      for (INTM i = 0; i<_n; ++i) {
         refCol(i,col);
         b._X[i] = beta*b._X[i]+alpha*col.dot(x);
      }
   }
};

template <typename T> inline void Matrix<T>::multTrans(
      const Vector<T>& x, Vector<T>& b, const Vector<bool>& active) const {
   b.setZeros();
   Vector<T> col;
   bool* pr_active=active.rawX();
   for (INTM i = 0; i<_n; ++i) {
      if (pr_active[i]) {
         this->refCol(i,col);
         b._X[i]=col.dot(x);
      }
   }
};

/// perform b = alpha*A*x+beta*b
template <typename T> inline void Matrix<T>::mult(const Vector<T>& x, 
      Vector<T>& b, const T a, const T c) const {
   //  assert(x._n == _n && b._n == _m);
   b.resize(_m);
   cblas_gemv<T>(CblasColMajor,CblasNoTrans,_m,_n,a,_X,_m,x._X,1,c,b._X,1);
};


/// perform b = alpha*A*x+beta*b
template <typename T> inline void Matrix<T>::mult_loop(const Vector<T>& x, 
      Vector<T>& b) const {
   b.resize(_m);
   for (int ii=0; ii<_m; ++ii) {
      b[ii]=cblas_dot<T>(_n,x._X,1,_X+ii,_m);
   }
};

/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename T> 
template <typename I> 
inline void Matrix<T>::mult(const SpVector<T,I>& x, 
      Vector<T>& b, const T a, const T a2) const {
   if (!a2) {
      b.setZeros();
   } else if (a2 != 1.0) {
      b.scal(a2);
   }
   if (a == 1.0) {
      for (INTM i = 0; i<x._L; ++i) {
         cblas_axpy<T>(_m,x._v[i],_X+x._r[i]*_m,1,b._X,1);
      }
   } else {
      for (INTM i = 0; i<x._L; ++i) {
         cblas_axpy<T>(_m,a*x._v[i],_X+x._r[i]*_m,1,b._X,1);
      }
   }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename T> inline void Matrix<T>::mult(const Matrix<T>& B, 
      Matrix<T>& C, const bool transA, const bool transB,
      const T a, const T b) const {
   CBLAS_TRANSPOSE trA,trB;
   INTM m,k,n;
   if (transA) {
      trA = CblasTrans;
      m = _n;
      k = _m;
   } else {
      trA= CblasNoTrans;
      m = _m;
      k = _n;
   }
   if (transB) {
      trB = CblasTrans;
      n = B._m; 
      //assert(B._n == k);
   } else {
      trB = CblasNoTrans;
      n = B._n; 
      //assert(B._m == k);
   }
   C.resize(m,n);
   cblas_gemm<T>(CblasColMajor,trA,trB,m,n,k,a,_X,_m,B._X,B._m,
         b,C._X,C._m);
};

/// perform C = a*B*A + b*C, possibly transposing A or B.
template <typename T>
inline void Matrix<T>::multSwitch(const Matrix<T>& B, Matrix<T>& C, 
      const bool transA, const bool transB,
      const T a, const T b) const {
   B.mult(*this,C,transB,transA,a,b);
};

/// perform C = A*B, when B is sparse
template <typename T>
template <typename I>
inline void Matrix<T>::mult(const SpMatrix<T,I>& B, Matrix<T>& C,
      const bool transA, const bool transB,
      const T a, const T b) const {
   if (transA) {
      if (transB) {
         C.resize(_n,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<T> rowC(B.m());
         Vector<T> colA;
         for (INTM i = 0; i<_n; ++i) {
            this->refCol(i,colA);
            B.mult(colA,rowC,a);
            C.addRow(i,rowC,a);
         }
      } else {
         C.resize(_n,B.n());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<T> colC;
         SpVector<T,I> colB;
         for (INTM i = 0; i<B.n(); ++i) {
            C.refCol(i,colC);
            B.refCol(i,colB);
            this->multTrans(colB,colC,a,T(1.0));
         }
      }
   } else {
      if (transB) {
         C.resize(_m,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<T> colA;
         SpVector<T,I> colB;
         for (INTM i = 0; i<_n; ++i) {
            this->refCol(i,colA);
            B.refCol(i,colB);
            C.rank1Update(colA,colB,a);
         }
      } else {
         C.resize(_m,B.n());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<T> colC;
         SpVector<T,I> colB;
         for (INTM i = 0; i<B.n(); ++i) {
            C.refCol(i,colC);
            B.refCol(i,colB);
            this->mult(colB,colC,a,T(1.0));
         }
      }
   };
}


/// mult by a diagonal matrix on the left
template <typename T>
   inline void Matrix<T>::multDiagLeft(const Vector<T>& diag) {
      if (diag.n() != _m)
         return;
      T* d = diag.rawX();
      for (INTM i = 0; i< _n; ++i) {
         for (INTM j = 0; j<_m; ++j) {
            _X[i*_m+j] *= d[j];
         }
      }
   };

/// mult by a diagonal matrix on the right
template <typename T> inline void Matrix<T>::multDiagRight(
      const Vector<T>& diag) {
   if (diag.n() != _n)
      return;
   T* d = diag.rawX();
   for (INTM i = 0; i< _n; ++i) {
      for (INTM j = 0; j<_m; ++j) {
         _X[i*_m+j] *= d[i];
      }
   }
};
/// mult by a diagonal matrix on the right
template <typename T> inline void Matrix<T>::AddMultDiagRight(
      const Vector<T>& diag, Matrix<T>& mat) {
   if (diag.n() != _n)
      return;
   mat.resize(_m,_n);
   //mat.setZeros();
   T* d = diag.rawX();
   for (INTM i = 0; i< _n; ++i) {
      cblas_axpy<T>(_m,d[i],_X+i*_m,1,mat._X+i*_m,1);
   }
};



/// C = A .* B, elementwise multiplication
template <typename T> inline void Matrix<T>::mult_elementWise(
      const Matrix<T>& B, Matrix<T>& C) const {
   assert(_n == B._n && _m == B._m);
   C.resize(_m,_n);
   vMul<T>(_n*_m,_X,B._X,C._X);
};

/// C = A .* B, elementwise multiplication
template <typename T> inline void Matrix<T>::div_elementWise(
      const Matrix<T>& B, Matrix<T>& C) const {
   assert(_n == B._n && _m == B._m);
   C.resize(_m,_n);
   vDiv<T>(_n*_m,_X,B._X,C._X);
};


/// XtX = A'*A
template <typename T> inline void Matrix<T>::XtX(Matrix<T>& xtx) const {
   xtx.resize(_n,_n);
   cblas_syrk<T>(CblasColMajor,CblasUpper,CblasTrans,_n,_m,T(1.0),
         _X,_m,T(),xtx._X,_n);
   xtx.fillSymmetric();
};

/// XXt = A*At
template <typename T> inline void Matrix<T>::XXt(Matrix<T>& xxt) const {
   xxt.resize(_m,_m);
   cblas_syrk<T>(CblasColMajor,CblasUpper,CblasNoTrans,_m,_n,T(1.0),
         _X,_m,T(),xxt._X,_m);
   xxt.fillSymmetric();
};

/// XXt = A*A' where A is an upper triangular matrix
template <typename T> inline void Matrix<T>::upperTriXXt(Matrix<T>& XXt, const INTM L) const {
   XXt.resize(L,L);
   for (INTM i = 0; i<L; ++i) {
      cblas_syr<T>(CblasColMajor,CblasUpper,i+1,T(1.0),_X+i*_m,1,XXt._X,L);
   }
   XXt.fillSymmetric();
}


/// extract the diagonal
template <typename T> inline void Matrix<T>::diag(Vector<T>& dv) const {
   INTM size_diag=MIN(_n,_m);
   dv.resize(size_diag);
   T* const d = dv.rawX();
   for (INTM i = 0; i<size_diag; ++i)
      d[i]=_X[i*_m+i];
};

/// set the diagonal
template <typename T> inline void Matrix<T>::setDiag(const Vector<T>& dv) {
   INTM size_diag=MIN(_n,_m);
   T* const d = dv.rawX();
   for (INTM i = 0; i<size_diag; ++i)
      _X[i*_m+i]=d[i];
};

/// set the diagonal
template <typename T> inline void Matrix<T>::setDiag(const T val) {
   INTM size_diag=MIN(_n,_m);
   for (INTM i = 0; i<size_diag; ++i)
      _X[i*_m+i]=val;
};


/// each element of the matrix is replaced by its exponential
template <typename T> inline void Matrix<T>::exp() {
   vExp<T>(_n*_m,_X,_X);
};

/// each element of the matrix is replaced by its exponential
template <typename T> inline void Matrix<T>::pow(const T a) {
   vPowx<T>(_n*_m,_X,a,_X);
};

template <typename T> inline void Matrix<T>::sqr() {
   vSqr<T>(_n*_m,_X,_X);
};

template <typename T> inline void Matrix<T>::Sqrt() {
   vSqrt<T>(_n*_m,_X,_X);
};

template <typename T> inline void Matrix<T>::Invsqrt() {
   vInvSqrt<T>(_n*_m,_X,_X);
};
/// return vec1'*A*vec2, where vec2 is sparse
template <typename T> 
template <typename I> 
inline T Matrix<T>::quad(const SpVector<T,I>& vec) const {
   T sum = T();
   INTM L = vec._L;
   I* r = vec._r;
   T* v = vec._v;
   for (INTM i = 0; i<L; ++i)
      for (INTM j = 0; j<L; ++j)
         sum += _X[r[i]*_m+r[j]]*v[i]*v[j];
   return sum;
};

template <typename T> 
template <typename I> 
inline void Matrix<T>::quad_mult(const Vector<T>& vec1,
      const SpVector<T,I>& vec2, Vector<T>& y, const T a, const T b) const {
   const INTM size_y= y.n();
   const INTM nn = _n/size_y;
   //y.resize(size_y);
   //y.setZeros();
   Matrix<T> tmp;
   for (INTM i = 0; i<size_y; ++i) {
      tmp.setData(_X+(i*nn)*_m,_m,nn);
      y[i]=b*y[i]+a*tmp.quad(vec1,vec2);
   }
}

/// return vec'*A*vec when vec is sparse
template <typename T> 
template <typename I> 
inline T Matrix<T>::quad(
      const Vector<T>& vec1, const SpVector<T,I>& vec) const {
   T sum = T();
   INTM L = vec._L;
   I* r = vec._r;
   T* v = vec._v;
   Vector<T> col;
   for (INTM i = 0; i<L; ++i) {
      this->refCol(r[i],col);
      sum += v[i]*col.dot(vec1);
   }
   return sum;
};

/// add alpha*mat to the current matrix
template <typename T> inline void Matrix<T>::add(const Matrix<T>& mat, const T alpha) {
   assert(mat._m == _m && mat._n == _n);
   cblas_axpy<T>(_n*_m,alpha,mat._X,1,_X,1);
};

/// add alpha*mat to the current matrix
template <typename T> inline void Matrix<T>::add_scal(const Matrix<T>& mat, const T alpha, const T beta) {
   assert(mat._m == _m && mat._n == _n);
   cblas_axpby<T>(_n*_m,alpha,mat._X,1,beta,_X,1);
};


/// add alpha*mat to the current matrix
template <typename T> inline T Matrix<T>::dot(const Matrix<T>& mat) const {
   assert(mat._m == _m && mat._n == _n);
   return cblas_dot<T>(_n*_m,mat._X,1,_X,1);
};


/// add alpha to the current matrix
template <typename T> inline void Matrix<T>::add(const T alpha) {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]+=alpha;
};

/// substract the matrix mat to the current matrix
template <typename T> inline void Matrix<T>::sub(const Matrix<T>& mat) {
   vSub<T>(_n*_m,_X,mat._X,_X);
};

/// compute the sum of the magnitude of the matrix values
template <typename T> inline T Matrix<T>::asum() const {
   return cblas_asum<T>(_n*_m,_X,1);
};

template <typename T> inline T Matrix<T>::sum() const {
   T sum=0;
   for (INTM i =0; i<_n*_m; ++i) sum+=_X[i];
   return sum;
};



/// returns the trace of the matrix
template <typename T> inline T Matrix<T>::trace() const {
   T sum=T();
   INTM m = MIN(_n,_m);
   for (INTM i = 0; i<m; ++i) 
      sum += _X[i*_m+i];
   return sum;
};

/// return ||A||_F
template <typename T> inline T Matrix<T>::normF() const {
   return cblas_nrm2<T>(_n*_m,_X,1);
};

template <typename T> inline T Matrix<T>::mean() const {
   Vector<T> vec;
   this->toVect(vec);
   return vec.mean();
};

template <typename T> inline T Matrix<T>::abs_mean() const {
   Vector<T> vec;
   this->toVect(vec);
   return vec.abs_mean();
};


/// return ||A||_F^2
template <typename T> inline T Matrix<T>::normFsq() const {
   return cblas_dot<T>(_n*_m,_X,1,_X,1);
};

/// return ||At||_{inf,2}
template <typename T> inline T Matrix<T>::norm_inf_2_col() const {
   Vector<T> col;
   T max = -1.0;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      T norm_col = col.nrm2();
      if (norm_col > max) 
         max = norm_col;
   }
   return max;
};

/// return ||At||_{1,2}
template <typename T> inline T Matrix<T>::norm_1_2_col() const {
   Vector<T> col;
   T sum = 0.0;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      sum += col.nrm2();
   }
   return sum;
};

/// returns the l2 norms of the columns
template <typename T> inline void Matrix<T>::norm_2_rows(
      Vector<T>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] += _X[i*_m+j]*_X[i*_m+j];
   for (INTM j = 0; j<_m; ++j) 
      norms[j]=sqrt(norms[j]);
};

/// returns the l2 norms of the columns
template <typename T> inline void Matrix<T>::norm_2sq_rows(
      Vector<T>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] += _X[i*_m+j]*_X[i*_m+j];
};


/// returns the l2 norms of the columns
template <typename T> inline void Matrix<T>::norm_2_cols(
      Vector<T>& norms) const {
   norms.resize(_n);
   Vector<T> col;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      norms[i] = col.nrm2();
   }
};


/// returns the linf norms of the columns
template <typename T> inline void Matrix<T>::norm_inf_cols(Vector<T>& norms) const {
   norms.resize(_n);
   Vector<T> col;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      norms[i] = col.fmaxval();
   }
};

/// returns the linf norms of the columns
template <typename T> inline void Matrix<T>::norm_inf_rows(Vector<T>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] = MAX(abs<T>(_X[i*_m+j]),norms[j]);
};

template <typename T> inline void Matrix<T>::get_sum_cols(Vector<T>& sum) const {
   sum.resize(_n);
   for (INTM i = 0; i<_n; ++i) {
      sum[i]=0;
      for (INTM j = 0; j<_m; ++j) 
         sum[i] += (_X[i*_m+j]);
   }
};

template <typename T> inline void Matrix<T>::dot_col(const Matrix<T>& mat, 
      Vector<T>& dots) const {
   dots.resize(_n);
   for (INTM i = 0; i<_n; ++i) 
      dots[i] = cblas_dot<T>(_m,_X+i*_m,1,mat._X+i*_m,1);
}

/// returns the linf norms of the columns
template <typename T> inline void Matrix<T>::norm_l1_rows(Vector<T>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] += abs<T>(_X[i*_m+j]);
};



/// returns the l2 norms of the columns
template <typename T> inline void Matrix<T>::norm_2sq_cols(
      Vector<T>& norms) const {
   norms.resize(_n);
   Vector<T> col;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      norms[i] = col.nrm2sq();
   }
};

template <typename T> 
inline void Matrix<T>::sum_cols(Vector<T>& sum) const {
   sum.resize(_m);
   sum.setZeros();
   Vector<T> tmp;
   for (INTM i = 0; i<_n; ++i) {
      this->refCol(i,tmp);
      sum.add(tmp);
   }
};

/// Compute the mean of the columns
template <typename T> inline void Matrix<T>::meanCol(Vector<T>& mean) const {
   Vector<T> ones(_n);
   ones.set(T(1.0/_n));
   this->mult(ones,mean,1.0,0.0);
};

/// Compute the mean of the rows
template <typename T> inline void Matrix<T>::meanRow(Vector<T>& mean) const {
   Vector<T> ones(_m);
   ones.set(T(1.0/_m));
   this->multTrans(ones,mean,1.0,0.0);
};


/// fill the matrix with the row given
template <typename T> inline void Matrix<T>::fillRow(const Vector<T>& row) {
   for (INTM i = 0; i<_n; ++i) {
      T val = row[i];
      for (INTM j = 0; j<_m; ++j) {
         _X[i*_m+j]=val;
      }
   }
};

/// fill the matrix with the row given
template <typename T> inline void Matrix<T>::extractRow(const INTM j,
      Vector<T>& row) const {
   row.resize(_n);
   for (INTM i = 0; i<_n; ++i) {
      row[i]=_X[i*_m+j];
   }
};

/// fill the matrix with the row given
template <typename T> inline void Matrix<T>::setRow(const INTM j,
      const Vector<T>& row) {
   for (INTM i = 0; i<_n; ++i) {
      _X[i*_m+j]=row[i];
   }
};

/// fill the matrix with the row given
template <typename T> inline void Matrix<T>::addRow(const INTM j,
      const Vector<T>& row, const T a) {
   if (a==1.0) {
      for (INTM i = 0; i<_n; ++i) {
         _X[i*_m+j]+=row[i];
      }
   } else {
      for (INTM i = 0; i<_n; ++i) {
         _X[i*_m+j]+=a*row[i];
      }
   }
};


/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::softThrshold(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.softThrshold(nu);
};

/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::fastSoftThrshold(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.fastSoftThrshold(nu);
};
/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::fastSoftThrshold(Matrix<T>& output, const T nu) const {
   output.resize(_m,_n,false);
   Vector<T> vec, vec2;
   toVect(vec);
   output.toVect(vec2);
   vec.fastSoftThrshold(vec2,nu);
};




/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::hardThrshold(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.hardThrshold(nu);
};


/// perform thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::thrsmax(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.thrsmax(nu);
};

/// perform thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::thrsmin(const T nu) {
   Vector<T> vec;
   toVect(vec);
   vec.thrsmin(nu);
};


/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::inv_elem() {
   Vector<T> vec;
   toVect(vec);
   vec.inv();
};

/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::blockThrshold(const T nu,
      const INTM sizeGroup) {
   for (INTM i = 0; i<_n; ++i) {
      INTM j;
      for (j = 0; j<_m-sizeGroup+1; j+=sizeGroup) {
         T nrm=0;
         for (INTM k = 0; k<sizeGroup; ++k)
            nrm += _X[i*_m +j+k]*_X[i*_m +j+k];
         nrm=sqrt(nrm);
         if (nrm < nu) {
            for (INTM k = 0; k<sizeGroup; ++k)
               _X[i*_m +j+k]=0;
         } else {
            T scal = (nrm-nu)/nrm;
            for (INTM k = 0; k<sizeGroup; ++k)
               _X[i*_m +j+k]*=scal;
         }
      }
      j -= sizeGroup;
      for ( ; j<_m; ++j)
         _X[j]=softThrs<T>(_X[j],nu);
   }
}

template <typename T> inline void Matrix<T>::sparseProject(Matrix<T>& Y, 
      const T thrs,   const int mode, const T lambda_1,
      const T lambda_2, const T lambda_3, const bool pos,
      const int numThreads) {

   int NUM_THREADS=init_omp(numThreads);
   Vector<T>* XXT= new Vector<T>[NUM_THREADS];
   for (int i = 0; i<NUM_THREADS; ++i) {
      XXT[i].resize(_m);
   }

   int i;
#pragma omp parallel for private(i) 
   for (i = 0; i< _n; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      Vector<T> Xi;
      this->refCol(i,Xi);
      Vector<T> Yi;
      Y.refCol(i,Yi);
      Vector<T>& XX = XXT[numT];
      XX.copy(Xi);
      XX.sparseProject(Yi,thrs,mode,lambda_1,lambda_2,lambda_3,pos);
   }
   delete[](XXT);
};


/// perform soft-thresholding of the matrix, with the threshold nu
template <typename T> inline void Matrix<T>::thrsPos() {
   Vector<T> vec;
   toVect(vec);
   vec.thrsPos();
};


/// perform A <- A + alpha*vec1*vec2'
template <typename T> inline void Matrix<T>::rank1Update(
      const Vector<T>& vec1, const Vector<T>& vec2, const T alpha) {
   cblas_ger<T>(CblasColMajor,_m,_n,alpha,vec1._X,1,vec2._X,1,_X,_m);
};

/// perform A <- A + alpha*vec1*vec2', when vec1 is sparse
template <typename T> 
template <typename I> 
inline void Matrix<T>::rank1Update(
      const SpVector<T,I>& vec1, const Vector<T>& vec2, const T alpha) {
   I* r = vec1._r;
   T* v = vec1._v;
   T* X2 = vec2._X;
   assert(vec2._n == _n);
   if (alpha == 1.0) {
      for (INTM i = 0; i<_n; ++i) {
         for (INTM j = 0; j<vec1._L; ++j) {
            _X[i*_m+r[j]] += v[j]*X2[i];
         }
      }
   } else {
      for (INTM i = 0; i<_n; ++i) {
         for (INTM j = 0; j<vec1._L; ++j) {
            _X[i*_m+r[j]] += alpha*v[j]*X2[i];
         }
      }
   }
};

template <typename T>
template <typename I>
inline void Matrix<T>::rank1Update_mult(const Vector<T>& vec1, 
      const Vector<T>& vec1b,
      const SpVector<T,I>& vec2,
      const T alpha) {
   const INTM nn = vec1b.n();
   const INTM size_A = _n/nn;
   Matrix<T> tmp;
   for (INTM i = 0; i<nn; ++i) {
      tmp.setData(_X+i*size_A*_m,_m,size_A);
      tmp.rank1Update(vec1,vec2,alpha*vec1b[i]);
   }
};

/// perform A <- A + alpha*vec1*vec2', when vec1 is sparse
template <typename T>
template <typename I>
inline void Matrix<T>::rank1Update(
      const SpVector<T,I>& vec1, const SpVector<T,I>& vec2, const T alpha) {
   I* r = vec1._r;
   T* v = vec1._v;
   T* v2 = vec2._v;
   I* r2 = vec2._r;
   if (alpha == 1.0) {
      for (INTM i = 0; i<vec2._L; ++i) {
         for (INTM j = 0; j<vec1._L; ++j) {
            _X[r2[i]*_m+r[j]] += v[j]*v2[i];
         }
      }
   } else {
      for (INTM i = 0; i<vec2._L; ++i) {
         for (INTM j = 0; j<vec1._L; ++j) {
            _X[r[i]*_m+r[j]] += alpha*v[j]*v2[i];
         }
      }
   }
};


/// perform A <- A + alpha*vec1*vec2', when vec2 is sparse
template <typename T> 
template <typename I> 
inline void Matrix<T>::rank1Update(
      const Vector<T>& vec1, const SpVector<T,I>& vec2, const T alpha) {
   I* r = vec2._r;
   T* v = vec2._v;
   Vector<T> Xi;
   for (INTM i = 0; i<vec2._L; ++i) {
      this->refCol(r[i],Xi);
      Xi.add(vec1,v[i]*alpha);
   }
};

/// perform A <- A + alpha*vec1*vec1', when vec1 is sparse
template <typename T> 
template <typename I> 
inline void Matrix<T>::rank1Update(
      const SpVector<T,I>& vec1, const T alpha) {
   I* r = vec1._r;
   T* v = vec1._v;
   if (alpha == 1.0) {
      for (INTM i = 0; i<vec1._L; ++i) {
         for (INTM j = 0; j<vec1._L; ++j) {
            _X[r[i]*_m+r[j]] += v[j]*v[i];
         }
      }
   } else {
      for (INTM i = 0; i<vec1._L; ++i) {
         for (INTM j = 0; j<vec1._L; ++j) {
            _X[_m*r[i]+r[j]] += alpha*v[j]*v[i];
         }
      }
   }
};

/// compute x, such that b = Ax, 
template <typename T> inline void Matrix<T>::conjugateGradient(
      const Vector<T>& b, Vector<T>& x, const T tol, const int itermax) const {
   Vector<T> R,P,AP;
   R.copy(b);
   this->mult(x,R,T(-1.0),T(1.0));
   P.copy(R);
   int k = 0;
   T normR = R.nrm2sq();
   T alpha;
   while (normR > tol && k < itermax) {
      this->mult(P,AP);
      alpha = normR/P.dot(AP);
      x.add(P,alpha);
      R.add(AP,-alpha);
      T tmp = R.nrm2sq();
      P.scal(tmp/normR);
      normR = tmp;
      P.add(R,T(1.0));
      ++k;
   };
};

template <typename T> inline void Matrix<T>::drop(char* fileName) const {
   std::ofstream f;
   f.precision(12);
   f.flags(std::ios_base::scientific);
   f.open(fileName, ofstream::trunc);
   std::cout << "Matrix written in " << fileName << std::endl;
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = 0; j<_m; ++j) 
         f << _X[i*_m+j] << " ";
      f << std::endl;
   }
   f.close();
};

/// compute a Nadaraya Watson estimator
template <typename T> inline void Matrix<T>::NadarayaWatson(
      const Vector<INTM>& ind, const T sigma) {
   if (ind.n() != _n) return;

   init_omp(MAX_THREADS);

   const INTM Ngroups=ind.maxval();
   INTM i;
#pragma omp parallel for private(i)
   for (i = 1; i<=Ngroups; ++i) {
      Vector<INTM> indicesGroup(_n);
      INTM count = 0;
      for (INTM j = 0; j<_n; ++j)
         if (ind[j] == i) indicesGroup[count++]=j;
      Matrix<T> Xm(_m,count);
      Vector<T> col, col2;
      for (INTM j= 0; j<count; ++j) {
         this->refCol(indicesGroup[j],col);
         Xm.refCol(j,col2);
         col2.copy(col);
      }
      Vector<T> norms;
      Xm.norm_2sq_cols(norms);
      Matrix<T> weights;
      Xm.XtX(weights);
      weights.scal(T(-2.0));
      Vector<T> ones(Xm.n());
      ones.set(T(1.0));
      weights.rank1Update(ones,norms);
      weights.rank1Update(norms,ones);
      weights.scal(-sigma);
      weights.exp();
      Vector<T> den;
      weights.mult(ones,den);
      den.inv();
      weights.multDiagRight(den);
      Matrix<T> num;
      Xm.mult(weights,num);
      for (INTM j= 0; j<count; ++j) {
         this->refCol(indicesGroup[j],col);
         num.refCol(j,col2);
         col.copy(col2);
      }
   }
};

/// make a sparse copy of the current matrix
template <typename T> inline void Matrix<T>::toSparse(SpMatrix<T>& out) const {
   out.clear();
   INTM count=0;
   INTM* pB;
#pragma omp critical
   {
      pB=new INTM[_n+1];
   }
   INTM* pE=pB+1;
   for (INTM i = 0; i<_n*_m; ++i) 
      if (_X[i] != 0) ++count;
   INTM* r;
   T* v;
#pragma omp critical
   {
      r=new INTM[count];
      v=new T[count];
   }
   count=0;
   for (INTM i = 0; i<_n; ++i) {
      pB[i]=count;
      for (INTM j = 0; j<_m; ++j) {
         if (_X[i*_m+j] != 0) {
            v[count]=_X[i*_m+j];
            r[count++]=j;
         }
      }
      pE[i]=count;
   }
   out._v=v;
   out._r=r;
   out._pB=pB;
   out._pE=pE;
   out._m=_m;
   out._n=_n;
   out._nzmax=count;
   out._externAlloc=false;
};

/// make a sparse copy of the current matrix
template <typename T> inline void Matrix<T>::toSparseTrans(
      SpMatrix<T>& out) {
   out.clear();
   INTM count=0;
   INTM* pB;
#pragma omp critical
   {
      pB=new INTM[_m+1];
   }
   INTM* pE=pB+1;
   for (INTM i = 0; i<_n*_m; ++i) 
      if (_X[i] != 0) ++count;
   INTM* r;
   T* v;
#pragma omp critical
   {
      r=new INTM[count];
      v=new T[count];
   }
   count=0;
   for (INTM i = 0; i<_m; ++i) {
      pB[i]=count;
      for (INTM j = 0; j<_n; ++j) {
         if (_X[i+j*_m] != 0) {
            v[count]=_X[j*_m+i];
            r[count++]=j;
         }
      }
      pE[i]=count;
   }
   out._v=v;
   out._r=r;
   out._pB=pB;
   out._pE=pE;
   out._m=_n;
   out._n=_m;
   out._nzmax=count;
   out._externAlloc=false;
};

/// make a reference of the matrix to a vector vec 
template <typename T> inline void Matrix<T>::toVect(
      Vector<T>& vec) const {
   vec.clear();
   vec._externAlloc=true;
   vec._n=_n*_m;
   vec._X=_X;
};


/* ************************************
 * Implementation of the class OptimInfo 
 * ************************************/

/// Constructor with existing data X of an m x n OptimInfo
template <typename T> OptimInfo<T>::OptimInfo(T* X, INTM nclass, INTM m, INTM n) :
   _externAlloc(true), _X(X), _nclass(nclass), _m(m), _n(n) {  };


/// Constructor for a new m x n OptimInfo
template <typename T> OptimInfo<T>::OptimInfo(INTM nclass, INTM m, INTM n) :
   _externAlloc(false), _nclass(nclass), _m(m), _n(n)  {
#pragma omp critical
      {
         _X= new T[_nclass*_n*_m];
      }
   };

/// Empty constructor
template <typename T> OptimInfo<T>::OptimInfo() :
   _externAlloc(false), _X(NULL), _nclass(0), _m(0), _n(0) { };

/// Destructor
template <typename T> OptimInfo<T>::~OptimInfo() {
   clear();
};

/// Return a modifiable reference to X(i,j,k)
template <typename T> inline T& OptimInfo<T>::operator()(const INTM i, const INTM j, const INTM k) {
   return _X[i*_m*_n + k*_m+j];
};

/// Return the value X(i,j,k)
template <typename T> inline T OptimInfo<T>::operator()(const INTM i, const INTM j, const INTM k) const {
   return _X[i*_m*_n + k*_m+j];
};

/// Print the OptimInfo to std::cout
template <typename T> inline void OptimInfo<T>::print(const string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _m << " x " << _n << std::endl;
   for (INTM i = 0; i<_m; ++i) {
      for (INTM j = 0; j<_n; ++j) {
          for (INTM k = 0; k<_nclass; ++k) {
         printf("%10.5g ",static_cast<double>(_X[i*_m*_n + k*_m+j]));
         }
      printf("\n ");
      }
      printf("\n ");
   }
   printf("\n ");
};

/// Print the OptimInfo to std::cout
template <typename T> inline void OptimInfo<T>::dump(const string& name) const {
   ofstream f; 
   const char * cname = name.c_str();
   f.open(cname);
   f.precision(20);
   std::cerr << name << std::endl;
   f << _m << " x " << _n << std::endl;
   for (INTM i = 0; i<_m; ++i) {
      for (INTM j = 0; j<_n; ++j) {
          for (INTM k = 0; k<_nclass; ++k) {
         f << static_cast<double>(_X[i*_m*_n + k*_m+j]) << " ";
         }
      f << std::endl;
      }
      f << std::endl;
   }
   f << std::endl;
   f.close();
};

/// Set all the values to zero
template <typename T> inline void OptimInfo<T>::setZeros() {
   memset(_X,0,_nclass*_n*_m*sizeof(T));
};

/// Resize the optimInfo
template <typename T> inline void OptimInfo<T>::resize(INTM nclass, INTM m, INTM n, const bool set_zeros) {
   if (_nclass==nclass && _n==n && _m==m) return;
   clear();
   _nclass=nclass;
   _n=n;
   _m=m;
   _externAlloc=false;
#pragma omp critical
   {
      _X=new T[_nclass*_n*_m];
   }
   if (set_zeros)
      setZeros();
};

/// Clear the optimInfo
template <typename T> inline void OptimInfo<T>::clear() {
   if (!_externAlloc) delete[](_X);
   _nclass=0;
   _n=0;
   _m=0;
   _X=NULL;
   _externAlloc=true;
};

/// make a copy of the optimInfo optim in the current optim
template <typename T> inline void OptimInfo<T>::copy(const OptimInfo<T>& optim) {
   if (_X != optim._X) {
      resize(optim._nclass, optim._m,optim._n);
      memcpy(_X,optim._X,_nclass*_m*_n*sizeof(T));
   }
};

/// Change the data in the optimInfo
template <typename T> inline void OptimInfo<T>::setData(T* X, INTM nclass, INTM m, INTM n) {
   clear();
   _X=X;
   _nclass=nclass;
   _m=m;
   _n=n;
   _externAlloc=true;
};

/// add alpha*optim to the current optim info at a given index
template <typename T> inline void OptimInfo<T>::add(const OptimInfo<T>& optim, const int index, const T alpha) {
   assert(optim._m == _m && optim._n == _n);
   for (INTT i = 0; i<_m * _n; ++i){
       //FIXME maybe slow
      _X[index * _m * _n + i] += alpha*optim[i];
   }
};

#endif
