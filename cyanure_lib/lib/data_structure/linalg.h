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

#include "../misc.h"
#include "../BLAS/cblas_alt_template.h"
#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif
#include "../utils/macro.h"
#include "../logger.h"
#include "../BLAS/configure_blas.h"

#undef max
#undef min

/// Dense OptimInfo class
template<typename floating_type> class OptimInfo;
/// Sparse OptimInfo class
template<typename floating_type, typename I = INTM> class SpOptimInfo;
/// Dense Matrix class
template<typename floating_type> class Matrix;
/// Sparse Matrix class
template<typename floating_type, typename I = INTM> class SpMatrix;
/// Dense Vector class
template<typename floating_type> class Vector;
/// Sparse Vector class
template<typename floating_type, typename I = INTM> class SpVector;

template <typename floating_type> 
static inline bool isZero(const floating_type lambda_1) {
   return static_cast<double>(abs<floating_type>(lambda_1)) < 1e-99;
}

template <typename floating_type> 
static inline bool isEqual(const floating_type lambda_1, const floating_type lambda_2) {
   return static_cast<double>(abs<floating_type>(lambda_1-lambda_2)) < 1e-99;
}


template <typename floating_type>
static inline floating_type softThrs(const floating_type x, const floating_type lambda_1) {
   if (x > lambda_1) {
      return x-lambda_1;
   } else if (x < -lambda_1) {
      return x+lambda_1;
   } else {
      return 0;
   }
};

template <typename floating_type>
static inline floating_type fastSoftThrs(const floating_type x, const floating_type lambda_1) {
    return x + floating_type(0.5)*(abs<floating_type>(x-lambda_1) - abs<floating_type>(x+lambda_1));
};


template <typename floating_type>
static inline floating_type hardThrs(const floating_type x, const floating_type lambda_1) {
   return (x > lambda_1 || x < -lambda_1) ? x : 0;
};

template <typename floating_type>
static inline floating_type xlogx(const floating_type x) {
   if (x < -1e-20) {
      return INFINITY;
   } else if (x < 1e-20) {
      return 0;
   } else {
      return x*alt_log<floating_type>(x);
   }
}

template <typename floating_type>
static inline floating_type logexp(const floating_type x) {
   if (x < -30) {
      return 0;
   } else if (x < 30) {
      return alt_log<floating_type>( floating_type(1.0) + exp_alt<floating_type>( x ) );
   } else {
      return x;
   }
}

template <typename floating_type>
static inline floating_type logexp2(const floating_type x) {
   return (x > 0) ? x + log_alt<floating_type>(floating_type(1.0)+ exp_alt<floating_type>(-x)) :
      log( floating_type(1.0) + exp_alt<floating_type>( x ) );
}

template <typename floating_type>
static floating_type solve_binomial(const floating_type a, const floating_type b, const floating_type c) {
   const floating_type delta = b*b-4*a*c;
   return (-b + alt_sqrt<floating_type>(delta))/(2*a); // returns single largest solution, assiming delta > 0;
};

template <typename floating_type>
static floating_type solve_binomial2(const floating_type a, const floating_type b, const floating_type c) {
   const floating_type delta = b*b-4*a*c;
   return (-b - alt_sqrt<floating_type>(delta))/(2*a); // returns single largest solution, assiming delta > 0;
};

/// Class OptimInfo
template<typename floating_type> class OptimInfo {
    public:
    typedef floating_type value_type;
    typedef Vector<floating_type> col_type;
    typedef INTM index_type;
    typedef Vector<floating_type> element;

    /// Constructor with existing data X of an nclass x m x n matrix
    OptimInfo(floating_type* X, INTM nclass, INTM m, INTM n);
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
    inline floating_type& operator()(const INTM i, const INTM j, const INTM k);
    /// Return the value X(i,j,k)
    inline floating_type operator()(const INTM i, const INTM j, const INTM k) const;
    /// Return a modifiable reference to X(i) (1D indexing)
    inline floating_type& operator[](const INTM index) { return _X[index]; };
    /// Return the value X(i) (1D indexing)
    inline floating_type operator[](const INTM index) const { return _X[index]; };
    /// Copy the column i into x
    inline void copyCol(const INTM i, Vector<floating_type>& x) const;
    /// make a copy of the OptimInfo optim in the current OptimInfo
   inline void copy(const OptimInfo<floating_type>& optim);
   /// Set all the values to zero
   inline void setZeros();
   /// Resize the optiminfo
   inline void resize(INTM nclass,INTM m, INTM n, const bool set_zeros = true);
   /// add alpha*optimInfo to the current matrix
   inline void add(const OptimInfo<floating_type>& mat, const int index, const floating_type alpha = 1.0);

   /// Change the data in the optimInfo
   inline void setData(floating_type* X, INTM nclass, INTM m, INTM n);

    /// Debugging function
   /// Print the matrix to std::cout
   inline void print(const string& name) const;
   inline void dump(const string& name) const;

    /// clear the vector
   inline void clear();

   typedef Vector<floating_type> col;
   typedef Matrix<floating_type> mat;
   static const bool is_sparse = false;

   protected:
   /// Forbid lazy copies
   explicit OptimInfo<floating_type>(const OptimInfo<floating_type>& matrix);
   /// Forbid lazy copies
   OptimInfo<floating_type>& operator=(const OptimInfo<floating_type>& matrix);

   /// is the data allocation external or not
   bool _externAlloc;
   /// pointer to the data
   floating_type* _X;
   /// number of class
   INTM _nclass;
   /// number of rows
   INTM _m;
   /// number of columns
   INTM _n;

};

/// Class Matrix
template<typename floating_type> class Matrix {
   friend class SpMatrix<floating_type>;
   public:
   typedef floating_type value_type;
   typedef Vector<floating_type> col_type;
   typedef INTM index_type;
   typedef Vector<floating_type> element;

   /// Constructor with existing data X of an m x n matrix
   Matrix(floating_type* X, INTM m, INTM n);
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
   inline floating_type& operator()(const INTM i, const INTM j);
   /// Return the value X(i,j)
   inline floating_type operator()(const INTM i, const INTM j) const;
   /// Return a modifiable reference to X(i) (1D indexing)
   inline floating_type& operator[](const INTM index) { return _X[index]; };
   /// Return the value X(i) (1D indexing)
   inline floating_type operator[](const INTM index) const { return _X[index]; };
   /// Copy the column i into x
   inline void copyCol(const INTM i, Vector<floating_type>& x) const;
   /// Copy the column i into x
   inline void copyRow(const INTM i, Vector<floating_type>& x) const;
   inline void scalRow(const INTM i, const floating_type s) const;
   inline void copyToRow(const INTM i, const Vector<floating_type>& x);
   /// Copy the column i into x
   inline void extract_rawCol(const INTM i, floating_type* x) const;
   /// Copy the column i into x
   virtual void add_rawCol(const INTM i, floating_type* DtXi, const floating_type a) const;
   /// Copy the column i into x
   inline void getData(Vector<floating_type>& data, const INTM i) const;
   /// Reference the column i into the vector x
   inline void refCol(INTM i, Vector<floating_type>& x) const;
   /// Reference the column i to i+n into the Matrix mat
   inline void refSubMat(INTM i, INTM n, Matrix<floating_type>& mat) const;
   /// extract a sub-matrix of a symmetric matrix
   inline void subMatrixSym(const Vector<INTM>& indices, 
         Matrix<floating_type>& subMatrix) const;
   /// reference a modifiable reference to the data, DANGEROUS
   inline floating_type* rawX() const { return _X; };
   /// return a non-modifiable reference to the data
   inline const floating_type* X() const { return _X; };
   /// make a copy of the matrix mat in the current matrix
   inline void copy(const Matrix<floating_type>& mat);
   /// make a copy of the matrix mat in the current matrix
   inline void copyTo(Matrix<floating_type>& mat) const { mat.copy(*this); };
   /// make a copy of the matrix mat in the current matrix
   inline void copyRef(const Matrix<floating_type>& mat);

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
   inline void setData(floating_type* X, INTM m, INTM n);
   /// Change the data in the matrix
   inline void refData(const Matrix<floating_type>& mat) {
      this->setData(mat.rawX(),mat.m(),mat.n());
   };
   /// modify _m
   inline void setm(const INTM m) { _m = m; }; //DANGEROUS
   /// modify _n
   inline void setn(const INTM n) { _n = n; }; //DANGEROUS
   /// Set all the values to zero
   inline void setZeros();
   /// Set all the values to a scalar
   inline void set(const floating_type a);
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
   inline void center(Vector<floating_type>& centers);
   /// scale the matrix by the a
   inline void scal(const floating_type a);
   /// make the matrix symmetric by copying the upper-right part
   /// into the lower-left part
   inline void fillSymmetric();
   inline void fillSymmetric2();
   /// change artificially the size of the matrix, DANGEROUS
   inline void fakeSize(const INTM m, const INTM n) { _n = n; _m=m;};
   /// whiten
   inline void whiten(const INTM V);
   /// whiten
   inline void whiten(Vector<floating_type>& mean, const bool pattern = false);
   /// whiten
   inline void whiten(Vector<floating_type>& mean, const Vector<floating_type>& mask);
   /// whiten
   inline void unwhiten(Vector<floating_type>& mean, const bool pattern = false);
   /// whiten
   inline void sum_cols(Vector<floating_type>& sum) const;

   /// Analysis functions
   /// Check wether the columns of the matrix are normalized or not
   inline bool isNormalized() const;
   /// return the 1D-index of the value of greatest magnitude
   inline INTM fmax() const;
   /// return the 1D-index of the value of greatest magnitude
   inline floating_type fmaxval() const;
   /// return the 1D-index of the value of lowest magnitude
   inline INTM fmin() const;

   // Algebric operations
   /// Transpose the current matrix and put the result in the matrix
   /// trans
   inline void transpose(Matrix<floating_type>& trans) const;
   /// A <- -A
   inline void neg();
   /// add one to the diagonal
   inline void incrDiag();
   inline void addDiag(const Vector<floating_type>& diag);
   inline void addDiag(const floating_type diag);
   inline void addToCols(const Vector<floating_type>& diag);
   inline void addVecToCols(const Vector<floating_type>& diag, const floating_type a = 1.0);
   /// perform a rank one approximation uv' using the power method
   /// u0 is an initial guess for u (can be empty).
   inline void svdRankOne(const Vector<floating_type>& u0,
         Vector<floating_type>& u, Vector<floating_type>& v) const;
   inline void singularValues(Vector<floating_type>& u) const;
   inline void svd(Matrix<floating_type>& U, Vector<floating_type>& S, Matrix<floating_type>&V) const;
   inline void svd2(Matrix<floating_type>& U, Vector<floating_type>& S, const int num = -1, const int method = 0) const;
   inline void SymEig(Matrix<floating_type>& U, Vector<floating_type>& S) const;
   inline void InvsqrtMat(Matrix<floating_type>& out, const floating_type lambda_1 = 0) const;
   inline void sqrtMat(Matrix<floating_type>& out) const;
//   inline void Inv(Matrix<floating_type>& out) const;

   /// find the eigenvector corresponding to the largest eigenvalue
   /// when the current matrix is symmetric. u0 is the initial guess.
   /// using two iterations of the power method
   inline void eigLargestSymApprox(const Vector<floating_type>& u0,
         Vector<floating_type>& u) const;
   /// find the eigenvector corresponding to the eivenvalue with the 
   /// largest magnitude when the current matrix is symmetric,
   /// using the power method. It 
   /// returns the eigenvalue. u0 is an initial guess for the 
   /// eigenvector.
   inline floating_type eigLargestMagnSym(const Vector<floating_type>& u0, 
         Vector<floating_type>& u) const;
   /// returns the value of the eigenvalue with the largest magnitude
   /// using the power iteration.
   inline floating_type eigLargestMagnSym() const;
   /// inverse the matrix when it is symmetric
   inline void invSym();
   inline void invSymPos();
   /// perform b = alpha*A'x + beta*b
   inline void multTrans(const Vector<floating_type>& x, Vector<floating_type>& b,
         const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
   /// perform b = alpha*A'x + beta*b
   inline void multTrans(const Vector<floating_type>& x, Vector<floating_type>& b,
         const Vector<bool>& active) const;
   /// perform b = A'x, when x is sparse
   template <typename I>
   inline void multTrans(const SpVector<floating_type,I>& x, Vector<floating_type>& b, const floating_type alpha =1.0, const floating_type beta = 0.0) const;
   /// perform b = alpha*A*x+beta*b
   inline void mult(const Vector<floating_type>& x, Vector<floating_type>& b, 
         const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
   inline void mult_loop(const Vector<floating_type>& x, Vector<floating_type>& b) const;

   /// perform b = alpha*A*x + beta*b, when x is sparse
   template <typename I>
   inline void mult(const SpVector<floating_type,I>& x, Vector<floating_type>& b, 
         const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
   template <typename I>
   inline void mult_loop(const SpVector<floating_type,I>& x, Vector<floating_type>& b) const {
      this->mult(x,b);
   }
   /// perform C = a*A*B + b*C, possibly transposing A or B.
   inline void mult(const Matrix<floating_type>& B, Matrix<floating_type>& C, 
         const bool transA = false, const bool transB = false,
         const floating_type a = 1.0, const floating_type b = 0.0) const;
   /// perform C = a*B*A + b*C, possibly transposing A or B.
   inline void multSwitch(const Matrix<floating_type>& B, Matrix<floating_type>& C, 
         const bool transA = false, const bool transB = false,
         const floating_type a = 1.0, const floating_type b = 0.0) const;
   /// perform C = A*B, when B is sparse
   template <typename I>
   inline void mult(const SpMatrix<floating_type,I>& B, Matrix<floating_type>& C, const bool transA = false,
         const bool transB = false, const floating_type a = 1.0,
         const floating_type b = 0.0) const;
   /// mult by a diagonal matrix on the left
   inline void multDiagLeft(const Vector<floating_type>& diag);
   /// mult by a diagonal matrix on the right
   inline void multDiagRight(const Vector<floating_type>& diag);
   /// mult by a diagonal matrix on the right
   inline void AddMultDiagRight(const Vector<floating_type>& diag, Matrix<floating_type>& mat);
   /// C = A .* B, elementwise multiplication
   inline void mult_elementWise(const Matrix<floating_type>& B, Matrix<floating_type>& C) const;
   inline void div_elementWise(const Matrix<floating_type>& B, Matrix<floating_type>& C) const;
   /// XtX = A'*A
   inline void XtX(Matrix<floating_type>& XtX) const;
   /// XXt = A*A'
   inline void XXt(Matrix<floating_type>& XXt) const;
   /// XXt = A*A' where A is an upper triangular matrix
   inline void upperTriXXt(Matrix<floating_type>& XXt, 
         const INTM L) const;
   /// extract the diagonal
   inline void diag(Vector<floating_type>& d) const;
   /// set the diagonal
   inline void setDiag(const Vector<floating_type>& d);
   /// set the diagonal
   inline void setDiag(const floating_type val);
   /// each element of the matrix is replaced by its exponential
   inline void exp();
   /// each element of the matrix is replaced by its square root
   inline void pow(const floating_type a);
   inline void Sqrt();
   inline void Invsqrt();
   inline void sqr();
   /// return vec1'*A*vec2, where vec2 is sparse
   template <typename I>
   inline floating_type quad(const Vector<floating_type>& vec1, const SpVector<floating_type,I>& vec2) const;
   /// return vec1'*A*vec2, where vec2 is sparse
   template <typename I>
   inline void quad_mult(const Vector<floating_type>& vec1, const SpVector<floating_type,I>& vec2,
         Vector<floating_type>& y, const floating_type a = 1.0, const floating_type b = 0.0) const;
   /// return vec'*A*vec when vec is sparse
   template <typename I>
   inline floating_type quad(const SpVector<floating_type,I>& vec) const;
   /// add alpha*mat to the current matrix
   inline void add(const Matrix<floating_type>& mat, const floating_type alpha = 1.0);
   /// add alpha*mat to the current matrix
   inline void add_scal(const Matrix<floating_type>& mat, const floating_type alpha = 1.0, const floating_type beta = 1.0);
   /// add alpha to the current matrix
   inline void add(const floating_type alpha);
   /// add alpha*mat to the current matrix
   inline floating_type dot(const Matrix<floating_type>& mat) const;
   /// substract the matrix mat to the current matrix
   inline void sub(const Matrix<floating_type>& mat);
   /// inverse the elements of the matrix
   inline void inv_elem();
   /// inverse the elements of the matrix
   inline void inv() { this->inv_elem(); };
   /// return the trace of the matrix
   inline floating_type trace() const;
   /// compute the sum of the magnitude of the matrix values
   inline floating_type asum() const;
   /// compute the sum of the magnitude of the matrix values
   inline floating_type sum() const;
   /// return ||A||_F
   inline floating_type normF() const;
   /// whiten
   inline floating_type mean() const;
   /// whiten
   inline floating_type abs_mean() const;
   /// whiten
   /// return ||A||_F^2
   inline floating_type normFsq() const;
   /// return ||A||_F^2
   inline floating_type nrm2sq() const { return this->normFsq(); };
   /// return ||At||_{inf,2} (max of l2 norm of the columns)
   inline floating_type norm_inf_2_col() const;
   /// return ||At||_{1,2} (max of l2 norm of the columns)
   inline floating_type norm_1_2_col() const;
   /// returns the l2 norms of the columns
   inline void norm_2_cols(Vector<floating_type>& norms) const;
   /// returns the l2 norms of the columns
   inline void norm_2_rows(Vector<floating_type>& norms) const;
   /// returns the linf norms of the columns
   inline void norm_inf_cols(Vector<floating_type>& norms) const;
   /// returns the linf norms of the columns
   inline void norm_inf_rows(Vector<floating_type>& norms) const;
   /// returns the linf norms of the columns
   inline void norm_l1_rows(Vector<floating_type>& norms) const;
   /// returns the linf norms of the columns
   inline void get_sum_cols(Vector<floating_type>& sum) const;
   /// returns the linf norms of the columns
   inline void dot_col(const Matrix<floating_type>& mat, Vector<floating_type>& dots) const;
   /// returns the l2 norms ^2 of the columns
   inline void norm_2sq_cols(Vector<floating_type>& norms) const;
   /// returns the l2 norms of the columns
   inline void norm_2sq_rows(Vector<floating_type>& norms) const;
   inline void thrsmax(const floating_type nu);
   inline void thrsmin(const floating_type nu);
   inline void thrsabsmin(const floating_type nu);
   /// perform soft-thresholding of the matrix, with the threshold nu
   inline void softThrshold(const floating_type nu);
   inline void fastSoftThrshold(const floating_type nu);
   inline void fastSoftThrshold(Matrix<floating_type>& output, const floating_type nu) const;
   inline void hardThrshold(const floating_type nu);
   /// perform soft-thresholding of the matrix, with the threshold nu
   inline void thrsPos();
   /// perform A <- A + alpha*vec1*vec2'
   inline void rank1Update(const Vector<floating_type>& vec1, const Vector<floating_type>& vec2,
         const floating_type alpha = 1.0);
   /// perform A <- A + alpha*vec1*vec2', when vec1 is sparse
   template <typename I>
   inline void rank1Update(const SpVector<floating_type,I>& vec1, const Vector<floating_type>& vec2,
         const floating_type alpha = 1.0);
   /// perform A <- A + alpha*vec1*vec2', when vec2 is sparse
   template <typename I>
   inline void rank1Update(const Vector<floating_type>& vec1, const SpVector<floating_type,I>& vec2,
         const floating_type alpha = 1.0);
   template <typename I>
   inline void rank1Update_mult(const Vector<floating_type>& vec1, const Vector<floating_type>& vec1b,
         const SpVector<floating_type,I>& vec2,
         const floating_type alpha = 1.0);
   /// perform A <- A + alpha*vec*vec', when vec2 is sparse
   template <typename I>
   inline void rank1Update(const SpVector<floating_type,I>& vec,
         const floating_type alpha = 1.0);
   /// perform A <- A + alpha*vec*vec', when vec2 is sparse
   template <typename I>
   inline void rank1Update(const SpVector<floating_type,I>& vec, const SpVector<floating_type,I>& vec2,
         const floating_type alpha = 1.0);
   /// Compute the mean of the columns
   inline void meanCol(Vector<floating_type>& mean) const;
   /// Compute the mean of the rows
   inline void meanRow(Vector<floating_type>& mean) const;
   /// fill the matrix with the row given
   inline void fillRow(const Vector<floating_type>& row);
   /// fill the matrix with the row given
   inline void extractRow(const INTM i, Vector<floating_type>& row) const;
   inline void setRow(const INTM i, const Vector<floating_type>& row);
   inline void addRow(const INTM i, const Vector<floating_type>& row, const floating_type a=1.0);
   /// compute x, such that b = Ax, WARNING this function needs to be u
   /// updated
   inline void conjugateGradient(const Vector<floating_type>& b, Vector<floating_type>& x,
         const floating_type tol = 1e-4, const int = 4) const;
   /// compute x, such that b = Ax, WARNING this function needs to be u
   /// updated, the temporary vectors are given.
   inline void drop(char* fileName) const;
   /// compute a Nadaraya Watson estimator
   inline void NadarayaWatson(const Vector<INTM>& ind, const floating_type sigma);
   /// performs soft-thresholding of the vector
   inline void blockThrshold(const floating_type nu, const INTM sizeGroup);
   /// performs sparse projections of the columns 
   inline void sparseProject(Matrix<floating_type>& out, const floating_type thrs,   const int mode = 1, const floating_type lambda_1 = 0,
         const floating_type lambda_2 = 0, const floating_type lambda_3 = 0, const bool pos = false, const int numThreads=-1);
   inline void transformFilter();

   /// Conversion
   /// make a sparse copy of the current matrix
   inline void toSparse(SpMatrix<floating_type>& matrix) const;
   /// make a sparse copy of the current matrix
   inline void toSparseTrans(SpMatrix<floating_type>& matrixTrans);
   /// make a reference of the matrix to a vector vec 
   inline void toVect(Vector<floating_type>& vec) const;
   /// Accessor
   inline INTM V() const { return 1;};
   /// extract the rows of a matrix corresponding to a binary mask
   inline void copyMask(Matrix<floating_type>& out, Vector<bool>& mask) const;

   typedef Vector<floating_type> col;
   static const bool is_sparse = false;

   protected:
   /// Forbid lazy copies
   explicit Matrix<floating_type>(const Matrix<floating_type>& matrix);
   /// Forbid lazy copies
   Matrix<floating_type>& operator=(const Matrix<floating_type>& matrix);

   /// is the data allocation external or not
   bool _externAlloc;
   /// pointer to the data
   floating_type* _X;
   /// number of rows
   INTM _m;
   /// number of columns
   INTM _n;

};



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
   inline void setData(floating_type* X, const INTM n) { this->setPointer(X,n); };
   inline void refData(const Vector<floating_type>& vec) { this->setPointer(vec.rawX(),vec.n()); };
   inline void refSubVec(INTM i, INTM n, Vector<floating_type>& mat) const { mat.setData(_X+i,n); };
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
   inline floating_type dot(const SpVector<floating_type,I>& x) const;
   /// A <- A + a*x
   inline void add(const Vector<floating_type>& x, const floating_type a = 1.0);
   /// A <- A + a*x
   template <typename I>
   inline void add(const SpVector<floating_type,I>& x, const floating_type a = 1.0);
   /// adds a to each value in the vector
   inline void add(const floating_type a);
   /// A <- b*A + a*x
   inline void add_scal(const Vector<floating_type>& x, const floating_type a = 1.0, const floating_type b = 0);
   /// A <- b*A + a*x
   template <typename I>
      inline void add_scal(const SpVector<floating_type,I>& x, const floating_type a = 1.0, const floating_type b = 0);
   /// A <- A - x
   inline void sub(const Vector<floating_type>& x);
   /// A <- A + a*x
   template <typename I>
   inline void sub(const SpVector<floating_type,I>& x);
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
   inline void mult_elementWise(const Vector<floating_type>& B, Vector<floating_type>& C) const { C.mult(*this,B); };
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
   inline void sparseProject(Vector<floating_type>& out, const floating_type thrs,   const int mode = 1, const floating_type lambda_1 = 0,
         const floating_type lambda_2 = 0, const floating_type lambda_3 = 0, const bool pos = false);
   inline void project_sft(const Vector<int>& labels, const int clas);
   inline void project_sft_binary(const Vector<floating_type>& labels);
   /// projects the vector onto the l1 ball of radius thrs,
   /// projects the vector onto the l1 ball of radius thrs,
   /// returns true if the returned vector is null
   inline void l1l2project(Vector<floating_type>& out, const floating_type thrs, const floating_type gamma, const bool pos = false) const;
   inline void fusedProject(Vector<floating_type>& out, const floating_type lambda_1, const floating_type lambda_2, const int itermax);
   inline void fusedProjectHomotopy(Vector<floating_type>& out, const floating_type lambda_1,const floating_type lambda_2,const floating_type lambda_3 = 0,
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



/// Sparse Matrix class, CSC format
template<typename floating_type, typename I> class SpMatrix {
   friend class Matrix<floating_type>;
   friend class SpVector<floating_type,I>;
   public:
   typedef floating_type value_type;
   typedef SpVector<floating_type,I> col_type;
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
   inline void refCol(I i, SpVector<floating_type,I>& vec) const;
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
   inline void print(const string& name) const;
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
   inline void add_direct(const SpMatrix<floating_type,I>& mat, const floating_type a);
   inline void copy_direct(const SpMatrix<floating_type,I>& mat);
   inline floating_type dot_direct(const SpMatrix<floating_type,I>& mat) const;

   /// Modifiers
   /// clear the matrix
   inline void clear();
   /// resize the matrix
   inline void resize(const I m, const I n, const I nzmax);
   /// scale the matrix by a
   inline void scal(const floating_type a) const;
   inline floating_type abs_mean() const;

   /// Algebraic operations
   /// aat <- A*A'
   inline void AAt(Matrix<floating_type>& aat) const;
   /// aat <- A(:,indices)*A(:,indices)'
   inline void AAt(Matrix<floating_type>& aat, const Vector<I>& indices) const;
   /// aat <- sum_i w_i A(:,i)*A(:,i)'
   inline void wAAt(const Vector<floating_type>& w, Matrix<floating_type>& aat) const;
   /// XAt <- X*A'
   inline void XAt(const Matrix<floating_type>& X, Matrix<floating_type>& XAt) const;
   /// XAt <- X(:,indices)*A(:,indices)'
   inline void XAt(const Matrix<floating_type>& X, Matrix<floating_type>& XAt, 
         const Vector<I>& indices) const;
   /// XAt <- sum_i w_i X(:,i)*A(:,i)'
   inline void wXAt( const Vector<floating_type>& w, const Matrix<floating_type>& X, 
         Matrix<floating_type>& XAt, const int numthreads=-1) const;
   inline void XtX(Matrix<floating_type>& XtX) const;

   /// y <- A'*x
   inline void multTrans(const Vector<floating_type>& x, Vector<floating_type>& y,
         const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
   inline void multTrans(const SpVector<floating_type,I>& x, Vector<floating_type>& y,
         const floating_type alpha = 1.0, const floating_type beta = 0.0) const;
   /// perform b = alpha*A*x + beta*b, when x is sparse
   inline void mult(const SpVector<floating_type,I>& x, Vector<floating_type>& b, 
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
   inline void mult(const SpMatrix<floating_type,I>& B, Matrix<floating_type>& C, const bool transA = false,
         const bool transB = false, const floating_type a = 1.0,
         const floating_type b = 0.0) const;
   /// make a copy of the matrix mat in the current matrix
   inline void copyTo(Matrix<floating_type>& mat) const { this->toFull(mat); };
   /// dot product;
   inline floating_type dot(const Matrix<floating_type>& x) const;
   inline void copyRow(const I i, Vector<floating_type>& x) const;
   inline void sum_cols(Vector<floating_type>& sum) const;
   inline void copy(const SpMatrix<floating_type,I>& mat);

   /// Conversions
   /// copy the sparse matrix into a dense matrix
   inline void toFull(Matrix<floating_type>& matrix) const;
   /// copy the sparse matrix into a dense transposed matrix
   inline void toFullTrans(Matrix<floating_type>& matrix) const;

   /// use the data from v, r for _v, _r
   inline void convert(const Matrix<floating_type>&v, const Matrix<I>& r,
         const I K);
   /// use the data from v, r for _v, _r
   inline void convert2(const Matrix<floating_type>&v, const Vector<I>& r,
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

   typedef SpVector<floating_type,I> col;
   static const bool is_sparse = true;

   private:
   /// forbid copy constructor
   explicit SpMatrix(const SpMatrix<floating_type,I>& matrix);
   SpMatrix<floating_type,I>& operator=(const SpMatrix<floating_type,I>& matrix);

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

/// Sparse vector class
template <typename floating_type, typename I> class SpVector {
   friend class Matrix<floating_type>;
   friend class SpMatrix<floating_type,I>;
   friend class Vector<floating_type>;
   public:
   typedef floating_type value_type;
   /// Constructor, of the sparse vector of size L.
   SpVector(floating_type* v, I* r, I L, I nzmax);
   /// Constructor, allocates nzmax slots
   SpVector(I nzmax);
   /// Empty constructor
   SpVector();

   /// Destructor
   ~SpVector();

   /// Accessors
   /// returns the length of the vector
   inline floating_type nzmax() const { return _nzmax; };
   /// returns the length of the vector
   inline floating_type length() const { return _L; };
   /// computes the sum of the magnitude of the elements
   inline floating_type asum() const;
   /// computes the l2 norm ^2 of the vector
   inline floating_type nrm2sq() const;
   /// computes the l2 norm  of the vector
   inline floating_type nrm2() const;
   /// computes the linf norm  of the vector
   inline floating_type fmaxval() const;
   /// print the vector to std::cerr
   inline void print(const string& name) const;
   inline void refIndices(Vector<I>& indices) const;
   /// creates a reference on the vector val
   inline void refVal(Vector<floating_type>& val) const;
   /// access table r
   inline I r(const I i) const { return _r[i]; };
   /// access table r
   inline floating_type v(const I i) const { return _v[i]; };
   inline floating_type* rawX() const { return _v; };
   inline I* rawR() const { return _r; };

   /// 
   inline I L() const { return _L; };
   /// 
   inline void setL(const I L) { _L=L; };
   /// a <- a.^2
   inline void sqr();
   /// dot product
   inline floating_type dot(const SpVector<floating_type,I>& vec) const;
   /// dot product
   inline floating_type dot(const Vector<floating_type>& vec) const;
   /// dot product
   inline void scal(const floating_type a);

   /// Modifiers
   /// clears the vector
   inline void clear();
   /// resizes the vector
   inline void resize(const I nzmax);

   /// resize the vector as a sparse matrix
   void inline toSpMatrix(SpMatrix<floating_type,I>& out,
         const I m, const I n) const;
  /// resize the vector as a sparse matrix
   void inline toFull(Vector<floating_type>& out) const;
   inline void getIndices(Vector<int>& ind) const;

   private:
   /// forbids lazy copies
   explicit SpVector(const SpVector<floating_type,I>& vector);
   SpVector<floating_type,I>& operator=(const SpVector<floating_type,I>& vector);

   /// external allocation 
   bool _externAlloc;
   /// data
   floating_type* _v;
   /// indices
   I* _r;
   /// length
   I _L;
   /// maximum number of nonzeros elements
   I _nzmax;
};

/// Class for dense vector
template<typename floating_type, typename I> class LazyVector {
   public:
      LazyVector(Vector<floating_type>& x, const Vector<floating_type>& z, const int n) : _x(x), _z(z), _n(n+1), _p(x.n()) { 
         _current_time=0;
         _dates.resize(_p);
         _dates.setZeros();
         _stats1.resize(n+1);
         _stats2.resize(n+1);
         _stats1[0]=floating_type(1.0);
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
      void inline add_scal(const floating_type a, const floating_type b) { // performs x <- a(x - b z) 
         if (_current_time == _n)
            update();
         _current_time++;
         _stats2[_current_time]=_stats2[_current_time-1] + a/_stats1[_current_time-1];
         _stats1[_current_time]=_stats1[_current_time-1]*b;
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
      DoubleLazyVector(Vector<floating_type>& x, const Vector<floating_type>& z1, const Vector<floating_type>& z2, const int n) : _x(x), _z1(z1), _z2(z2), _n(n+1), _p(x.n()) { 
         _current_time=0;
         _dates.resize(_p);
         _dates.setZeros();
         _stats1.resize(n+1);
         _stats2.resize(n+1);
         _stats3.resize(n+1);
         _stats1[0]=floating_type(1.0);
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
      void inline add_scal(const floating_type a, const floating_type b, const floating_type c) {
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
      Vector<floating_type>& _x;
      const Vector<floating_type>& _z1;
      const Vector<floating_type>& _z2;
      const int _n;
      const int _p;
      Vector<floating_type> _stats1, _stats2, _stats3;
      Vector<int> _dates;
      int _current_time;
};



/* ***********************************
 * Implementation of the class Vector
 * ***********************************/


/// Empty constructor
template <typename floating_type> Vector<floating_type>::Vector() :
   _externAlloc(true), _X(NULL),  _n(0) {  };

/// Constructor. Create a new vector of size n
template <typename floating_type> Vector<floating_type>::Vector(INTM n) :
   _externAlloc(false), _n(n) {
#pragma omp critical
      {
         _X=new floating_type[_n];
      }
   };

/// Constructor with existing data
template <typename floating_type> Vector<floating_type>::Vector(floating_type* X, INTM n) :
   _externAlloc(true), _X(X),  _n(n) {  };

/// Copy constructor
template <typename floating_type> Vector<floating_type>::Vector(const Vector<floating_type>& vec) :
   _externAlloc(false), _n(vec._n) {
#pragma omp critical
      {
         _X=new floating_type[_n];
      }
      cblas_copy<floating_type>(_n,vec._X,1,_X,1);
   };

/// Destructor
template <typename floating_type> Vector<floating_type>::~Vector() {
   clear();
};

/// Print the matrix to std::cout
template <typename floating_type> inline void Vector<floating_type>::print(const string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _n << std::endl;
   for (INTM j = 0; j<_n; ++j) {
      fprintf( stderr, "%10.5g ",static_cast<double>(_X[j]));
   }
   fprintf( stderr, "\n ");
};

/// Print the matrix to std::cout
template <typename floating_type> inline void Vector<floating_type>::dump(const string& name) const {
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
template <typename floating_type> inline INTM Vector<floating_type>::max() const {
   INTM imax=0;
   floating_type max=_X[0];
   for (INTM j = 1; j<_n; ++j) {
      floating_type cur = _X[j];
      if (cur > max) {
         imax=j;
         max = cur;
      }
   }
   return imax;
};

/// returns the index of the minimum value
template <typename floating_type> inline INTM Vector<floating_type>::min() const {
   INTM imin=0;
   floating_type min=_X[0];
   for (INTM j = 1; j<_n; ++j) {
      floating_type cur = _X[j];
      if (cur < min) {
         imin=j;
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
   floating_type first=log10(a);
   floating_type last=log10(b);
   floating_type step = (last-first)/(n-1);
   this->resize(n);
   _X[0]=first;
   for (INTM i = 1; i<_n; ++i)
      _X[i]=_X[i-1]+step;
   for (INTM i = 0; i<_n; ++i)
      _X[i]=pow(floating_type(10.0),_X[i]);
}

template <typename floating_type>
inline INTM Vector<floating_type>::nnz() const {
   INTM sum=0;
   for (INTM i = 0; i<_n; ++i) 
      if (_X[i] != floating_type()) ++sum;
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
template <typename floating_type> inline INTM Vector<floating_type>::fmax() const {
   return cblas_iamax<floating_type>(_n,_X,1);
};

/// returns the index of the value with smallest magnitude
template <typename floating_type> inline INTM Vector<floating_type>::fmin() const {
   return cblas_iamin<floating_type>(_n,_X,1);
};

/// returns a reference to X[index]
template <typename floating_type> inline floating_type& Vector<floating_type>::operator[] (const INTM i) {
   assert(i>=0 && i<_n);
   return _X[i];
};

/// returns X[index]
template <typename floating_type> inline floating_type Vector<floating_type>::operator[] (const INTM i) const {
   assert(i>=0 && i<_n);
   return _X[i];
};

/// make a copy of x
template <typename floating_type> inline void Vector<floating_type>::copy(const Vector<floating_type>& x) {
   if (_X != x._X) {
      this->resize(x.n());
      //cblas_copy<floating_type>(_n,x._X,1,_X,1);
      memcpy(_X,x._X,_n*sizeof(floating_type));
   }
};

/// make a copy of x
template <typename floating_type> inline void Vector<floating_type>::copyRef(const Vector<floating_type>& x) {
   this->setData(x.rawX(),x.n());
};


/// Set all values to zero
template <typename floating_type> inline void Vector<floating_type>::setZeros() {
   memset(_X,0,_n*sizeof(floating_type));
};

/// resize the vector
template <typename floating_type> inline void Vector<floating_type>::resize(const INTM n, const bool set_zeros) {
   if (_n == n) return;
   clear();
#pragma omp critical
   {
      _X=new floating_type[n];
   }
   _n=n;
   _externAlloc=false;
   if (set_zeros)
      this->setZeros();
};

/// change the data of the vector
template <typename floating_type> inline void Vector<floating_type>::setPointer(floating_type* X, const INTM n) {
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
template <typename floating_type> inline void Vector<floating_type>::setAleat() {
   for (INTM i = 0; i<_n; ++i) _X[i]=normalDistrib<floating_type>();
};

/// clear the vector
template <typename floating_type> inline void Vector<floating_type>::clear() {
   if (!_externAlloc) delete[](_X);
   _n=0;
   _X=NULL;
   _externAlloc=true;
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::softThrshold(const floating_type nu) {
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
template <typename floating_type> inline void Vector<floating_type>::fastSoftThrshold(const floating_type nu) {
    //#pragma omp parallel for
    for (INTM i = 0; i<_n; ++i)
    {
        _X[i]=fastSoftThrs(_X[i],nu);
    }
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::fastSoftThrshold(Vector<floating_type>& output, const floating_type nu) const {
   output.resize(_n,false);
//#pragma omp parallel for
   for (INTM i = 0; i<_n; ++i) 
      output[i]=fastSoftThrs(_X[i],nu);
};

/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::softThrsholdScal(Vector<floating_type>& out, const floating_type nu, const floating_type s) {
   floating_type* Y = out.rawX();
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
template <typename floating_type> inline void Vector<floating_type>::hardThrshold(const floating_type nu) {
   for (INTM i = 0; i<_n; ++i) {
      if (!(_X[i] > nu || _X[i] < -nu)) {
         _X[i] = 0;
      }
   }
};


/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsmax(const floating_type nu) {
//#pragma omp parallel for private(i)
   for (INTM i = 0; i<_n; ++i) 
      if (_X[i] < nu) _X[i]=nu;
}

/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsmin(const floating_type nu) {
   for (INTM i = 0; i<_n; ++i) 
      _X[i]=MIN(_X[i],nu);
}

/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsabsmin(const floating_type nu) {
   for (INTM i = 0; i<_n; ++i) 
      _X[i]=MAX(MIN(_X[i],nu),-nu);
}

/// performs thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrshold(const floating_type nu) {
   for (INTM i = 0; i<_n; ++i) 
      if (abs<floating_type>(_X[i]) < nu) 
         _X[i]=0;
}
/// performs soft-thresholding of the vector
template <typename floating_type> inline void Vector<floating_type>::thrsPos() {
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
template <typename floating_type> inline void Vector<floating_type>::set(const floating_type val) {
   for (INTM i = 0; i<_n; ++i) _X[i]=val;
};

/// returns ||A||_2
template <typename floating_type> inline floating_type Vector<floating_type>::nrm2() const {
   return cblas_nrm2<floating_type>(_n,_X,1);
};

/// returns ||A||_2^2
template <typename floating_type> inline floating_type Vector<floating_type>::nrm2sq() const {
   return cblas_dot<floating_type>(_n,_X,1,_X,1);
};

/// returns  A'x
template <typename floating_type> inline floating_type Vector<floating_type>::dot(const Vector<floating_type>& x) const {
   assert(_n == x._n);
   return cblas_dot<floating_type>(_n,_X,1,x._X,1);
};

/// returns A'x, when x is sparse
template <typename floating_type> 
template <typename I> 
inline floating_type Vector<floating_type>::dot(const SpVector<floating_type,I>& x) const {
   floating_type sum=0;
   const I* r = x.rawR();
   const floating_type* v = x.rawX();
   for (INTT i = 0; i<x._L; ++i) {
      sum += _X[r[i]]*v[i];
   }
   return sum;
   //return cblas_doti<floating_type>(x._L,x._v,x._r,_X);
};

/// A <- A + a*x
template <typename floating_type> inline void Vector<floating_type>::add(const Vector<floating_type>& x, const floating_type a) {
   assert(_n == x._n);
   cblas_axpy<floating_type>(_n,a,x._X,1,_X,1);
};

template <typename floating_type> inline void Vector<floating_type>::add_scal(const Vector<floating_type>& x, const floating_type a, const floating_type b) {
   assert(_n == x._n);
   cblas_axpby<floating_type>(_n,a,x._X,1,b,_X,1);
};

/// A <- A + a*x
template <typename floating_type> 
template <typename I> 
inline void Vector<floating_type>::add(const SpVector<floating_type,I>& x,
      const floating_type a) {
   if (a == 1.0) {
      for (INTM i = 0; i<x._L; ++i)
         _X[x._r[i]]+=x._v[i];
   } else {
      for (INTM i = 0; i<x._L; ++i)
         _X[x._r[i]]+=a*x._v[i];
   }
};

/// A <- A + a*x
template <typename floating_type> 
template <typename I>
inline void Vector<floating_type>::add_scal(const SpVector<floating_type,I>& x,
      const floating_type a, const floating_type b) {
   if (b != floating_type(1.0)) {
      if (b==0) {
         this->setZeros();
      } else {
         this->scal(b);
      }   
   }
   if (a == floating_type(1.0)) {
      for (I i = 0; i<x._L; ++i)
         _X[x._r[i]]+=x._v[i];
   } else {
      for (I i = 0; i<x._L; ++i)
         _X[x._r[i]]+=a*x._v[i];
   }
};



/// adds a to each value in the vector
template <typename floating_type> inline void Vector<floating_type>::add(const floating_type a) {
   for (INTM i = 0; i<_n; ++i) _X[i]+=a;
};

/// A <- A - x
template <typename floating_type> inline void Vector<floating_type>::sub(const Vector<floating_type>& x) {
   assert(_n == x._n);
   vSub<floating_type>(_n,_X,x._X,_X);
};

/// A <- A + a*x
template <typename floating_type> 
template <typename I> 
inline void Vector<floating_type>::sub(const SpVector<floating_type,I>& x) {
   for (INTM i = 0; i<x._L; ++i)
      _X[x._r[i]]-=x._v[i];
};

/// A <- A ./ x
template <typename floating_type> inline void Vector<floating_type>::div(const Vector<floating_type>& x) {
   assert(_n == x._n);
   vDiv<floating_type>(_n,_X,x._X,_X);
};

/// A <- x ./ y
template <typename floating_type> inline void Vector<floating_type>::div(const Vector<floating_type>& x, const Vector<floating_type>& y) {
   assert(_n == x._n);
   vDiv<floating_type>(_n,x._X,y._X,_X);
};


/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::sqr(const Vector<floating_type>& x) {
   this->resize(x._n);
   vSqr<floating_type>(_n,x._X,_X);
}

/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::sqr() {
   vSqr<floating_type>(_n,_X,_X);
}

/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Invsqrt(const Vector<floating_type>& x) {
   this->resize(x._n);
   vInvSqrt<floating_type>(_n,x._X,_X);
}
/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Sqrt(const Vector<floating_type>& x) {
   this->resize(x._n);
   vSqrt<floating_type>(_n,x._X,_X);
}
/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Invsqrt() {
   vInvSqrt<floating_type>(_n,_X,_X);
}
/// A <- x .^ 2
template <typename floating_type> inline void Vector<floating_type>::Sqrt() {
   vSqrt<floating_type>(_n,_X,_X);
}


/// A <- 1./x
template <typename floating_type> inline void Vector<floating_type>::inv(const Vector<floating_type>& x) {
   this->resize(x.n());
   vInv<floating_type>(_n,x._X,_X);
};

/// A <- 1./A
template <typename floating_type> inline void Vector<floating_type>::inv() {
   vInv<floating_type>(_n,_X,_X);
};

/// A <- x .* y
template <typename floating_type> inline void Vector<floating_type>::mult(const Vector<floating_type>& x,
      const Vector<floating_type>& y) {
   this->resize(x.n());
   vMul<floating_type>(_n,x._X,y._X,_X);
};
;

/// normalize the vector
template <typename floating_type> inline void Vector<floating_type>::normalize() {
   floating_type norm=nrm2();
   if (norm > EPSILON) scal(1.0/norm);
};

/// normalize the vector
template <typename floating_type> inline void Vector<floating_type>::normalize2(const floating_type thrs) {
   floating_type norm=nrm2();
   if (norm > thrs) scal(thrs/norm);
};

/// whiten
template <typename floating_type> inline void Vector<floating_type>::whiten(
      Vector<floating_type>& meanv, const bool pattern) {
   if (pattern) {
      const INTM n =static_cast<INTM>(sqrt(static_cast<floating_type>(_n)));
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
         floating_type mean = 0;
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
template <typename floating_type> inline void Vector<floating_type>::whiten(
      Vector<floating_type>& meanv, const Vector<floating_type>& mask) {
   const INTM V = meanv.n();
   const INTM sizePatch=_n/V;
   for (INTM j = 0; j<V; ++j) {
      floating_type mean = 0;
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
template <typename floating_type> inline void Vector<floating_type>::whiten(const INTM V) {
   const INTM sizePatch=_n/V;
   for (INTM j = 0; j<V; ++j) {
      floating_type mean = 0;
      for (INTM k = 0; k<sizePatch; ++k) {
         mean+=_X[sizePatch*j+k];
      }
      mean /= sizePatch;
      for (INTM k = 0; k<sizePatch; ++k) {
         _X[sizePatch*j+k]-=mean;
      }
   }
};

template <typename floating_type> inline floating_type Vector<floating_type>::KL(const Vector<floating_type>& Y) {
   floating_type sum = 0;
   floating_type* prY = Y.rawX();
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] > 1e-20) {
         if (prY[i] < 1e-60) {
            sum += 1e200;
         } else {
            sum += _X[i]*log_alt<floating_type>(_X[i]/prY[i]);
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
      const INTM n =static_cast<INTM>(sqrt(static_cast<floating_type>(_n)));
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
         floating_type mean = meanv[j];
         for (INTM k = 0; k<sizePatch; ++k) {
            _X[sizePatch*j+k]+=mean;
         }
      }
   }
};


/// return the mean
template <typename floating_type> inline floating_type Vector<floating_type>::mean() const {
   return this->sum()/_n;
}

template <typename floating_type> inline floating_type Vector<floating_type>::abs_mean() const {
   return this->asum()/_n;
};

template <typename floating_type> inline floating_type Vector<floating_type>::mean_non_uniform(const Vector<floating_type>& qi) const {
   Vector<floating_type> tmp;
   tmp.copy(*this);
   tmp.mult(qi,tmp);
   return tmp.sum();
};

/// return the std
template <typename floating_type> inline floating_type Vector<floating_type>::std() {
   floating_type E = this->mean();
   floating_type std=0;
   for (INTM i = 0; i<_n; ++i) {
      floating_type tmp=_X[i]-E;
      std += tmp*tmp;
   }
   std /= _n;
   return sqr_alt<floating_type>(std);
}

/// scale the vector by a
template <typename floating_type> inline void Vector<floating_type>::scal(const floating_type a) {
   return cblas_scal<floating_type>(_n,a,_X,1);
};

/// A <- -A
template <typename floating_type> inline void Vector<floating_type>::neg() {
   for (INTM i = 0; i<_n; ++i) _X[i]=-_X[i];
};

/// replace each value by its exponential
template <typename floating_type> inline void Vector<floating_type>::exp() {
   vExp<floating_type>(_n,_X,_X);
};

/// replace each value by its absolute value
template <typename floating_type> inline void Vector<floating_type>::abs_vec() {
   vAbs<floating_type>(_n,_X,_X);
};

/// replace each value by its logarithm
template <typename floating_type> inline void Vector<floating_type>::log() {
   for (INTM i=0; i<_n; ++i) _X[i]=alt_log<floating_type>(_X[i]);
};

/// replace each value by its exponential
template <typename floating_type> inline void Vector<floating_type>::logexp() {
   for (INTM i = 0; i<_n; ++i) {
      _X[i]=logexp2(_X[i]);
      /*if (_X[i] < -30) {
         _X[i]=0;
      } else if (_X[i] < 30) {
         _X[i]= alt_log<floating_type>( floating_type(1.0) + exp_alt<floating_type>( _X[i] ) );
      }*/
   }
};

template <typename floating_type> inline floating_type Vector<floating_type>::logsumexp() {
   floating_type mm=this->maxval();
   this->add(-mm);
   this->exp();
   return mm+alt_log<floating_type>(this->asum());
};

/// replace each value by its exponential
template <typename floating_type> inline floating_type Vector<floating_type>::softmax(const int y) {
   this->add(-_X[y]);
   _X[y]=-INFINITY;
   floating_type max=this->maxval();
   if (max > 30) {
      return max;
   } else if (max < -30) {
      return 0;
   } else {
      _X[y]=floating_type(0.0);
      this->exp();
      return alt_log<floating_type>(this->sum());
   }
};

/// computes the sum of the magnitudes of the vector
template <typename floating_type> inline floating_type Vector<floating_type>::asum() const {
   return cblas_asum<floating_type>(_n,_X,1);
};

template <typename floating_type> inline floating_type Vector<floating_type>::lzero() const {
   INTM count=0;
   for (INTM i = 0; i<_n; ++i) 
      if (_X[i] != 0) ++count;
   return count;
};


template <typename floating_type> inline floating_type Vector<floating_type>::afused() const {
   floating_type sum = 0;
   for (INTM i = 1; i<_n; ++i) {
      sum += abs<floating_type>(_X[i]-_X[i-1]);
   }
   return sum;
}
/// returns the sum of the vector
template <typename floating_type> inline floating_type Vector<floating_type>::sum() const {
   floating_type sum=floating_type();
   for (INTM i = 0; i<_n; ++i) sum +=_X[i]; 
   return sum;
};

/// puts in signs, the sign of each poINTM in the vector
template <typename floating_type> inline void Vector<floating_type>::sign(Vector<floating_type>& signs) const {
   floating_type* prSign=signs.rawX();
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
template <typename floating_type> inline void Vector<floating_type>::l1project(Vector<floating_type>& out,
      const floating_type thrs, const bool simplex) const {
   out.copy(*this);
   if (simplex) {
      out.thrsPos();
   } else {
      vAbs<floating_type>(_n,out._X,out._X);
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
      swap(prU[0],prU[sizeU/2]);
      floating_type pivot = prU[0];
      INTM sizeG=1;
      floating_type sumG=pivot;

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
   floating_type lambda_1 = (sum-thrs)/sum_card;
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
   if (thrs==0) {
      out.setZeros();
      return;
   }
   vAbs<floating_type>(_n,out._X,out._X);
   out.div(weights);
   Vector<INTM> keys(_n);
   for (INTM i = 0; i<_n; ++i) keys[i]=i;
   out.sort2(keys,false);
   floating_type sum1=0;
   floating_type sum2=0;
   floating_type lambda_1=0;
   for (INTM i = 0; i<_n; ++i) {
      const floating_type lambda_old=lambda_1;
      const floating_type fact=weights[keys[i]]*weights[keys[i]];
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


template <typename floating_type>
inline void Vector<floating_type>::project_sft_binary(const Vector<floating_type>& y) {
   floating_type mean = this->mean();
   Vector<floating_type> ztilde, xtilde;
   ztilde.resize(_n);
   int count=0;
   if (mean > 0) {
      for (int ii=0; ii<_n; ++ii) 
         if (y[ii] > 0) {
            count++;
            ztilde[ii]=_X[ii]+floating_type(1.0);
         } else {
            ztilde[ii]= _X[ii];
         }
      ztilde.l1project(xtilde,floating_type(count));
      for (int ii=0; ii<_n; ++ii) 
         _X[ii] = y[ii] > 0 ? xtilde[ii]-floating_type(1.0) : xtilde[ii];
   } else {
      for (int ii=0; ii<_n; ++ii) 
         if (y[ii] > 0) {
            ztilde[ii]=-_X[ii];
         } else {
            count++;
            ztilde[ii]=- _X[ii] + floating_type(1.0);
         }
      ztilde.l1project(xtilde,floating_type(count));
      for (int ii=0; ii<_n; ++ii) 
         _X[ii] = y[ii] > 0 ? -xtilde[ii] :  -xtilde[ii]+floating_type(1.0);
   }
};

template <typename floating_type>
inline void Vector<floating_type>::project_sft(const Vector<int>& labels, const int clas) {
   Vector<floating_type> y(_n);
   for (int ii=0; ii<_n; ++ii) y[ii] = labels[ii]==clas ? floating_type(1.0) : -floating_type(1.0);
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
      this->l1project(out,thrs,pos);
   } else if (mode == 2) {
      /// min_u ||b-u||_2^2 / ||u||_2^2 + lambda_1||u||_1 <= thrs
      if (lambda_1 > 1e-10) {
         this->scal(lambda_1);
         this->l1l2project(out,thrs,2.0/(lambda_1*lambda_1),pos);
         this->scal(floating_type(1.0/lambda_1));
         out.scal(floating_type(1.0/lambda_1));
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
      floating_type nrm=out.nrm2sq();
      if (nrm > thrs)
         out.scal(sqr_alt<floating_type>(thrs/nrm));
   } else if (mode == 5) {
      /// min_u 0.5||b-u||_2^2  + lambda_1||u||_1 +lambda_2 Fused(u) / ||u||_2^2 <= thrs
      //      this->fusedProject(out,lambda_1,lambda_2,100);
      //      floating_type nrm=out.nrm2sq();
      //      if (nrm > thrs)
      //         out.scal(sqr_alt<floating_type>(thrs/nrm));
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
template <typename floating_type>
inline void Vector<floating_type>::l1l2projectb(Vector<floating_type>& out, const floating_type thrs, const floating_type gamma, const bool pos,
      const int mode) {
   if (mode == 1) {
      /// min_u ||b-u||_2^2 / ||u||_2^2 + gamma ||u||_1 <= thrs
      this->scal(gamma);
      this->l1l2project(out,thrs,2.0/(gamma*gamma),pos);
      this->scal(floating_type(1.0/gamma));
      out.scal(floating_type(1.0/gamma));
   } else if (mode == 2) {
      /// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
      this->l1l2project(out,thrs,gamma,pos);
   } else if (mode == 3) {
      /// min_u 0.5||b-u||_2^2  + gamma||u||_1 / ||u||_2^2 <= thrs
      out.copy(*this);
      if (pos) 
         out.thrsPos();
      out.softThrshold(gamma);
      floating_type nrm=out.nrm2();
      if (nrm > thrs)
         out.scal(thrs/nrm);
   }
}

/// returns true if the returned vector is null
/// min_u ||b-u||_2^2 / ||u||_1 + (gamma/2) ||u||_2^2 <= thrs
template <typename floating_type>
   inline void Vector<floating_type>::l1l2project(Vector<floating_type>& out, const floating_type thrs, const floating_type gamma, const bool pos) const {
      if (gamma == 0) 
         return this->l1project(out,thrs,pos);
      out.copy(*this);
      if (pos) {
         out.thrsPos();
      } else {
         vAbs<floating_type>(_n,out._X,out._X);
      }
      floating_type norm = out.sum() + gamma*out.nrm2sq();
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
         swap(prU[0],prU[sizeU/2]);
         floating_type pivot = prU[0];
         INTM sizeG=1;
         floating_type sumG=pivot+0.5*gamma*pivot*pivot;

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
      floating_type a = gamma*gamma*thrs+0.5*gamma*sum_card;
      floating_type b = 2*gamma*thrs+sum_card;
      floating_type c=thrs-sum;
      floating_type delta = b*b-4*a*c;
      floating_type lambda_1 = (-b+sqrt(delta))/(2*a);

      out.copy(*this);
      if (pos) {
         out.thrsPos();
      }
      out.fastSoftThrshold(lambda_1);
      out.scal(floating_type(1.0/(1+lambda_1*gamma)));
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

template <typename floating_type>
inline void Vector<floating_type>::fusedProjectHomotopy(Vector<floating_type>& alpha, 
      const floating_type lambda_1,const floating_type lambda_2,const floating_type lambda_3,
      const bool penalty) {
   floating_type* pr_DtR=_X;
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
   Vector<INTM> ind(K+1);
   Vector<bool> signs(K);
   ind.set(K);
   INTM* pr_ind = ind.rawX();
   bool* pr_signs = signs.rawX();

   /// Computation of DtR
   floating_type sumBeta = this->sum();

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
   pr_c[0]=floating_type(1.0)/K;
   INTM currentInd=this->fmax();
   floating_type currentLambda=abs<floating_type>(pr_DtR[currentInd]);
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
         scores.scal(floating_type(1.0/(1.0+lambda_3*currentLambda/lambda_2)));
         if (lambda_1*scores.asum()+lambda_2*scores.afused()+0.5*
               lambda_3*scores.nrm2sq() >= floating_type(1.0)) break;
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
         pr_c[j-1]=floating_type(1.0)/(pr_ind[j]-pr_ind[j-1]);
         pr_c[j]=floating_type(1.0)/(pr_ind[j+1]-pr_ind[j]);
      }

      // Compute u
      pr_u[0]= pr_signs[1] ? -pr_c[0] : pr_c[0];
      if (i == 1) {
         pr_u[1]=pr_signs[1] ? pr_c[0]+pr_c[1] : -pr_c[0]-pr_c[1];
      } else {
         pr_u[1]=pr_signs[1] ? pr_c[0]+pr_c[1] : -pr_c[0]-pr_c[1];
         pr_u[1]+=pr_signs[2] ? -pr_c[1] : pr_c[1];
         for (INTM j = 2; j<i; ++j) {
            pr_u[j]=2*fusedHomotopyAux<floating_type>(pr_signs[j-1],
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
      floating_type max_step1 = INFINITY;
      if (penalty) {
         max_step1 = currentLambda-lambda_2;
      } 

      /// Check changes of sign
      floating_type max_step2 = INFINITY;
      INTM step_out = -1;
      for (INTM j = 1; j<=i; ++j) {
         floating_type ratio = -pr_gamma[pr_ind[j]]/pr_u[j];
         if (ratio > 0 && ratio <= max_step2) {
            max_step2=ratio;
            step_out=j;
         }
      }
      floating_type max_step3 = INFINITY;
      /// Check new variables entering the active set
      for (INTM j = 1; j<K; ++j) {
         floating_type sc1 = (currentLambda-pr_DtR[j])/(floating_type(1.0)-pr_DDu[j]);
         floating_type sc2 = (currentLambda+pr_DtR[j])/(floating_type(1.0)+pr_DDu[j]);
         if (sc1 <= 1e-10) sc1=INFINITY;
         if (sc2 <= 1e-10) sc2=INFINITY;
         pr_scores[j]= MIN(sc1,sc2);
      }
      for (INTM j = 0; j<=i; ++j) {
         pr_scores[pr_ind[j]]=INFINITY;
      }
      currentInd = scores.fmin();
      max_step3 = pr_scores[currentInd];
      floating_type step = MIN(max_step1,MIN(max_step3,max_step2));
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
         pr_c[step_out-1]=floating_type(1.0)/(pr_ind[step_out]-pr_ind[step_out-1]);
         pr_c[step_out]=floating_type(1.0)/(pr_ind[step_out+1]-pr_ind[step_out]);
         i-=2;
         newAtom=false;
      } else {
         newAtom=true;
      }
   }

   if (penalty) {
      alpha.softThrshold(lambda_1);
      alpha.scal(floating_type(1.0/(1.0+lambda_3)));
   } else {
      alpha.softThrshold(lambda_1*currentLambda/lambda_2);
      alpha.scal(floating_type(1.0/(1.0+lambda_3*currentLambda/lambda_2)));
   }
};

template <typename floating_type>
inline void Vector<floating_type>::fusedProject(Vector<floating_type>& alpha, const floating_type lambda_1, const floating_type lambda_2,
      const int itermax) {
   floating_type* pr_alpha= alpha.rawX();
   floating_type* pr_beta=_X;
   const INTM K = alpha.n();

   floating_type total_alpha =alpha.sum();
   /// Modification of beta
   for (INTM i = K-2; i>=0; --i) 
      pr_beta[i]+=pr_beta[i+1];

   for (INTM i = 0; i<itermax; ++i) {
      floating_type sum_alpha=0;
      floating_type sum_diff = 0;
      /// Update first coordinate
      floating_type gamma_old=pr_alpha[0];
      pr_alpha[0]=(K*gamma_old+pr_beta[0]-
            total_alpha)/K;
      floating_type diff = pr_alpha[0]-gamma_old;
      sum_diff += diff;
      sum_alpha += pr_alpha[0];
      total_alpha +=K*diff;

      /// Update alpha_j
      for (INTM j = 1; j<K; ++j) {
         pr_alpha[j]+=sum_diff;
         floating_type gamma_old=pr_alpha[j]-pr_alpha[j-1];
         floating_type gamma_new=softThrs((K-j)*gamma_old+pr_beta[j]-
               (total_alpha-sum_alpha),lambda_2)/(K-j);
         pr_alpha[j]=pr_alpha[j-1]+gamma_new;
         floating_type diff = gamma_new-gamma_old;
         sum_diff += diff;
         sum_alpha+=pr_alpha[j];
         total_alpha +=(K-j)*diff;
      }
   }
   alpha.softThrshold(lambda_1);

};

/// sort the vector
template <typename floating_type>
inline void Vector<floating_type>::sort(const bool mode) {
   if (mode) {
      lasrt<floating_type>(incr,_n,_X);
   } else {
      lasrt<floating_type>(decr,_n,_X);
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
   quick_sort(key.rawX(),_X,(INTM)0,_n-1,mode);
};


template <typename floating_type>
inline void Vector<floating_type>::sort2(Vector<floating_type>& out, Vector<INTM>& key, const bool mode) const {
   out.copy(*this);
   out.sort2(key,mode);
}

template <typename floating_type>
inline void Vector<floating_type>::applyBayerPattern(const int offset) {
   INTM sizePatch=_n/3;
   INTM n = static_cast<INTM>(sqrt(static_cast<floating_type>(sizePatch)));
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
template <typename floating_type> inline void Vector<floating_type>::toSparse(
      SpVector<floating_type>& vec) const {
   INTM L=0;
   floating_type* v = vec._v;
   INTM* r = vec._r;
   for (INTM i = 0; i<_n; ++i) {
      if (_X[i] != floating_type()) {
         v[L]=_X[i];
         r[L++]=i;
      }
   }
   vec._L=L;
};


template <typename floating_type>
inline void Vector<floating_type>::copyMask(Vector<floating_type>& out, Vector<bool>& mask) const {
   out.resize(_n);
   INTM pointer=0;
   for (INTM i = 0; i<_n; ++i) {
      if (mask[i])
         out[pointer++]=_X[i];
   }
   out.setn(pointer);
};

template <typename floating_type>
inline void Matrix<floating_type>::copyMask(Matrix<floating_type>& out, Vector<bool>& mask) const {
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
template <typename floating_type, typename I> SpMatrix<floating_type,I>::SpMatrix(floating_type* v, I* r, I* pB, I* pE,
      I m, I n, I nzmax) :
   _externAlloc(true), _v(v), _r(r), _pB(pB), _pE(pE), _m(m), _n(n), _nzmax(nzmax)
{ };

/// Constructor, new m x n matrix, with at most nzmax non-zeros values
template <typename floating_type, typename I> SpMatrix<floating_type,I>::SpMatrix(I m, I n, I nzmax) :
   _externAlloc(false), _m(m), _n(n), _nzmax(nzmax) {
#pragma omp critical
      {
         _v=new floating_type[nzmax];
         _r=new I[nzmax];
         _pB=new I[_n+1];
      }
      _pE=_pB+1;
   };

/// Empty constructor
template <typename floating_type, typename I> SpMatrix<floating_type,I>::SpMatrix() :
   _externAlloc(true), _v(NULL), _r(NULL), _pB(NULL), _pE(NULL),
   _m(0),_n(0),_nzmax(0) { };


template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::copy(const SpMatrix<floating_type,I>& mat) {
   this->resize(mat._m,mat._n,mat._nzmax);
   memcpy(_v,mat._v,_nzmax*sizeof(floating_type));
   memcpy(_r,mat._r,_nzmax*sizeof(I));
   memcpy(_pB,mat._pB,(_n+1)*sizeof(I));
}


/// Destructor
template <typename floating_type, typename I> SpMatrix<floating_type,I>::~SpMatrix() {
   clear();
};

/// reference the column i Io vec
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::refCol(I i, 
      SpVector<floating_type,I>& vec) const {
   if (vec._nzmax > 0) vec.clear();
   vec._v=_v+_pB[i];
   vec._r=_r+_pB[i];
   vec._externAlloc=true;
   vec._L=_pE[i]-_pB[i];
   vec._nzmax=vec._L;
};

/// print the sparse matrix
template<typename floating_type, typename I> inline void SpMatrix<floating_type,I>::print(const string& name) const {
   cerr << name;
   cerr << _m << " x " << _n << " , " << _nzmax;
   for (I i = 0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         cerr << "(" <<_r[j] << "," << i << ") = " << _v[j];
      }
   }
};

template<typename floating_type, typename I>
inline floating_type SpMatrix<floating_type,I>::operator[](const I index) const {
   const I num_col=(index/_m);
   const I num_row=index -num_col*_m;
   floating_type val = 0;
   for (I j = _pB[num_col]; j<_pB[num_col+1]; ++j) {
      if (_r[j]==num_row) {
         val=_v[j];
         break;
      }
   }
   return val;
};
template<typename floating_type, typename I>
void SpMatrix<floating_type,I>::getData(Vector<floating_type>& data, const I index) const {
   data.resize(_m);
   data.setZeros();
   for (I i = _pB[index]; i< _pB[index+1]; ++i) 
      data[_r[i]]=_v[i];
};

template <typename floating_type, typename I>
void SpMatrix<floating_type,I>::setData(floating_type* v, I* r, I* pB, I* pE, I m, I n, I nzmax) {
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
template <typename floating_type, typename I> inline floating_type SpMatrix<floating_type,I>::asum() const {
   return cblas_asum<floating_type>(_pB[_n],_v,1);
};

/// compute the sum of the matrix elements
template <typename floating_type, typename I> inline floating_type SpMatrix<floating_type,I>::normFsq() const {
   return cblas_dot<floating_type>(_pB[_n],_v,1,_v,1);
};

template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::add_direct(const SpMatrix<floating_type,I>& mat, const floating_type a) {
   Vector<floating_type> v2(mat._v,mat._nzmax);
   Vector<floating_type> v1(_v,_nzmax);
   v1.add(v2,a);
}

template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::copy_direct(const SpMatrix<floating_type,I>& mat) {
   Vector<floating_type> v2(mat._v,_pB[_n]);
   Vector<floating_type> v1(_v,_pB[_n]);
   v1.copy(v2);
}

template <typename floating_type, typename I>
inline floating_type SpMatrix<floating_type,I>::dot_direct(const SpMatrix<floating_type,I>& mat) const {
   Vector<floating_type> v2(mat._v,_pB[_n]);
   Vector<floating_type> v1(_v,_pB[_n]);
   return v1.dot(v2);
}

/// clear the matrix
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::clear() {
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
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::resize(const I m, 
      const I n, const I nzmax) {
   if (n == _n && m == _m && nzmax == _nzmax) return;
   this->clear();
   _n=n;
   _m=m;
   _nzmax=nzmax;
   _externAlloc=false;
#pragma omp critical
   {
      _v = new floating_type[nzmax];
      _r = new I[nzmax];
      _pB = new I[_n+1];
   }
   _pE = _pB+1;
   for (I i = 0; i<=_n; ++i) _pB[i]=0;
};

/// resize the matrix
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::scal(const floating_type a) const {
   cblas_scal<floating_type>(_pB[_n],a,_v,1);
};

///// resize the matrix
template <typename floating_type, typename I> inline floating_type SpMatrix<floating_type,I>::abs_mean() const {
   Vector<floating_type> vec(_v,_pB[_n]);
   return vec.abs_mean();
};


/// y <- A'*x
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::multTrans(const Vector<floating_type>& x, Vector<floating_type>& y,
      const floating_type alpha, const floating_type beta) const {
   y.resize(_n);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   const floating_type* prX = x.rawX();
#pragma omp parallel for
   for (I i = 0; i<_n; ++i) {
      floating_type sum=floating_type();
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         sum+=_v[j]*prX[_r[j]];
      }
      y[i] += alpha*sum;
   }
};

/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::multTrans(const SpVector<floating_type,I>& x, Vector<floating_type>& y, 
      const floating_type alpha, const floating_type beta) const {
   y.resize(_n);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   floating_type* prY = y.rawX();
   SpVector<floating_type,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      prY[i] += alpha*x.dot(col);
   }
};


/// y <- A*x
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::mult(const Vector<floating_type>& x, Vector<floating_type>& y,
      const floating_type alpha, const floating_type beta) const {
   y.resize(_m);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   const floating_type* prX = x.rawX();
   for (I i = 0; i<_n; ++i) {
      floating_type sca=alpha* prX[i];
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         y[_r[j]] += sca*_v[j];
      }
   }
};


/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::mult(const SpVector<floating_type,I>& x, Vector<floating_type>& y, 
      const floating_type alpha, const floating_type beta) const {
   y.resize(_m);
   if (beta) {
      y.scal(beta);
   } else {
      y.setZeros();
   }
   floating_type* prY = y.rawX();
   for (I i = 0; i<x.L(); ++i) {
      I ind=x.r(i);
      floating_type val = alpha * x.v(i);
      for (I j = _pB[ind]; j<_pE[ind]; ++j) {
         prY[_r[j]] += val *_v[j];
      }
   }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::mult(const Matrix<floating_type>& B, Matrix<floating_type>& C, 
      const bool transA, const bool transB,
      const floating_type a, const floating_type b) const {
   if (transA) {
      if (transB) {
         C.resize(_n,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<floating_type,I> tmp;
         Vector<floating_type> row(B.m());
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
         SpVector<floating_type,I> tmp;
         Vector<floating_type> row(B.n());
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
         Vector<floating_type> row(B.n());
         Vector<floating_type> col;
         for (I i = 0; i<B.m(); ++i) {
            B.copyRow(i,row);
            C.refCol(i,col);
            this->mult(row,col,a,floating_type(1.0));
         }
      } else {
         C.resize(_m,B.n());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<floating_type> colB;
         Vector<floating_type> colC;
         for (I i = 0; i<B.n(); ++i) {
            B.refCol(i,colB);
            C.refCol(i,colC);
            this->mult(colB,colC,a,floating_type(1.0));
         }
      }
   }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::mult(const SpMatrix<floating_type,I>& B, Matrix<floating_type>& C, 
      const bool transA, const bool transB,
      const floating_type a, const floating_type b) const {
   if (transA) {
      if (transB) {
         C.resize(_n,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         SpVector<floating_type,I> tmp;
         Vector<floating_type> row(B.m());
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
         SpVector<floating_type,I> tmp;
         Vector<floating_type> row(B.n());
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
         SpVector<floating_type,I> colB;
         SpVector<floating_type,I> colA;
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
         SpVector<floating_type,I> colB;
         Vector<floating_type> colC;
         for (I i = 0; i<B.n(); ++i) {
            B.refCol(i,colB);
            C.refCol(i,colC);
            this->mult(colB,colC,a);
         }
      }
   }
};

/// perform C = a*B*A + b*C, possibly transposing A or B.
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::multSwitch(const Matrix<floating_type>& B, Matrix<floating_type>& C, 
      const bool transA, const bool transB,
      const floating_type a, const floating_type b) const {
   B.mult(*this,C,transB,transA,a,b);
};

template <typename floating_type, typename I>
inline floating_type SpMatrix<floating_type,I>::dot(const Matrix<floating_type>& x) const {
   floating_type sum=0;
   for (I i = 0; i<_n; ++i)
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         sum+=_v[j]*x(_r[j],j);
      }
   return sum;
};


template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::copyRow(const I ind, Vector<floating_type>& x) const {
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

template <typename floating_type, typename I> 
inline void SpMatrix<floating_type,I>::addVecToCols(
      const Vector<floating_type>& vec, const floating_type a) {
   const floating_type* pr_vec = vec.rawX();
   if (isEqual(a,floating_type(1.0))) {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += pr_vec[_r[j]];
   } else {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += a*pr_vec[_r[j]];
   }
};

template <typename floating_type, typename I> 
inline void SpMatrix<floating_type,I>::addVecToColsWeighted(
      const Vector<floating_type>& vec, const floating_type* weights, const floating_type a) {
   const floating_type* pr_vec = vec.rawX();
   if (isEqual(a,floating_type(1.0))) {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += pr_vec[_r[j]]*weights[j-_pB[i]];
   } else {
      for (I i = 0; i<_n; ++i) 
         for (I j = _pB[i]; j<_pE[i]; ++j) 
            _v[j] += a*pr_vec[_r[j]]*weights[j-_pB[i]];
   }
};

template <typename floating_type, typename I> 
inline void SpMatrix<floating_type,I>::sum_cols(Vector<floating_type>& sum) const {
   sum.resize(_m);
   sum.setZeros();
   SpVector<floating_type,I> tmp;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,tmp);
      sum.add(tmp);
   }
};

/// aat <- A*A'
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::AAt(Matrix<floating_type>& aat) const {
   I i,j,k;
   I K=_m;
   I M=_n;

   /* compute alpha alpha^floating_type */
   aat.resize(K,K);
   int NUM_THREADS=init_omp(MAX_THREADS);
   floating_type* aatT=new floating_type[NUM_THREADS*K*K];
   for (j = 0; j<NUM_THREADS*K*K; ++j) aatT[j]=floating_type();

#pragma omp parallel for private(i,j,k)
   for (i = 0; i<M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      floating_type* write_area=aatT+numT*K*K;
      for (j = _pB[i]; j<_pE[i]; ++j) {
         for (k = _pB[i]; k<=j; ++k) {
            write_area[_r[j]*K+_r[k]]+=_v[j]*_v[k];
         }
      }
   }

   cblas_copy<floating_type>(K*K,aatT,1,aat._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<floating_type>(K*K,1.0,aatT+K*K*i,1,aat._X,1);
   aat.fillSymmetric();
   delete[](aatT);
}

template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::XtX(Matrix<floating_type>& XtX) const {
   XtX.resize(_n,_n);
   XtX.setZeros();
   SpVector<floating_type,I> col;
   Vector<floating_type> col_out;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      XtX.refCol(i,col_out);
      this->multTrans(col,col_out);
   }
};


/// aat <- A(:,indices)*A(:,indices)'
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::AAt(Matrix<floating_type>& aat,
      const Vector<I>& indices) const {
   I i,j,k;
   I K=_m;
   I M=indices.n();

   /* compute alpha alpha^floating_type */
   aat.resize(K,K);
   int NUM_THREADS=init_omp(MAX_THREADS);
   floating_type* aatT=new floating_type[NUM_THREADS*K*K];
   for (j = 0; j<NUM_THREADS*K*K; ++j) aatT[j]=floating_type();

#pragma omp parallel for private(i,j,k)
   for (i = 0; i<M; ++i) {
      I ii = indices[i];
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      floating_type* write_area=aatT+numT*K*K;
      for (j = _pB[ii]; j<_pE[ii]; ++j) {
         for (k = _pB[ii]; k<=j; ++k) {
            write_area[_r[j]*K+_r[k]]+=_v[j]*_v[k];
         }
      }
   }

   cblas_copy<floating_type>(K*K,aatT,1,aat._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<floating_type>(K*K,1.0,aatT+K*K*i,1,aat._X,1);
   aat.fillSymmetric();
   delete[](aatT);
}

/// aat <- sum_i w_i A(:,i)*A(:,i)'
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::wAAt(const Vector<floating_type>& w,
      Matrix<floating_type>& aat) const {
   I i,j,k;
   I K=_m;
   I M=_n;

   /* compute alpha alpha^floating_type */
   aat.resize(K,K);
   int NUM_THREADS=init_omp(MAX_THREADS);
   floating_type* aatT=new floating_type[NUM_THREADS*K*K];
   for (j = 0; j<NUM_THREADS*K*K; ++j) aatT[j]=floating_type();

#pragma omp parallel for private(i,j,k)
   for (i = 0; i<M; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      floating_type* write_area=aatT+numT*K*K;
      for (j = _pB[i]; j<_pE[i]; ++j) {
         for (k = _pB[i]; k<=j; ++k) {
            write_area[_r[j]*K+_r[k]]+=w._X[i]*_v[j]*_v[k];
         }
      }
   }

   cblas_copy<floating_type>(K*K,aatT,1,aat._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<floating_type>(K*K,1.0,aatT+K*K*i,1,aat._X,1);
   aat.fillSymmetric();
   delete[](aatT);
}

/// XAt <- X*A'
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::XAt(const Matrix<floating_type>& X,
      Matrix<floating_type>& XAt) const {
   I j,i;
   I n=X._m;
   I K=_m;
   I M=_n;

   XAt.resize(n,K);
   /* compute X alpha^floating_type */
//   int NUM_THREADS=init_omp(MAX_THREADS);
   //floating_type* XatT=new floating_type[NUM_THREADS*n*K];
   //for (j = 0; j<NUM_THREADS*n*K; ++j) XatT[j]=floating_type();

//#pragma omp parallel for private(i,j)
   for (i = 0; i<M; ++i) {
//#ifdef _OPENMP
//      int numT=omp_get_thread_num();
//#else
//      int numT=0;
//#endif
//      floating_type* write_area=XatT+numT*n*K;
      for (j = _pB[i]; j<_pE[i]; ++j) {
         cblas_axpy<floating_type>(n,_v[j],X._X+i*n,1,XAt._X+_r[j]*n,1);
      }
   }
 //  cblas_copy<floating_type>(n*K,XatT,1,XAt._X,1);
//   for (i = 1; i<NUM_THREADS; ++i) 
//      cblas_axpy<floating_type>(n*K,1.0,XatT+n*K*i,1,XAt._X,1);
//   delete[](XatT);
};

/// XAt <- X(:,indices)*A(:,indices)'
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::XAt(const Matrix<floating_type>& X,
      Matrix<floating_type>& XAt, const Vector<I>& indices) const {
   I j,i;
   I n=X._m;
   I K=_m;
   I M=indices.n();

   XAt.resize(n,K);
   /* compute X alpha^floating_type */
   int NUM_THREADS=init_omp(MAX_THREADS);
   floating_type* XatT=new floating_type[NUM_THREADS*n*K];
   for (j = 0; j<NUM_THREADS*n*K; ++j) XatT[j]=floating_type();

#pragma omp parallel for private(i,j)
   for (i = 0; i<M; ++i) {
      I ii = indices[i];
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      floating_type* write_area=XatT+numT*n*K;
      for (j = _pB[ii]; j<_pE[ii]; ++j) {
         cblas_axpy<floating_type>(n,_v[j],X._X+i*n,1,write_area+_r[j]*n,1);
      }
   }

   cblas_copy<floating_type>(n*K,XatT,1,XAt._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<floating_type>(n*K,1.0,XatT+n*K*i,1,XAt._X,1);
   delete[](XatT);
};

/// XAt <- sum_i w_i X(:,i)*A(:,i)'
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::wXAt(const Vector<floating_type>& w,
      const Matrix<floating_type>& X, Matrix<floating_type>& XAt, const int numThreads) const {
   I j,l,i;
   I n=X._m;
   I K=_m;
   I M=_n;
   I Mx = X._n;
   I numRepX= M/Mx;
   assert(numRepX*Mx == M);
   XAt.resize(n,K);
   /* compute X alpha^floating_type */
   int NUM_THREADS=init_omp(numThreads);
   floating_type* XatT=new floating_type[NUM_THREADS*n*K];
   for (j = 0; j<NUM_THREADS*n*K; ++j) XatT[j]=floating_type();

#pragma omp parallel for private(i,j,l)
   for (i = 0; i<Mx; ++i) {
#ifdef _OPENMP
      int numT=omp_get_thread_num();
#else
      int numT=0;
#endif
      floating_type * write_area=XatT+numT*n*K;
      for (l = 0; l<numRepX; ++l) {
         I ind=numRepX*i+l;
         if (w._X[ind] != 0)
            for (j = _pB[ind]; j<_pE[ind]; ++j) {
               cblas_axpy<floating_type>(n,w._X[ind]*_v[j],X._X+i*n,1,write_area+_r[j]*n,1);
            }
      }
   }

   cblas_copy<floating_type>(n*K,XatT,1,XAt._X,1);
   for (i = 1; i<NUM_THREADS; ++i) 
      cblas_axpy<floating_type>(n*K,1.0,XatT+n*K*i,1,XAt._X,1);
   delete[](XatT);
};

/// copy the sparse matrix into a dense matrix
template<typename floating_type, typename I> inline void SpMatrix<floating_type,I>::toFull(Matrix<floating_type>& matrix) const {
   matrix.resize(_m,_n);
   matrix.setZeros();
   floating_type* out = matrix._X;
   for (I i=0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         out[i*_m+_r[j]]=_v[j];
      }
   }
};

/// copy the sparse matrix into a full dense matrix
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::toFullTrans(
      Matrix<floating_type>& matrix) const {
   matrix.resize(_n,_m);
   matrix.setZeros();
   floating_type* out = matrix._X;
   for (I i=0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         out[i+_r[j]*_n]=_v[j];
      }
   }
};


/// use the data from v, r for _v, _r
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::convert(const Matrix<floating_type>&vM, 
      const Matrix<I>& rM, const I K) {
   const I M = rM.n();
   const I L = rM.m();
   const I* r = rM.X();
   const floating_type* v = vM.X();
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
template <typename floating_type, typename I> inline void SpMatrix<floating_type,I>::convert2(
      const Matrix<floating_type>&vM, const Vector<I>& rv, const I K) {
   const I M = vM.n();
   const I L = vM.m();
   I* r = rv.rawX();
   const floating_type* v = vM.X();
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
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::normalize() {
   SpVector<floating_type,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      const floating_type norm = col.nrm2sq();
      if (norm > 1e-10)
         col.scal(floating_type(1.0)/col.nrm2sq());
   }
};

/// returns the l2 norms ^2 of the columns
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::normalize_rows() {
   Vector<floating_type> norms(_m);
   norms.setZeros();
   for (I i = 0; i<_n; ++i) {
      for (I j = _pB[i]; j<_pE[i]; ++j) {
         norms[_r[j]] += _v[j]*_v[j];
      }
   }
   norms.Sqrt();
   for (I i = 0; i<_m; ++i) 
      norms[i] = norms[i] < 1e-10 ? floating_type(1.0) : floating_type(1.0)/norms[i];
   for (I i = 0; i<_n; ++i) 
      for (I j = _pB[i]; j<_pE[i]; ++j) 
         _v[j] *= norms[_r[j]];
};




/// returns the l2 norms ^2 of the columns
template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::norm_2sq_cols(Vector<floating_type>& norms) const {
   norms.resize(_n);
   SpVector<floating_type,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      norms[i] = col.nrm2sq();
   }
};


template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::norm_0_cols(Vector<floating_type>& norms) const {
   norms.resize(_n);
   SpVector<floating_type,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      norms[i] = static_cast<floating_type>(col.length());
   }
};

template <typename floating_type, typename I>
inline void SpMatrix<floating_type,I>::norm_1_cols(Vector<floating_type>& norms) const {
   norms.resize(_n);
   SpVector<floating_type,I> col;
   for (I i = 0; i<_n; ++i) {
      this->refCol(i,col);
      norms[i] =col.asum();
   }
};


/* ***************************
 * Implementation of SpVector 
 * ***************************/


/// Constructor, of the sparse vector of size L.
template <typename floating_type, typename I> SpVector<floating_type,I>::SpVector(floating_type* v, I* r, I L, I nzmax) :
   _externAlloc(true), _v(v), _r(r), _L(L), _nzmax(nzmax)  { };

/// Constructor, allocates nzmax slots
template <typename floating_type, typename I> SpVector<floating_type,I>::SpVector(I nzmax) :
   _externAlloc(false), _L(0), _nzmax(nzmax) {
#pragma omp critical
      {
         _v = new floating_type[nzmax];
         _r = new I[nzmax];
      }
   };

/// Empty constructor
template <typename floating_type, typename I> SpVector<floating_type,I>::SpVector() : _externAlloc(true), _v(NULL), _r(NULL), _L(0),
   _nzmax(0) { };


/// Destructor
template <typename floating_type, typename I> SpVector<floating_type,I>::~SpVector() { clear(); };


/// computes the sum of the magnitude of the elements
template <typename floating_type, typename I> inline floating_type SpVector<floating_type,I>::asum() const {
   return cblas_asum<floating_type>(_L,_v,1);
};

/// computes the l2 norm ^2 of the vector
template <typename floating_type, typename I> inline floating_type SpVector<floating_type,I>::nrm2sq() const {
   return cblas_dot<floating_type>(_L,_v,1,_v,1);
};

/// computes the l2 norm of the vector
template <typename floating_type, typename I> inline floating_type SpVector<floating_type,I>::nrm2() const {
   return cblas_nrm2<floating_type>(_L,_v,1);
};

/// computes the l2 norm of the vector
template <typename floating_type, typename I> inline floating_type SpVector<floating_type,I>::fmaxval() const {
   Vector<floating_type> tmp(_v,_L);
   return tmp.fmaxval();
};

/// print the vector to std::cerr
template <typename floating_type, typename I> inline void SpVector<floating_type,I>::print(const string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _nzmax << std::endl;
   for (I i = 0; i<_L; ++i)
      cerr << "(" <<_r[i] << ", " <<  _v[i] << ")";
};

/// create a reference on the vector r
template <typename floating_type, typename I> inline void SpVector<floating_type,I>::refIndices(
      Vector<I>& indices) const {
   indices.setPointer(_r,_L);   
};

template <typename floating_type, typename I> inline void SpVector<floating_type,I>::getIndices(Vector<int>& indices) const {
//   indices.resize(_L);
   indices.setn(_L);
   for (int ii=0; ii<_L; ++ii)
      indices[ii]=_r[ii]; 
};

/// creates a reference on the vector val
template <typename floating_type, typename I> inline void SpVector<floating_type,I>::refVal(
      Vector<floating_type>& val) const {
   val.setPointer(_v,_L);   
};

/// a <- a.^2
template <typename floating_type, typename I> inline void SpVector<floating_type,I>::sqr() {
   vSqr<floating_type>(_L,_v,_v);
};

template <typename floating_type, typename I>
inline void SpVector<floating_type,I>::scal(const floating_type a) {
   cblas_scal<floating_type>(_L,a,_v,1);
};

template <typename floating_type, typename I>
inline floating_type SpVector<floating_type,I>::dot(const SpVector<floating_type,I>& vec) const {
   floating_type sum=floating_type();
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

template <typename floating_type, typename I>
inline floating_type SpVector<floating_type,I>::dot(const Vector<floating_type>& vec) const {
   //return cblas_doti(_L,_v,_r,vec.rawX());
   floating_type sum=floating_type();
   for (int countI=0; countI < _L; ++countI)
      sum+=_v[countI]*vec[_r[countI]];
   return sum;
};

/// clears the vector
template <typename floating_type, typename I> inline void SpVector<floating_type,I>::clear() {
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
template <typename floating_type, typename I> inline void SpVector<floating_type,I>::resize(const I nzmax) {
   if (_nzmax != nzmax) {
      clear();
      _nzmax=nzmax;
      _L=0;
      _externAlloc=false;
#pragma omp critical
      {
         _v=new floating_type[nzmax];
         _r=new I[nzmax];
      }
   }
};

template <typename floating_type, typename I> void inline SpVector<floating_type,I>::toSpMatrix(
      SpMatrix<floating_type,I>& out, const I m, const I n) const {
   out.resize(m,n,_L);
   cblas_copy<floating_type>(_L,_v,1,out._v,1);
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

template <typename floating_type, typename I> void inline SpVector<floating_type,I>::toFull(Vector<floating_type>& out)
   const {
      out.setZeros();
      floating_type* X = out.rawX();
      for (I i = 0; i<_L; ++i)
         X[_r[i]]=_v[i];
   };


/* ************************************
 * Implementation of the class Matrix 
 * ************************************/

/// Constructor with existing data X of an m x n matrix
template <typename floating_type> Matrix<floating_type>::Matrix(floating_type* X, INTM m, INTM n) :
   _externAlloc(true), _X(X), _m(m), _n(n) {  };


/// Constructor for a new m x n matrix
template <typename floating_type> Matrix<floating_type>::Matrix(INTM m, INTM n) :
   _externAlloc(false), _m(m), _n(n)  {
#pragma omp critical
      {
         _X= new floating_type[_n*_m];
      }
   };

/// Empty constructor
template <typename floating_type> Matrix<floating_type>::Matrix() :
   _externAlloc(false), _X(NULL), _m(0), _n(0) { };

/// Destructor
template <typename floating_type> Matrix<floating_type>::~Matrix() {
   clear();
};

/// Return a modifiable reference to X(i,j)
template <typename floating_type> inline floating_type& Matrix<floating_type>::operator()(const INTM i, const INTM j) {
   return _X[j*_m+i];
};

/// Return the value X(i,j)
template <typename floating_type> inline floating_type Matrix<floating_type>::operator()(const INTM i, const INTM j) const {
   return _X[j*_m+i];
};

/// Print the matrix to std::cout
template <typename floating_type> inline void Matrix<floating_type>::print(const string& name) const {
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
template <typename floating_type> inline void Matrix<floating_type>::dump(const string& name) const {
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
template <typename floating_type> inline void Matrix<floating_type>::copyCol(const INTM i, Vector<floating_type>& x) const {
   assert(i >= 0 && i<_n);
   x.resize(_m);
   cblas_copy<floating_type>(_m,_X+i*_m,1,x._X,1);
};

/// Copy the column i INTMo x
template <typename floating_type> inline void Matrix<floating_type>::copyRow(const INTM i, Vector<floating_type>& x) const {
   assert(i >= 0 && i<_m);
   x.resize(_n);
   cblas_copy<floating_type>(_n,_X+i,_m,x._X,1);
};

/// Copy the column i INTMo x
template <typename floating_type> inline void Matrix<floating_type>::scalRow(const INTM i, const floating_type s) const {
   assert(i >= 0 && i<_m);
   for (int ii=0; ii<_n; ++ii)
      _X[i+ii*_m] *= s;
};


/// Copy the column i INTMo x
template <typename floating_type> inline void Matrix<floating_type>::copyToRow(const INTM i, const Vector<floating_type>& x) {
   assert(i >= 0 && i<_m);
   cblas_copy<floating_type>(_n,x._X,1,_X+i,_m);
};

/// Copy the column i INTMo x
template <typename floating_type> inline void Matrix<floating_type>::extract_rawCol(const INTM i, floating_type* x) const {
   assert(i >= 0 && i<_n);
   cblas_copy<floating_type>(_m,_X+i*_m,1,x,1);
};

/// Copy the column i INTMo x
template <typename floating_type> inline void Matrix<floating_type>::add_rawCol(const INTM i, floating_type* x, const floating_type a) const {
   assert(i >= 0 && i<_n);
   cblas_axpy<floating_type>(_m,a,_X+i*_m,1,x,1);
};

/// Copy the column i INTMo x
template <typename floating_type> inline void Matrix<floating_type>::getData(Vector<floating_type>& x, const INTM i) const {
   this->copyCol(i,x);
};

/// Reference the column i into the vector x
template <typename floating_type> inline void Matrix<floating_type>::refCol(INTM i, Vector<floating_type>& x) const {
   assert(i >= 0 && i<_n);
   x.clear();
   x._X=_X+i*_m;
   x._n=_m;
   x._externAlloc=true; 
};

/// Reference the column i to i+n INTMo the Matrix mat
template <typename floating_type> inline void Matrix<floating_type>::refSubMat(INTM i, INTM n, Matrix<floating_type>& mat) const {
   mat.setData(_X+i*_m,_m,n);
}

/// Check wether the columns of the matrix are normalized or not
template <typename floating_type> inline bool Matrix<floating_type>::isNormalized() const {
   for (INTM i = 0; i<_n; ++i) {
      floating_type norm=cblas_nrm2<floating_type>(_m,_X+_m*i,1);
      if (fabs(norm - 1.0) > 1e-6) return false;
   }
   return true;
};

/// clean a dictionary matrix
template <typename floating_type>
inline void Matrix<floating_type>::clean() {
   this->normalize();
   Matrix<floating_type> G;
   this->XtX(G);
   floating_type* prG = G._X;
   /// remove the diagonal
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = i+1; j<_n; ++j) {
         if (prG[i*_n+j] > 0.99) {
            // remove nasty column j and put random values inside
            Vector<floating_type> col;
            this->refCol(j,col);
            col.setAleat();
            col.normalize();
         }
      }
   }
};

/// return the 1D-index of the value of greatest magnitude
template <typename floating_type> inline INTM Matrix<floating_type>::fmax() const {
   return cblas_iamax<floating_type>(_n*_m,_X,1);
};

/// return the value of greatest magnitude
template <typename floating_type> inline floating_type Matrix<floating_type>::fmaxval() const {
   return _X[cblas_iamax<floating_type>(_n*_m,_X,1)];
};


/// return the 1D-index of the value of lowest magnitude
template <typename floating_type> inline INTM Matrix<floating_type>::fmin() const {
   return cblas_iamin<floating_type>(_n*_m,_X,1);
};

/// extract a sub-matrix of a symmetric matrix
template <typename floating_type> inline void Matrix<floating_type>::subMatrixSym(
      const Vector<INTM>& indices, Matrix<floating_type>& subMatrix) const {
   INTM L = indices.n();
   subMatrix.resize(L,L);
   floating_type* out = subMatrix._X;
   INTM* rawInd = indices.rawX();
   for (INTM i = 0; i<L; ++i)
      for (INTM j = 0; j<=i; ++j)
         out[i*L+j]=_X[rawInd[i]*_n+rawInd[j]];
   subMatrix.fillSymmetric();
};

/// Resize the matrix
template <typename floating_type> inline void Matrix<floating_type>::resize(INTM m, INTM n, const bool set_zeros) {
   if (_n==n && _m==m) return;
   clear();
   _n=n;
   _m=m;
   _externAlloc=false;
#pragma omp critical
   {
      _X=new floating_type[_n*_m];
   }
   if (set_zeros)
      setZeros();
};

/// Change the data in the matrix
template <typename floating_type> inline void Matrix<floating_type>::setData(floating_type* X, INTM m, INTM n) {
   clear();
   _X=X;
   _m=m;
   _n=n;
   _externAlloc=true;
};

/// Set all the values to zero
template <typename floating_type> inline void Matrix<floating_type>::setZeros() {
   memset(_X,0,_n*_m*sizeof(floating_type));
};

/// Set all the values to a scalar
template <typename floating_type> inline void Matrix<floating_type>::set(const floating_type a) {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]=a;
};

/// Clear the matrix
template <typename floating_type> inline void Matrix<floating_type>::clear() {
   if (!_externAlloc) delete[](_X);
   _n=0;
   _m=0;
   _X=NULL;
   _externAlloc=true;
};

/// Put white Gaussian noise in the matrix 
template <typename floating_type> inline void Matrix<floating_type>::setAleat() {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]=normalDistrib<floating_type>();
};

/// set the matrix to the identity
template <typename floating_type> inline void Matrix<floating_type>::eye() {
   this->setZeros();
   for (INTM i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] = floating_type(1.0);
};

/// Normalize all columns to unit l2 norm
template <typename floating_type> inline void Matrix<floating_type>::normalize() {
   //floating_type constant = 1.0/sqrt(_m);
   for (INTM i = 0; i<_n; ++i) {
      floating_type norm=cblas_nrm2<floating_type>(_m,_X+_m*i,1);
      if (norm > 1e-10) {
         floating_type invNorm=1.0/norm;
         cblas_scal<floating_type>(_m,invNorm,_X+_m*i,1);
      }  else {
         // for (INTM j = 0; j<_m; ++j) _X[_m*i+j]=constant;
         Vector<floating_type> d;
         this->refCol(i,d);
         d.setAleat();
         d.normalize();
      } 
   }
};

/// Normalize all columns which l2 norm is greater than one.
template <typename floating_type> inline void Matrix<floating_type>::normalize2() {
   for (INTM i = 0; i<_n; ++i) {
      floating_type norm=cblas_nrm2<floating_type>(_m,_X+_m*i,1);
      if (norm > 1.0) {
         floating_type invNorm=1.0/norm;
         cblas_scal<floating_type>(_m,invNorm,_X+_m*i,1);
      } 
   }
};

/// center the matrix
template <typename floating_type> inline void Matrix<floating_type>::center() {
   for (INTM i = 0; i<_n; ++i) {
      Vector<floating_type> col;
      this->refCol(i,col);
      floating_type sum = col.sum();
      col.add(-sum/static_cast<floating_type>(_m));
   }
};

/// center the matrix
template <typename floating_type> inline void Matrix<floating_type>::center_rows() {
   Vector<floating_type> mean_rows(_m);
   mean_rows.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         mean_rows[j] += _X[i*_m+j];
   mean_rows.scal(floating_type(1.0)/_n);
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         _X[i*_m+j] -= mean_rows[j];
};

/// center the matrix
template <typename floating_type> inline void Matrix<floating_type>::normalize_rows() {
   Vector<floating_type> norm_rows(_m);
   norm_rows.setZeros();
   for (INTM i = 0; i<_n; ++i)
      for (INTM j = 0; j<_m; ++j)
         norm_rows[j] += _X[i*_m+j]*_X[i*_m+j];
   for (INTM j = 0; j<_m; ++j)
      norm_rows[j]  = norm_rows[j] < floating_type(1e-10) ? floating_type(1e-10) : floating_type(1.0)/sqrt(norm_rows[j]);
   this->multDiagLeft(norm_rows);
};

/// center the matrix and keep the center values
template <typename floating_type> inline void Matrix<floating_type>::center(Vector<floating_type>& centers) {
   centers.resize(_n);
   for (INTM i = 0; i<_n; ++i) {
      Vector<floating_type> col;
      this->refCol(i,col);
      floating_type sum = col.sum()/static_cast<floating_type>(_m);
      centers[i]=sum;
      col.add(-sum);
   }
};

/// scale the matrix by the a
template <typename floating_type> inline void Matrix<floating_type>::scal(const floating_type a) {
    cblas_scal<floating_type>(_n*_m,a,_X,1);
};

/// make a copy of the matrix mat in the current matrix
template <typename floating_type> inline void Matrix<floating_type>::copy(const Matrix<floating_type>& mat) {
   if (_X != mat._X) {
      resize(mat._m,mat._n);
      //   cblas_copy<floating_type>(_m*_n,mat._X,1,_X,1);
      memcpy(_X,mat._X,_m*_n*sizeof(floating_type));
   }
};

/// make a copy of the matrix mat in the current matrix
template <typename floating_type> inline void Matrix<floating_type>::copyRef(const Matrix<floating_type>& mat) {
   this->setData(mat.rawX(),mat.m(),mat.n());
};

/// make the matrix symmetric by copying the upper-right part
/// INTMo the lower-left part
template <typename floating_type> inline void Matrix<floating_type>::fillSymmetric() {
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j =0; j<i; ++j) {
         _X[j*_m+i]=_X[i*_m+j];
      }
   }
};
template <typename floating_type> inline void Matrix<floating_type>::fillSymmetric2() {
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j =0; j<i; ++j) {
         _X[i*_m+j]=_X[j*_m+i];
      }
   }
};


template <typename floating_type> inline void Matrix<floating_type>::whiten(const INTM V) {
   const INTM sizePatch=_m/V;
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = 0; j<V; ++j) {
         floating_type mean = 0;
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

template <typename floating_type> inline void Matrix<floating_type>::whiten(Vector<floating_type>& mean, const bool pattern) {
   mean.setZeros();
   if (pattern) {
      const INTM n =static_cast<INTM>(sqrt(static_cast<floating_type>(_m)));
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
      mean.scal(floating_type(1.0)/(_n*sizePatch));
      for (INTM i = 0; i<_n; ++i) {
         for (INTM j = 0; j<V; ++j) {
            for (INTM k = 0; k<sizePatch; ++k) {
               _X[i*_m+sizePatch*j+k]-=mean[j];
            }
         }
      }
   }
};

template <typename floating_type> inline void Matrix<floating_type>::whiten(Vector<floating_type>& mean, const
      Vector<floating_type>& mask) {
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


template <typename floating_type> inline void Matrix<floating_type>::unwhiten(Vector<floating_type>& mean, const bool pattern) {
   if (pattern) {
      const INTM n =static_cast<INTM>(sqrt(static_cast<floating_type>(_m)));
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
template <typename floating_type> inline void Matrix<floating_type>::transpose(Matrix<floating_type>& trans) const {
   trans.resize(_n,_m);
   floating_type* out = trans._X;
   for (INTM i = 0; i<_n; ++i)
      for (INTM j = 0; j<_m; ++j)
         out[j*_n+i] = _X[i*_m+j];
};

/// A <- -A
template <typename floating_type> inline void Matrix<floating_type>::neg() {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]=-_X[i];
};

template <typename floating_type> inline void Matrix<floating_type>::incrDiag() {
   for (INTM i = 0; i<MIN(_n,_m); ++i) ++_X[i*_m+i];
};

template <typename floating_type> inline void Matrix<floating_type>::addDiag(
      const Vector<floating_type>& diag) {
   floating_type* d= diag.rawX();
   for (INTM i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] += d[i];
};

template <typename floating_type> inline void Matrix<floating_type>::addDiag(
      const floating_type diag) {
   for (INTM i = 0; i<MIN(_n,_m); ++i) _X[i*_m+i] += diag;
};

template <typename floating_type> inline void Matrix<floating_type>::addToCols(
      const Vector<floating_type>& cent) {
   Vector<floating_type> col;
   for (INTM i = 0; i<_n; ++i) {
      this->refCol(i,col);      
      col.add(cent[i]);
   }
};

template <typename floating_type> inline void Matrix<floating_type>::addVecToCols(
      const Vector<floating_type>& vec, const floating_type a) {
   Vector<floating_type> col;
   for (INTM i = 0; i<_n; ++i) {
      this->refCol(i,col);      
      col.add(vec,a);
   }
};

/// perform a rank one approximation uv' using the power method
/// u0 is an initial guess for u (can be empty).
template <typename floating_type> inline void Matrix<floating_type>::svdRankOne(const Vector<floating_type>& u0,
      Vector<floating_type>& u, Vector<floating_type>& v) const {
   int i;
   const int max_iter=MAX(_m,MAX(_n,200));
   const floating_type eps=1e-10;
   u.resize(_m);
   v.resize(_n);
   floating_type norm=u0.nrm2();
   Vector<floating_type> up(u0);
   if (norm < EPSILON) up.setAleat();
   up.normalize();
   multTrans(up,v);
   for (i = 0; i<max_iter; ++i) {
      mult(v,u);
      norm=u.nrm2();
      u.scal(1.0/norm);
      multTrans(u,v);
      floating_type theta=u.dot(up);
      if (i > 10 && (1 - fabs(theta)) < eps) break;
      up.copy(u);
   }
};

template <typename floating_type> inline void Matrix<floating_type>::svd2(Matrix<floating_type>& U, Vector<floating_type>& S, const int num, const int method) const {
   const INTM num_eig= (num == -1 || method <= 1) ? MIN(_m,_n) : MIN(MIN(_m,num),_n);
   S.resize(num_eig);
   U.resize(_m,num_eig);
   if (method==0) {
      // gesv
      floating_type* vv = NULL;
      Matrix<floating_type> copyX;
      copyX.copy(*this);
      gesvd<floating_type>(reduced,no,_m,_n,copyX._X,_m,S.rawX(),U.rawX(),_m,vv,1);
   } else if (method==1) {
      // syev
      if (_m == num_eig) {
         this->XXt(U);
         syev<floating_type>(allV,lower,_m,U.rawX(),_m,S.rawX());
      } else {
         Matrix<floating_type> XXt(_m,_m);
         this->XXt(XXt); // in fact should do XtX, but will do that later
         Vector<floating_type> ss(_m);
         syev<floating_type>(allV,lower,_m,XXt.rawX(),_m,ss.rawX());
         memcpy(U.rawX(),XXt.rawX()+(_m-num_eig)*_m,_m*num_eig*sizeof(floating_type));
         memcpy(S.rawX(),ss.rawX()+_m-num_eig,num_eig*sizeof(floating_type));
      }
      S.thrsPos();
      S.Sqrt();
   } else if (method==2) {
      // syevr
      Matrix<floating_type> XXt(_m,_m);
      this->XXt(XXt); // in fact should do XtX, but will do that later
      if (_m == num_eig) {
         syevr(allV,rangeAll,lower,_m,XXt.rawX(),_m,floating_type(0),floating_type(0),0,0,S.rawX(),U.rawX(),_m);
      } else {
         Vector<floating_type> ss(_m);
         syevr(allV,range,lower,_m,XXt.rawX(),_m,floating_type(0),floating_type(0),_m-num_eig+1,_m,ss.rawX(),U.rawX(),_m);
         memcpy(S.rawX(),ss.rawX(),num_eig*sizeof(floating_type));
      }
      S.thrsPos();
      for (int ii=0; ii<S.n(); ++ii)
         S[ii]=alt_sqrt<floating_type>(S[ii]);
      //S.Sqrt();
   } 
   if (method==1 || method==2) {
      Vector<floating_type> col, col2;
      Vector<floating_type> tmpcol(_m);
      const int n=U.n();
      for (int ii=0; ii<n/2; ++ii) {
         floating_type tmp=S[n-ii-1];
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

template <typename floating_type> inline void Matrix<floating_type>::SymEig(Matrix<floating_type>& U, Vector<floating_type>& S) const {
   const int num_eig=_m;
   S.resize(_m);
   U.resize(_m,_m);
   syevr(allV,rangeAll,lower,_m,_X,_m,floating_type(0),floating_type(0),0,0,S.rawX(),U.rawX(),_m);
   S.thrsPos();
}

template <typename floating_type> inline void Matrix<floating_type>::InvsqrtMat(Matrix<floating_type>& out, const floating_type lambda_1) const {
   const int num_eig=_m;
   Vector<floating_type> S;
   S.resize(_m);
   Matrix<floating_type> U, U2;
   U.resize(_m,_m);
   syevr(allV,rangeAll,lower,_m,_X,_m,floating_type(0),floating_type(0),0,0,S.rawX(),U.rawX(),_m);
   S.thrsPos();
   //for (int ii=0; ii<_m; ++ii) S[ii]=sqrt(S[ii])/(S[ii]+lambda_1);
   //for (int ii=0; ii<_m; ++ii) S[ii]= S[ii] > 1e-6 ? floating_type(1.0)/S[ii] : 0;
   for (int ii=0; ii<_m; ++ii) S[ii]= S[ii] > 1e-6 ? floating_type(1.0)/sqrt(S[ii]+lambda_1) : 0;
   U2.copy(U);
   U2.multDiagRight(S);
   U2.mult(U,out,false,true);
}

template <typename floating_type> inline void Matrix<floating_type>::sqrtMat(Matrix<floating_type>& out) const {
   const int num_eig=_m;
   Vector<floating_type> S;
   S.resize(_m);
   Matrix<floating_type> U, U2;
   U.resize(_m,_m);
   syevr(allV,rangeAll,lower,_m,_X,_m,floating_type(0),floating_type(0),0,0,S.rawX(),U.rawX(),_m);
   S.thrsPos();
   S.Sqrt();
   U2.copy(U);
   U2.multDiagRight(S);
   U2.mult(U,out,false,true);
}



template <typename floating_type> inline void Matrix<floating_type>::singularValues(Vector<floating_type>& u) const {
   u.resize(MIN(_m,_n));
   if (_m > 10*_n) {
      Matrix<floating_type> XtX;
      this->XtX(XtX);
      syev<floating_type>(no,lower,_n,XtX.rawX(),_n,u.rawX());
      u.thrsPos();
      u.Sqrt();
   } else if (_n > 10*_m) { 
      Matrix<floating_type> XXt;
      this->XXt(XXt);
      syev<floating_type>(no,lower,_m,XXt.rawX(),_m,u.rawX());
      u.thrsPos();
      u.Sqrt();
   } else {
      floating_type* vu = NULL;
      floating_type* vv = NULL;
      Matrix<floating_type> copyX;
      copyX.copy(*this);
      gesvd<floating_type>(no,no,_m,_n,copyX._X,_m,u.rawX(),vu,1,vv,1);
   }
};

template <typename floating_type> inline void Matrix<floating_type>::svd(Matrix<floating_type>& U, Vector<floating_type>& S, Matrix<floating_type>&V) const {
   const INTM num_eig=MIN(_m,_n);
   S.resize(num_eig);
   U.resize(_m,num_eig);
   V.resize(num_eig,_n);
   if (_m > 10*_n) {
      Matrix<floating_type> Vt(_n,_n);
      this->XtX(Vt);
      syev<floating_type>(allV,lower,_n,Vt.rawX(),_n,S.rawX());
      S.thrsPos();
      S.Sqrt();
      this->mult(Vt,U);
      Vt.transpose(V);
      Vector<floating_type> inveigs;
      inveigs.copy(S);
      for (INTM i = 0; i<num_eig; ++i) 
         if (S[i] > 1e-10) {
            inveigs[i]=floating_type(1.0)/S[i];
         } else {
            inveigs[i]=floating_type(1.0);
         }
      U.multDiagRight(inveigs);
   } else if (_n > 10*_m) {
      this->XXt(U);
      syev<floating_type>(allV,lower,_m,U.rawX(),_m,S.rawX());
      S.thrsPos();
      S.Sqrt();
      U.mult(*this,V,true,false);
      Vector<floating_type> inveigs;
      inveigs.copy(S);
      for (INTM i = 0; i<num_eig; ++i) 
         if (S[i] > 1e-10) {
            inveigs[i]=floating_type(1.0)/S[i];
         } else {
            inveigs[i]=floating_type(1.0);
         }
      V.multDiagLeft(inveigs);
   } else {
      Matrix<floating_type> copyX;
      copyX.copy(*this);
      gesvd<floating_type>(reduced,reduced,_m,_n,copyX._X,_m,S.rawX(),U.rawX(),_m,V.rawX(),num_eig);
   }
};

/// find the eigenvector corresponding to the largest eigenvalue
/// when the current matrix is symmetric. u0 is the initial guess.
/// using two iterations of the power method
template <typename floating_type> inline void Matrix<floating_type>::eigLargestSymApprox(
      const Vector<floating_type>& u0, Vector<floating_type>& u) const {
   int i,j;
   const int max_iter=100;
   const floating_type eps=10e-6;
   u.copy(u0);
   floating_type norm = u.nrm2();
   floating_type theta;
   u.scal(1.0/norm);
   Vector<floating_type> up(u);
   Vector<floating_type> uor(u);
   floating_type lambda_1=floating_type();

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
template <typename floating_type> inline floating_type Matrix<floating_type>::eigLargestMagnSym(
      const Vector<floating_type>& u0, Vector<floating_type>& u) const {
   const int max_iter=1000;
   const floating_type eps=10e-6;
   u.copy(u0);
   floating_type norm = u.nrm2();
   u.scal(1.0/norm);
   Vector<floating_type> up(u);
   floating_type lambda_1=floating_type();

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
template <typename floating_type> inline floating_type Matrix<floating_type>::eigLargestMagnSym() const {
   const int max_iter=1000;
   const floating_type eps=10e-6;
   Vector<floating_type> u(_m);
   u.setAleat();
   floating_type norm = u.nrm2();
   u.scal(1.0/norm);
   Vector<floating_type> up(u);
   floating_type lambda_1=floating_type();
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
template <typename floating_type> inline void Matrix<floating_type>::invSym() {
   sytri<floating_type>(upper,_n,_X,_n);
   this->fillSymmetric();
};
template <typename floating_type> inline void Matrix<floating_type>::invSymPos() {
   potri<floating_type>(upper,_n,_X,_n);
   this->fillSymmetric();
};

/// perform b = alpha*A'x + beta*b
template <typename floating_type> inline void Matrix<floating_type>::multTrans(const Vector<floating_type>& x, 
      Vector<floating_type>& b, const floating_type a, const floating_type c) const {
   b.resize(_n);
   //   assert(x._n == _m && b._n == _n);
   cblas_gemv<floating_type>(CblasColMajor,CblasTrans,_m,_n,a,_X,_m,x._X,1,c,b._X,1);
};

/// perform b = A'x, when x is sparse
template <typename floating_type> 
template <typename I> 
inline void Matrix<floating_type>::multTrans(const SpVector<floating_type,I>& x, 
      Vector<floating_type>& b, const floating_type alpha, const floating_type beta) const {
   b.resize(_n);
   Vector<floating_type> col;
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

template <typename floating_type> inline void Matrix<floating_type>::multTrans(
      const Vector<floating_type>& x, Vector<floating_type>& b, const Vector<bool>& active) const {
   b.setZeros();
   Vector<floating_type> col;
   bool* pr_active=active.rawX();
   for (INTM i = 0; i<_n; ++i) {
      if (pr_active[i]) {
         this->refCol(i,col);
         b._X[i]=col.dot(x);
      }
   }
};

/// perform b = alpha*A*x+beta*b
template <typename floating_type> inline void Matrix<floating_type>::mult(const Vector<floating_type>& x, 
      Vector<floating_type>& b, const floating_type a, const floating_type c) const {
   //  assert(x._n == _n && b._n == _m);
   b.resize(_m);
   cblas_gemv<floating_type>(CblasColMajor,CblasNoTrans,_m,_n,a,_X,_m,x._X,1,c,b._X,1);
};


/// perform b = alpha*A*x+beta*b
template <typename floating_type> inline void Matrix<floating_type>::mult_loop(const Vector<floating_type>& x, 
      Vector<floating_type>& b) const {
   b.resize(_m);
   for (int ii=0; ii<_m; ++ii) {
      b[ii]=cblas_dot<floating_type>(_n,x._X,1,_X+ii,_m);
   }
};

/// perform b = alpha*A*x + beta*b, when x is sparse
template <typename floating_type> 
template <typename I> 
inline void Matrix<floating_type>::mult(const SpVector<floating_type,I>& x, 
      Vector<floating_type>& b, const floating_type a, const floating_type a2) const {
   if (!a2) {
      b.setZeros();
   } else if (a2 != 1.0) {
      b.scal(a2);
   }
   if (a == 1.0) {
      for (INTM i = 0; i<x._L; ++i) {
         cblas_axpy<floating_type>(_m,x._v[i],_X+x._r[i]*_m,1,b._X,1);
      }
   } else {
      for (INTM i = 0; i<x._L; ++i) {
         cblas_axpy<floating_type>(_m,a*x._v[i],_X+x._r[i]*_m,1,b._X,1);
      }
   }
};

/// perform C = a*A*B + b*C, possibly transposing A or B.
template <typename floating_type> inline void Matrix<floating_type>::mult(const Matrix<floating_type>& B, 
      Matrix<floating_type>& C, const bool transA, const bool transB,
      const floating_type a, const floating_type b) const {
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
   cblas_gemm<floating_type>(CblasColMajor,trA,trB,m,n,k,a,_X,_m,B._X,B._m,
         b,C._X,C._m);
};

/// perform C = a*B*A + b*C, possibly transposing A or B.
template <typename floating_type>
inline void Matrix<floating_type>::multSwitch(const Matrix<floating_type>& B, Matrix<floating_type>& C, 
      const bool transA, const bool transB,
      const floating_type a, const floating_type b) const {
   B.mult(*this,C,transB,transA,a,b);
};

/// perform C = A*B, when B is sparse
template <typename floating_type>
template <typename I>
inline void Matrix<floating_type>::mult(const SpMatrix<floating_type,I>& B, Matrix<floating_type>& C,
      const bool transA, const bool transB,
      const floating_type a, const floating_type b) const {
   if (transA) {
      if (transB) {
         C.resize(_n,B.m());
         if (b) {
            C.scal(b);
         } else {
            C.setZeros();
         }
         Vector<floating_type> rowC(B.m());
         Vector<floating_type> colA;
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
         Vector<floating_type> colC;
         SpVector<floating_type,I> colB;
         for (INTM i = 0; i<B.n(); ++i) {
            C.refCol(i,colC);
            B.refCol(i,colB);
            this->multTrans(colB,colC,a,floating_type(1.0));
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
         Vector<floating_type> colA;
         SpVector<floating_type,I> colB;
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
         Vector<floating_type> colC;
         SpVector<floating_type,I> colB;
         for (INTM i = 0; i<B.n(); ++i) {
            C.refCol(i,colC);
            B.refCol(i,colB);
            this->mult(colB,colC,a,floating_type(1.0));
         }
      }
   };
}


/// mult by a diagonal matrix on the left
template <typename floating_type>
   inline void Matrix<floating_type>::multDiagLeft(const Vector<floating_type>& diag) {
      if (diag.n() != _m)
         return;
      floating_type* d = diag.rawX();
      for (INTM i = 0; i< _n; ++i) {
         for (INTM j = 0; j<_m; ++j) {
            _X[i*_m+j] *= d[j];
         }
      }
   };

/// mult by a diagonal matrix on the right
template <typename floating_type> inline void Matrix<floating_type>::multDiagRight(
      const Vector<floating_type>& diag) {
   if (diag.n() != _n)
      return;
   floating_type* d = diag.rawX();
   for (INTM i = 0; i< _n; ++i) {
      for (INTM j = 0; j<_m; ++j) {
         _X[i*_m+j] *= d[i];
      }
   }
};
/// mult by a diagonal matrix on the right
template <typename floating_type> inline void Matrix<floating_type>::AddMultDiagRight(
      const Vector<floating_type>& diag, Matrix<floating_type>& mat) {
   if (diag.n() != _n)
      return;
   mat.resize(_m,_n);
   //mat.setZeros();
   floating_type* d = diag.rawX();
   for (INTM i = 0; i< _n; ++i) {
      cblas_axpy<floating_type>(_m,d[i],_X+i*_m,1,mat._X+i*_m,1);
   }
};



/// C = A .* B, elementwise multiplication
template <typename floating_type> inline void Matrix<floating_type>::mult_elementWise(
      const Matrix<floating_type>& B, Matrix<floating_type>& C) const {
   assert(_n == B._n && _m == B._m);
   C.resize(_m,_n);
   vMul<floating_type>(_n*_m,_X,B._X,C._X);
};

/// C = A .* B, elementwise multiplication
template <typename floating_type> inline void Matrix<floating_type>::div_elementWise(
      const Matrix<floating_type>& B, Matrix<floating_type>& C) const {
   assert(_n == B._n && _m == B._m);
   C.resize(_m,_n);
   vDiv<floating_type>(_n*_m,_X,B._X,C._X);
};


/// XtX = A'*A
template <typename floating_type> inline void Matrix<floating_type>::XtX(Matrix<floating_type>& xtx) const {
   xtx.resize(_n,_n);
   cblas_syrk<floating_type>(CblasColMajor,CblasUpper,CblasTrans,_n,_m,floating_type(1.0),
         _X,_m,floating_type(),xtx._X,_n);
   xtx.fillSymmetric();
};

/// XXt = A*At
template <typename floating_type> inline void Matrix<floating_type>::XXt(Matrix<floating_type>& xxt) const {
   xxt.resize(_m,_m);
   cblas_syrk<floating_type>(CblasColMajor,CblasUpper,CblasNoTrans,_m,_n,floating_type(1.0),
         _X,_m,floating_type(),xxt._X,_m);
   xxt.fillSymmetric();
};

/// XXt = A*A' where A is an upper triangular matrix
template <typename floating_type> inline void Matrix<floating_type>::upperTriXXt(Matrix<floating_type>& XXt, const INTM L) const {
   XXt.resize(L,L);
   for (INTM i = 0; i<L; ++i) {
      cblas_syr<floating_type>(CblasColMajor,CblasUpper,i+1,floating_type(1.0),_X+i*_m,1,XXt._X,L);
   }
   XXt.fillSymmetric();
}


/// extract the diagonal
template <typename floating_type> inline void Matrix<floating_type>::diag(Vector<floating_type>& dv) const {
   INTM size_diag=MIN(_n,_m);
   dv.resize(size_diag);
   floating_type* const d = dv.rawX();
   for (INTM i = 0; i<size_diag; ++i)
      d[i]=_X[i*_m+i];
};

/// set the diagonal
template <typename floating_type> inline void Matrix<floating_type>::setDiag(const Vector<floating_type>& dv) {
   INTM size_diag=MIN(_n,_m);
   floating_type* const d = dv.rawX();
   for (INTM i = 0; i<size_diag; ++i)
      _X[i*_m+i]=d[i];
};

/// set the diagonal
template <typename floating_type> inline void Matrix<floating_type>::setDiag(const floating_type val) {
   INTM size_diag=MIN(_n,_m);
   for (INTM i = 0; i<size_diag; ++i)
      _X[i*_m+i]=val;
};


/// each element of the matrix is replaced by its exponential
template <typename floating_type> inline void Matrix<floating_type>::exp() {
   vExp<floating_type>(_n*_m,_X,_X);
};

/// each element of the matrix is replaced by its exponential
template <typename floating_type> inline void Matrix<floating_type>::pow(const floating_type a) {
   vPowx<floating_type>(_n*_m,_X,a,_X);
};

template <typename floating_type> inline void Matrix<floating_type>::sqr() {
   vSqr<floating_type>(_n*_m,_X,_X);
};

template <typename floating_type> inline void Matrix<floating_type>::Sqrt() {
   vSqrt<floating_type>(_n*_m,_X,_X);
};

template <typename floating_type> inline void Matrix<floating_type>::Invsqrt() {
   vInvSqrt<floating_type>(_n*_m,_X,_X);
};
/// return vec1'*A*vec2, where vec2 is sparse
template <typename floating_type> 
template <typename I> 
inline floating_type Matrix<floating_type>::quad(const SpVector<floating_type,I>& vec) const {
   floating_type sum = floating_type();
   INTM L = vec._L;
   I* r = vec._r;
   floating_type* v = vec._v;
   for (INTM i = 0; i<L; ++i)
      for (INTM j = 0; j<L; ++j)
         sum += _X[r[i]*_m+r[j]]*v[i]*v[j];
   return sum;
};

template <typename floating_type> 
template <typename I> 
inline void Matrix<floating_type>::quad_mult(const Vector<floating_type>& vec1,
      const SpVector<floating_type,I>& vec2, Vector<floating_type>& y, const floating_type a, const floating_type b) const {
   const INTM size_y= y.n();
   const INTM nn = _n/size_y;
   //y.resize(size_y);
   //y.setZeros();
   Matrix<floating_type> tmp;
   for (INTM i = 0; i<size_y; ++i) {
      tmp.setData(_X+(i*nn)*_m,_m,nn);
      y[i]=b*y[i]+a*tmp.quad(vec1,vec2);
   }
}

/// return vec'*A*vec when vec is sparse
template <typename floating_type> 
template <typename I> 
inline floating_type Matrix<floating_type>::quad(
      const Vector<floating_type>& vec1, const SpVector<floating_type,I>& vec) const {
   floating_type sum = floating_type();
   INTM L = vec._L;
   I* r = vec._r;
   floating_type* v = vec._v;
   Vector<floating_type> col;
   for (INTM i = 0; i<L; ++i) {
      this->refCol(r[i],col);
      sum += v[i]*col.dot(vec1);
   }
   return sum;
};

/// add alpha*mat to the current matrix
template <typename floating_type> inline void Matrix<floating_type>::add(const Matrix<floating_type>& mat, const floating_type alpha) {
   assert(mat._m == _m && mat._n == _n);
   cblas_axpy<floating_type>(_n*_m,alpha,mat._X,1,_X,1);
};

/// add alpha*mat to the current matrix
template <typename floating_type> inline void Matrix<floating_type>::add_scal(const Matrix<floating_type>& mat, const floating_type alpha, const floating_type beta) {
   assert(mat._m == _m && mat._n == _n);
   cblas_axpby<floating_type>(_n*_m,alpha,mat._X,1,beta,_X,1);
};


/// add alpha*mat to the current matrix
template <typename floating_type> inline floating_type Matrix<floating_type>::dot(const Matrix<floating_type>& mat) const {
   assert(mat._m == _m && mat._n == _n);
   return cblas_dot<floating_type>(_n*_m,mat._X,1,_X,1);
};


/// add alpha to the current matrix
template <typename floating_type> inline void Matrix<floating_type>::add(const floating_type alpha) {
   for (INTM i = 0; i<_n*_m; ++i) _X[i]+=alpha;
};

/// substract the matrix mat to the current matrix
template <typename floating_type> inline void Matrix<floating_type>::sub(const Matrix<floating_type>& mat) {
   vSub<floating_type>(_n*_m,_X,mat._X,_X);
};

/// compute the sum of the magnitude of the matrix values
template <typename floating_type> inline floating_type Matrix<floating_type>::asum() const {
   return cblas_asum<floating_type>(_n*_m,_X,1);
};

template <typename floating_type> inline floating_type Matrix<floating_type>::sum() const {
   floating_type sum=0;
   for (INTM i =0; i<_n*_m; ++i) sum+=_X[i];
   return sum;
};



/// returns the trace of the matrix
template <typename floating_type> inline floating_type Matrix<floating_type>::trace() const {
   floating_type sum=floating_type();
   INTM m = MIN(_n,_m);
   for (INTM i = 0; i<m; ++i) 
      sum += _X[i*_m+i];
   return sum;
};

/// return ||A||_F
template <typename floating_type> inline floating_type Matrix<floating_type>::normF() const {
   return cblas_nrm2<floating_type>(_n*_m,_X,1);
};

template <typename floating_type> inline floating_type Matrix<floating_type>::mean() const {
   Vector<floating_type> vec;
   this->toVect(vec);
   return vec.mean();
};

template <typename floating_type> inline floating_type Matrix<floating_type>::abs_mean() const {
   Vector<floating_type> vec;
   this->toVect(vec);
   return vec.abs_mean();
};


/// return ||A||_F^2
template <typename floating_type> inline floating_type Matrix<floating_type>::normFsq() const {
   return cblas_dot<floating_type>(_n*_m,_X,1,_X,1);
};

/// return ||At||_{inf,2}
template <typename floating_type> inline floating_type Matrix<floating_type>::norm_inf_2_col() const {
   Vector<floating_type> col;
   floating_type max = -1.0;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      floating_type norm_col = col.nrm2();
      if (norm_col > max) 
         max = norm_col;
   }
   return max;
};

/// return ||At||_{1,2}
template <typename floating_type> inline floating_type Matrix<floating_type>::norm_1_2_col() const {
   Vector<floating_type> col;
   floating_type sum = 0.0;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      sum += col.nrm2();
   }
   return sum;
};

/// returns the l2 norms of the columns
template <typename floating_type> inline void Matrix<floating_type>::norm_2_rows(
      Vector<floating_type>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] += _X[i*_m+j]*_X[i*_m+j];
   for (INTM j = 0; j<_m; ++j) 
      norms[j]=sqrt(norms[j]);
};

/// returns the l2 norms of the columns
template <typename floating_type> inline void Matrix<floating_type>::norm_2sq_rows(
      Vector<floating_type>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] += _X[i*_m+j]*_X[i*_m+j];
};


/// returns the l2 norms of the columns
template <typename floating_type> inline void Matrix<floating_type>::norm_2_cols(
      Vector<floating_type>& norms) const {
   norms.resize(_n);
   Vector<floating_type> col;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      norms[i] = col.nrm2();
   }
};


/// returns the linf norms of the columns
template <typename floating_type> inline void Matrix<floating_type>::norm_inf_cols(Vector<floating_type>& norms) const {
   norms.resize(_n);
   Vector<floating_type> col;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      norms[i] = col.fmaxval();
   }
};

/// returns the linf norms of the columns
template <typename floating_type> inline void Matrix<floating_type>::norm_inf_rows(Vector<floating_type>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] = MAX(abs<floating_type>(_X[i*_m+j]),norms[j]);
};

template <typename floating_type> inline void Matrix<floating_type>::get_sum_cols(Vector<floating_type>& sum) const {
   sum.resize(_n);
   for (INTM i = 0; i<_n; ++i) {
      sum[i]=0;
      for (INTM j = 0; j<_m; ++j) 
         sum[i] += (_X[i*_m+j]);
   }
};

template <typename floating_type> inline void Matrix<floating_type>::dot_col(const Matrix<floating_type>& mat, 
      Vector<floating_type>& dots) const {
   dots.resize(_n);
   for (INTM i = 0; i<_n; ++i) 
      dots[i] = cblas_dot<floating_type>(_m,_X+i*_m,1,mat._X+i*_m,1);
}

/// returns the linf norms of the columns
template <typename floating_type> inline void Matrix<floating_type>::norm_l1_rows(Vector<floating_type>& norms) const {
   norms.resize(_m);
   norms.setZeros();
   for (INTM i = 0; i<_n; ++i) 
      for (INTM j = 0; j<_m; ++j) 
         norms[j] += abs<floating_type>(_X[i*_m+j]);
};



/// returns the l2 norms of the columns
template <typename floating_type> inline void Matrix<floating_type>::norm_2sq_cols(
      Vector<floating_type>& norms) const {
   norms.resize(_n);
   Vector<floating_type> col;
   for (INTM i = 0; i<_n; ++i) {
      refCol(i,col);
      norms[i] = col.nrm2sq();
   }
};

template <typename floating_type> 
inline void Matrix<floating_type>::sum_cols(Vector<floating_type>& sum) const {
   sum.resize(_m);
   sum.setZeros();
   Vector<floating_type> tmp;
   for (INTM i = 0; i<_n; ++i) {
      this->refCol(i,tmp);
      sum.add(tmp);
   }
};

/// Compute the mean of the columns
template <typename floating_type> inline void Matrix<floating_type>::meanCol(Vector<floating_type>& mean) const {
   Vector<floating_type> ones(_n);
   ones.set(floating_type(1.0/_n));
   this->mult(ones,mean,1.0,0.0);
};

/// Compute the mean of the rows
template <typename floating_type> inline void Matrix<floating_type>::meanRow(Vector<floating_type>& mean) const {
   Vector<floating_type> ones(_m);
   ones.set(floating_type(1.0/_m));
   this->multTrans(ones,mean,1.0,0.0);
};


/// fill the matrix with the row given
template <typename floating_type> inline void Matrix<floating_type>::fillRow(const Vector<floating_type>& row) {
   for (INTM i = 0; i<_n; ++i) {
      floating_type val = row[i];
      for (INTM j = 0; j<_m; ++j) {
         _X[i*_m+j]=val;
      }
   }
};

/// fill the matrix with the row given
template <typename floating_type> inline void Matrix<floating_type>::extractRow(const INTM j,
      Vector<floating_type>& row) const {
   row.resize(_n);
   for (INTM i = 0; i<_n; ++i) {
      row[i]=_X[i*_m+j];
   }
};

/// fill the matrix with the row given
template <typename floating_type> inline void Matrix<floating_type>::setRow(const INTM j,
      const Vector<floating_type>& row) {
   for (INTM i = 0; i<_n; ++i) {
      _X[i*_m+j]=row[i];
   }
};

/// fill the matrix with the row given
template <typename floating_type> inline void Matrix<floating_type>::addRow(const INTM j,
      const Vector<floating_type>& row, const floating_type a) {
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
template <typename floating_type> inline void Matrix<floating_type>::softThrshold(const floating_type nu) {
   Vector<floating_type> vec;
   toVect(vec);
   vec.softThrshold(nu);
};

/// perform soft-thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::fastSoftThrshold(const floating_type nu) {
    Vector<floating_type> vec;
    toVect(vec);
    vec.fastSoftThrshold(nu);
};
/// perform soft-thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::fastSoftThrshold(Matrix<floating_type>& output, const floating_type nu) const {
    output.resize(_m,_n,false);
    Vector<floating_type> vec, vec2;
    toVect(vec);
    output.toVect(vec2);
    vec.fastSoftThrshold(vec2,nu);
};




/// perform soft-thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::hardThrshold(const floating_type nu) {
   Vector<floating_type> vec;
   toVect(vec);
   vec.hardThrshold(nu);
};


/// perform thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::thrsmax(const floating_type nu) {
   Vector<floating_type> vec;
   toVect(vec);
   vec.thrsmax(nu);
};

/// perform thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::thrsmin(const floating_type nu) {
   Vector<floating_type> vec;
   toVect(vec);
   vec.thrsmin(nu);
};


/// perform soft-thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::inv_elem() {
   Vector<floating_type> vec;
   toVect(vec);
   vec.inv();
};

/// perform soft-thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::blockThrshold(const floating_type nu,
      const INTM sizeGroup) {
   for (INTM i = 0; i<_n; ++i) {
      INTM j;
      for (j = 0; j<_m-sizeGroup+1; j+=sizeGroup) {
         floating_type nrm=0;
         for (INTM k = 0; k<sizeGroup; ++k)
            nrm += _X[i*_m +j+k]*_X[i*_m +j+k];
         nrm=sqrt(nrm);
         if (nrm < nu) {
            for (INTM k = 0; k<sizeGroup; ++k)
               _X[i*_m +j+k]=0;
         } else {
            floating_type scal = (nrm-nu)/nrm;
            for (INTM k = 0; k<sizeGroup; ++k)
               _X[i*_m +j+k]*=scal;
         }
      }
      j -= sizeGroup;
      for ( ; j<_m; ++j)
         _X[j]=softThrs<floating_type>(_X[j],nu);
   }
}

template <typename floating_type> inline void Matrix<floating_type>::sparseProject(Matrix<floating_type>& Y, 
      const floating_type thrs,   const int mode, const floating_type lambda_1,
      const floating_type lambda_2, const floating_type lambda_3, const bool pos,
      const int numThreads) {

   int NUM_THREADS=init_omp(numThreads);
   Vector<floating_type>* XXT= new Vector<floating_type>[NUM_THREADS];
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
      Vector<floating_type> Xi;
      this->refCol(i,Xi);
      Vector<floating_type> Yi;
      Y.refCol(i,Yi);
      Vector<floating_type>& XX = XXT[numT];
      XX.copy(Xi);
      XX.sparseProject(Yi,thrs,mode,lambda_1,lambda_2,lambda_3,pos);
   }
   delete[](XXT);
};


/// perform soft-thresholding of the matrix, with the threshold nu
template <typename floating_type> inline void Matrix<floating_type>::thrsPos() {
   Vector<floating_type> vec;
   toVect(vec);
   vec.thrsPos();
};


/// perform A <- A + alpha*vec1*vec2'
template <typename floating_type> inline void Matrix<floating_type>::rank1Update(
      const Vector<floating_type>& vec1, const Vector<floating_type>& vec2, const floating_type alpha) {
   cblas_ger<floating_type>(CblasColMajor,_m,_n,alpha,vec1._X,1,vec2._X,1,_X,_m);
};

/// perform A <- A + alpha*vec1*vec2', when vec1 is sparse
template <typename floating_type> 
template <typename I> 
inline void Matrix<floating_type>::rank1Update(
      const SpVector<floating_type,I>& vec1, const Vector<floating_type>& vec2, const floating_type alpha) {
   I* r = vec1._r;
   floating_type* v = vec1._v;
   floating_type* X2 = vec2._X;
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

template <typename floating_type>
template <typename I>
inline void Matrix<floating_type>::rank1Update_mult(const Vector<floating_type>& vec1, 
      const Vector<floating_type>& vec1b,
      const SpVector<floating_type,I>& vec2,
      const floating_type alpha) {
   const INTM nn = vec1b.n();
   const INTM size_A = _n/nn;
   Matrix<floating_type> tmp;
   for (INTM i = 0; i<nn; ++i) {
      tmp.setData(_X+i*size_A*_m,_m,size_A);
      tmp.rank1Update(vec1,vec2,alpha*vec1b[i]);
   }
};

/// perform A <- A + alpha*vec1*vec2', when vec1 is sparse
template <typename floating_type>
template <typename I>
inline void Matrix<floating_type>::rank1Update(
      const SpVector<floating_type,I>& vec1, const SpVector<floating_type,I>& vec2, const floating_type alpha) {
   I* r = vec1._r;
   floating_type* v = vec1._v;
   floating_type* v2 = vec2._v;
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
template <typename floating_type> 
template <typename I> 
inline void Matrix<floating_type>::rank1Update(
      const Vector<floating_type>& vec1, const SpVector<floating_type,I>& vec2, const floating_type alpha) {
   I* r = vec2._r;
   floating_type* v = vec2._v;
   Vector<floating_type> Xi;
   for (INTM i = 0; i<vec2._L; ++i) {
      this->refCol(r[i],Xi);
      Xi.add(vec1,v[i]*alpha);
   }
};

/// perform A <- A + alpha*vec1*vec1', when vec1 is sparse
template <typename floating_type> 
template <typename I> 
inline void Matrix<floating_type>::rank1Update(
      const SpVector<floating_type,I>& vec1, const floating_type alpha) {
   I* r = vec1._r;
   floating_type* v = vec1._v;
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
template <typename floating_type> inline void Matrix<floating_type>::conjugateGradient(
      const Vector<floating_type>& b, Vector<floating_type>& x, const floating_type tol, const int itermax) const {
   Vector<floating_type> R,P,AP;
   R.copy(b);
   this->mult(x,R,floating_type(-1.0),floating_type(1.0));
   P.copy(R);
   int k = 0;
   floating_type normR = R.nrm2sq();
   floating_type alpha;
   while (normR > tol && k < itermax) {
      this->mult(P,AP);
      alpha = normR/P.dot(AP);
      x.add(P,alpha);
      R.add(AP,-alpha);
      floating_type tmp = R.nrm2sq();
      P.scal(tmp/normR);
      normR = tmp;
      P.add(R,floating_type(1.0));
      ++k;
   };
};

template <typename floating_type> inline void Matrix<floating_type>::drop(char* fileName) const {
   std::ofstream f;
   f.precision(12);
   f.flags(std::ios_base::scientific);
   f.open(fileName, ofstream::trunc);
   logging(logINFO) << "Matrix written in " << fileName;
   for (INTM i = 0; i<_n; ++i) {
      for (INTM j = 0; j<_m; ++j) 
         f << _X[i*_m+j] << " ";
      f << std::endl;
   }
   f.close();
};

/// compute a Nadaraya Watson estimator
template <typename floating_type> inline void Matrix<floating_type>::NadarayaWatson(
      const Vector<INTM>& ind, const floating_type sigma) {
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
      Matrix<floating_type> Xm(_m,count);
      Vector<floating_type> col, col2;
      for (INTM j= 0; j<count; ++j) {
         this->refCol(indicesGroup[j],col);
         Xm.refCol(j,col2);
         col2.copy(col);
      }
      Vector<floating_type> norms;
      Xm.norm_2sq_cols(norms);
      Matrix<floating_type> weights;
      Xm.XtX(weights);
      weights.scal(floating_type(-2.0));
      Vector<floating_type> ones(Xm.n());
      ones.set(floating_type(1.0));
      weights.rank1Update(ones,norms);
      weights.rank1Update(norms,ones);
      weights.scal(-sigma);
      weights.exp();
      Vector<floating_type> den;
      weights.mult(ones,den);
      den.inv();
      weights.multDiagRight(den);
      Matrix<floating_type> num;
      Xm.mult(weights,num);
      for (INTM j= 0; j<count; ++j) {
         this->refCol(indicesGroup[j],col);
         num.refCol(j,col2);
         col.copy(col2);
      }
   }
};

/// make a sparse copy of the current matrix
template <typename floating_type> inline void Matrix<floating_type>::toSparse(SpMatrix<floating_type>& out) const {
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
   floating_type* v;
#pragma omp critical
   {
      r=new INTM[count];
      v=new floating_type[count];
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
template <typename floating_type> inline void Matrix<floating_type>::toSparseTrans(
      SpMatrix<floating_type>& out) {
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
   floating_type* v;
#pragma omp critical
   {
      r=new INTM[count];
      v=new floating_type[count];
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
template <typename floating_type> inline void Matrix<floating_type>::toVect(
      Vector<floating_type>& vec) const {
   vec.clear();
   vec._externAlloc=true;
   vec._n=_n*_m;
   vec._X=_X;
};


/* ************************************
 * Implementation of the class OptimInfo 
 * ************************************/

/// Constructor with existing data X of an m x n OptimInfo
template <typename floating_type> OptimInfo<floating_type>::OptimInfo(floating_type* X, INTM nclass, INTM m, INTM n) :
   _externAlloc(true), _X(X), _nclass(nclass), _m(m), _n(n) {  };


/// Constructor for a new m x n OptimInfo
template <typename floating_type> OptimInfo<floating_type>::OptimInfo(INTM nclass, INTM m, INTM n) :
   _externAlloc(false), _nclass(nclass), _m(m), _n(n)  {
#pragma omp critical
      {
         _X= new floating_type[_nclass*_n*_m];
      }
   };

/// Empty constructor
template <typename floating_type> OptimInfo<floating_type>::OptimInfo() :
   _externAlloc(false), _X(NULL), _nclass(0), _m(0), _n(0) { };

/// Destructor
template <typename floating_type> OptimInfo<floating_type>::~OptimInfo() {
   clear();
};

/// Return a modifiable reference to X(i,j,k)
template <typename floating_type> inline floating_type& OptimInfo<floating_type>::operator()(const INTM i, const INTM j, const INTM k) {
   return _X[i*_m*_n + k*_m+j];
};

/// Return the value X(i,j,k)
template <typename floating_type> inline floating_type OptimInfo<floating_type>::operator()(const INTM i, const INTM j, const INTM k) const {
   return _X[i*_m*_n + k*_m+j];
};

/// Print the OptimInfo to std::cout
template <typename floating_type> inline void OptimInfo<floating_type>::print(const string& name) const {
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
template <typename floating_type> inline void OptimInfo<floating_type>::dump(const string& name) const {
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
template <typename floating_type> inline void OptimInfo<floating_type>::setZeros() {
   memset(_X,0,_nclass*_n*_m*sizeof(floating_type));
};

/// Resize the optimInfo
template <typename floating_type> inline void OptimInfo<floating_type>::resize(INTM nclass, INTM m, INTM n, const bool set_zeros) {
   if (_nclass==nclass && _n==n && _m==m) return;
   clear();
   _nclass=nclass;
   _n=n;
   _m=m;
   _externAlloc=false;
#pragma omp critical
   {
      _X=new floating_type[_nclass*_n*_m];
   }
   if (set_zeros)
      setZeros();
};

/// Clear the optimInfo
template <typename floating_type> inline void OptimInfo<floating_type>::clear() {
   if (!_externAlloc) delete[](_X);
   _nclass=0;
   _n=0;
   _m=0;
   _X=NULL;
   _externAlloc=true;
};

/// make a copy of the optimInfo optim in the current optim
template <typename floating_type> inline void OptimInfo<floating_type>::copy(const OptimInfo<floating_type>& optim) {
   if (_X != optim._X) {
      resize(optim._nclass, optim._m,optim._n);
      memcpy(_X,optim._X,_nclass*_m*_n*sizeof(floating_type));
   }
};

/// Change the data in the optimInfo
template <typename floating_type> inline void OptimInfo<floating_type>::setData(floating_type* X, INTM nclass, INTM m, INTM n) {
   clear();
   _X=X;
   _nclass=nclass;
   _m=m;
   _n=n;
   _externAlloc=true;
};

/// add alpha*optim to the current optim info at a given index
template <typename floating_type> inline void OptimInfo<floating_type>::add(const OptimInfo<floating_type>& optim, const int index, const floating_type alpha) {
   assert(optim._m == _m && optim._n == _n);
   for (INTT i = 0; i<_m * _n; ++i){
       //FIXME maybe slow
      _X[index * _m * _n + i] += alpha*optim[i];
   }
};

#endif
