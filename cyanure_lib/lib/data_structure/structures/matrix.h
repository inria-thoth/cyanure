#ifndef MATRIX_H
#define MATRIX_H

#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif

#include "../declare_structures.h"



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
   inline void print(const std::string& name) const;
   inline void dump(const std::string& name) const;


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
template <typename floating_type> inline void Matrix<floating_type>::print(const std::string& name) const {
   logging(logERROR) << name;
   logging(logERROR) << _m;
   for (INTM i = 0; i<_m; ++i) {
      for (INTM j = 0; j<_n; ++j) {
         printf("%10.5g ",static_cast<double>(_X[j*_m+i]));
      }
      printf("\n ");
   }
   printf("\n ");
};

/// Print the matrix to std::cout
template <typename floating_type> inline void Matrix<floating_type>::dump(const std::string& name) const {
   std::ofstream f; 
   const char * cname = name.c_str();
   f.open(cname);
   f.precision(20);
   logging(logERROR) << name;
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
         logging(logERROR) << "eigLargestSymApprox failed";
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
   f.open(fileName, std::ofstream::trunc);
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


#endif