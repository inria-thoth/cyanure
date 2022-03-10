#ifndef OPTIM_INFO_H
#define OPTIM_INFO_H

#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif

#include "../declare_structures.h"



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
   inline void print(const std::string& name) const;
   inline void dump(const std::string& name) const;

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
template <typename floating_type> inline void OptimInfo<floating_type>::print(const std::string& name) const {
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
template <typename floating_type> inline void OptimInfo<floating_type>::dump(const std::string& name) const {
   std::ofstream f; 
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