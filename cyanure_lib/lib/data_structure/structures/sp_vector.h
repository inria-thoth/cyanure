#ifndef SP_VECTOR_H
#define SP_VECTOR_H

#include <fstream>
#ifdef WINDOWS
#include <string>
#else
#include <cstring>
#endif

#include "../declare_structures.h"


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
   inline void print(const std::string& name) const;
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
template <typename floating_type, typename I> inline void SpVector<floating_type,I>::print(const std::string& name) const {
   std::cerr << name << std::endl;
   std::cerr << _nzmax << std::endl;
   for (I i = 0; i<_L; ++i)
      std::cerr << "(" <<_r[i] << ", " <<  _v[i] << ")";
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



#endif