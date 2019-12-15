#ifndef DATA_H 
#define DATA_H 

#include "linalg.h"

#define GET_WB(a)  Matrix<T> W; Vector<T> b; get_wb(a,W,b); 
#define GET_w(a)  Vector<T> w; get_w(a,w); 

template <typename M, typename D>
class Data {
   public:
      typedef D variable_type;
      typedef typename D::value_type value_type;
      typedef value_type T;

      Data(const M& X, const bool intercept) : _X(X), _scale_intercept(T(1.0)), _intercept(intercept) { };
      virtual ~Data() { };

      virtual void pred(const D& input, D& output) const = 0;
      virtual void add_dual_pred(const D& input, D& output,const T a=T(1.0), const T b=T(1.0)) const = 0;
      virtual void print() const = 0;
      inline void get_coordinates(const int ind, Vector<typename M::index_type>& indices) const { 
         if (_X.is_sparse) {
            typename M::col_type col;
            _X.refCol(ind,col);
            col.refIndices(indices);
         }
      };
      virtual void pred(const int ind, const D& input, typename D::element& output) const = 0;
      bool is_sparse() const { return _X.is_sparse; };
      virtual void set_intercept(const D& x0, D& x) = 0;
      virtual void reverse_intercept(D& x) = 0;
      inline bool intercept() const { return _intercept; };
      inline void norms_data(Vector<T>& norms) {
         if (_norms.n()==0) {
            _norms.resize(_X.n());
            _X.norm_2sq_cols(_norms);
            if (_intercept)
               norms.add(_scale_intercept*_scale_intercept);
         } 
         norms.copy(_norms);
      };
      inline T norms(const int ind) {
         return _norms[ind];
      };

      
   protected:
      const M& _X; 
      T _scale_intercept; 
      const bool _intercept; 
      Vector<T> _norms;

   private:
      explicit Data<M,D>(const Data<M,D>& data);
      Data<M,D>& operator=(const Data<M,D>& data);
};

template <typename M>
class DataLinear final : public Data<M, Vector<typename M::value_type> > {
   typedef typename M::value_type T;
   typedef Vector<T> D;
   using Data<M,D>::_X;
   using Data<M,D>::_scale_intercept;
   using Data<M,D>::_intercept;
   public:
      typedef M data_type;
      typedef Vector<T> variable_type;
      DataLinear(const M& X, const bool
            intercept = false) : Data<M,D>(X,intercept) { };

      inline void pred(const Vector<T>& input, Vector<T>& output) const {
         if (_intercept) {
            GET_w(input);
            _X.multTrans(w,output);
            output.add(input[input.n()-1]*_scale_intercept);
         } else {
            _X.multTrans(input,output);
         }
      };
      inline void pred(const int ind, const Vector<T>& input, T& output) const {
         output=this->pred(ind,input);
      };
      inline T pred(const int ind, const Vector<T>& input) const {
         typename M::col_type col;
         _X.refCol(ind,col);
         if (_intercept) {
            GET_w(input);
            return col.dot(w)+ input[input.n()-1]*_scale_intercept;
         } else {
            return col.dot(input);
         }
      };
      inline void add_dual_pred(const Vector<T>& input, Vector<T>& output,const T a=T(1.0), const T b=T(1.0)) const {
         if (_intercept) {
            const int m = _X.m();
            output.resize(m+1);
            GET_w(output);
            _X.mult(input,w,a,b);
            output[m] = _scale_intercept*a*input.sum() + b*output[m];
         } else {
            _X.mult(input,output,a,b);
         }
      };
      inline void add_dual_pred(const int ind, Vector<T>& output,const T a=T(1.0), const T b=T(1.0)) const {
         typename M::col_type col;
         _X.refCol(ind,col);
         if (_intercept) {
            const int m = _X.m();
            output.resize(m+1);
            GET_w(output);
            w.add_scal(col,a,b);
            output[m] = a*_scale_intercept + b*output[m];
         } else {
            output.resize(_X.m());
            output.add_scal(col,a,b);
         }
      };
      virtual void print() const  {
         cout << "Matrix X, n=" << _X.n() <<  ", p=" << _X.m() << endl; 
      };
      virtual void reverse_intercept(Vector<T>& x) {
         if (_scale_intercept != T(1.0))
            x[x.n()-1] *= _scale_intercept;
      };
      virtual void set_intercept(const Vector<T>& x0, Vector<T>& x) {
         _scale_intercept=sqrt(T(0.1)*_X.normFsq()/_X.n());
         x.copy(x0);
         x[x.n()-1] /= _scale_intercept;
      };


   private:
      inline void get_w(const Vector<T>& input, Vector<T>& w) const {
         const int n = input.n();
         input.refSubVec(0,n-1,w);
      };
};

/// prediction = W*X
template <typename M>
class DataMatrixLinear final : public Data<M, Matrix<typename M::value_type> > {
   typedef M data_type;
   typedef typename M::value_type T;
   typedef Matrix<T> D;
   using Data<M,D>::_X;
   using Data<M,D>::_scale_intercept;
   using Data<M,D>::_intercept;
   
   public:
   typedef Matrix<T> variable_type;
      DataMatrixLinear(const M& X, const bool
            intercept = false) : Data<M,D>(X,intercept) { 
         _ones.resize(_X.n());
         _ones.set(T(1.0));
      };

      // _X  is  p x n
      // input is nclass x p
      // output is nclass x n
      inline void pred(const Matrix<T>& input, Matrix<T>& output) const {
         if (_intercept) {
            GET_WB(input);
            W.mult(_X,output);
            output.rank1Update(b,_ones);
         } else {
            input.mult(_X,output);
         }
      };
      inline void pred(const int ind, const Matrix<T>& input, Vector<T>& output) const {
         typename M::col_type col;
         _X.refCol(ind,col);
         if (_intercept) {
            GET_WB(input);
            W.mult(col,output);
            output.add(b,_scale_intercept);
         } else {
            input.mult(col,output);
         }
      };
      inline void add_dual_pred(const Matrix<T>& input, Matrix<T>& output,const T a1=T(1.0), const T a2=T(1.0)) const {
         if (_intercept) { 
            output.resize(input.m(),_X.m()+1);
            GET_WB(output);
            input.mult(_X,W,false,true,a1,a2); //  W = input * X.T =  (X* input.T).T
            input.mult(_ones,b,a1,a2);
         } else {
            input.mult(_X,output,false,true,a1,a2);
         }
      };
      inline void add_dual_pred(const int ind, const Vector<T>& input, Matrix<T>& output,const T a=T(1.0), const T bb=T(1.0)) const {
         typename M::col_type col;
         _X.refCol(ind,col);
         if (bb != T(1.0)) output.scal(bb);
         if (_intercept) {
            output.resize(input.n(),_X.m()+1);
            GET_WB(output);
            W.rank1Update(input,col,a);
            b.add(input,a*_scale_intercept);
         } else {
            output.rank1Update(input,col,a);
         }
      };
      virtual void print() const  {
         cout << "Matrix X, n=" << _X.n() <<  ", p=" << _X.m() << endl; 
      };
      virtual DataLinear<M>* toDataLinear() const {
         return new DataLinear<M>(_X,_intercept);
      };
      virtual void reverse_intercept(Matrix<T>& x) {
         const int m=x.m();
         const int n=x.n();
         if (_scale_intercept != T(1.0))
            for (int ii=0; ii<n; ++ii)
               x[ii*m+m-1] *= _scale_intercept;
      };
      virtual void set_intercept(const Matrix<T>& x0, Matrix<T>& x) {
         _scale_intercept=sqrt(T(0.1)*_X.normFsq()/_X.n());
         _ones.set(_scale_intercept);
         x.copy(x0);
         const int m=x.m();
         const int n=x.n();
         for (int ii=0; ii<n; ++ii)
            x[ii*m+m-1] /= _scale_intercept;
      };



   private:
      inline void get_wb(const Matrix<T>& input, Matrix<T>& W, Vector<T>& b) const {
         const int p = input.n();
         input.refSubMat(0,p-1,W);
         input.refCol(p-1,b);
      };
      Vector<T> _ones;
};



#endif

