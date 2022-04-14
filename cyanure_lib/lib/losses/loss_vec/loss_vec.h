#ifndef LOSS_VEC_H
#define LOSS_VEC_H

#include "../loss.h"

template <typename M>
class LinearLossVec : public Loss< M, VECM, VECM > {
   public:
      typedef typename M::value_type T;
      typedef Loss<M, VECM, VECM> loss_type;
      //using loss_type::_data2;

      LinearLossVec(DataLinear<M>& data, const Vector<T>& y) : loss_type(data,y), _data(data) { };
      inline void add_grad(const Vector<T>& input, const INTM i, Vector<T>& output, const T a = T(1.0)) const {
         T s=scal_grad(input,i);
         _data.add_dual_pred(i,output,a*s);
      };
      inline void double_add_grad(const Vector<T>& input1, const Vector<T>& input2, const INTM i, Vector<T>& output, const T eta1 = T(1.0), const T eta2 = -T(1.0), const T dummy = T(1.0)) const {
         T res1=scal_grad(input1,i);
         T res2=scal_grad(input2,i);
         if (res1 || res2) 
            _data.add_dual_pred(i,output,eta1*res1+eta2*res2);
      };
      virtual void add_feature(Vector<T>& output, const INTM i, const T& s) const {
         _data.add_dual_pred(i,output,s);
      };
      virtual void add_feature(const Vector<T>& input, Vector<T>& output, const T s) const { 
         _data.add_dual_pred(input,output,s,T(1.0));
      }
      virtual T scal_grad(const Vector<T>& input, const INTM ii) const {
         T s; 
         scal_grad(input,ii,s);
         return s;
      }
      virtual void scal_grad(const Vector<T>& input, const INTM i, T& output) const = 0;
      DataLinear<M>& data() const { return _data;};

   protected:
      DataLinear<M>& _data;
};

#endif