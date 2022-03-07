#ifndef SAFE_LOGISTIC_LOSS_H
#define SAFE_LOGISTIC_LOSS_H

#include "../loss_vec.h"

template <typename M>
class SafeLogisticLoss final : public LinearLossVec<M> {
   public:
      typedef typename M::value_type T;
      using LinearLossVec<M>::_data;
      using LinearLossVec<M>::_y;
      using LinearLossVec<M>::scal_grad;

      SafeLogisticLoss(DataLinear<M>& data, const Vector<T>& y) : LinearLossVec<M>(data,y) { 
         this->_id=SAFE_LOGISTIC;
      };

      inline T eval(const Vector<T>& input) const {
         Vector<T> tmp;
         _data.pred(input,tmp);
         tmp.mult(_y,tmp);
         const int n = tmp.n();
         for (int ii=0; ii<n; ++ii) 
            tmp[ii] = tmp[ii] <= T(1.0) ? exp_alt<T>(tmp[ii]-T(1.0)) - tmp[ii] : 0;
         return tmp.sum()/tmp.n(); 
      };
      inline T eval(const Vector<T>& input, const INTM i) const {
         const T res = _y[i]*_data.pred(i,input);
         return res <= T(1.0) ? exp_alt<T>(res-T(1.0)) - res : 0;
      };
      inline void print() const {
         logging(logINFO) << "Safe Logistic Loss is used";
      };
      inline T fenchel(const Vector<T>& input) const {
         T sum=0;
         const int n = input.n();
         for (int ii = 0; ii<n; ++ii) {  
            T prod = _y[ii]*input[ii];
            sum += xlogx(1.0+prod); 
         }
         return sum/n;
      };
      inline void scal_grad(const Vector<T>& input, const INTM i, T& s) const {
         const T y = _y[i];
         const T ss = y*_data.pred(i,input);
         s=ss <= T(1.0) ? y*(exp_alt<T>(ss-T(1.0))-T(1.0)) : 0; 
      };

   private:
      inline void get_grad_aux(const Vector<T>& input, Vector<T>& grad1) const {
         _data.pred(input,grad1);
         grad1.mult(_y,grad1);
         grad1.add(-T(1.0));
         grad1.exp();
         grad1.add(-T(1.0));
         grad1.thrsmin(0);
         grad1.mult(_y,grad1);
      };
      inline T lipschitz_constant() const { return T(1.0);};
      inline void get_dual_constraints(Vector<T>& grad1) const {
         if (_data.intercept()) 
            grad1.project_sft_binary(_y);
      }
};

#endif