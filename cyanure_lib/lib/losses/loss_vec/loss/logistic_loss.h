#ifndef LOGISTIC_LOSS_H
#define LOGISTIC_LOSS_H

#include "../loss_vec.h"

template <typename M>
class LogisticLoss final : public LinearLossVec<M> {
   public:
      typedef typename M::value_type T;
      using LinearLossVec<M>::_data;
      using LinearLossVec<M>::_y;

      LogisticLoss(DataLinear<M>& data, const Vector<T>& y) : LinearLossVec<M>(data,y) { 
         this->_id=LOGISTIC;
      };

      inline T eval(const Vector<T>& input, const INTM i) const {
         const T res = _y[i]*_data.pred(i,input);
         return logexp2<T>(-res);
      };
      inline T eval(const Vector<T>& input) const {
         Vector<T> tmp;
         _data.pred(input,tmp);
         tmp.mult(_y,tmp);
         tmp.neg();
         tmp.logexp();
         return tmp.sum()/tmp.n(); 
      };
      inline void print() const {
         logging(logINFO) << "Logistic Loss is used";
      };
      inline T fenchel(const Vector<T>& input) const {
         T sum=0;
         const int n = input.n();
         for (int ii = 0; ii<n; ++ii) {  
            T prod = _y[ii]*input[ii];
            sum += xlogx(1.0+prod)+xlogx(-prod); 
         }
         return sum/n;
      };
      inline void scal_grad(const Vector<T>& input, const INTM i, T& s) const {
         const T y = _y[i];
         const T ss = _data.pred(i,input);
         s=-y/(T(1.0)+exp_alt<T>(y*ss));
      };

   private:
      inline void get_grad_aux(const Vector<T>& input, Vector<T>& grad1) const {
         _data.pred(input,grad1);
         grad1.mult(_y,grad1);
         grad1.exp();
         grad1.add(T(1.0));
         grad1.inv();
         grad1.mult(_y,grad1);
         grad1.neg();
      };
      inline T lipschitz_constant() const { return T(0.25);};
      inline void get_dual_constraints(Vector<T>& grad1) const {
         if (_data.intercept()) 
            grad1.project_sft_binary(_y);
      }
};

#endif