#ifndef HINGE_LOSS_H
#define HINGE_LOSS_H

#include "../loss_vec.h"

template <typename M>
class HingeLoss final : public LinearLossVec<M> {
   public:
      typedef typename M::value_type T;
      using LinearLossVec<M>::_data;
      using LinearLossVec<M>::_y;
      HingeLoss(DataLinear<M>& data, const Vector<T>& y) : LinearLossVec<M>(data,y) { 
         this->_id=HINGE;
      };

      inline T eval(const Vector<T>& input) const {
         Vector<T> tmp;
         _data.pred(input,tmp);
         tmp.mult(_y,tmp);
         tmp.neg();
         tmp.add(T(1.0));
         tmp.thrsPos();
         return tmp.sum()/tmp.n(); 
      };
      inline T eval(const Vector<T>& input, const INTM i) const {
         return MAX(T(1.0)-_y[i]*_data.pred(i,input),0);
      };

      inline void print() const {
         logging(logINFO) << "Hinge Loss is used";
      };

      inline T fenchel(const Vector<T>& input) const {
         const int n = input.n();
         return input.dot(_y)/n; 
      };

      inline void scal_grad(const Vector<T>& input, const INTM i, T& s) const {
         const T y = _y[i];
         const T ss = (T(1.0)-y*_data.pred(i,input))/_data.norms(i);
         s = -y*MIN(MAX(ss,0),T(1.0));
      };

   private:
      inline void get_grad_aux(const Vector<T>& input, Vector<T>& grad1) const {
         _data.pred(input,grad1);
         const int n = grad1.n();
         for (int ii=0; ii<n; ++ii) {
            const T ss = (T(1.0)-_y[ii]*grad1[ii])/_data.norms(ii);
            grad1[ii]=-_y[ii]*MIN(MAX(ss,0),T(1.0));
         }
      };

      inline T lipschitz_constant() const { return T(1.0);}; 

      inline void get_dual_constraints(Vector<T>& grad1) const {
         if (_data.intercept()) 
            grad1.project_sft_binary(_y);
      }
};

#endif