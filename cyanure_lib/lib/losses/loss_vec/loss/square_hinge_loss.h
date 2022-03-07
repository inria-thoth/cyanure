#ifndef SQUARE_HINGE_LOSS_H
#define SQUARE_HINGE_LOSS_H

#include "../loss_vec.h"

template <typename M>
class SquaredHingeLoss final : public LinearLossVec<M> {
   public:
      typedef typename M::value_type T;
      using LinearLossVec<M>::_data;
      using LinearLossVec<M>::_y;
      SquaredHingeLoss(DataLinear<M>& data, const Vector<T>& y) : LinearLossVec<M>(data,y) { 
         this->_id=SQHINGE;
      };

      inline T eval(const Vector<T>& input) const {
         Vector<T> tmp;
         _data.pred(input,tmp);
         tmp.mult(_y,tmp);
         tmp.neg();
         tmp.add(T(1.0));
         tmp.thrsPos();
         tmp.sqr();
         return T(0.5)*tmp.sum()/tmp.n(); 
      };
      inline T eval(const Vector<T>& input, const INTM i) const {
         const T res = MAX(T(1.0)-_y[i]*_data.pred(i,input),0);
         return T(0.5)*res*res;
      };

      inline void print() const {
         logging(logINFO) << "Squared Hinge Loss is used";
      };
      inline T fenchel(const Vector<T>& input) const {
         const int n = input.n();
         return T(0.5)*input.nrm2sq()/n + input.dot(_y)/n; 
      };
      inline void scal_grad(const Vector<T>& input, const INTM i, T& s) const {
         const T y = _y[i];
         const T ss = _data.pred(i,input);
         s=y*ss > 1 ? 0 : ss-y;
      };

   private:
      inline void get_grad_aux(const Vector<T>& input, Vector<T>& grad1) const {
         _data.pred(input,grad1);
         grad1.mult(_y,grad1);
         grad1.neg();
         grad1.add(T(1.0));
         grad1.thrsPos();
         grad1.neg();
         grad1.mult(_y,grad1);
      };
      inline T lipschitz_constant() const { return T(1.0);};
      inline void get_dual_constraints(Vector<T>& grad1) const {
         if (_data.intercept()) {
            T sumpos=0;
            T sumneg=0;
            const int n = grad1.n();
            for (int ii=0; ii<n; ++ii)
               if (grad1[ii] < 0) {
                  sumneg+=grad1[ii];
               } else {
                  sumpos+=grad1[ii];
               }
            if (sumpos > -sumneg) {
               const T scal = -sumneg/sumpos;
               for (int ii=0; ii<n; ++ii)
                  if (grad1[ii] > 0) grad1[ii] *= scal;
            } else {
               const T scal = -sumpos/sumneg;
               for (int ii=0; ii<n; ++ii)
                  if (grad1[ii] < 0) grad1[ii] *= scal;
            }
         }
      }
};

#endif