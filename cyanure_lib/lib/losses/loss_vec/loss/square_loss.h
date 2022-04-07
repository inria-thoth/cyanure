#ifndef SQUARE_LOSS_VEC_H
#define SQUARE_LOSS_VEC_H

#include "../loss_vec.h"

template <typename M>
class SquareLoss final : public LinearLossVec<M> {
   public:
      typedef typename M::value_type T;
      using LinearLossVec<M>::_data;
      using LinearLossVec<M>::_y;
      SquareLoss(DataLinear<M>& data, const Vector<T>& y) : LinearLossVec<M>(data,y) {
         this->_id=SQUARE;
      };

      inline T eval(const Vector<T>& input) const {
         Vector<T> tmp;
         _data.pred(input,tmp);
         tmp.sub(_y);
         return T(0.5)*tmp.nrm2sq()/tmp.n();
      };
      inline T eval(const Vector<T>& input, const INTM i) const {
         const T res = _y[i] - _data.pred(i,input);
         return T(0.5)*res*res;
      };
      inline void print() const {
         logging(logINFO) << "Square Loss is used";
      };
      inline T fenchel(const Vector<T>& input) const {
         return 0.5*input.nrm2sq()/input.n()+input.dot(_y)/input.n();
      };
      inline void scal_grad(const Vector<T>& input, const INTM i, T& s) const {
         s=_data.pred(i,input) - _y[i];
      };

   private:
      inline void get_grad_aux(const Vector<T>& input, Vector<T>& grad1) const {
         _data.pred(input,grad1);
         grad1.sub(_y);
      };
      inline T lipschitz_constant() const { return T(1.0);};
      inline void get_dual_constraints(Vector<T>& grad1) const {
         if (_data.intercept())
            grad1.add(-grad1.mean());
      };
};

#endif