#ifndef SQUARE_LOSS_MAT_H
#define SQUARE_LOSS_MAT_H

#include "../loss_mat.h"

template <typename M>
class SquareLossMat final : public LinearLossMat<M, Matrix<typename M::value_type> > {
   public:
      typedef typename M::value_type T;
      using LinearLossMat<M, Matrix<T> >::_data;
      using LinearLossMat<M, Matrix<T> >::_y;
      SquareLossMat(DataMatrixLinear<M>& data, const Matrix<T>& y) : LinearLossMat<M, Matrix<T> >(data,y) { 
         this->_id=SQUARE;
      };

      inline T eval(const Matrix<T>& input) const {
         Matrix<T> tmp;
         _data.pred(input,tmp); 
         tmp.sub(_y);  
         return T(0.5)*tmp.normFsq()/(tmp.n());
      };
      inline T eval(const Matrix<T>& input, const INTM i) const {
         Vector<T> tmp, col;
         _data.pred(i,input,tmp); 
         _y.refCol(i,col);
         tmp.sub(col);  
         return T(0.5)*tmp.nrm2sq();
      };
      inline void print() const {
         logging(logINFO) << "Square Loss is used";
      };
      inline T fenchel(const Matrix<T>& input) const {
         return 0.5*input.normFsq()/(input.n())+input.dot(_y)/(input.n());
      };

   protected:
      inline void get_grad_aux(const Matrix<T>& input, Matrix<T>& grad1) const {
         _data.pred(input,grad1);
         grad1.sub(_y);
      };
      inline void scal_grad(const Matrix<T>& input, const INTM i, Vector<T>& output) const {
         _data.pred(i,input,output);
         Vector<T> ycol;
         _y.refCol(i,ycol);
         output.sub(ycol);
      };
      inline T lipschitz_constant() const { return T(1.0);};
      inline void get_dual_constraints(Matrix<T>& grad1) const {
         if (_data.intercept()) 
            grad1.center_rows();
      }
};

#endif