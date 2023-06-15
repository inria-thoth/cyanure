#ifndef LOSS_MAT_H
#define LOSS_MAT_H

#include "../loss.h"


template <typename M, typename L>
class LinearLossMat : public Loss< M, L, Matrix<typename M::value_type> > {
   public:
      typedef Loss< M, L, Matrix<typename M::value_type> > loss_type;
      typedef typename M::value_type T;

      LinearLossMat(DataMatrixLinear<M>& data, const L& y) : loss_type(data,y), _data(data) { };
      virtual void add_grad(const Matrix<T>& input, const INTM i, Matrix<T>& output, const T a = T(1.0)) const {
         Vector<T> sgrad;
         scal_grad(input,i,sgrad);
         _data.add_dual_pred(i,sgrad,output,a);
      };
      inline void double_add_grad(const Matrix<T>& input1, const Matrix<T>& input2, const INTM i, Matrix<T>& output, const T eta1 = T(1.0), const T eta2 = -T(1.0), const T dummy = T(1.0)) const {
         Vector<T> sgrad2;
         Vector<T> sgrad1;
         scal_grad(input1,i,sgrad1);
         scal_grad(input2,i,sgrad2);
         sgrad1.add_scal(sgrad2,eta2,eta1);
         _data.add_dual_pred(i,sgrad1,output);
      };
      virtual bool transpose() const { 
         return true; 
      };
      virtual void  add_feature(const Matrix<T>& input, Matrix<T>& output, const T s) const { 
         _data.add_dual_pred(input,output,s,T(1.0));
      }
      virtual void  add_feature(Matrix<T>& output, const INTM i, const Vector<T>& s) const { 
         _data.add_dual_pred(i,s,output,T(1.0),T(1.0));
      };
  
      virtual void scal_grad(const Matrix<T>& input, const INTM i, Vector<T>& output) const = 0;

      DataMatrixLinear<M>& data() const { return _data;};

   protected:
      DataMatrixLinear<M>& _data;
};

template <typename loss_type>
class LossMat : public LinearLossMat<typename loss_type::data_type, Matrix<typename loss_type::value_type> > {
   public:
      typedef typename loss_type::value_type T;
      typedef typename loss_type::data_type M;
      typedef typename loss_type::label_type L;
      typedef LinearLossMat<M, Matrix<T> > base_loss;

      LossMat(DataMatrixLinear<M>& data, const Matrix<T>& y) : base_loss(data,y), _N(y.m()) {
         _losses=new loss_type*[_N];
         _datas=new DataLinear<M>*[_N];
         _n = y.n();
         y.transpose(_yT);
         Vector<T> ycol;
         for (int i = 0; i<_N; ++i) {
            _datas[i]=data.toDataLinear(); 
            _yT.refCol(i,ycol);
            _losses[i]=new loss_type(*(_datas[i]),ycol);
         }
         this->_id=_losses[0]->id();
      };
      virtual ~LossMat() {
         for (int i = 0; i<_N; ++i) {
            delete(_losses[i]);
            delete(_datas[i]);
            _losses[i]=NULL;
            _datas[i]=NULL;
         }
         delete[](_losses);
         delete[](_datas);
      };

      inline T eval(const Matrix<T>& input) const {
         T sum=0;
#pragma omp parallel for reduction(+ : sum)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col;
            input.refCol(ii,col);
            sum += _losses[ii]->eval(col);
         }
         return sum;
      };
      inline T eval(const Matrix<T>& input, const INTM i) const {
         T sum=0;
#pragma omp parallel for reduction(+ : sum) num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col;
            input.refCol(ii,col);
            sum += _losses[ii]->eval(col,i);
         }
         return sum;
      };
      inline void add_grad(const Matrix<T>& input, const INTM i, Matrix<T>& output, const T eta = T(1.0)) const {
         output.resize(input.m(),input.n());
#pragma omp parallel for num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col_input, col_output;
            input.refCol(ii,col_input);
            output.refCol(ii,col_output);
            _losses[ii]->add_grad(col_input,i,col_output,eta);
         }
      };
      inline void double_add_grad(const Matrix<T>& input1, const Matrix<T>& input2, const INTM i, Matrix<T>& output, const T eta1 = T(1.0), const T eta2 = -T(1.0), const T dummy =T(1.0)) const {
         output.resize(input1.m(),input1.n());
#pragma omp parallel for num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col_input1, col_input2, col_output;
            input1.refCol(ii,col_input1);
            input2.refCol(ii,col_input2);
            output.refCol(ii,col_output);
            _losses[ii]->double_add_grad(col_input1,col_input2,i,col_output,eta1,eta2,dummy);
         }
      };
      inline void grad(const Matrix<T>& input, Matrix<T>& output) const {
         output.resize(input.m(),input.n());
#pragma omp parallel for ordered
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col_input, col_output;
            input.refCol(ii,col_input);
            output.refCol(ii,col_output);
            _losses[ii]->grad(col_input,col_output);
         }
      }
      inline void print() const {
         logging(logINFO) << "Loss for matrices";
         _losses[0]->print();
      };
      inline bool provides_fenchel() const {
         return _losses[0]->provides_fenchel();
      };
      inline void get_dual_variable(const Matrix<T>& input, Matrix<T>& grad1, Matrix<T>& grad2) const {
         grad1.resize(_n,input.n());
         grad2.resize(input.m(),input.n());
#pragma omp parallel for ordered         
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col1, col2, col3;
            input.refCol(ii,col1);
            grad1.refCol(ii,col2);
            grad2.refCol(ii,col3);
            _losses[ii]->get_dual_variable(col1,col2,col3);
         }
      };
      inline T fenchel(const Matrix<T>& input) const {  
         T sum=0;
#pragma omp parallel for reduction(+ : sum) 
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col;
            input.copyCol(ii,col);
            sum+=_losses[ii]->fenchel(col);
         }
         return sum;
      };
      inline T lipschitz() const {
         return _losses[0]->lipschitz();
      }
      inline void lipschitz(Vector<T>& Li) const {
         _losses[0]->lipschitz(Li);
      };
      // input; nclass x n
      // output: p x nclass
      virtual void  add_feature(const Matrix<T>& input, Matrix<T>& output, const T s) const { 
#pragma omp parallel for ordered
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col1, col2;
            input.copyRow(ii,col1);
            output.refCol(ii,col2);
            _losses[ii]->add_feature(col1,col2,s);
         }
      }
      virtual void  add_feature(Matrix<T>& output, const INTM i, const Vector<T>& s) const { 
#pragma omp parallel for num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col;
            output.refCol(ii,col);
            _losses[ii]->add_feature(col,i,s[ii]);
         } 
      };
      virtual void scal_grad(const Matrix<T>& input, const INTM i, Vector<T>& output) const {
         output.resize(_N);
#pragma omp parallel for num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col;
            input.refCol(ii,col);
            _losses[ii]->scal_grad(col,i,output[ii]);
         }
      };
      virtual bool transpose() const { 
         return false; 
      };
      virtual void get_grad_aux(const Matrix<T>& input, Matrix<T>& grad1) const { 
         logging(logERROR) << "Not used";
      };
      virtual T lipschitz_constant() const {
         logging(logERROR) << "Not used";
         return 0;
      };
      virtual void get_dual_constraints(Matrix<T>& grad1) const {
         logging(logERROR) << "Not used";
      };

   protected:
      int _N;
      int _n;
      loss_type** _losses;
      DataLinear<M>** _datas;
      Matrix<T> _yT;
};

#endif
