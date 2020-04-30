#ifndef LOSS_H 
#define LOSS_H 

#include "data.h"
#define VECM Vector<typename M::value_type>

enum loss_t { SQUARE, LOGISTIC, HINGE, SQHINGE, SAFE_LOGISTIC, MULTI_LOGISTIC, PPA, INCORRECT_LOSS };

static bool is_loss_for_matrices(const loss_t& loss) {
   return loss==SQUARE || loss==MULTI_LOGISTIC;
}

static bool is_regression_loss(const loss_t& loss) {
   return loss==SQUARE;
}

static loss_t loss_from_string(char* loss) {
   if (strcmp(loss,"square")==0) return SQUARE;
   if (strcmp(loss,"logistic")==0) return LOGISTIC;
   if (strcmp(loss,"sqhinge")==0) return SQHINGE;
   if (strcmp(loss,"hinge")==0) return HINGE;
   if (strcmp(loss,"safe-logistic")==0) return SAFE_LOGISTIC;
   if (strcmp(loss,"multiclass-logistic")==0) return MULTI_LOGISTIC;
   return INCORRECT_LOSS;
}

template <typename M, typename L, typename D>
class Loss {
   public:
      typedef typename D::value_type T;
      typedef typename D::value_type value_type;
      typedef D variable_type;
      typedef M data_type;
      typedef L label_type;
      typedef typename M::index_type index_type;

      Loss(Data<M,D>& data, const L& y) : _data2(data), _y(y) { };
      virtual ~Loss() { };
      static bool is_sparse() { return M::is_sparse; };

      /// functions that should be implemented in derived classes (see also in protected section)
      virtual T eval(const D& input) const = 0;
      virtual T eval(const D& input, const INTM i) const = 0;
      virtual void add_grad(const D& input, const INTM i, D& output, const T eta = T(1.0)) const = 0;// should be safe if output=input1
      virtual void scal_grad(const D& input, const INTM i, typename D::element& output) const = 0;
      virtual void add_feature(const D& input, D& output, const T s) const  = 0;
      virtual void add_feature(D& output, const INTM i, const typename D::element& s) const = 0;
      virtual T fenchel(const D& input) const = 0;
      virtual void print() const = 0;

      /// virtual functions that may be reimplemented, if needed
      virtual void double_add_grad(const D& input1, const D& input2, const INTM i, D& output, const T eta1 = T(1.0), const T eta2 = -T(1.0), const T dummy =T(1.0)) const {
         add_grad(input1,i,output,eta1);
         add_grad(input2,i,output,eta2);
      };
      virtual bool provides_fenchel() const { return true; };
      virtual T lipschitz() const {
         Vector<T> norms;
         _data2.norms_data(norms);
         return lipschitz_constant()*norms.maxval(); 
      };
      virtual bool transpose() const { 
         return false; 
      };
      virtual void grad(const D& input, D& output) const {
         D tmp;
         get_grad_aux(input,tmp);
         _data2.add_dual_pred(tmp,output,T(1.0)/tmp.n(),0);
      };
      virtual T eval_random_minibatch(const D& input, const INTM minibatch) const {
         T sum=0;
         const int n = this->n();
         for (int ii=0; ii<minibatch; ++ii) 
            sum +=eval(input,random() % n);
         return sum/minibatch;
      }
      virtual void grad_random_minibatch(const D& input, D& grad, const INTM minibatch) const {
         const int n = this->n();
         for (int ii=0; ii<minibatch; ++ii) { 
            this->add_grad(input, random() % n,grad,ii==0 ? 0 : T(1.0));
         }
         grad.scal(T(1.0)/minibatch);
      }
      virtual void lipschitz(Vector<T>& Li) const {
         _data2.norms_data(Li);
         Li.scal(lipschitz_constant());
      };
      virtual T kappa() const { return 0; };
      virtual void set_anchor_point(const D& z) {  };
      virtual void get_anchor_point(D& z) const {  };
      virtual void get_dual_variable(const D& input, D& grad1, D& grad2) const {
         get_grad_aux(input,grad1);
         get_dual_constraints(grad1);
         _data2.add_dual_pred(grad1,grad2,T(1.0)/grad1.n(),0);
      };

      /// non-virtual function classes
      inline loss_t id() const { return _id; };
      inline INTM n() const { return _y.n(); };
      inline void get_coordinates(const int ind, Vector<typename M::index_type>& indices) const { // todo change integers here 
         _data2.get_coordinates(ind,indices);
      };
      inline void set_intercept(const D& x0, D& x) { _data2.set_intercept(x0,x); };
      inline void reverse_intercept(D& x) { _data2.reverse_intercept(x); };
      inline bool intercept() const {
         return _data2.intercept();
      }
      inline void check_grad(const D& input, D& output) const { 
         output.copy(input);
         D x1, x2;
         x1.copy(input);
         x2.copy(input);
         for (int ii=0; ii<input.size(); ++ii)  {
            x1[ii] -= 1e-7;
            x2[ii] += 1e-7;
            output[ii]=(this->eval(x2)-this->eval(x1))/(2e-7);
            x1[ii] += 1e-7;
            x2[ii] -= 1e-7;
         }
      };
      //Data<M,D>& data() const { return _data; };
      const L& y() const { return _y; };

   protected:
      virtual void get_grad_aux(const D& input, D& grad1) const = 0;
      virtual T lipschitz_constant() const = 0;
      virtual void get_dual_constraints(D& grad1) const = 0;

      Data<M,D>& _data2;
      const L& _y;
      loss_t _id;

   private:
      explicit Loss<M,L,D>(const Loss<M,L,D>& loss);
      Loss<M,L,D>& operator=(const Loss<M,L,D>& loss);
};

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
         if (res1 || res2) _data.add_dual_pred(i,output,eta1*res1+eta2*res2);
      };
      virtual void add_feature(Vector<T>& output, const INTM i, const T& s) const {
         _data.add_dual_pred(i,output,s);
      };
      virtual void add_feature(const Vector<T>& input, Vector<T>& output, const T s) const { 
         _data.add_dual_pred(input,output,s,T(1.0));
      }
      virtual T scal_grad(const Vector<T>& input, const INTM ii) const {
         T s; scal_grad(input,ii,s);
         return s;
      }
      virtual void scal_grad(const Vector<T>& input, const INTM i, T& output) const = 0;
      DataLinear<M>& data() const { return _data;};

   protected:
      DataLinear<M>& _data;
};

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
         Vector<T> sgrad1, sgrad2;
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
         cout << "Square Loss is used" << endl;
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
         cout << "Logistic Loss is used" << endl;
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
         cout << "Squared Hinge Loss is used" << endl;
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
         cout << "Hinge Loss is used" << endl;
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
         cout << "Safe Logistic Loss is used" << endl;
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


template <typename loss_type>
class ProximalPointLoss final : public loss_type {
   public:
      typedef typename loss_type::value_type T;
      typedef typename loss_type::variable_type D;

      ProximalPointLoss(const loss_type& loss, const D& z, const T kappa) 
         : loss_type(loss.data(), loss.y()), _loss(loss), _kappa(kappa)  { 
            _z.copy(z);
            this->_id=PPA;
         };
      virtual ~ProximalPointLoss() { };

      inline T eval(const D& input) const {
         D tmp;
         tmp.copy(input);
         tmp.sub(_z);
         return _loss.eval(input)+T(0.5)*_kappa*tmp.nrm2sq();
      };
      inline T eval(const D& input, const INTM i) const {
         D tmp;
         tmp.copy(input);
         tmp.sub(_z);
         return _loss.eval(input,i)+T(0.5)*_kappa*tmp.nrm2sq();
      };
      inline void grad(const D& input, D& output) const {
         _loss.grad(input,output);
         output.add(input,_kappa);
         output.add(_z,-_kappa);
      };
      inline void add_grad(const D& input, const INTM i, D& output, const T eta = T(1.0)) const {
         _loss.add_grad(input,i,output,eta);
         output.add(input,_kappa*eta);
         output.add(_z,-_kappa*eta);
      };
      inline void double_add_grad(const D& input1, const D& input2, const INTM i, D& output, const T eta1 = T(1.0), const T eta2 = -T(1.0), const T dummy = T(1.0)) const {
         _loss.double_add_grad(input1,input2,i,output,eta1,eta2);
         if (dummy) {
            output.add(input1,dummy*_kappa*eta1);
            output.add(input2,dummy*_kappa*eta2);
            if (abs<T>(eta1+eta2) > EPSILON)
               output.add(_z,-_kappa*dummy*(eta1+eta2));
         }
      }
      virtual T eval_random_minibatch(const D& input, const INTM minibatch) const {
         const T sum=_loss.eval_random_minibatch(input,minibatch);
         D tmp;
         tmp.copy(input);
         tmp.sub(_z);
         return sum+T(0.5)*_kappa*tmp.nrm2sq();
      };
      virtual void grad_random_minibatch(const D& input, D& grad, const INTM minibatch) const {
         _loss.grad_random_minibatch(input,grad,minibatch);
         grad.add(input,_kappa);
         grad.add(_z,-_kappa);
      };
      inline void print() const {
         cout << "Proximal point loss with" << endl;
         _loss.print();
      }
      virtual bool provides_fenchel() const { return false; };
      virtual T fenchel(const D& input) const { return 0; };
      virtual T lipschitz() const { 
         return _loss.lipschitz() + _kappa;
      };
      virtual void lipschitz(Vector<T>& Li) const {
         _loss.lipschitz(Li);
         Li.add(_kappa);
      };
      virtual void scal_grad(const D& input, const INTM i, typename D::element& output) const  { 
         _loss.scal_grad(input,i,output);
      };
      virtual void  add_feature(const D& input, D& output, const T s) const { 
         _loss.add_feature(input,output,s);
      };
      virtual void  add_feature(D& output, const INTM i, const typename D::element& s) const {
         _loss.add_feature(output,i,s);
      };
      virtual void set_anchor_point(const D& z) { _z.copy(z); };
      virtual void get_anchor_point(D& z) const { z.copy(_z); };
      virtual T kappa() const { return _kappa; };
      virtual bool transpose() const { 
         return _loss.transpose();
      };
   protected:
      virtual void get_grad_aux(const D& input, D& grad1) const { 
         cerr << "Not used" << endl;
      };
      virtual T lipschitz_constant() const {
         cerr << "Not used" << endl;
         return 0;
      };
      virtual void get_dual_constraints(D& grad1) const {
         cerr << "Not used" << endl;
      };

   
   private:
      const loss_type& _loss;
      const T _kappa;
      D _z;
};
template <typename loss_type>
class LossMat : public LinearLossMat<typename loss_type::data_type, Matrix<typename loss_type::value_type> > {
//class LossMat : public Loss< typename loss_type::data_type, Matrix<typename loss_type::value_type>, Matrix<typename loss_type::value_type> > {
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
//#pragma omp parallel for reduction(+ : sum) num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col;
            input.refCol(ii,col);
            sum += _losses[ii]->eval(col,i);
         }
         return sum;
      };
      inline void add_grad(const Matrix<T>& input, const INTM i, Matrix<T>& output, const T eta = T(1.0)) const {
         output.resize(input.m(),input.n());
//#pragma omp parallel for num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col_input, col_output;
            input.refCol(ii,col_input);
            output.refCol(ii,col_output);
            _losses[ii]->add_grad(col_input,i,col_output,eta);
         }
      };
      inline void double_add_grad(const Matrix<T>& input1, const Matrix<T>& input2, const INTM i, Matrix<T>& output, const T eta1 = T(1.0), const T eta2 = -T(1.0), const T dummy =T(1.0)) const {
         output.resize(input1.m(),input1.n());
//#pragma omp parallel for num_threads(2)
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
#pragma omp parallel for 
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col_input, col_output;
            input.refCol(ii,col_input);
            output.refCol(ii,col_output);
            _losses[ii]->grad(col_input,col_output);
         }
      }
      inline void print() const {
         cout << "Loss for matrices" << endl;
         _losses[0]->print();
      };
      inline bool provides_fenchel() const {
         return _losses[0]->provides_fenchel();
      };
      inline void get_dual_variable(const Matrix<T>& input, Matrix<T>& grad1, Matrix<T>& grad2) const {
         grad1.resize(_n,input.n());
         grad2.resize(input.m(),input.n());
#pragma omp parallel for 
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
#pragma omp parallel for 
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col1, col2;
            input.copyRow(ii,col1);
            output.refCol(ii,col2);
            _losses[ii]->add_feature(col1,col2,s);
         }
      }
      virtual void  add_feature(Matrix<T>& output, const INTM i, const Vector<T>& s) const { 
//#pragma omp parallel for num_threads(2)
         for (int ii=0; ii<_N; ++ii) {
            Vector<T> col;
            output.refCol(ii,col);
            _losses[ii]->add_feature(col,i,s[ii]);
         } 
      };
      virtual void scal_grad(const Matrix<T>& input, const INTM i, Vector<T>& output) const {
         output.resize(_N);
//#pragma omp parallel for num_threads(2)
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
         cerr << "Not used" << endl;
      };
      virtual T lipschitz_constant() const {
         cerr << "Not used" << endl;
         return 0;
      };
      virtual void get_dual_constraints(Matrix<T>& grad1) const {
         cerr << "Not used" << endl;
      };

   protected:
      int _N;
      int _n;
      loss_type** _losses;
      DataLinear<M>** _datas;
      Matrix<T> _yT;
};

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
         cout << "Square Loss is used" << endl;
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

template <typename M>
class MultiClassLogisticLoss final : public LinearLossMat<M, Vector<int> > {
   typedef typename M::value_type T;
   using LinearLossMat<M, Vector<int> >::_data;
   using LinearLossMat<M, Vector<int> >::_y;
   public:
      MultiClassLogisticLoss(DataMatrixLinear<M>& data, const Vector<int>& y) : LinearLossMat<M, Vector<int> >(data,y) { 
         _nclasses=y.maxval()+1;
         this->_id=MULTI_LOGISTIC;
      };

      inline T eval(const Matrix<T>& input) const {
         Matrix<T> tmp;
         _data.pred(input,tmp); 
         const int n = tmp.n();
         T sum=0;
#pragma omp parallel for reduction(+:sum) schedule(static)
         for (int ii = 0; ii<n; ++ii) {
            Vector<T> col;
            tmp.refCol(ii,col);
            col.add(-col[_y[ii]]);
            sum+=col.logsumexp();
         }
         return sum / n;
      };
      inline T eval(const Matrix<T>& input, const INTM i) const {
         Vector<T> tmp;
         _data.pred(i,input,tmp); 
         tmp.add(-tmp[_y[i]]);
         return tmp.logsumexp();
      }
      inline void print() const {
         cout << "Multiclass logistic Loss is used" << endl;
      };
     inline T fenchel(const Matrix<T>& input) const {
        T sum = 0;
        const int n = input.n();
#pragma omp parallel for reduction(+:sum) schedule(static) 
        for (int i = 0; i<n; ++i) {
           const int clas = _y[i];
           for (int j = 0; j<_nclasses; ++j) {
              if (j == clas) {
                 sum += xlogx(input[i*_nclasses+j]+1.0);
              } else {
                 sum += xlogx(input[i*_nclasses+j]);
              }
           }
        }
        return sum/n;
     };

   private:
     int _nclasses; 
     inline void get_grad_aux2(Vector<T>& col, const int ind) const {
        col.add(-col[ind]);
        const T mm = col.maxval();
        col.add(-mm);
        col.exp();
        col.scal(T(1.0)/col.asum());
        col[ind]=0;
        col[ind]=-col.asum();
     }
     inline void get_grad_aux(const Matrix<T>& input, Matrix<T>& grad1) const {
         _data.pred(input,grad1);
         const int n = grad1.n();
#pragma omp parallel for schedule(static,16)
         for (int ii = 0; ii<n; ++ii) {
            Vector<T> col;
            grad1.refCol(ii,col);
            get_grad_aux2(col,_y[ii]);
         }
      };
      inline void scal_grad(const Matrix<T>& input, const INTM i, Vector<T>& col) const {
          _data.pred(i,input,col);
          get_grad_aux2(col,_y[i]);
      };
      inline T lipschitz_constant() const { return T(0.25);};
      inline void get_dual_constraints(Matrix<T>& grad1) const {
         // scale grad1 by 1/Nclasses
         if (_data.intercept()) {
            Vector<T> row;
            for (int i = 0; i<grad1.m(); ++i) {
               grad1.extractRow(i,row);
               row.project_sft(_y,i);
               grad1.setRow(i,row);
            }
         }
      }
};




#endif


