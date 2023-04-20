#ifndef LOSS_H 
#define LOSS_H 

#include "../data_structure/data.h"
#include "../timer.h"


#define VECM Vector<typename M::value_type>

enum loss_t { SQUARE, LOGISTIC, HINGE, SQHINGE, SAFE_LOGISTIC, MULTI_LOGISTIC, PPA, INCORRECT_LOSS };

static loss_t loss_from_string(char* loss) {
   if (strcmp(loss,"square")==0) 
    return SQUARE;
   if (strcmp(loss,"logistic")==0) 
    return LOGISTIC;
   if (strcmp(loss,"sqhinge")==0) 
    return SQHINGE;
   if (strcmp(loss,"hinge")==0) 
    return HINGE;
   if (strcmp(loss,"safe-logistic")==0) 
    return SAFE_LOGISTIC;
   if (strcmp(loss,"multiclass-logistic")==0) 
    return MULTI_LOGISTIC;
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
            sum +=eval(input,random_r() % n);
         return sum/minibatch;
      }
      virtual void grad_random_minibatch(const D& input, D& grad, const INTM minibatch) const {
         const int n = this->n();
         for (int ii=0; ii<minibatch; ++ii) { 
            this->add_grad(input, random_r() % n,grad,ii==0 ? 0 : T(1.0));
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
         logging(logINFO) << "Proximal point loss with";
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
         logging(logERROR) << "Not used";
      };
      virtual T lipschitz_constant() const {
         logging(logERROR) << "Not used";
         return 0;
      };
      virtual void get_dual_constraints(D& grad1) const {
         logging(logERROR) << "Not used";
      };

   
   private:
      const loss_type& _loss;
      const T _kappa;
      D _z;
};


#endif


