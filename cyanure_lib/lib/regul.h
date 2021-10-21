#ifndef REGUL_H 
#define REGUL_H

#include "linalg.h"
Timer timer_global, timer_global2, timer_global3;

enum regul_t { L2, L1, ELASTICNET, L1BALL, L2BALL, FUSEDLASSO, L1L2, L1LINF, NONE, L1L2_L1, INCORRECT_REG };

static bool is_regul_for_matrices(const regul_t& reg) {
   return reg==L1L2 || reg==L1LINF;
}

template <typename T> struct ParamModel { 
   ParamModel() { regul=NONE; lambda=0; lambda2=0; lambda3=0; intercept=false; loss=SQUARE; };
   loss_t loss;
   regul_t regul;
   T lambda;
   T lambda2;
   T lambda3;
   bool intercept;
};

template <typename T> 
void clean_param_model(ParamModel<T>& param) {
   if (param.regul==FUSEDLASSO && param.lambda==0) {
         param.regul=ELASTICNET; 
         param.lambda=param.lambda2;
         param.lambda2=param.lambda3; 
   };
   if (param.regul==ELASTICNET) {
      if (param.lambda==0) { param.regul=L2; param.lambda=param.lambda2; };
      if (param.lambda2==0) param.regul=L1;
      if (param.lambda==0 && param.lambda2==0) param.regul=NONE;
   } else {
      if (param.lambda==0)
         param.regul=NONE;
   }
}

static regul_t regul_from_string(char* regul) {
   if (strcmp(regul,"l1")==0) return L1;
   if (strcmp(regul,"l1-ball")==0) return L1BALL;
   if (strcmp(regul,"fused-lasso")==0) return FUSEDLASSO;
   if (strcmp(regul,"l2")==0) return L2;
   if (strcmp(regul,"l2-ball")==0) return L2BALL;
   if (strcmp(regul,"elastic-net")==0) return ELASTICNET;
   if (strcmp(regul,"l1l2")==0) return L1L2;
   if (strcmp(regul,"l1l2+l1")==0) return L1L2_L1;
   if (strcmp(regul,"l1linf")==0) return L1LINF;
   if (strcmp(regul,"none")==0) return NONE;
   return INCORRECT_REG;
}


template <typename D, typename I>
class Regularizer {
   public:
      typedef typename D::value_type T;
      typedef I index_type;

      Regularizer(const ParamModel<T>& model) : _intercept(model.intercept), _id(model.regul) { };
      virtual ~Regularizer() { };

      virtual void prox(const D& input, D& output, const T eta) const = 0; // should be able to do inplace with output=input
      virtual T eval(const D& input) const = 0;
      virtual T fenchel(D& grad1, D& grad2) const = 0;
      virtual void print() const = 0;


      virtual bool is_lazy() const { return false; };
      virtual void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const { };
      
      virtual bool provides_fenchel() const { return true; };
      virtual regul_t id() const { return _id;};
      virtual bool intercept() const { return _intercept;};
      virtual T strong_convexity() const { return 0; };
      virtual T lambda() const { return 0;};

   protected:
      const bool _intercept;

   private:
      explicit Regularizer<D,I>(const Regularizer<D,I>& reg);
      Regularizer<D,I>& operator=(const Regularizer<D,I>& reg);
      const regul_t _id;
};

template <typename D, typename I>
class None final : public Regularizer<D,I> {
   public:
      typedef typename D::value_type T;

      None(const ParamModel<T>& model) : Regularizer<D,I>(model) {  };
      virtual void prox(const D& input, D& output, const T eta) const {
         output.copy(input);
      };
      inline T eval(const D& input) const { return 0; };
      inline T fenchel(D& grad1, D& grad2) const { return 0; };
      bool provides_fenchel() const { return false; };
      void print() const { 
         cout << "No regularization" << endl;
      }
};

template <typename D, typename I>
class Ridge final : public Regularizer<D,I> {
   public:
      typedef typename D::value_type T;

      Ridge(const ParamModel<T>& model) : Regularizer<D,I>(model), _lambda(model.lambda) {  };

      inline void prox(const D& input, D& output, const T eta) const {
         output.copy(input);
         output.scal(T(1.0/(1.0+_lambda*eta)));
         if (this->_intercept) {
            const int n = input.n();
            output[n-1]=input[n-1];
         }
      };
      inline T eval(const D& input) const { 
         const int n = input.n();
         const T res = input.nrm2sq();
         return (this->_intercept ? T(0.5)*_lambda*(res - input[n-1]*input[n-1]) : T(0.5)*_lambda*res);
      };
      inline T fenchel(D& grad1, D& grad2) const { 
         return (this->_intercept & (abs<T>(grad2[grad2.n()-1]) >
                  1e-6)) ? INFINITY : this->eval(grad2)/(_lambda*_lambda); 
      };
      void print() const { 
         cout << "L2 regularization" << endl;
      }
      virtual T strong_convexity() const { return this->_intercept ? 0 : _lambda; };
      virtual T lambda() const { return _lambda; };
      inline void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const { 
         const T scal = T(1.0)/(T(1.0)+_lambda*eta);
         const int p = input.n();
         const int r = indices.n();
         for (int jj=0; jj<r; ++jj) 
            output[indices[jj]]=scal*input[indices[jj]];  
         if (this->_intercept) output[p-1]=input[p-1];
      };
      virtual bool is_lazy() const { return true; };

   private:
      const T _lambda;
};

template <typename D, typename I>
class Lasso final : public Regularizer<D,I> {
   public:
      typedef typename D::value_type T;

      Lasso(const ParamModel<T>& model) : Regularizer<D,I>(model), _lambda(model.lambda) {  };
      inline void prox(const D& input, D& output, const T eta) const {
         input.fastSoftThrshold(output,eta*_lambda);
         if (this->_intercept) {
            const int n = input.n();
            output[n-1]=input[n-1];
         }
      };
      inline T eval(const D& input) const { 
         const int n = input.n();
         const T res = input.asum();
         return (this->_intercept ? _lambda*(res - abs<T>(input[n-1])) : _lambda*res);
      };
      inline T fenchel(D& grad1, D& grad2) const { 
         const T mm = grad2.fmaxval();
         if (mm > _lambda) 
            grad1.scal(_lambda/mm);
         return (this->_intercept & (abs<T>(grad2[grad2.n()-1]) >
                  1e-6)) ? INFINITY : 0; 
      };
      void print() const { 
         cout << "L1 regularization" << endl;
      }
      virtual T lambda() const { return _lambda;};
      inline void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const { 
         const int p = input.n();
         const int r = indices.n();
         const T thrs=_lambda*eta;
//#pragma omp parallel for
         for (int jj=0; jj<r; ++jj) 
            output[indices[jj]]=fastSoftThrs(input[indices[jj]],thrs);;  
         if (this->_intercept) output[p-1]=input[p-1];
      };
      virtual bool is_lazy() const { return true; };

   private:
      const T _lambda;
};

template <typename D, typename I>
class ElasticNet final : public Regularizer<D,I> {
   public:
      typedef typename D::value_type T;

      // min_x 0.5|y-x|^2 + lambda_1 |x|  + 0.5 lambda_2 x^2
      // min_x - y x + 0.5 x^2  + lambda_1 |x|  + 0.5 lambda_2 x^2
      // min_x - y x + 0.5 (1+lambda_2) x^2  + lambda_1 |x|  
      // min_x - y/(1+lambda2) x + 0.5  x^2  + lambda_1/(1+lambda2) |x|  
      ElasticNet(const ParamModel<T>& model) : Regularizer<D,I>(model), _lambda(model.lambda), _lambda2(model.lambda2) { 
      };
      inline void prox(const D& input, D& output, const T eta) const {
         output.copy(input);
         output.fastSoftThrshold(_lambda*eta);
         output.scal(T(1.0)/(1+_lambda2*eta));
         if (this->_intercept) {
            const int n = input.n();
            output[n-1]=input[n-1];
         }
      };
      inline T eval(const D& input) const { 
         const int n = input.n();
         const T res = _lambda*input.asum() + T(0.5)*_lambda2*input.nrm2sq();
         return (this->_intercept ? res - _lambda*abs<T>(input[n-1]) - T(0.5)*_lambda2*input[n-1]*input[n-1] : res);
      };
      // max_x xy - lambda_1 |x| - 0.5 lambda_2 x^2
      // - min_x - xy + lambda_1 |x| + 0.5 lambda_2 x^2
      // -(1/lambda2) min_x - xy/lambda2  + lambda_1/lambda_2 |x| + 0.5 x^2
      // x^* = prox_(l1 lambda_1/lambda_2) [ y/lambda2] 
      // x^* = prox_(l1 lambda_1) [ y] /_lambda2
      inline T fenchel(D& grad1, D& grad2) const { 
         D tmp;
         tmp.copy(grad2);
         grad2.fastSoftThrshold(_lambda);
         const int n = grad2.n();
         T res0 = _lambda*grad2.asum()/_lambda2 + T(0.5)*grad2.nrm2sq()/_lambda2;
         if (this->_intercept) res0 -= _lambda*abs<T>(grad2[n-1])/_lambda2 - T(0.5)*grad2[n-1]*grad2[n-1]/_lambda2;
         const T res = tmp.dot(grad2)/_lambda2 - res0;
         return (this->_intercept & (abs<T>(tmp[tmp.n()-1]) >
                  1e-6)) ? INFINITY : res; 
      };
      void print() const { 
         cout << "Elastic Net regularization" << endl;
      }
      virtual T strong_convexity() const { return this->_intercept ? 0 : _lambda2; };
      virtual T lambda() const { return _lambda;};
      inline void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const { 
         const int p = input.n();
         const int r = indices.n();
         const T thrs=_lambda*eta;
         const T scal = T(1.0)/(T(1.0)+_lambda2*eta);
#pragma omp parallel for
         for (int jj=0; jj<r; ++jj) 
            output[indices[jj]]=scal*fastSoftThrs(input[indices[jj]],thrs);;  
         if (this->_intercept) output[p-1]=input[p-1];
      };
      virtual bool is_lazy() const { return true; };


   private:
      const T _lambda;
      const T _lambda2;
};

template <typename D, typename I>
class L1Ball final : public Regularizer<D,I> {
   public:
      typedef typename D::value_type T;

      L1Ball(const ParamModel<T>& model) : Regularizer<D,I>(model), _lambda(model.lambda) { };
      inline void prox(const D& input, D& output, const T eta) const {
         D tmp;
         tmp.copy(input);
         if (this->_intercept) {
            tmp[tmp.n()-1]=0;
            tmp.sparseProject(output,_lambda,1,0,0,0,false);
            output[output.n()-1] = input[output.n()-1];
         } else {
            tmp.sparseProject(output,_lambda,1,0,0,0,false);
         }
      };
      inline T eval(const D& input) const { return 0; };
      inline T fenchel(D& grad1, D& grad2) const { 
         Vector<T> output;
         output.copy(grad2);
         if (this->_intercept) output[output.n()-1]=0;
         return _lambda*(output.fmaxval());
      };
      void print() const { 
         cout << "L1 ball regularization" << endl;
      }
      virtual T lambda() const { return _lambda;};

   private:
      const T _lambda;
};

template <typename D, typename I>
class L2Ball final : public Regularizer<D,I> {
   public:
      typedef typename D::value_type T;

      L2Ball(const ParamModel<T>& model) : Regularizer<D,I>(model), _lambda(model.lambda) { };
      inline void prox(const D& input, D& output, const T eta) const {
         D tmp;
         tmp.copy(input);
         if (this->_intercept) {
            tmp[tmp.n()-1]=0;
            const T nrm = tmp.nrm2();
            if (nrm > _lambda)
               tmp.scal(_lambda/nrm);
            output[output.n()-1] = input[output.n()-1];
         } else {
            const T nrm = tmp.nrm2();
            if (nrm > _lambda)
               tmp.scal(_lambda/nrm);
         }
      };
      inline T eval(const D& input) const { return 0; };
      inline T fenchel(D& grad1, D& grad2) const { 
         Vector<T> output;
         output.copy(grad2);
         if (this->_intercept) output[output.n()-1]=0;
         return _lambda*(output.nrm2());
      };
      void print() const { 
         cout << "L1 ball regularization" << endl;
      }
      virtual T lambda() const { return _lambda;};

   private:
      const T _lambda;
};


template <typename D, typename I>
class FusedLasso final : public Regularizer<D,I> {
   public:
      typedef typename D::value_type T;

      FusedLasso(const ParamModel<T>& model) : Regularizer<D,I>(model), _lambda(model.lambda), _lambda2(model.lambda2), _lambda3(model.lambda3) { };
      inline void prox(const D& x, D& output, const T eta) const {
         output.resize(x.n());
         Vector<T> copyx;
         copyx.copy(x);
         copyx.fusedProjectHomotopy(output,_lambda2,_lambda,_lambda3,true);
      };
      inline T eval(const D& x) const {  
         T sum = T();
         const int maxn = this->_intercept ? x.n()-1 : x.n();
         for (int i = 0; i<maxn-1; ++i)
            sum += _lambda*abs(x[i+1]-x[i]) + _lambda2*abs(x[i]) + T(0.5)*_lambda3*x[i]*x[i];
         sum += _lambda2*abs(x[maxn-1])+0.5*_lambda3*x[maxn-1]*x[maxn-1];
         return sum;
      };
      inline T fenchel(D& grad1, D& grad2) const { return 0; };
      void print() const { 
         cout << "Fused Lasso regularization" << endl;
      }
      bool provides_fenchel() const { return false; };
      virtual T strong_convexity() const { return this->_intercept ? 0 : _lambda3; };
      virtual T lambda() const { return _lambda;};

   private:
      const T _lambda;
      const T _lambda2;
      const T _lambda3;
};

template <typename Reg>
class RegMat final : public Regularizer< Matrix<typename Reg::T>, typename Reg::index_type > {
   public:
      typedef typename Reg::T T;
      typedef typename Reg::index_type I;
      RegMat(const ParamModel<T>& model, const int num_cols, const bool transpose) : Regularizer< Matrix<T>, I >(model), _N(num_cols), _transpose(transpose) {
         _regs=new Reg*[_N];
         for (int i = 0; i<_N; ++i)
            _regs[i]=new Reg(model);
      };
      virtual ~RegMat() {
         for (int i = 0; i<_N; ++i) {
            delete(_regs[i]);
            _regs[i]=NULL;
         }
         delete[](_regs);
      };
      void inline prox(const Matrix<T>& x, Matrix<T>& y, const T eta) const {
         y.copy(x);
         int i;
#pragma omp parallel for private(i)
         for (i = 0; i<_N; ++i) {
            Vector<T> colx, coly;
            if (_transpose) {
               x.copyRow(i,colx);
               y.copyRow(i,coly);
            } else {
               x.refCol(i,colx);
               y.refCol(i,coly);
            }
            _regs[i]->prox(colx,coly,eta);
            if (_transpose) 
               y.copyToRow(i,coly);
         }
      };
      T inline eval(const Matrix<T>& x) const {
         T sum = 0;
#pragma omp parallel for reduction(+ : sum)
         for (int i = 0; i<_N; ++i) {
            Vector<T> col;
            if (_transpose) {
               x.copyRow(i,col);
            } else {
               x.refCol(i,col);
            }
            const T val = _regs[i]->eval(col);
            sum += val;
         }
         return sum;
      };
      T inline fenchel(Matrix<T>& grad1, Matrix<T>& grad2) const {
         T sum=0;
#pragma omp parallel for reduction(+ : sum)
         for (int i = 0; i<_N; ++i) {
            Vector<T> col1, col2;
            if (_transpose) {
               grad1.copyRow(i,col1);
               grad2.copyRow(i,col2);
            } else {
               grad1.refCol(i,col1);
               grad2.refCol(i,col2);
            }
            const T fench=_regs[i]->fenchel(col1,col2);
            sum+=fench;
            if (_transpose) {
               grad1.copyToRow(i,col1);
               grad2.copyToRow(i,col2);
            }
         }
         return sum;
      };
      virtual bool provides_fenchel() const {
         bool ok=true;
         for (int i = 0; i<_N; ++i)
            ok = ok && _regs[i]->provides_fenchel();
         return ok;
      };
      void print() const {
         cout << "Regularization for matrices" << endl;
         _regs[0]->print();
      };
      virtual T lambda() const { return _regs[0]->lambda();};
      inline void lazy_prox(const Matrix<T>& input, Matrix<T>& output, const Vector<I>& indices, const T eta) const { 
#pragma omp parallel for 
         for (int i = 0; i<_N; ++i) {
            Vector<T> colx, coly;
            output.refCol(i,coly);
            if (_transpose) {
               input.copyRow(i,colx);
            } else {
               input.refCol(i,colx);
            }
            _regs[i]->lazy_prox(colx,coly,indices,eta);
         }
      };
      virtual bool is_lazy() const { return _regs[0]->is_lazy(); };

   protected:
      int _N;
      Reg** _regs;
      bool _transpose;
};


template <typename Reg>
class RegVecToMat final : public Regularizer< Matrix<typename Reg::T>, typename Reg::index_type > {
   public:
      typedef typename Reg::T T;
      typedef typename Reg::index_type I;
      typedef Matrix<T> D;
      RegVecToMat(const ParamModel<T>& model) : Regularizer<D,I>(model), _intercept(model.intercept) {
         ParamModel<T> model2=model;
         model2.intercept=false;
         _reg=new Reg(model2);
      };
      ~RegVecToMat() { delete(_reg);};

      inline void prox(const D& input, D& output, const T eta) const {
         Vector<T> w1, w2, b1, b2;
         output.resize(input.m(),input.n());
         get_wb(input,w1,b1);
         get_wb(output,w2,b2);
         _reg->prox(w1,w2,eta);
         if (_intercept) b2.copy(b1);
      };

      inline T eval(const D& input) const { 
         Vector<T> w, b;
         get_wb(input,w,b);
         return _reg->eval(w);
      }
      
      inline T fenchel(D& grad1, D& grad2) const { 
         Vector<T> g1;
         grad1.toVect(g1);
         Vector<T> w, b;
         get_wb(grad2,w,b);
         return (this->_intercept && ((b.nrm2sq()) > 1e-7) ? INFINITY : _reg->fenchel(g1,w));
      };
      void print() const { 
         _reg->print();
      }
      virtual T strong_convexity() const { 
         return _intercept ? 0 : _reg->strong_convexity(); 
      };
      virtual T lambda() const { return _reg->lambda(); };
      inline void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const { 
         Vector<T> w1, w2, b1, b2;
         output.resize(input.m(),input.n());
         get_wb(input,w1,b1);
         get_wb(output,w2,b2);
         _reg->lazy_prox(w1,w2,indices,eta);
         if (_intercept) b2.copy(b1);
      };
      virtual bool is_lazy() const { return _reg->is_lazy(); };

   private:
      inline void get_wb(const Matrix<T>& input, Vector<T>& w, Vector<T>& b) const {
         const int p = input.n();
         Matrix<T> W;
         if (_intercept) {
            input.refSubMat(0,p-1,W);
            input.refCol(p-1,b);
         } else {
            input.refSubMat(0,p,W);
         }
         W.toVect(w);
      };
      Reg* _reg;
      const bool _intercept;
};


template <typename T>
struct normL2 {
   public:
      typedef T value_type;
      normL2(const ParamModel<T>& model) : _lambda(model.lambda) { };

      inline void prox(Vector<T>& x, const T thrs) const {
         const T nrm=x.nrm2();
         const T thrs2 = thrs*_lambda;
         if (nrm > thrs2) {
            x.scal((nrm-thrs2)/nrm);
         } else {
            x.setZeros();
         }
      };
      inline T eval(const Vector<T>& x) const {
         return _lambda*x.nrm2();
      };
      static inline void print() {
         cout << "L2";
      };
      inline T eval_dual(const Vector<T>& x) const {
         return x.nrm2()/_lambda;
      };
   private:
      const T _lambda;
};

template <typename T>
struct normLinf {
   public:
      typedef T value_type;
      normLinf(const ParamModel<T>& model) : _lambda(model.lambda) { };

      inline void prox(Vector<T>& x, const T thrs) const {
         Vector<T> z;
         x.l1project(z,thrs*_lambda);
         x.sub(z);
      };
      inline T eval(const Vector<T>& x) const {
         return _lambda*x.fmaxval();
      };
      static inline void print() {
         cout << "LInf";
      }
      inline T eval_dual(const Vector<T>& x) const {
         return x.asum()/_lambda;
      };
   private:
      const T _lambda;
};


template <typename T>
struct normL2_L1 {
   public:
      typedef T value_type;
      normL2_L1(const ParamModel<T>& model) : _lambda(model.lambda), _lambda2(model.lambda2) { };

      inline void prox(Vector<T>& x, const T thrs) const {
         x.fastSoftThrshold(x,thrs*_lambda2);
         const T nrm=x.nrm2();
         const T thrs2 = thrs*_lambda;
         if (nrm > thrs2) {
            x.scal((nrm-thrs2)/nrm);
         } else {
            x.setZeros();
         }
      };
      inline T eval(const Vector<T>& x) const {
         return _lambda*x.nrm2()+_lambda2*x.asum();
      };
      static inline void print() {
         cout << "L2+L1";
      };
      inline T eval_dual(const Vector<T>& x) const {
         Vector<T> sorted_x;
         sorted_x.copy(x);
         sorted_x.abs_vec();
         sorted_x.sort(false);
         const int n = x.n();
         T lambda_gamma=0;
         T sum_sq=0;
         T sum_lin=0;
         for (int ii=0; ii<n; ++ii) {
            lambda_gamma=sorted_x[ii];
            sum_lin+=lambda_gamma;
            sum_sq+=lambda_gamma*lambda_gamma;
            const T lambda_mu = _lambda*lambda_gamma/(_lambda2);
            if (sum_sq - 2*lambda_gamma*sum_lin + (ii+1)*lambda_gamma*lambda_gamma >= lambda_mu*lambda_mu) {
               sum_lin-=lambda_gamma;
               sum_sq -=lambda_gamma*lambda_gamma;
               const T dual=solve_binomial2((ii)*_lambda2*_lambda2-_lambda*_lambda,-2*sum_lin*_lambda2,sum_sq);
               return dual;
            }
         }
         return 0;
      };
   private:
      const T _lambda;
      const T _lambda2;
};



template <typename N, typename I>
class MixedL1LN final : public Regularizer< Matrix<typename N::value_type>, I > {
   public: 
      typedef typename N::value_type T;
      typedef Matrix<T> D;
      MixedL1LN(const ParamModel<T>& model, const int nclass, const bool transpose) :
         Regularizer<D,I>(model), _transpose(transpose), _lambda(model.lambda), _norm(model) { };
      inline void prox(const D& x, D& y, const T eta) const {
         const int n = x.n();
         const int m = x.m();
         y.copy(x);
         if (_transpose) {
            const int nn = this->_intercept ? n-1 : n;
#pragma omp parallel for
            for (int i = 0; i<nn; ++i) {
               Vector<T> col;
               y.refCol(i,col);
               _norm.prox(col,eta);
            }
         } else {
            const int nn = this->_intercept ? m-1 : m;
#pragma omp parallel for
            for (int i = 0; i<nn; ++i) {
               Vector<T> row;
               y.copyRow(i,row);
               _norm.prox(row,eta);
               y.copyToRow(i,row);
            }
         }
      };
      T inline eval(const D& x) const {
         T sum=0;
         const int n = x.n();
         const int m = x.m();
         if (_transpose) {
            const int nn = this->_intercept ? n-1 : n;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i<nn; ++i) {
               Vector<T> col;
               x.refCol(i,col);
               sum+=_norm.eval(col);
            }
         } else {
            const int nn = this->_intercept ? m-1 : m;
#pragma omp parallel for reduction(+:sum)
            for (int i = 0; i<nn; ++i) {
               Vector<T> row;
               x.copyRow(i,row);
               sum+=_norm.eval(row);
            }
         }
         return sum;
      }
      // grad1 is nclasses * n
      inline T fenchel(D& grad1, D& grad2) const { 
         const int n = grad2.n();
         const int m = grad2.m();
         T res=0;
         T mm=0;
         if (_transpose) {
            const int nn = this->_intercept ? n-1 : n;
            for (int i = 0; i<nn; ++i) {
               Vector<T> col;
               grad2.refCol(i,col);
               mm = MAX(_norm.eval_dual(col),mm);
            }
            Vector<T> col;
            if (this->_intercept) {
               grad2.refCol(nn,col);
               if (col.nrm2sq() > T(1e-7)) res=INFINITY; 
            }
         } else {
            const int nn = this->_intercept ? m-1 : m;
            for (int i = 0; i<nn; ++i) {
               Vector<T> row;
               grad2.copyRow(i,row);
               mm = MAX(_norm.eval_dual(row),mm);
            }
            Vector<T> col;
            if (this->_intercept) {
               grad2.copyRow(nn,col);
               if (col.nrm2sq() > T(1e-7)) res=INFINITY; 
            }
         }
         if (mm > T(1.0))  
            grad1.scal(T(1.0)/mm);
         return res;
      };

      void print() const { 
         cout << "Mixed L1-";
         N::print();
         cout << " norm regularization" << endl;
      }
      inline T lambda() const { return _lambda; };
      inline void lazy_prox(const D& input, D& output, const Vector<I>& indices, const T eta) const { 
         output.resize(input.m(),input.n());
         const int r = indices.n();
         const int m = input.m();
         const int n = input.n();
         if (_transpose) {
#pragma omp parallel for
            for (int i = 0; i<r; ++i) {
               const int ind=indices[i];
               Vector<T> col, col1;
               input.refCol(ind,col1);
               output.refCol(ind,col);
               col.copy(col1);
               _norm.prox(col,eta);
            }
            if (this->_intercept) {
               Vector<T> col, col1;
               input.refCol(n-1,col1);
               output.refCol(n-1,col);
               col.copy(col1);
            }
         } else {
#pragma omp parallel for
            for (int i = 0; i<r; ++i) {
               const int ind=indices[i];
               Vector<T> col;
               input.copyRow(ind,col);
               _norm.prox(col,eta);
               output.copyToRow(ind,col);
            }
            if (this->_intercept) {
               Vector<T> col;
               input.copyRow(m-1,col);
               output.copyToRow(m-1,col);
            }

         }
      };
      virtual bool is_lazy() const { return true; };

   private:
      const bool _transpose;
      const T _lambda;
      N _norm;
};

template <typename T, typename I>
using MixedL1L2=MixedL1LN< normL2<T>, I >;

template <typename T, typename I>
using MixedL1Linf=MixedL1LN< normLinf<T> , I>;

template <typename T, typename I>
using MixedL1L2_L1=MixedL1LN< normL2_L1<T>, I >;

#endif
