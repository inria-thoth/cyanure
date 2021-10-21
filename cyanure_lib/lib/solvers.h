#ifndef SOLVERS_H
#define SOLVERS_H

#include "loss.h"
#include "regul.h"
#include "list.h"

#define USING_SOLVER \
   typedef typename loss_type::variable_type D; \
   typedef typename loss_type::value_type T; \
   typedef typename loss_type::index_type I;\
   typedef loss_type LT; \
   using Solver<loss_type>::_L; \
   using Solver<loss_type>::_loss;  \
   using Solver<loss_type>::_regul; \
   using Solver<loss_type>::_Li; \
   using Solver<loss_type>::_verbose; \
   using Solver<loss_type>::_max_iter_backtracking; 

#define USING_INCREMENTAL_SOLVER \
   USING_SOLVER; \
   using IncrementalSolver<loss_type>::_minibatch; \
   using IncrementalSolver<loss_type>::_non_uniform_sampling; \
   using IncrementalSolver<loss_type>::_n; \
   using IncrementalSolver<loss_type>::_qi; 

#define USING_SVRG_SOLVER \
      USING_INCREMENTAL_SOLVER; \
      using SVRG_Solver<loss_type>::_xtilde;\
      using SVRG_Solver<loss_type>::_gtilde;

#define USING_ACC_SVRG_SOLVER \
      using Acc_SVRG_Solver<loss_type,allow_acc>::_y;\
      using Acc_SVRG_Solver<loss_type,allow_acc>::_etak;\
      using Acc_SVRG_Solver<loss_type,allow_acc>::_gammak;\
      using Acc_SVRG_Solver<loss_type,allow_acc>::_mu;\
      using Acc_SVRG_Solver<loss_type,allow_acc>::_deltak;\
      using Acc_SVRG_Solver<loss_type,allow_acc>::_thetak;\
      using Acc_SVRG_Solver<loss_type,allow_acc>::_accelerated_solver;\
      USING_SVRG_SOLVER;
enum solver_t { ISTA, CATALYST_ISTA, QNING_ISTA, FISTA, SAGA, SVRG, SVRG_UNIFORM, CATALYST_SVRG, ACC_SVRG, MISO, CATALYST_MISO, QNING_SVRG, QNING_MISO, AUTO, INCORRECT_SOLVER };

solver_t solver_from_string(char* regul) {
   if (strcmp(regul,"ista")==0) return ISTA;
   if (strcmp(regul,"catalyst-ista")==0) return CATALYST_ISTA;
   if (strcmp(regul,"qning-ista")==0) return QNING_ISTA;
   if (strcmp(regul,"fista")==0) return FISTA;
   if (strcmp(regul,"saga")==0) return SAGA;
   if (strcmp(regul,"svrg")==0) return SVRG;
   if (strcmp(regul,"catalyst-svrg")==0) return CATALYST_SVRG;
   if (strcmp(regul,"qning-svrg")==0) return QNING_SVRG;
   if (strcmp(regul,"qning-miso")==0) return QNING_MISO;
   if (strcmp(regul,"acc-svrg")==0) return ACC_SVRG;
   if (strcmp(regul,"miso")==0) return MISO;
   if (strcmp(regul,"catalyst-miso")==0) return CATALYST_MISO;
   if (strcmp(regul,"svrg-uniform")==0) return SVRG_UNIFORM;
   if (strcmp(regul,"auto")==0) return AUTO;
   return INCORRECT_SOLVER;
}

template <typename T> struct ParamSolver { 
   ParamSolver() { nepochs=100; it0=10; tol=T(1e-3); verbose=false; solver=FISTA; 
   max_iter_backtracking=500; minibatch=1; threads=-1; non_uniform_sampling=true; l_memory=20; freq_restart=50; };
   int nepochs;
   T tol;
   int it0;
   bool verbose;
   solver_t solver;
   int max_iter_backtracking;
   int minibatch;
   int threads;
   bool non_uniform_sampling;
   int l_memory;
   int freq_restart;
};

template <typename loss_type> 
class Solver {
   public:
      typedef typename loss_type::variable_type D;
      typedef typename loss_type::value_type T;
      typedef typename loss_type::index_type I;
      
      Solver(const loss_type& loss, const Regularizer<D,I>& regul, const
            ParamSolver<T>& param) : _loss(loss), _regul(regul) { 
         _verbose=param.verbose;
         _it0=MAX(param.it0,1);
         _tol=param.tol;
         _nepochs=param.nepochs;
         _max_iter_backtracking=param.max_iter_backtracking;
         _best_dual=-INFINITY;
         _best_primal=INFINITY;
         _duality = _loss.provides_fenchel() && regul.provides_fenchel();
         _optim_info.resize(6,MAX(param.nepochs/_it0,1));
         _L=0;
      };
      virtual ~Solver() { };

      virtual void solve(const D& x0, D& x) {
         _time.start();
         x.copy(x0);
         if (!_duality && _nepochs > 1) _xold.copy(x0);
         solver_init(x0);
         if (_verbose) {
            cout << "*********************************" << endl;
            print();
            _loss.print(); 
            _regul.print();
         }

         for (int it = 1; it<=_nepochs; ++it) {
            if ((it % _it0) == 0) 
               if (test_stopping_criterion(x,it))
                  break;
            solver_aux(x);
         }
         _time.stop();
         if (_verbose) 
            _time.printElapsed();
         if (_best_primal != INFINITY) x.copy(_bestx);
      }
      void get_optim_info(Matrix<T>& optim) const {
         int count=0;
         for (int ii=0; ii<_optim_info.n(); ++ii)
            if (_optim_info(0,ii) != 0) ++count;
         if (count > 0) {
            optim.resize(6,count);
         }
         for (int ii=0; ii<count; ++ii)
            for (int jj=0; jj<6; ++jj)
               optim(jj,ii)=_optim_info(jj,ii);
      };

      void eval(const D& x) {
         test_stopping_criterion(x,1);
         _optim_info(5,0)=0;
      };

      virtual void set_dual_variable(const D& dual0) { };
      virtual void save_state() { };
      virtual void restore_state() { };

   private:
      inline T get_dual(const D& x) const {
         if (!_regul.provides_fenchel() || !_loss.provides_fenchel()) {
            cerr << "Error: no duality gap available" << endl;
            return -INFINITY;
         }
         D grad1, grad2;
         _loss.get_dual_variable(x,grad1,grad2);
         const T dual = - _regul.fenchel(grad1,grad2);
         return dual - _loss.fenchel(grad1);
      };

      inline bool test_stopping_criterion(const D& x, const int it) {
         const T primal =_loss.eval(x) + _regul.eval(x);
         _best_primal = MIN(_best_primal,primal);
         const int ii=MAX(it/_it0-1,0);
         const double sec = _time.getElapsed();
         Vector<T> optim;
         _optim_info.refCol(ii,optim);
         if (_best_primal == primal)
            _bestx.copy(x);
         if (_verbose) {
            if (primal==_best_primal) {
               cout << "Epoch: " << it << ", primal objective: " << primal << ", time: " << sec << endl;
            } else {
               cout << "Epoch: " << it << ", primal objective: " << primal << ", best primal: " << _best_primal << ", time: " << sec << endl;
            }
         }
         optim[0]=it; optim[1]=primal; optim[5]=sec;
         if (_duality) {
            const T dual=get_dual(x);
            _best_dual=MAX(_best_dual,dual);
            const T duality_gap=(_best_primal-_best_dual)/abs<T>(_best_primal);
            if (_verbose) 
               cout << "Best relative duality gap: " << duality_gap << endl;
            optim[2]=_best_dual; optim[3]=duality_gap;
            return duality_gap < _tol;
         } else {
            _xold.sub(x);
            const T diff=sqrt(_xold.nrm2sq()/MAX(EPSILON,x.nrm2sq()));
            _xold.copy(x);
            optim[4]=diff;
            return diff < _tol;
         }
      }

   protected:
      virtual void solver_init(const D& x0) = 0;
      virtual void solver_aux(D& x) = 0;
      virtual void print() const = 0;
      virtual int minibatch() const { return 1; };
      bool _verbose;
      int _it0;
      int _nepochs;
      int _max_iter_backtracking;
      int _restart_frequency;
      T _tol;
      const loss_type& _loss;
      const Regularizer<D,I>& _regul;
      Timer _time;
      T _best_dual;
      T _best_primal;
      Matrix<T> _optim_info;
      bool _duality;
      D _xold;
      T _L;
      D _bestx;
      Vector<T> _Li;
};

template <typename loss_type> 
class ISTA_Solver: public Solver<loss_type> {
   public:
      USING_SOLVER;
      ISTA_Solver(const loss_type& loss, const Regularizer<D,I>& regul, const ParamSolver<T>& param, const Vector<T>* Li=NULL) : Solver<loss_type>(loss,regul,param) { 
         _L=0;
         if (Li) {
            _Li.copy(*Li);
            _L = _Li.maxval()/100;
         }
      };

   protected:
      virtual void solver_init(const D& x0) {
         if (_L==0) {
            _loss.lipschitz(_Li);
            _L = _Li.maxval()/100;
         }
      };
      virtual void solver_aux(D& x) {
         int iter=1;
         const T fx = _loss.eval(x);
         D grad, tmp, tmp2;
         _loss.grad(x,grad);
         while (iter < _max_iter_backtracking) {
            tmp2.copy(x);
            tmp2.add(grad,-T(1.0)/_L);
            _regul.prox(tmp2,tmp,T(1.0)/_L);
            const T fprox = _loss.eval(tmp);
            tmp2.copy(tmp);
            tmp2.sub(x);

            if (fprox <= fx + grad.dot(tmp2) + T(0.5)*_L*tmp2.nrm2sq() + EPSILON) {
               x.copy(tmp);
               break;
            }
            _L *= T(1.5);
            if (_verbose)
               cout << "new value for L: " << _L << endl;
            ++iter;
            if (iter == _max_iter_backtracking)
               cout << "Warning: maximum number of backtracking iterations has been reached" << endl;
         } 
      };
      void print() const {
         cout << "ISTA Solver" << endl;
      };
      T init_kappa_acceleration(const D& x0) {
         ISTA_Solver<loss_type>::solver_init(x0);
         return _L;
      };
};

template <typename loss_type> 
class FISTA_Solver final: public ISTA_Solver<loss_type> {
   public:
      USING_SOLVER;
      FISTA_Solver(const loss_type& loss, const Regularizer<D,I>& regul, const ParamSolver<T>& param) : ISTA_Solver<loss_type>(loss,regul,param) { };

   protected:
      virtual void solver_init(const D& x0) {
         ISTA_Solver<loss_type>::solver_init(x0);
         _t = T(1.0);
         _y.copy(x0);
      };
      virtual void solver_aux(D& x) {
         ISTA_Solver<loss_type>::solver_aux(_y);
         D diff;
         diff.copy(x);
         x.copy(_y);
         diff.sub(x);
         const T old_t=_t;
         _t=(1.0+sqrt(1+4*_t*_t))/2;
         _y.add(diff,(T(1.0)-old_t)/_t);
      };
      virtual void print() const {
         cout << "FISTA Solver" << endl;
      };
      
      T _t;
      D _y;
};

template <typename loss_type> 
class IncrementalSolver: public Solver<loss_type> {
   public:
      USING_SOLVER;
      IncrementalSolver(const loss_type& loss, const Regularizer<D,I>& regul, const ParamSolver<T>& param, const Vector<T>* Li=NULL) : Solver<loss_type>(loss,regul,param) { 
         _minibatch=param.minibatch;
         _non_uniform_sampling=param.non_uniform_sampling;
         if (Li) _Li.copy(*Li);
      };
   
   protected:
      virtual void solver_init(const D& x0) {
         if (_Li.n() == 0)
            _loss.lipschitz(_Li);
         _n=_Li.n();
         if (_L==0) {
            _qi.copy(_Li);
            _qi.scal(T(1.0)/_qi.sum());
            const T Lmean = _Li.mean();
            const T Lmax = _Li.maxval();
            _non_uniform_sampling = (_non_uniform_sampling && Lmean <= T(0.9)*Lmax);
            _L = _non_uniform_sampling ? Lmean : Lmax;
            if (_minibatch > 1)
               heuristic_L(x0);
            _oldL=_L;
            if (_non_uniform_sampling) 
               init_nonu_sampling();
         }
         this->check_mkl(x0);
      };
      void print() const {
         cout << "Incremental Solver ";
         if (_non_uniform_sampling) {
            cout << "with non uniform sampling" << endl;
         } else {
            cout << "with uniform sampling" << endl;
         }
         cout << "Lipschitz constant: " << _L << endl;
      };

      bool _non_uniform_sampling;
      int _minibatch;
      int _n;
      Vector<T> _qi;
      Vector<double> _Ui;
      Vector<int> _Ki;
      T _oldL;

      void init_nonu_sampling() {
         _Ui.resize(_n);
         for (int ii=0; ii<_n; ++ii) _Ui[ii]=static_cast<double>(_qi[ii]);
         _Ui.scal(_n/_Ui.asum());
         _Ki.resize(_n);
         _Ki.set(0);
         List<int> overfull;
         List<int> underfull;
         for (int ii=0; ii<_n; ++ii) {
            if (_Ui[ii] < double(1.0)) {
               underfull.push_back(ii);
            } else if (_Ui[ii] > double(1.0)) {
               overfull.push_back(ii);
            } 
         }
         while (underfull.size() > 0 && overfull.size() > 0) {
            const int indj = underfull.front();
            underfull.pop_front();
            const int indi = overfull.front();
            overfull.pop_front();
            _Ki[indj]=indi;
            _Ui[indi]=_Ui[indi] + _Ui[indj] - double(1.0);
            if (_Ui[indi] < double(1.0)) {
               underfull.push_back(indi);
            } else if (_Ui[indi] > double(1.0)) {
               overfull.push_back(indi);
            }
         }
      };
      int nonu_sampling() {
         const double x = static_cast<double>(random()-1)/RAND_MAX;
         const int ind = static_cast<int>(floor(_n*x))+1;
         const double y = _n*x+1-ind;
         if (y < _Ui[ind-1]) return ind-1;
         return _Ki[ind-1];
      };
      virtual int minibatch() const { return _minibatch; };
      T init_kappa_acceleration(const D& x0) {
         IncrementalSolver<loss_type>::solver_init(x0);
         const T mu = _regul.strong_convexity();
         return ((this->_oldL/(_n) - mu));
      };
      void check_mkl(const Vector<T>& x0) const { };
      void check_mkl(const Matrix<T>& x0) const { 
         if (x0.m() <= 15 || x0.n() <= 15) {
            set_mkl_sequential(); // TODO should be local
         }
      };


   private:
      void heuristic_L(const D& x) {
         if (_verbose) 
            cout << "Heuristic: Initial L=" << _L;
         const T Lmax=_L;
         _L /= _minibatch;
         int iter=0;
         D tmp, tmp2, grad;
         while (iter <= log(_minibatch)/log(2.0) && _L < Lmax) {
            tmp.copy(x);
            const T fx=_loss.eval_random_minibatch(tmp,_minibatch);
            _loss.grad_random_minibatch(tmp,grad,_minibatch);  // should do non uniform
            // compute grad and fx
            tmp.add(grad,-T(1.0)/_L);
            const T ftmp=_loss.eval_random_minibatch(tmp,_minibatch);
            tmp2.copy(tmp);
            tmp2.sub(x);
            const T s1 =fx + grad.dot(tmp2);
            const T s2=T(0.5)*tmp2.nrm2sq();
            if (ftmp > s1 + _L*s2) 
               _L = MIN( MAX(2.0*_L, (ftmp-s1)/s2),Lmax);
            ++iter;
         } 
         if (_verbose) 
            cout << ", Final L=" << _L << endl;
      }
};


template <typename loss_type> 
class SVRG_Solver: public IncrementalSolver<loss_type> {
   public:
      USING_INCREMENTAL_SOLVER;
      SVRG_Solver(const loss_type& loss, const Regularizer<D,I>& regul, const ParamSolver<T>& param, const Vector<T>* Li=NULL) : IncrementalSolver<loss_type>(loss,regul,param,Li) { 
      };

   protected:
      virtual void solver_init(const D& x0) {
         IncrementalSolver<loss_type>::solver_init(x0);
         _xtilde.copy(x0);
         _loss.grad(_xtilde,_gtilde);
      };

      virtual void solver_aux(D& x) {
         const int nn = _n/_minibatch;
         const T eta = T(1.0)/(3*_L);
         D tmp;
         for (int ii = 0; ii<nn; ++ii) {
            tmp.copy(x);
            tmp.add(_gtilde,-eta);
            for (int jj=0; jj<_minibatch; ++jj) {
               const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
               const T scal = _non_uniform_sampling ? T(1.0)/(_minibatch*_qi[ind]*_n) : T(1.0)/_minibatch;
               _loss.double_add_grad(x,_xtilde,ind,tmp,-scal*eta,scal*eta, jj==0 ? T(_minibatch) : 0); 
            }
            _regul.prox(tmp,x,eta);
            if (random() % nn == 0) {
               _xtilde.copy(x);
               _loss.grad(_xtilde,_gtilde);
            }
         }
      };
      void print() const {
         cout << "SVRG Solver" << endl;
         IncrementalSolver<loss_type>::print();
      };
      D _xtilde, _gtilde;
};

template <typename loss_type> 
class MISO_Solver: public IncrementalSolver<loss_type> {
   public:
      USING_INCREMENTAL_SOLVER;
      MISO_Solver(const loss_type& loss, const Regularizer<D,I>& regul, const ParamSolver<T>& param, const Vector<T>* Li=NULL) : IncrementalSolver<loss_type>(loss,regul,param,Li) { 
         _minibatch=1;
         _mu= _regul.id() == L2 ? _regul.strong_convexity() : 0; 
         _kappa = _loss.kappa();
         if (_loss.id() == PPA) _mu += _kappa;
         _isprox=(_regul.id() != L2 || _regul.intercept()) && _regul.id() != NONE;
         _is_lazy = _isprox && _regul.is_lazy() && _loss.is_sparse();
         _extern_zis=false;
         _count=0;
      };

      virtual void set_dual_variable(const D& dual0) {
         _extern_zis=true;
         _zis.copyRef(dual0);
      };
      
      virtual void solve(const D& y, D& x) {
         if (_count > 0 && (_count % 10) != 0) {
            D& ref_barz = _isprox ? _barz : x;
            ref_barz.add(_oldy,-_kappa/_mu); // necessary to have PPA loss here
            ref_barz.add(y,_kappa/_mu);
            const bool is_lazy = _isprox && _regul.is_lazy() && _loss.is_sparse();
            if (_isprox && !is_lazy) 
               _regul.prox(ref_barz,x,T(1.0)/_mu);
         } else if (_count==0) {
            x.copy(y); // just to have the right size
         }
         if (_loss.id()==PPA)
            _loss.get_anchor_point(_oldy);
         Solver<loss_type>::solve(x,x);
      };

      virtual void save_state() {
         _count2=_count;
         _barz2.copy(_barz);
         _zis2.copy(_zis);
         _oldy2.copy(_oldy);
      };
      virtual void restore_state() {
         _count=_count2;
         _barz.copy(_barz2);
         _zis.copy(_zis2);
         _oldy.copy(_oldy2);
      };


   protected:
      virtual void solver_init(const D& x0) {
         // initial point will be in fact _z of PPA
         if (_count==0) {
            IncrementalSolver<loss_type>::solver_init(x0);
            _delta=MIN(T(1.0), _n*_mu/(2*_L));
            if (_non_uniform_sampling) {
               const T beta=T(0.5)*_mu*_n;
               if (this->_Li.maxval() <= beta) {
                  _non_uniform_sampling=false;
                  _delta=T(1.0);
               } else if (this->_Li.minval() >= beta) {
                  // no change
               } else {
                  _qi.copy(this->_Li);
                  _qi.thrsmax(beta);
                  _qi.scal(T(1.0)/_qi.sum());
                  Vector<T> tmp;
                  tmp.copy(_qi);
                  tmp.inv();
                  tmp.mult(tmp,this->_Li);
                  _L = tmp.maxval()/_n; 
                  this->init_nonu_sampling();
                  _delta=MIN(_n*_qi.minval(), _n*_mu/(2*_L));
               }
            }
            if (_non_uniform_sampling)
               _delta=MIN(_delta,_n*_qi.minval());
            if (_isprox)
               _barz.copy(x0); // if PPA, x0 should be the anchor point and _barz = X*_Zis + x0 
            init_dual_variables(x0);
         } 
      };
      virtual void solver_aux(D& x) {
         D& ref_barz = _isprox ? _barz : x;
         if (_count++ % 10 == 0) {
            if (_loss.id()==PPA) {
               _loss.get_anchor_point(ref_barz);
               ref_barz.scal(_kappa/_mu);
            } else {
               ref_barz.setZeros(); 
            }
            if (_count > 1 || _extern_zis) _loss.add_feature(_zis,ref_barz,T(1.0)/(_n*_mu)); 
            if (_isprox && !_is_lazy) 
               _regul.prox(ref_barz,x,T(1.0)/_mu);  
         } 
         Vector<typename loss_type::index_type> indices;
         for (int ii = 0; ii<_n; ++ii) {
            const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
            const T scal = _non_uniform_sampling ? T(1.0)/(_qi[ind]*_n) : T(1.0);
            const T deltas=scal*_delta;
            if (_is_lazy) {
               _loss.get_coordinates(ind,indices);
               _regul.lazy_prox(ref_barz,x,indices,T(1.0)/_mu);
            }
            solver_aux_aux(x,ref_barz,ind,deltas);

            if (_isprox && (!_is_lazy || ii==_n-1)) 
               _regul.prox(ref_barz,x,T(1.0)/_mu);
         }
      };
      void print() const {
         cout << "MISO Solver" << endl;
         IncrementalSolver<loss_type>::print();
      };

   private:
      D _zis, _zis2;
      D _barz, _barz2;
      D _oldy, _oldy2;
      T _mu;
      T _kappa;
      T _delta;
      int _count, _count2;
      bool _perform_update_barz;
      bool _isprox, _is_lazy,_extern_zis;


      void inline init_dual_variables(const Vector<T>& x0) {
         if (_zis.n() != _n) {
            _zis.resize(_n);
            _zis.setZeros(); 
         }
      }
      void inline init_dual_variables(const Matrix<T>& x0) {
         const int nclasses=_loss.transpose() ? x0.m() : x0.n();
         if (_zis.n() != _n || _zis.m() != nclasses) {
            _zis.resize(nclasses,_n);
            _zis.setZeros(); 
         }
      }
      void inline solver_aux_aux(const Vector<T>& x,Vector<T>& ref_barz,const int ind, const T deltas) {
         const T oldzi = _zis[ind];
         _zis[ind] = (T(1.0)-deltas)*_zis[ind] + deltas*(-_loss.scal_grad(x,ind));
         _loss.add_feature(ref_barz,ind,(_zis[ind]-oldzi)/(_n*_mu));
      };
      void inline solver_aux_aux(const Matrix<T>& x, Matrix<T>& ref_barz,const int ind, const T deltas) {
         Vector<T> oldzi, newzi;
         _zis.copyCol(ind,oldzi);
         _zis.refCol(ind,newzi);
         _loss.scal_grad(x,ind,newzi);
         newzi.add_scal(oldzi,T(1.0)-deltas,-deltas);
         oldzi.sub(newzi);
         oldzi.scal(-T(1.0)/(_n*_mu));
         _loss.add_feature(ref_barz,ind,oldzi);
      };
};


template <typename SolverType> 
class Catalyst : public SolverType {
   public:
      typedef typename SolverType::LT loss_type;
      USING_SOLVER
      Catalyst(const loss_type& loss, const Regularizer<D,I>& regul, const ParamSolver<T>& param) : SolverType(loss,regul,param) { 
         _auxiliary_solver=NULL;
         _loss_ppa=NULL;
         _accelerated_solver=true;
         _freq_restart=regul.strong_convexity() > 0 ? param.nepochs+2 : param.freq_restart;
      }; 
      ~Catalyst() { 
         if(_auxiliary_solver) delete(_auxiliary_solver); 
         if(_loss_ppa) delete(_loss_ppa); 
      };
      virtual void set_dual_variable(const D& dual0) {
         _dual_var.copyRef(dual0);
      };

   protected:
      virtual void solver_init(const D& x0) {
         _kappa = this->init_kappa_acceleration(x0);
         _mu = _regul.strong_convexity();
         _count=0;
         _accelerated_solver=_kappa > 0; //this->_oldL/(_n) >= _mu;
         if (_accelerated_solver) {
            ParamSolver<T> param2;
            param2.nepochs=1;
            param2.it0=2;
            param2.verbose=false;
            param2.minibatch=this->minibatch();
            this->_Li.add(_kappa);
            _loss_ppa = new ProximalPointLoss<loss_type>(_loss,x0,_kappa);
            _auxiliary_solver = new SolverType(*_loss_ppa,_regul,param2,&this->_Li);
            if (_dual_var.size() > 0) 
               _auxiliary_solver->set_dual_variable(_dual_var);
            _y.copy(x0);
            _alpha=T(1.0);
         } else {
            if (_verbose)
               cout << "Switching to regular solver, problem is well conditioned" << endl;
            SolverType::solver_init(x0);
         }
      };
      virtual void solver_aux(D& x) {
         if (_accelerated_solver) {
            const T q = _mu/(_mu+_kappa);
            D xold;
            xold.copy(x);
            _auxiliary_solver->solve(_y,x);
            const T alphaold=_alpha;
            _alpha = solve_binomial(T(1.0),_alpha*_alpha-q,-_alpha*_alpha);
            T beta= alphaold*(T(1.0)-alphaold)/(alphaold*alphaold+_alpha);
            if (++_count % _freq_restart == 0) {
               beta=0;
               _alpha=T(1.0);
            }
            _y.copy(xold);
            _y.add_scal(x,T(1.0)+beta,-beta);
            _loss_ppa->set_anchor_point(_y);
         } else {
            SolverType::solver_aux(x);
         }
      };
      void print() const {
         cout << "Catalyst Accelerator" << endl;
         SolverType::print();
      };

      int _count, _freq_restart;
      T _kappa, _alpha, _mu;
      D _y, _dual_var;
      bool _accelerated_solver;
      SolverType* _auxiliary_solver;
      ProximalPointLoss<loss_type>* _loss_ppa;
};


template <typename SolverType> 
class QNing final: public Catalyst<SolverType> {
   public:
      typedef typename SolverType::LT loss_type;
      USING_SOLVER
      using Catalyst<SolverType>::_kappa;
      using Catalyst<SolverType>::_accelerated_solver;
      using Catalyst<SolverType>::_auxiliary_solver;
      using Catalyst<SolverType>::_loss_ppa;
      using Catalyst<SolverType>::_y;
      QNing(const loss_type& loss, const Regularizer<D,I>& regul, const
            ParamSolver<T>& param) : Catalyst<SolverType>(loss,regul,param),
      _l_memory(param.l_memory) { 
         _skipping_steps=0;
         _line_search_steps=0;
      }; 

      virtual void solve(const D& x0, D& x) {
         Solver<loss_type>::solve(x0,x);
         if (_verbose)
            cout << "Total additional line search steps: " << _line_search_steps << endl;
         if (_verbose)
            cout << "Total skipping l-bfgs steps: " << _skipping_steps << endl;
      };

   protected:

      virtual void solver_init(const D& x0) {
         Catalyst<SolverType>::solver_init(x0);
         if (_accelerated_solver) {
            _h0=T(1.0)/_kappa;
            _m=0;
            if (_verbose)
               cout << "Memory parameter: " << _l_memory << endl;
            _ys.resize(x0.size(),_l_memory);
            _ss.resize(x0.size(),_l_memory);
            _rhos.resize(_l_memory);
            _etak=T(1.0);
            _skipping_steps=0;
            _line_search_steps=0;
         }
      };
      
      virtual void solver_aux(D& x) {
         if (_accelerated_solver) {
            if (_gk.size() == 0) 
               get_gradient(x);

            // update variable _y and test
            D oldyk; oldyk.copy(_y);
            D oldxk; oldxk.copy(x);
            T oldFk=_Fk;
            D oldgk; oldgk.copy(_gk);
            D g; get_lbfgs_direction(g);

            const int max_iter=5;
            _auxiliary_solver->save_state();
            for (int ii=1; ii<=max_iter; ++ii) {
               _y.copy(oldyk);
               _y.add(g,-_etak);
               _y.add(oldgk,-(T(1.0)-_etak)/_kappa);
               get_gradient(x); // _gk = kappa(x-y)
               if (_etak == 0 || _Fk <= oldFk - (T(0.25)/_kappa)*oldgk.nrm2sq()) break;
               if (_Fk > 1.05*oldFk) {
                  _auxiliary_solver->restore_state();
                  x.copy(oldxk);
               }
               _etak /= 2;
               _line_search_steps++;
               if (ii==max_iter-1 || _etak < T(0.1)) {
                  _etak=0;
               }
            }
            if (_Fk > 1.05*oldFk) {
               _auxiliary_solver->restore_state();
               x.copy(oldxk);
               reset_lbfgs();
            } else {
               oldyk.add_scal(_y,T(1.0),-T(1.0));
               oldgk.add_scal(_gk,T(1.0),-T(1.0));
               update_lbfgs_matrix(oldyk,oldgk);
            }
            _etak=MAX(MIN(T(1.0),_etak*T(1.2)),T(0.1));
         } else {
            SolverType::solver_aux(x);
         }
      };
      void print() const {
         cout << "QNing Accelerator" << endl;
         SolverType::print();
      };
   private:
      inline void get_lbfgs_direction(Vector<T>& g) const {
         g.copy(_gk);
         get_lbfgs_direction_aux(g);
      };
      inline void get_lbfgs_direction(Matrix<T>& g) const {
         g.copy(_gk);
         Vector<T> gg;
         g.toVect(gg);
         get_lbfgs_direction_aux(gg);
      };

      inline void get_lbfgs_direction_aux(Vector<T>& g) const {
         // two-loop recursion algorithm
         Vector<T> alphas(_l_memory);
         Vector<T> cols, coly;
         T gamma=T(1.0)/_kappa;
         for (int ii = _m-1; ii>= MAX(_m-_l_memory,0); --ii) {
            const int ind = ii % _l_memory;
            _ss.refCol(ind,cols);
            _ys.refCol(ind,coly);
            if (ii==_m-1)
               gamma=cols.dot(coly)/coly.nrm2sq();
            alphas[ind]=_rhos[ind]*cols.dot(g);
            g.add(coly,-alphas[ind]);
         }
         g.scal(gamma);
         for (int ii = MAX(_m-_l_memory,0); ii<= _m-1; ++ii) {
            const int ind = ii % _l_memory;
            _ss.refCol(ind,cols);
            _ys.refCol(ind,coly);
            const T beta = _rhos[ind]*coly.dot(g);
            g.add(cols,alphas[ind]-beta);
         }
      };
      inline void update_lbfgs_matrix(const Matrix<T>& sk, const Matrix<T>& yk) {
         Vector<T> skk, ykk;
         sk.toVect(skk);
         yk.toVect(ykk);
         update_lbfgs_matrix(skk,ykk);
      };
      inline void update_lbfgs_matrix(const Vector<T>& sk, const Vector<T>& yk) {
         const T theta=sk.dot(yk);
         if (theta > T(1e-12)) {
            Vector<T> coly, cols;
            const int ind=_m % _l_memory;
            _ys.refCol(ind,coly); coly.copy(yk);
            _ss.refCol(ind,cols); cols.copy(sk);
            _rhos[ind]=T(1.0)/theta;
            _m++;
         } else {
            _skipping_steps++;
            //if (_skipping_steps % 10 == 0)
            //   reset_lbfgs();
         }
      };
      void reset_lbfgs() {
         _m=0;
      };
      void get_gradient(D& x) {
         _loss_ppa->set_anchor_point(_y);
         _auxiliary_solver->solve(_y,x);
         _gk.copy(_y);
         _gk.add_scal(x,-_kappa,_kappa);
         _Fk=_loss_ppa->eval(x)+_regul.eval(x);
      };

      T _h0;
      int _l_memory;
      INTM _m;
      Matrix<T> _ys, _ss;
      Vector<T> _rhos;
      D _gk, _xk;
      T _Fk;
      T _etak;
      int _skipping_steps, _line_search_steps;
};


template <typename loss_type, bool allow_acc = true> 
class Acc_SVRG_Solver: public SVRG_Solver<loss_type> {
   public:
      USING_SVRG_SOLVER;
      Acc_SVRG_Solver(const loss_type& loss, const Regularizer<D,I>& regul, const
            ParamSolver<T>& param, const Vector<T>* Li=NULL) : SVRG_Solver<loss_type>(loss,regul,param,Li) {
         _accelerated_solver=allow_acc;
      };

      virtual void solver_init(const D& x0) {
         IncrementalSolver<loss_type>::solver_init(x0);
         _mu = _regul.strong_convexity();
         _nn=_n/_minibatch;
         _accelerated_solver=allow_acc && (T(20)*this->_oldL/_nn > _mu);
         if (_accelerated_solver) {
            _gammak = MAX(_L/(_nn),_mu);
            update_acceleration_parameters();
            _xtilde.copy(x0);
            _y.copy(x0);
            _loss.grad(_xtilde,_gtilde);
         } else {
            if (_verbose && allow_acc) 
               cout << "Problem is well conditioned, switching to regular solver" << endl;
            SVRG_Solver<loss_type>::solver_init(x0);
         }
      };

      virtual void solver_aux(D& x) {
         if (_accelerated_solver) {
            for (int ii=0; ii<_nn; ++ii) {
               x.copy(_y);
               x.add(_gtilde,-_etak);
               for (int jj=0; jj<_minibatch; ++jj) {
                  const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
                  const T scal = _non_uniform_sampling ? T(1.0)/(_qi[ind]*_n*_minibatch) : T(1.0)/_minibatch;
                  _loss.double_add_grad(_y,_xtilde,ind,x,-scal*_etak,scal*_etak);
               }
               _regul.prox(x,x,_etak);

               const T alphak=_mu*_deltak/_gammak;
               const T betak=_deltak/(_gammak*_etak);
               const T a=(T(1.0)-alphak)/_thetak + alphak;
               update_acceleration_parameters();
               if (random() % _nn == 0) {
                  _y.add_scal(_xtilde,(T(1.0)-a)*_thetak,_thetak*(a-betak));
                  _y.add(x,betak*_thetak + T(1.0)-_thetak);
                  _xtilde.copy(x);
                  _loss.grad(_xtilde,_gtilde);
               } else {
                  _y.add_scal(_xtilde,T(1.0)-_thetak*a,_thetak*(a-betak));
                  _y.add(x,betak*_thetak);
               }
            };
         } else {
            SVRG_Solver<loss_type>::solver_aux(x);
         }
      };

   protected:
      void print() const {
         cout << "Accelerated SVRG Solver" << endl;
         if (!_accelerated_solver) 
            cout << "Problem is well conditioned, switching to regular solver" << endl;
         IncrementalSolver<loss_type>::print();
      };

      bool _accelerated_solver;
      T _gammak, _mu, _deltak, _etak, _thetak;
      D _y;
      int _nn;

      void update_acceleration_parameters() {
         _deltak = MIN( solve_binomial(T(9.0)*_nn*_L/T(5.0), _gammak-_mu, -_gammak), T(1.0)/(3*_nn));
         _gammak = (T(1.0)-_deltak)*_gammak+_mu*_deltak;
         _etak = MIN(T(1.0)/(3*_L), T(1.0)/(15*_gammak*_nn));  
         _thetak=(3*_nn*_deltak-5*_mu*_etak)/(3-5*_mu*_etak);
      };
};

template <typename loss_type, bool allow_acc=true> 
class SVRG_Solver_FastRidge: public Acc_SVRG_Solver< loss_type, allow_acc > {
   public:
      USING_ACC_SVRG_SOLVER;

      SVRG_Solver_FastRidge(const loss_type& loss, const Regularizer<D,I>& regul, const
            ParamSolver<T>& param, const Vector<T>* Li = NULL) : Acc_SVRG_Solver<loss_type,allow_acc>(loss,regul,param,Li),
      _is_lazy(loss_type::is_sparse()) {
         if (param.minibatch > 1)
            cerr << "Minibatch is not compatible with lazy updates" << endl;
         _minibatch=1;
      };
      virtual void solver_init(const D& x0) {
         Acc_SVRG_Solver<loss_type,allow_acc>::solver_init(x0);
         if (_loss.id() == PPA) {
            const T kappa = _loss.kappa();
            _gtilde.add(_xtilde,-kappa);// now gtilde has the right value
         }
      };

      /// define auxiliary solver ?
      virtual void solver_aux(D& x) {
         if (_accelerated_solver) {
            const T lambda=_regul.lambda(); 
            DoubleLazyVector<T,I>* lazyy = NULL;
            Vector<I> indices;
            if (_is_lazy) {
               lazyy=new DoubleLazyVector<T,I>(_y,_xtilde,_gtilde,_n);
            }
            for (int ii=0; ii<_n; ++ii) {
               const T alphak=_mu*_deltak/_gammak;
               const T betak=_deltak/(_gammak*_etak);
               const T a=(T(1.0)-alphak)/_thetak + alphak;
               const T scalprox = T(1.0)/(T(1.0)+lambda*_etak);
               const T eta=_etak;
               const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
               const T scaleta = _non_uniform_sampling ? eta/(_qi[ind]*_n) : eta;
               this->update_acceleration_parameters();
               const bool update_xtilde = random() % _n == 0;
               const T coeffy=_thetak*(a-betak);
               const T coeffx= update_xtilde ? betak*_thetak + T(1.0)-_thetak :betak*_thetak;  
               const T coeffxtilde = update_xtilde ? (T(1.0)-a)*_thetak : T(1.0)-_thetak*a;

               if (update_xtilde || ii==_n-1) {
                  if (_is_lazy) lazyy->update();
                  x.copy(_y);
                  _loss.double_add_grad(_y,_xtilde,ind,x,-scaleta,scaleta);
                  x.add_scal(_gtilde,-scalprox*eta,scalprox);
                  _y.add_scal(_xtilde,coeffxtilde,coeffy);
                  _y.add(x,coeffx);
               } else {
                  const T coeff_add_grad=scaleta * scalprox*coeffx/(coeffy+scalprox*coeffx);
                  if (_is_lazy) {
                     _loss.get_coordinates(ind, indices);
                     lazyy->update(indices);
                  }
                  _loss.double_add_grad(_y,_xtilde,ind,_y,-coeff_add_grad,coeff_add_grad); 
                  if (_is_lazy) {
                     lazyy->add_scal(coeffxtilde,-scalprox*eta*coeffx,coeffy+scalprox*coeffx); 
                  } else {
                     _y.add_scal(_gtilde,-scalprox*eta*coeffx,coeffy+scalprox*coeffx);
                     _y.add(_xtilde,coeffxtilde);
                  }
               }
               if (update_xtilde) {
                  _xtilde.copy(x);
                  _loss.grad(_xtilde,_gtilde);
               } 
            };
            if (_is_lazy)
               delete(lazyy);
         } else if (_loss.id() == PPA) {
            /// we will optimize implicitly  f(xtilde) - kappa <x , z> + (mu+kappa)/2|x|^2  
            /// meaning, we want gtilde to be  Df(xtilde) - kappa z
            LazyVector<T,I>* lazyx = NULL;  
            Vector<I> indices;
            if (_is_lazy) {
               //indices.resize(x.n());
               lazyx=new LazyVector<T,I>(x,_gtilde,_n);
            }
            const T eta = T(1.0)/(3*(_L-_loss.kappa()));
            const T lambda=_regul.lambda()+_loss.kappa(); // take care of 0.5(mu+kappa)|x|^2, 
            for (int ii = 0; ii<_n; ++ii) {
               const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
               const T scal = _non_uniform_sampling ? T(1.0)/(_qi[ind]*_n) : T(1.0);
               if (_is_lazy) {
                  _loss.get_coordinates(ind, indices);
                  lazyx->update(indices);
                  _loss.double_add_grad(x,_xtilde,ind,x,-scal*eta,scal*eta,0); 
                  lazyx->add_scal(-eta,T(1.0)/(T(1.0)+eta*lambda));   
               } else {
                  _loss.double_add_grad(x,_xtilde,ind,x,-scal*eta,scal*eta,0);  // x <- x - s( D_i f(x) - D_i f(xtilde))
                  x.add_scal(_gtilde,-eta/(T(1.0)+eta*lambda),T(1.0)/(T(1.0)+eta*lambda));
               }

               if (random() % _n == 0) {
                  if (_is_lazy) lazyx->update();
                  _xtilde.copy(x);
                  _loss.grad(_xtilde,_gtilde);// gtilde will be equal to Df(xtilde) + kappa (xtilde- z)
                  _gtilde.add(_xtilde,-_loss.kappa());// now gtilde has the right value
               }
            }
            if (_is_lazy) {
               lazyx->update();
               delete(lazyx);
            }
         } else {
            LazyVector<T,I>* lazyx = NULL;
            Vector<I> indices;
            if (_is_lazy) {
               //indices.resize(x.n());
               lazyx=new LazyVector<T,I>(x,_gtilde,_n);
            }
            const T eta = T(1.0)/(3*_L);
            const T lambda=_regul.lambda(); // replace by lazyprox ?
            for (int ii = 0; ii<_n; ++ii) {
               const int ind = _non_uniform_sampling ? this->nonu_sampling() : random() % _n;
               const T scal = _non_uniform_sampling ? T(1.0)/(_qi[ind]*_n) : T(1.0);
               if (_is_lazy) {
                  _loss.get_coordinates(ind, indices);
                  lazyx->update(indices);
                  _loss.double_add_grad(x,_xtilde,ind,x,-scal*eta,scal*eta); 
                  lazyx->add_scal(-eta,T(1.0)/(T(1.0)+eta*lambda));   // replace by lazyprox ?
               } else {
                  _loss.double_add_grad(x,_xtilde,ind,x,-scal*eta,scal*eta); 
                  x.add_scal(_gtilde,-eta/(T(1.0)+eta*lambda),T(1.0)/(T(1.0)+eta*lambda));
               }
               if (random() % _n == 0) {
                  if (_is_lazy) lazyx->update();
                  _xtilde.copy(x);
                  _loss.grad(_xtilde,_gtilde);
               }
            }
            if (_is_lazy) {
               lazyx->update();
               delete(lazyx);
            }
         }
      };

   protected:
      void print() const {
         if (_accelerated_solver) {
            cout << "Accelerated SVRG Solver, ";
         } else {
            cout << "SVRG Solver, ";
         }
         if (_is_lazy) {
            cout << "specialized for sparse matrices and L2 regularization" << endl;
         } else {
            cout << "specialized for L2 regularization" << endl;
         }
         IncrementalSolver<loss_type>::print();
      };
   private:
      const bool _is_lazy;
};


template <typename loss_type>
Solver<loss_type>* get_solver(const loss_type& loss, const Regularizer<typename loss_type::variable_type, typename loss_type::index_type>& regul, const ParamSolver<typename loss_type::value_type>& param) {
   typedef typename loss_type::value_type T;
   Solver<loss_type>* solver;
   solver_t solver_type=param.solver;
   if (solver_type==AUTO) {
      const T L=loss.lipschitz();
      const int n = loss.n();
      const T lambda=regul.strong_convexity();
      if (n < 1000) {
         solver_type=QNING_ISTA;
      } else if (lambda < L/(100*n)) {
         solver_type=QNING_MISO;
      } else {
         solver_type=CATALYST_MISO;
      }
   }
   switch (solver_type) {
      case ISTA: solver= new ISTA_Solver<loss_type>(loss,regul,param); break;
      case QNING_ISTA: solver = new QNing< ISTA_Solver<loss_type> >(loss,regul,param); break;
      case CATALYST_ISTA: solver = new Catalyst< ISTA_Solver<loss_type> >(loss,regul,param); break;
      case FISTA: solver= new FISTA_Solver<loss_type>(loss,regul,param); break;
      case SVRG: solver= new SVRG_Solver<loss_type>(loss,regul,param); break;
      case MISO: solver= regul.strong_convexity() > 0 ?
                 new MISO_Solver<loss_type>(loss,regul,param) : 
                    new Catalyst< MISO_Solver<loss_type> >(loss,regul,param);
                 break;
      case SVRG_UNIFORM: {
                            ParamSolver<typename loss_type::value_type> param2=param;
                            param2.non_uniform_sampling=false;
                            solver= new SVRG_Solver<loss_type>(loss,regul,param2); break;
                         }
      case CATALYST_SVRG: solver = new Catalyst< SVRG_Solver<loss_type> >(loss,regul,param); break;
      case QNING_SVRG: solver = new QNing< SVRG_Solver<loss_type> >(loss,regul,param); break;
      case CATALYST_MISO: solver = new Catalyst< MISO_Solver<loss_type> >(loss,regul,param); break;
      case QNING_MISO: solver = new QNing< MISO_Solver<loss_type> >(loss,regul,param); break;
      case ACC_SVRG: solver = new Acc_SVRG_Solver<loss_type>(loss,regul,param); break;
      default: cerr << "Not implemented, performs nothing"; solver=NULL; 
   }
   return solver;
};


template <typename M>
void simple_erm(const M& X, const Vector<typename M::value_type>& y, const Vector<typename M::value_type>& w0, Vector<typename M::value_type>& w, Vector<typename M::value_type>& dual_variable, Matrix<typename M::value_type>& optim_info, const ParamSolver<typename M::value_type>& param, const ParamModel<typename M::value_type>& model) {
   init_omp(param.threads);
   typedef typename M::value_type T;
   typedef typename M::index_type I;
   typedef Vector<T> D;
   typedef LinearLossVec<M> loss_type; 
   if (model.intercept) {
      if (X.m()+1 != w0.n()) { 
         cerr << "Dimension of initial point is not consistent. With intercept, if X is m x n, w0 should be (n+1)-dimensional." << endl; 
         return;
      }
   } else {
      if (X.m() != w0.n()) { 
         cerr << "Dimension of initial point is not consistent. If X is m x n, w0 should be n-dimensional." << endl; 
         return;
      }
   }
   DataLinear<M> data(X,model.intercept); 
   if (param.verbose)
      data.print();
   LinearLossVec<M>* loss;
   switch (model.loss) {
      case SQUARE: loss = new SquareLoss<M>(data,y); break;
      case LOGISTIC: loss = new LogisticLoss<M>(data,y); break;
      case SQHINGE: loss = new SquaredHingeLoss<M>(data,y); break;
      //case HINGE: loss = new HingeLoss<M>(data,y); break;
      case SAFE_LOGISTIC: loss = new SafeLogisticLoss<M>(data,y); break;
      default: cerr << "Not implemented, square loss is chosen by default";
               loss = new SquareLoss<M>(data,y);
   }
   Regularizer<D,I>* regul;
   switch(model.regul) {
      case L2: regul = new Ridge<D,I>(model); break;
      case L1: regul = new Lasso<D,I>(model); break;
      case L1BALL: regul = new L1Ball<D,I>(model); break;
      case L2BALL: regul = new L2Ball<D,I>(model); break;
      case FUSEDLASSO: regul = new FusedLasso<D,I>(model); break;
      case ELASTICNET: regul = new ElasticNet<D,I>(model); break;
      case NONE: regul = new None<D,I>(model); break;
      default: cerr << "Not implemented, no regularization is chosen";
               regul = new None<D,I>(model);
   }
   Solver<loss_type>* solver;
   if (param.nepochs==0) {
      ParamSolver<typename D::value_type> param2=param;
      param2.verbose=false;
      solver= new ISTA_Solver<loss_type>(*loss,*regul,param2);
      solver->eval(w0);
      w.copy(w0);
   } else {
      if (param.solver==SVRG && model.regul==L2 && !model.intercept) {
         solver= new SVRG_Solver_FastRidge<loss_type,false>(*loss,*regul,param);
      } else if (param.solver==ACC_SVRG && model.regul==L2 && !model.intercept) {
         solver= new SVRG_Solver_FastRidge<loss_type,true>(*loss,*regul,param);
      } else if (param.solver==CATALYST_SVRG && model.regul==L2 && !model.intercept) {
         solver = new Catalyst< SVRG_Solver_FastRidge<loss_type,false> >(*loss,*regul,param); 
      } else if (param.solver==QNING_SVRG && model.regul==L2 && !model.intercept) {
         solver = new QNing< SVRG_Solver_FastRidge<loss_type,false> >(*loss,*regul,param); 
      } else {
         solver=get_solver<loss_type>(*loss,*regul,param);
         if (!solver) {
            w.copy(w0); 
            delete(loss);
            delete(regul);
            return;
         }
      }
      D new_w0;
      if (model.intercept) {
         data.set_intercept(w0,new_w0);
      } else {
         new_w0.copyRef(w0);
      }
      if (dual_variable.n() != 0)
         solver->set_dual_variable(dual_variable);
      solver->solve(new_w0,w);
      if (model.intercept) {
         data.reverse_intercept(w);
      }
   }
   if (model.regul==L1)
      for (int ii=0; ii<w.n(); ++ii) 
         if (abs<T>(w[ii]) < EPSILON) w[ii]=0;
   solver->get_optim_info(optim_info);
   delete(solver);
   delete(loss);
   delete(regul);
};


template <typename T, typename I>
Regularizer<Matrix<T>,I>* get_regul_mat(const ParamModel<T>& model, const int nclass, const bool transpose) {
   typedef Matrix<T> D;
   typedef Vector<T> V;
   Regularizer<D,I>* regul;
   switch(model.regul) {
      case L2: regul = transpose ? static_cast<Regularizer<D,I>*>(new RegVecToMat<Ridge<V,I> >(model))
               : new RegMat<Ridge<V,I> >(model,nclass,transpose); break;  
      case L1: regul = transpose ? static_cast<Regularizer<D,I>*>(new RegVecToMat<Lasso<V,I> >(model))
               : new RegMat<Lasso<V,I> >(model,nclass,transpose); break;
      case ELASTICNET: regul = transpose ? static_cast<Regularizer<D,I>*>(new RegVecToMat<ElasticNet <V,I> >(model))
                       : new RegMat<ElasticNet<V,I> >(model,nclass,transpose); break;
      case L1BALL: regul = new RegMat<L1Ball<V,I> >(model,nclass,transpose); break;
      case L2BALL: regul = new RegMat<L2Ball<V,I> >(model,nclass,transpose); break;
      case L1L2: regul = new MixedL1L2<T,I>(model,nclass,transpose); break;
      case L1L2_L1: regul = new MixedL1L2_L1<T,I>(model,nclass,transpose); break;
      case L1LINF: regul = new MixedL1Linf<T,I>(model,nclass,transpose); break;
      case FUSEDLASSO: regul = new RegMat<FusedLasso<V,I> >(model,nclass,transpose); break;
      case NONE: regul = new None<D,I>(model); break;
      default: cerr << "Not implemented, no regularization is chosen";
               regul = new None<D,I>(model);
   }
   return regul;
};

template <typename loss_type>
void solve_mat(loss_type& loss, const Regularizer<typename loss_type::variable_type, typename loss_type::index_type>& regul, const ParamSolver<typename loss_type::value_type>& param, const typename loss_type::variable_type& W0, typename loss_type::variable_type& W, Matrix<typename loss_type::value_type>& dual_variable,  Matrix<typename loss_type::value_type>& optim_info) {
   typedef typename loss_type::value_type T;
   typedef typename loss_type::variable_type D;
   Solver<loss_type>* solver;
   if (param.nepochs==0) {
      ParamSolver<T> param2=param;
      param2.verbose=false;
      solver= new ISTA_Solver<loss_type>(loss,regul,param2);
      if (loss.transpose()) {
         Matrix<T> W0T, WT;
         W0.transpose(W0T);
         solver->eval(W0T);
      } else {
         solver->eval(W0);
      }
      W.copy(W0);
   } else {
      solver=get_solver<loss_type>(loss,regul,param); 
      if (!solver) {
         W.copy(W0); 
         return;
      }
      D new_W0;
      if (loss.intercept()) {
         loss.set_intercept(W0,new_W0);
      } else {
         new_W0.copyRef(W0);
      }
      if (dual_variable.n() != 0)
         solver->set_dual_variable(dual_variable);
      if (loss.transpose()) {
         Matrix<T> W0T, WT;
         new_W0.transpose(W0T);
         solver->solve(W0T,WT);
         WT.transpose(W);
      } else {
         solver->solve(new_W0,W);
      }
      if (loss.intercept()) {
         loss.reverse_intercept(W);
      }
   }
   if (regul.id()==L1)
      for (INTM ii=0; ii<W.n(); ++ii) 
         for (INTM jj=0; jj<W.m(); ++jj) 
         if (abs<T>(W(jj,ii)) < EPSILON) W(jj,ii)=0;

   solver->get_optim_info(optim_info);
   delete(solver);
};

// X is p x n
// y is nclasses x n
// W0 is p x nclasses if no intercept (or p+1 x nclasses with intercept)
// prediction model is   W0^T X  gives  nclasses x n
template <typename M>
void multivariate_erm(const M& X, const Matrix<typename M::value_type>& y, const Matrix<typename M::value_type>& W0, Matrix<typename M::value_type>& W,  Matrix<typename M::value_type>& dual_variable, Matrix<typename M::value_type>& optim_info, const ParamSolver<typename M::value_type>& param, const ParamModel<typename M::value_type>& model) {
   typedef typename M::value_type T;
   typedef typename M::index_type I;
   if ((model.intercept && X.m()+1 != W0.m()) || (!model.intercept && X.m() != W0.m())) {
      cerr << "Dimension of initial point is not consistent." << endl; 
      return;
   }
   init_omp(param.threads);
   typedef Matrix<T> D;
   if (is_loss_for_matrices(model.loss) || is_regul_for_matrices(model.regul)) {
      DataMatrixLinear<M> data(X,model.intercept); 
      if (param.verbose)
         data.print();
      LinearLossMat< M, Matrix<T> >* loss;
      switch (model.loss) {
         case SQUARE: loss = new SquareLossMat<M>(data,y); break;
         case LOGISTIC: loss = new LossMat< LogisticLoss<M> >(data,y); break;
         case SQHINGE:  loss = new LossMat< SquaredHingeLoss<M> >(data,y); break;
         //case HINGE:  loss = new LossMat< HingeLoss<M> >(data,y); break;
         case SAFE_LOGISTIC: loss = new LossMat<SafeLogisticLoss<M> >(data,y); break;
         default: cerr << "Not implemented, square loss is chosen by default";
                  loss = new SquareLossMat<M>(data,y);
      }
      const int nclass=W0.n();
      Regularizer<D,I>* regul = get_regul_mat<T,I>(model,nclass,loss->transpose());
      solve_mat<LinearLossMat< M, Matrix<T> > >(*loss,*regul,param,W0,W,dual_variable,optim_info);
      delete(regul);
      delete(loss);
   } else {
      W.copy(W0);
      const int nclass=W0.n();
      const int it0=MAX(param.it0,1);
      optim_info.resize(6,MAX(param.nepochs/it0,1));
      optim_info.setZeros();
      ParamSolver<T> param2=param;
      param2.verbose=false;
      if (param.verbose) {
         DataMatrixLinear<M> data(X,model.intercept); 
         data.print();
      }
      Timer global_all;
      global_all.start();
#pragma omp parallel for
      for (int ii=0; ii<nclass; ++ii) {
         Vector<T> w0col, wcol, ycol, dualcol;
         Matrix<T> optim_info_col;
         W0.refCol(ii,w0col);
         W.refCol(ii,wcol);
         y.copyRow(ii,ycol);
         if (dual_variable.m() == nclass)
            dual_variable.copyRow(ii,dualcol);
         simple_erm(X,ycol,w0col,wcol,dualcol,optim_info_col,param2,model);
         if (dual_variable.m() == nclass)
            dual_variable.copyToRow(ii,dualcol);
#pragma omp critical 
         {
            optim_info.add(optim_info_col);
            if (param.verbose) {
               const int noptim=optim_info_col.n()-1;
               cout << "Solver " << ii << " has terminated after " << optim_info_col(0,noptim) << " epochs in " << optim_info_col(5,noptim) << " seconds" << endl;
               if (optim_info_col(4,noptim)==0) {
                  cout << "   Primal objective: " << optim_info_col(1,noptim) << ", relative duality gap: " << optim_info_col(3,noptim) << endl;
               } else {
                  cout << "   Primal objective: " << optim_info_col(1,noptim) << ", tol: " << optim_info_col(4,noptim) << endl;
               }
            }
         }
      }
      global_all.stop();
      if (param.verbose) {
         cout << "Time for the one-vs-all strategy" << endl;
         global_all.printElapsed();
      }
   }
};

template <typename M>
void multivariate_erm(const M& X, const Vector<int>& y, const Matrix<typename M::value_type>& W0, Matrix<typename M::value_type>& W, Matrix<typename M::value_type>& dual_variable, Matrix<typename M::value_type>& optim_info, const ParamSolver<typename M::value_type>& param, const ParamModel<typename M::value_type>& model) {
   typedef typename M::value_type T;
   typedef typename M::index_type I;
   if ((model.intercept && X.m()+1 != W0.m()) || (!model.intercept && X.m() != W0.m())) {
      cerr << "Dimension of initial point is not consistent." << endl; 
      return;
   }
   const int nclass=y.maxval()+1;
   if ((is_regression_loss(model.loss) || !is_loss_for_matrices(model.loss))) {
      const int n = y.n();
      Matrix<typename M::value_type> labels(nclass,n);
      labels.set(-(1.0));
      for (int ii=0; ii<n; ++ii) 
         labels(y[ii],ii)=(1.0);
      return multivariate_erm(X,labels,W0,W,dual_variable,optim_info,param,model); 
   }
   init_omp(param.threads);
   typedef Matrix<T> D;
   DataMatrixLinear<M> data(X,model.intercept); 
   if (param.verbose)
      data.print();
   LinearLossMat<M, Vector<int> >* loss;
   switch (model.loss) {
      case MULTI_LOGISTIC: loss= new MultiClassLogisticLoss<M>(data,y); break;
      default: cerr << "Not implemented, multilog loss is chosen by default";
               loss= new MultiClassLogisticLoss<M>(data,y);
   }
   Regularizer<D,I>* regul = get_regul_mat<T,I>(model,nclass,loss->transpose());
   solve_mat<LinearLossMat<M, Vector<int> > >(*loss,*regul,param,W0,W,dual_variable,optim_info);
   delete(regul);
   delete(loss);
};


#endif
