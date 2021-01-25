#include "utils/linalg.h"
#include "utils/wrapper_utils.h"
#include "utils/solvers.h"

template <typename T, typename I>
static PyArrayObject* erm(PyObject* inX, PyArrayObject* inY, PyArrayObject* inw0, PyArrayObject* inw, PyArrayObject* in_dual, const int nepochs, const int l_qning, const int f_restart, const T tol, const int it0, const bool verbose, char* solver, char* loss, char* regul, const T lambda, const T lambda2, const T lambda3, const bool intercept, const bool univariate, const int nthreads) {
   ParamSolver<T> param;
   param.nepochs=nepochs;
   param.l_memory=l_qning;
   param.freq_restart=f_restart;
   param.tol=tol;
   param.it0=it0;
   param.verbose=verbose;
   param.solver=solver_from_string(solver);
   param.non_uniform_sampling=false; // TODO: check if needs to be activated again
   param.threads=nthreads;
   ParamModel<T> model;
   model.loss=loss_from_string(loss); 
   model.regul=regul_from_string(regul);
   model.lambda=lambda;
   model.lambda2=lambda2;
   model.lambda3=lambda3;
   model.intercept=intercept;
   clean_param_model(model);
   Matrix<T> optim_info;

   if (univariate) {
      Vector<T> y, w0, w, dual_variable;
      if (!npyToVector<T>(inY,y,"Data y")); 
      if (!npyToVector<T>(inw0,w0,"x0"));
      if (!npyToVector<T>(inw,w,"x")); 
      if (reinterpret_cast<PyObject*>(in_dual) != Py_None && !npyToVector<T>(in_dual,dual_variable,"dual")); 
      if (w0.n() != w.n()) {
         PyErr_SetString(PyExc_TypeError, "Got wrong input size"); 
         return NULL;
      }
      if (isSparseMatrix(inX)) {
         SpMatrix<T,I> X;
         if (!npyToSpMatrix<T,I>(inX,X,"Data"));
         param.minibatch=MIN((int)floor((T(X.n())*T(X.m()))/T(X.nzmax())),X.n()/100); // aggressive strategy, but only uses minibatch if required
         simple_erm(X,y,w0,w,dual_variable,optim_info,param,model);
      } else {
         Matrix<T> X;
         param.minibatch=1;
         if (!npyToMatrix<T>((PyArrayObject*)inX,X,"Data X"));
         simple_erm(X,y,w0,w,dual_variable,optim_info,param,model);
      }
   } else {
      Matrix<T> w0, w, dual_variable;
      if (!npyToMatrix<T>(inw0,w0,"x0"));
      if (!npyToMatrix<T>(inw,w,"x")); 
      if (reinterpret_cast<PyObject*>(in_dual) != Py_None && !npyToMatrix<T>(in_dual,dual_variable,"dual")); 
      if (isSparseMatrix(inX)) {
         SpMatrix<T,I> X;
         if (!npyToSpMatrix<T,I>(inX,X,"Data"));
         param.minibatch=MIN((int)floor((T(X.n())*T(X.m()))/T(X.nzmax())),X.n()/100); 
         if (array_type(inY) == getTypeNumber<int>()) {
            Vector<int> y;
            if (!npyToVector<int>(inY,y,"Data y")); 
            multivariate_erm(X,y,w0,w,dual_variable,optim_info,param,model);
         } else {
            Matrix<T> y;
            if (!npyToMatrix<T>(inY,y,"Data y")); 
            multivariate_erm(X,y,w0,w,dual_variable,optim_info,param,model);
         }
      } else {
         Matrix<T> X;
         param.minibatch=1;
         if (!npyToMatrix<T>((PyArrayObject*)inX,X,"Data X"));
         if (array_type(inY) == getTypeNumber<int>()) {
            Vector<int> y;
            if (!npyToVector<int>(inY,y,"Data y")); 
            multivariate_erm(X,y,w0,w,dual_variable,optim_info,param,model);
         } else {
            Matrix<T> y;
            if (!npyToMatrix<T>(inY,y,"Data y")); 
            multivariate_erm(X,y,w0,w,dual_variable,optim_info,param,model);
         }
      }
   }
   PyArrayObject* out= create_np_matrix<T>(optim_info.m(),optim_info.n());
   Matrix<T> outm;
   npyToMatrix(out,outm,"optim info");
   outm.copy(optim_info);
   return out;
};

static PyObject* erm_(PyObject* self, PyObject* args, PyObject* keywds)
{
   PyObject* X=NULL;
   PyArrayObject* y=NULL;
   PyArrayObject* w0=NULL;
   PyArrayObject* w=NULL;
   PyArrayObject* dual=NULL;
   PyArrayObject* optim_info=NULL;
   int nepochs = 1000;
   int l_qning = 20;
   int f_restart = 50;
   double tol = 1e-3;
   int it0=10;
   int verbose = 1;
   int univariate = 1;
   int nthreads = 1;
   int seed = 0;
   double lambda=0;
   double lambda2=0;
   double lambda3=0;
   int intercept=0;
   char* regul=(char*)"none";
   char* solver=(char*)"ista";
   char* loss=(char*)"square";
   static char* kwlist[] = { (char*)"", (char*)"", (char*)"", (char*)"", (char*)"dual_variable", (char*)"loss", (char*)"penalty", (char*)"solver", (char*)"lambd", (char*)"lambd2", (char*)"lambd3", (char*)"intercept", (char*)"tol", (char*)"it0", (char*)"nepochs", (char*)"l_qning", (char*)"f_restart", (char*)"verbose", (char*)"univariate", (char*)"nthreads", (char*)"seed", NULL };
   const char* format = (const char*)"OOOO|Osssdddpdiiiippii";
   if (!PyArg_ParseTupleAndKeywords(args, keywds,format,kwlist,&X,&y,&w0,&w,&dual,&loss,&regul,&solver,&lambda,&lambda2,&lambda3,&intercept,&tol,&it0,&nepochs,&l_qning,&f_restart,&verbose,&univariate,&nthreads,&seed))
      return NULL;
   it0= it0 <= 0 ? -1 : MIN(it0,nepochs);
   srandom(seed);
   int T, I;
   getTypeObject((PyObject*)X,T,I);
   if (T == getTypeNumber<float>() && I == getTypeNumber<int>()) {
      optim_info=erm<float,int>(X,y,w0,w,dual,nepochs,l_qning,f_restart,tol,it0,verbose,solver,loss,regul,lambda,lambda2,lambda3,intercept,univariate,nthreads);
   } else if (T == getTypeNumber<float>() && I == getTypeNumber<long long int>()) {
      optim_info=erm<float,long long int>(X,y,w0,w,dual,nepochs,l_qning,f_restart,tol,it0,verbose,solver,loss,regul,lambda,lambda2,lambda3,intercept,univariate,nthreads);
   } else if (T == getTypeNumber<double>() && I == getTypeNumber<int>()) {
      optim_info=erm<double,int>(X,y,w0,w,dual,nepochs,l_qning,f_restart,tol,it0,verbose,solver,loss,regul,lambda,lambda2,lambda3,intercept,univariate,nthreads);
   } else if (T == getTypeNumber<double>() && I == getTypeNumber<long long int>()) {
      optim_info=erm<double,long long int>(X,y,w0,w,dual,nepochs,l_qning,f_restart,tol,it0,verbose,solver,loss,regul,lambda,lambda2,lambda3,intercept,univariate,nthreads);
   } else {
      PyErr_SetString(PyExc_TypeError, ("Got wrong data type: ")); 
      //PyErr_SetString(PyExc_TypeError, ("Got wrong data type: "+std::to_string(T)).c_str()); 
      return NULL;
   }
   return PyArray_Return(optim_info);
};


template <typename T, typename I>
static void preprocess_generic(PyObject* in, const bool centering, const bool normalize, const bool columns = true) {
   if (isSparseMatrix(in)) {
      SpMatrix<T,I> X;
      if (!npyToSpMatrix<T,I>(in,X,"Data"));
      if (columns) {
         if (normalize) X.normalize();
      } else {
         if (normalize) X.normalize_rows();
      }
   } else {
      Matrix<T> X;
      if (!npyToMatrix<T>((PyArrayObject*)in,X,"Data"));
      if (columns) {
         if (centering) X.center();
         if (normalize) X.normalize();
      } else {
         if (centering) X.center_rows();
         if (normalize) X.normalize_rows();
      }
   }
};

static PyObject* preprocess_(PyObject* self, PyObject* args, PyObject* keywds)
{
   PyObject* Ip=NULL;
   int centering = false;
   int normalize = false;
   int columns = true;
   static char* kwlist[] = { (char*)"", (char*)"centering", (char*)"normalize", (char*)"columns", NULL };
   const char* format = (const char*)"O|ppp";
   if (!PyArg_ParseTupleAndKeywords(args, keywds,format,kwlist,&Ip,&centering,&normalize,&columns))
      return NULL;

   int T, I;
   getTypeObject(Ip,T,I);
   if (T == getTypeNumber<float>() && I == getTypeNumber<int>()) {
      preprocess_generic<float,int>(Ip, centering,normalize,columns);
      Py_RETURN_NONE;
   } else if ((T == getTypeNumber<float>() && I == getTypeNumber<long long int>())) {
      preprocess_generic<float,long long int>(Ip, centering,normalize,columns);
      Py_RETURN_NONE;
   } else if ((T == getTypeNumber<double>() && I == getTypeNumber<int>())) {
      preprocess_generic<double,int>(Ip, centering,normalize,columns);
      Py_RETURN_NONE;
   } else if ((T == getTypeNumber<double>() && I == getTypeNumber<long long int>())) {
      preprocess_generic<double,long long int>(Ip, centering,normalize,columns);
      Py_RETURN_NONE;
   } else {
      PyErr_SetString(PyExc_TypeError, ("Got wrong data type: ")); 
      //PyErr_SetString(PyExc_TypeError, ("Got wrong data type: "+std::to_string(T)).c_str()); 
      return NULL;
   }
};

static PyMethodDef methods[] = {
   {"preprocess_",  (PyCFunction)preprocess_, METH_VARARGS | METH_KEYWORDS, "Preprocessing function."},
   {"erm_",  (PyCFunction)erm_, METH_VARARGS | METH_KEYWORDS, "Univariate regression or classification."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef module = {
   PyModuleDef_HEAD_INIT,
   "cyanure_wrap",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   methods
};

PyMODINIT_FUNC PyInit_cyanure_wrap(void) {
   PyObject* m = PyModule_Create(&module);
   if (m == NULL)
      return NULL;
   import_array();
   return m;
}

