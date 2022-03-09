#include <string>

#include "lib/data_structure/linalg.h"
#include "lib/convert_cxx_to_python.h"
#include "lib/solvers/solver.h"
#include "lib/error_management/exception.h"
#include "lib/erm/erm.h"
#include "lib/erm/simple_erm.h"
#include "lib/erm/multi_erm.h"

template <typename M, typename sparse_type>
static PyArrayObject* erm(PyObject* inX, PyArrayObject* inY, PyArrayObject* inw0, PyArrayObject* inw, PyArrayObject* in_dual, const int max_iter, const int limited_memory_qning, const int fista_restart, const M tol, const int duality_gap_interval, const bool verbose, char* solver, char* loss, char* regul, const M lambda_1, const M lambda_2, const M lambda_3, const bool intercept, const bool univariate, const int n_threads)
{
    ParamSolver<M> param;
    param.max_iter = max_iter;
    param.l_memory = limited_memory_qning;
    param.freq_restart = fista_restart;
    param.tol = tol;
    param.duality_gap_interval = duality_gap_interval;
    param.verbose = verbose;
    param.solver = solver_from_string(solver);
    // TODO: check if needs to be activated again --> demander à Julien comment vérifier
    param.non_uniform_sampling = false;
    param.threads = n_threads;
    ParamModel<M> model;
    model.loss = loss_from_string(loss);
    model.regul = regul_from_string(regul);
    model.lambda_1 = lambda_1;
    model.lambda_2 = lambda_2;
    model.lambda_3 = lambda_3;
    model.intercept = intercept;
    clean_param_model(model);
    OptimInfo<M> optim_info;
    try
    {
        if (univariate)
        {
            Vector<M> y, w0, w, dual_variable;
            if (!npyToVector<M>(inY, y, "Data y"))
                ;
            if (!npyToVector<M>(inw0, w0, "x0"))
                ;
            if (!npyToVector<M>(inw, w, "x"))
                ;
            if (reinterpret_cast<PyObject*>(in_dual) != Py_None && !npyToVector<M>(in_dual, dual_variable, "dual"))
                ;
            if (w0.n() != w.n())
            {
                PyErr_SetString(PyExc_TypeError, "Got wrong input size");
                return NULL;
            }
            if (isSparseMatrix(inX))
            {
                SpMatrix<M, sparse_type> X;
                if (!npyToSpMatrix<M, sparse_type>(inX, X, "Data"))
                    ;
                param.minibatch = MIN((int)floor((M(X.n()) * M(X.m())) / M(X.nzmax())), X.n() / 100); // aggressive strategy, but only uses minibatch if required
                SIMPLE_ERM<SpMatrix<M, sparse_type>, LinearLossVec<SpMatrix<M, sparse_type>>> problem_configuration(w0, w, dual_variable, optim_info, param, model);
                problem_configuration.solve_problem(X, y);
            }
            else
            {
                Matrix<M> X;
                param.minibatch = 1;
                if (!npyToMatrix<M>((PyArrayObject*)inX, X, "Data X"))
                    ;
                SIMPLE_ERM<Matrix<M>, LinearLossVec<Matrix<M>>> problem_configuration(w0, w, dual_variable, optim_info, param, model);
                problem_configuration.solve_problem(X, y);

            }

        }
        else
        {
            Matrix<M> w0, w, dual_variable;
            if (!npyToMatrix<M>(inw0, w0, "x0"))
                ;
            if (!npyToMatrix<M>(inw, w, "x"))
                ;
            if (reinterpret_cast<PyObject*>(in_dual) != Py_None && !npyToMatrix<M>(in_dual, dual_variable, "dual"))
                ;
            if (isSparseMatrix(inX))
            {
                SpMatrix<M, sparse_type> X;
                if (!npyToSpMatrix<M, sparse_type>(inX, X, "Data"))
                    ;
                param.minibatch = MIN((int)floor((M(X.n()) * M(X.m())) / M(X.nzmax())), X.n() / 100);
                if (array_type(inY) == getTypeNumber<int>())
                {
                    Vector<int> y;
                    if (!npyToVector<int>(inY, y, "Data y"))
                        ;
                    MULTI_ERM<SpMatrix<M, sparse_type>, LinearLossMat<SpMatrix<M, sparse_type>, Vector<int>>> problem_configuration(w0, w, dual_variable, optim_info, param, model);
                    problem_configuration.solve_problem_vector(X, y);
                }
                else
                {
                    Matrix<M> y;
                    if (!npyToMatrix<M>(inY, y, "Data y"))
                        ;
                    MULTI_ERM<SpMatrix<M, sparse_type>, LinearLossMat<SpMatrix<M, sparse_type>, Matrix<M>>> problem_configuration(w0, w, dual_variable, optim_info, param, model);
                    problem_configuration.solve_problem_matrix(X, y);
                }
            }
            else
            {
                Matrix<M> X;
                param.minibatch = 1;
                if (!npyToMatrix<M>((PyArrayObject*)inX, X, "Data X"))
                    ;
                if (array_type(inY) == getTypeNumber<int>())
                {
                    Vector<int> y;
                    if (!npyToVector<int>(inY, y, "Data y"))
                        ;

                    MULTI_ERM<Matrix<M>, LinearLossMat<Matrix<M>, Vector<int>>> problem_configuration(w0, w, dual_variable, optim_info, param, model);
                    problem_configuration.solve_problem_vector(X, y);
                }
                else
                {
                    Matrix<M> y;
                    if (!npyToMatrix<M>(inY, y, "Data y"))
                        ;
                    MULTI_ERM<Matrix<M>, LinearLossMat<Matrix<M>, Matrix<M>>> problem_configuration(w0, w, dual_variable, optim_info, param, model);
                    problem_configuration.solve_problem_matrix(X, y);
                }
            }
        }
        PyArrayObject* out = create_np_optim_info<M>(optim_info.nclass(), optim_info.m(), optim_info.n());
        OptimInfo<M> outm;
        npyToOptimInfo(out, outm, "optim info");
        outm.copy(optim_info);
        return out;
    }
    catch (NotImplementedException e)
    {
        PyObject* tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyBytes_FromString(e.what()));
        PyTuple_SetItem(tuple, 1, PyBytes_FromString(solver));
        PyErr_SetObject(PyExc_NotImplementedError, tuple);
        return NULL;
    }
    catch (ValueError e)
    {
        PyErr_SetObject(PyExc_ValueError, PyBytes_FromString(e.what()));
        return NULL;
    }
    catch (...)
    {
        PyErr_SetObject(PyExc_SystemError, PyBytes_FromString("Unexpected exception in C++"));
        return NULL;
    }
};

static PyObject* erm_(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* X = NULL;
    PyArrayObject* y = NULL;
    PyArrayObject* w0 = NULL;
    PyArrayObject* w = NULL;
    PyArrayObject* dual = NULL;
    PyArrayObject* optim_info = NULL;
    int max_iter = 1000;
    int limited_memory_qning = 20;
    int fista_restart = 50;
    double tol = 1e-3;
    int duality_gap_interval = 10;
    int verbose = 1;
    int univariate = 1;
    int n_threads = 1;
    int seed = 0;
    double lambda_1 = 0;
    double lambda_2 = 0;
    double lambda_3 = 0;
    int intercept = 0;
    char* regul = (char*)"none";
    char* solver = (char*)"ista";
    char* loss = (char*)"square";
    static char* kwlist[] = { (char*)"", (char*)"", (char*)"", (char*)"", (char*)"dual_variable", (char*)"loss", (char*)"penalty", (char*)"solver", (char*)"lambda_1", (char*)"lambda_2", (char*)"lambda_3", (char*)"intercept", (char*)"tol", (char*)"duality_gap_interval", (char*)"max_iter", (char*)"limited_memory_qning", (char*)"fista_restart", (char*)"verbose", (char*)"univariate", (char*)"n_threads", (char*)"seed", NULL };
    const char* format = (const char*)"OOOO|Osssdddpdiiiippii";
    if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist, &X, &y, &w0, &w, &dual, &loss, &regul, &solver, &lambda_1, &lambda_2, &lambda_3, &intercept, &tol, &duality_gap_interval, &max_iter, &limited_memory_qning, &fista_restart, &verbose, &univariate, &n_threads, &seed))
        return NULL;
    duality_gap_interval = duality_gap_interval <= 0 ? -1 : MIN(duality_gap_interval, max_iter);
    srandom(seed);
    int M, sparse_type;
    getTypeObject((PyObject*)X, M, sparse_type);
    if (M == getTypeNumber<float>() && sparse_type == getTypeNumber<int>())
    {
        optim_info = erm<float, int>(X, y, w0, w, dual, max_iter, limited_memory_qning, fista_restart, tol, duality_gap_interval, verbose, solver, loss, regul, lambda_1, lambda_2, lambda_3, intercept, univariate, n_threads);
    }
    else if (M == getTypeNumber<float>() && sparse_type == getTypeNumber<long long int>())
    {
        optim_info = erm<float, long long int>(X, y, w0, w, dual, max_iter, limited_memory_qning, fista_restart, tol, duality_gap_interval, verbose, solver, loss, regul, lambda_1, lambda_2, lambda_3, intercept, univariate, n_threads);
    }
    else if (M == getTypeNumber<double>() && sparse_type == getTypeNumber<int>())
    {
        optim_info = erm<double, int>(X, y, w0, w, dual, max_iter, limited_memory_qning, fista_restart, tol, duality_gap_interval, verbose, solver, loss, regul, lambda_1, lambda_2, lambda_3, intercept, univariate, n_threads);
    }
    else if (M == getTypeNumber<double>() && sparse_type == getTypeNumber<long long int>())
    {
        optim_info = erm<double, long long int>(X, y, w0, w, dual, max_iter, limited_memory_qning, fista_restart, tol, duality_gap_interval, verbose, solver, loss, regul, lambda_1, lambda_2, lambda_3, intercept, univariate, n_threads);
    }
    else
    {
        // PyErr_SetString(PyExc_TypeError, ("Got wrong data type: " + std::to_string(M)).c_str()));
        PyErr_SetString(PyExc_TypeError, ("Got wrong data type: " + std::to_string(M)).c_str());
        return NULL;
    }
    return PyArray_Return(optim_info);
};

template <typename M, typename sparse_type>
static void preprocess_generic(PyObject* in, const bool centering, const bool normalize, const bool columns = true)
{
    if (isSparseMatrix(in))
    {
        SpMatrix<M, sparse_type> X;
        if (!npyToSpMatrix<M, sparse_type>(in, X, "Data"))
            ;
        if (columns)
        {
            if (normalize)
                X.normalize();
        }
        else
        {
            if (normalize)
                X.normalize_rows();
        }
    }
    else
    {
        Matrix<M> X;
        if (!npyToMatrix<M>((PyArrayObject*)in, X, "Data"))
            ;
        if (columns)
        {
            if (centering)
                X.center();
            if (normalize)
                X.normalize();
        }
        else
        {
            if (centering)
                X.center_rows();
            if (normalize)
                X.normalize_rows();
        }
    }
};

static PyObject* preprocess_(PyObject* self, PyObject* args, PyObject* keywds)
{
    PyObject* Ip = NULL;
    int centering = false;
    int normalize = false;
    int columns = true;
    static char* kwlist[] = { (char*)"", (char*)"centering", (char*)"normalize", (char*)"columns", NULL };
    const char* format = (const char*)"O|ppp";
    if (!PyArg_ParseTupleAndKeywords(args, keywds, format, kwlist, &Ip, &centering, &normalize, &columns))
        return NULL;

    int M, sparse_type;
    getTypeObject(Ip, M, sparse_type);
    if (M == getTypeNumber<float>() && sparse_type == getTypeNumber<int>())
    {
        preprocess_generic<float, int>(Ip, centering, normalize, columns);
        Py_RETURN_NONE;
    }
    else if ((M == getTypeNumber<float>() && sparse_type == getTypeNumber<long long int>()))
    {
        preprocess_generic<float, long long int>(Ip, centering, normalize, columns);
        Py_RETURN_NONE;
    }
    else if ((M == getTypeNumber<double>() && sparse_type == getTypeNumber<int>()))
    {
        preprocess_generic<double, int>(Ip, centering, normalize, columns);
        Py_RETURN_NONE;
    }
    else if ((M == getTypeNumber<double>() && sparse_type == getTypeNumber<long long int>()))
    {
        preprocess_generic<double, long long int>(Ip, centering, normalize, columns);
        Py_RETURN_NONE;
    }
    else
    {
        // PyErr_SetString(PyExc_TypeError, ("Got wrong data type: " + std::to_string(M)).c_str()));
        PyErr_SetString(PyExc_TypeError, ("Got wrong data type: " + std::to_string(M)).c_str());
        return NULL;
    }
};

static PyMethodDef methods[] = {
    {"preprocess_", (PyCFunction)preprocess_, METH_VARARGS | METH_KEYWORDS, "Preprocessing function."},
    {"erm_", (PyCFunction)erm_, METH_VARARGS | METH_KEYWORDS, "Univariate regression or classification."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "cyanure_wrap", /* name of module */
    NULL,           /* module documentation, may be NULL */
    -1,             /* size of per-interpreter state of the module,
                       or -1 if the module keeps state in global variables. */
    methods };

PyMODINIT_FUNC PyInit_cyanure_wrap(void)
{
    PyObject* m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;
    import_array();
    return m;
}
