#ifndef WRAPPER_UTILS_H
#define WRAPPER_UTILS_H

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "linalg.h"

#define array_type(a)          (int)(PyArray_TYPE(a))
#define PyArray_SimpleNewF(nd, dims, typenum) PyArray_New(&PyArray_Type, nd, dims, typenum, NULL, NULL, 0, NPY_ARRAY_FARRAY, NULL)
#define check_array(a,npy_type) (!is_array(a) || !require_contiguous(a) || !require_native(a) || array_type(a)!=npy_type)
#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS(a))
#define array_is_native(a)     (PyArray_ISNOTSWAPPED(a))
#define array_numdims(a)       (((PyArrayObject *)a)->nd)
#define is_array(a)            ((a) && PyArray_Check((PyArrayObject *)a))

template <typename T> inline string getTypeName();
template <> inline string getTypeName<int>() { return "intc"; };
template <> inline string getTypeName<unsigned char>() { return "uint8"; };
template <> inline string getTypeName<float>() { return "float32"; };
template <> inline string getTypeName<double>() { return "float64"; };

template <typename T> inline int getTypeNumber();
template <> inline int getTypeNumber<int>() { return NPY_INT; };
template <> inline int getTypeNumber<long long int>() { return NPY_INT64; };
template <> inline int getTypeNumber<unsigned char>() { return NPY_UINT8; };
template <> inline int getTypeNumber<float>() { return NPY_FLOAT32; };
template <> inline int getTypeNumber<double>() { return NPY_FLOAT64; };

/* Test whether a python object is contiguous.  If array is
 * contiguous, return 1.  Otherwise, set the python error string and
 * return 0.
 */
int require_contiguous(PyArrayObject* ary) {
   int contiguous = 1;
   if (!array_is_contiguous(ary)) {
      PyErr_SetString(PyExc_TypeError,
            "Array must be contiguous.  A non-contiguous array was given");
      contiguous = 0;
   }
   return contiguous;
}

/* Require that a numpy array is not byte-swapped.  If the array is
 * not byte-swapped, return 1.  Otherwise, set the python error string
 * and return 0.
 */
int require_native(PyArrayObject* ary) {
   int native = 1;
   if (!array_is_native(ary))  {
      PyErr_SetString(PyExc_TypeError,
            "Array must have native byteorder.  "
            "A byte-swapped array was given");
      native = 0;
   }
   return native;
}


bool isSparseMatrix(PyObject* input) {
   return PyObject_HasAttrString(input, "indptr");
}

void getTypeObject(PyObject* input, int& T, int& I) {
   if (isSparseMatrix(input)) {
      PyArrayObject* data = (PyArrayObject *) PyObject_GetAttrString(input, "data");
      PyArrayObject* indptr= (PyArrayObject *) PyObject_GetAttrString(input, "indptr");
      T=PyArray_TYPE((PyArrayObject*) data);
      I=PyArray_TYPE((PyArrayObject*) indptr);
   } else {
      T=PyArray_TYPE((PyArrayObject*) input);
      I=getTypeNumber<INTM>();
   }
}

template <typename T, typename I>
static int npyToSpMatrix(PyObject* array, SpMatrix<T,I>& matrix, string obj_name) {
   if (array==NULL) {
      return 1;
   }
   PyArrayObject* indptr = (PyArrayObject *) PyObject_GetAttrString(array, "indptr");
   PyArrayObject* indices = (PyArrayObject *) PyObject_GetAttrString(array, "indices");
   PyArrayObject* data = (PyArrayObject *) PyObject_GetAttrString(array, "data");
   PyObject* shape = PyObject_GetAttrString(array, "shape");
   if (check_array(indptr,getTypeNumber<I>())) {
      PyErr_SetString(PyExc_TypeError,"spmatrix arg1: indptr array should be 1d int's");
      return 0;
   }
   if (check_array(indices,getTypeNumber<I>()))  {
      PyErr_SetString(PyExc_TypeError,"spmatrix arg1: indices array should be 1d int's");
      return 0;
   }
   if (check_array(data, getTypeNumber<T>()))  {
      PyErr_SetString(PyExc_TypeError,"spmatrix arg1: data array should be 1d and match datatype");
      return 0;
   }
   if (!PyTuple_Check(shape)) {
      PyErr_SetString(PyExc_TypeError,"shape should be a tuple");
      return 0;
   }
   I m =PyLong_AsLong(PyTuple_GetItem(shape, 0));
   I n =PyLong_AsLong(PyTuple_GetItem(shape, 1));
   I *pB = (I*)PyArray_DATA(indptr);
   I*pE = pB + 1;
   I nzmax = (I)PyArray_SIZE(data);
   Py_DECREF(indptr);
   Py_DECREF(indices);
   Py_DECREF(data);
   Py_DECREF(shape);
   matrix.setData((T *)PyArray_DATA(data),(I*)PyArray_DATA(indices),pB,pE,m,n,nzmax);

   return 1;
}



template <typename T>
static int npyToMatrix(PyArrayObject* array, Matrix<T>& matrix, string obj_name) {
   if (array==NULL) {
      return 1;
   }
   if(!(PyArray_NDIM(array) == 2 &&
            PyArray_TYPE(array) == getTypeNumber<T>() &&
            (PyArray_FLAGS(array) & NPY_ARRAY_F_CONTIGUOUS))) {
      PyErr_SetString(PyExc_TypeError, (obj_name + " matrices should be f-contiguous 2D "+getTypeName<T>()+" array").c_str());
      return 0;
   }
   T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
   const npy_intp *shape = PyArray_DIMS(array);
   npy_intp m = shape[0];
   npy_intp n = shape[1];
   matrix.setData(rawX, m, n);
   return 1;
}

template <typename T>
static int npyToVector(PyArrayObject* array, Vector<T>& vector, string obj_name) {
   if (array==NULL) {
      return 1;
   }
   T *rawX =  reinterpret_cast<T*>(PyArray_DATA(array));
   const npy_intp *shape = PyArray_DIMS(array);
   npy_intp n = shape[0];

   if(!(PyArray_NDIM(array) == 1 &&
            PyArray_TYPE(array) == getTypeNumber<T>() &&
            (PyArray_FLAGS(array) & NPY_ARRAY_ALIGNED))) {
      PyErr_SetString(PyExc_TypeError, (obj_name + " should be aligned 1D "+getTypeName<T>()+" array").c_str());
      return 0;
   }
   vector.setData(rawX, n);
   return 1;
}

template <typename T>
inline PyArrayObject* create_np_vector(const int n) {
   int nd=1;
   npy_intp dims[1]={n};
   return (PyArrayObject*) PyArray_SimpleNew(nd,dims,getTypeNumber<T>());
}

template <typename T>
inline PyArrayObject* create_np_matrix(const int m, const int n) {
   int nd=2;
   npy_intp dims[2]={m,n};
   return (PyArrayObject*) PyArray_SimpleNewF(nd,dims,getTypeNumber<T>());
}

template <typename T>
inline PyArrayObject* create_np_map(const Vector<int>& size) {
   int nd=3;
   int m = size[0];
   int n = size[1];
   int V = size.n() == 2 ? 1 : size[2];
   npy_intp dims[3]={m,n,V};
   return (PyArrayObject*) PyArray_SimpleNew(nd,dims,getTypeNumber<T>()); // maps are not fortran-arrays
}


#endif
