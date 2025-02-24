#ifndef LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_
#define LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_

#ifdef LIBYT_INIT_NUMPY
#define PY_ARRAY_UNIQUE_SYMBOL LIBYT_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL LIBYT_UFUNC_API
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "yt_type.h"

enum class NumPyStatus : int { kNumPyFailed = 0, kNumPySuccess = 1 };

struct NumPyArray {
  yt_dtype data_dtype = YT_DTYPE_UNKNOWN;
  int ndim = 0;
  npy_intp* data_dims = nullptr;
  void* data_ptr = nullptr;
};

namespace numpy_controller {
NumPyStatus InitializeNumPy();
PyObject* ArrayToNumPyArray(int dim, npy_intp* npy_dim, yt_dtype data_dtype,
                            void* data_ptr, bool readonly = false,
                            bool owned_by_python = false);
NumPyArray GetNumPyArrayInfo(PyObject* py_array);
}  // namespace numpy_controller

#endif  // LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_
