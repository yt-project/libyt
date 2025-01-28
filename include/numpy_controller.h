#ifndef LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_
#define LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_

#include <Python.h>
#include <numpy/arrayobject.h>

#include "yt_type.h"

enum class NumPyStatus : int { kNumPyFailed = 0, kNumPySuccess = 1 };

class NumPyController {
public:
    static NumPyStatus InitializeNumPy();
    static PyObject* ArrayToNumPyArray(int dim, npy_intp* npy_dim, yt_dtype data_dtype, void* data_ptr,
                                       bool readonly = false, bool owned_by_python = false);
};

#endif  // LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_
