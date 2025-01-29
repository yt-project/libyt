#ifndef LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_
#define LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_

#include <Python.h>
#include <numpy/arrayobject.h>

#include "yt_type.h"

enum class NumPyStatus : int { kNumPyFailed = 0, kNumPySuccess = 1 };

template<int N>
struct NumPyArray {
    int dim = N;
    npy_intp npy_dim[N];
    yt_dtype npy_type;
    void* data_ptr = nullptr;
};

namespace numpy_controller {
NumPyStatus InitializeNumPy();
PyObject* ArrayToNumPyArray(int dim, npy_intp* npy_dim, yt_dtype data_dtype, void* data_ptr, bool readonly = false,
                            bool owned_by_python = false);
template<int N>
void GetNumPyArrayInfo(PyObject* py_array, NumPyArray<N>* npy_array_info);
}  // namespace numpy_controller

#endif  // LIBYT_PROJECT_INCLUDE_NUMPY_CONTROLLER_H_
