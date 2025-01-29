#include "numpy_controller.h"

#include "yt_prototype.h"

//-------------------------------------------------------------------------------------------------------
// Namespace     : numpy_controller
//
// Notes         :  1. Initialize NumPy API.
//                  2. Currently, this method is only for unit test.
//                     This is because NumPy API initialization only imports the api within a translation unit.
//                     Since we compile it to a libyt library and make unit test link to it,
//                     it is in separate translation unit, we need to have a public API to initialize NumPy API
//                     within libyt library itself.
//-------------------------------------------------------------------------------------------------------
NumPyStatus numpy_controller::InitializeNumPy() {
    if (PyArray_API == nullptr) {
        import_array1(NumPyStatus::kNumPyFailed);
    }
    return NumPyStatus::kNumPySuccess;
}

//-------------------------------------------------------------------------------------------------------
// Namespace     : numpy_controller
//
// Notes         :  1. Create a NumPy array from existing pointer.
//                  2. Depending on the usage, we should Py_DECREF the returned object, once it is attached
//                     to something, say a dictionary.
//-------------------------------------------------------------------------------------------------------
PyObject* numpy_controller::ArrayToNumPyArray(int dim, npy_intp* npy_dim, yt_dtype data_dtype, void* data_ptr,
                                              bool readonly, bool owned_by_python) {
    int npy_dtype;
    get_npy_dtype(data_dtype, &npy_dtype);
    PyObject* py_data = PyArray_SimpleNewFromData(dim, npy_dim, npy_dtype, data_ptr);

    if (readonly) {
        PyArray_CLEARFLAGS((PyArrayObject*)py_data, NPY_ARRAY_WRITEABLE);
    }

    if (owned_by_python) {
        PyArray_ENABLEFLAGS((PyArrayObject*)py_data, NPY_ARRAY_OWNDATA);
    }

    return py_data;
}
