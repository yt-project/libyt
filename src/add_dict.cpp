#ifndef USE_PYBIND11
#include <typeinfo>

#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  AddScalarToDict
// Description :  Auxiliary function for adding a scalar item to a Python dictionary
//
// Note        :  1. Overloaded with various data types: float, double, int, long, unsigned int, unsigned long
//                   ==> (float,double)                        are converted to double internally
//                       (int,long,unsigned int,unsigned long) are converted to long internally
//                       (long long)                           are converted to long long internally
//
// Parameter   :  dict  : Target Python dictionary
//                key   : Dictionary key
//                value : Value to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template<typename T>
int AddScalarToDict(PyObject* dict, const char* key, const T value) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!PyDict_Check(dict)) {
        YT_ABORT("This is not a dict object (key = \"%s\", value = \"%.5g\")!\n", key, (double)value);
    }

    PyObject* py_obj;

    if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {
        py_obj = PyFloat_FromDouble((double)value);
    } else if (typeid(T) == typeid(int) || typeid(T) == typeid(long) || typeid(T) == typeid(unsigned int) ||
               typeid(T) == typeid(unsigned long)) {
        py_obj = PyLong_FromLong((long)value);
    } else if (typeid(T) == typeid(long long)) {
        py_obj = PyLong_FromLongLong((long long)value);
    } else {
        YT_ABORT(
            "Unsupported data type (only support float, double, int, long, long long, unsigned int, unsigned long)!\n");
    }

    if (PyDict_SetItemString(dict, key, py_obj) != 0) {
        Py_DECREF(py_obj);
        YT_ABORT("Inserting a dictionary item with value \"%.5g\" and key \"%s\" ... failed!\n", (double)value, key);
    }

    Py_DECREF(py_obj);

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  AddVectorNToDict
// Description :  Auxiliary function for adding an n-element vector item to a Python dictionary
//
// Note        :  1. Overloaded with various data types: float, double, int, long, unsigned int, unsigned long
//                   ==> (float,double)                        are converted to double internally
//                       (int,long,unsigned int,unsigned long) are converted to long internally
//                       (long long)                           are converted to long long internally
//
// Parameter   :  dict   : Target Python dictionary
//                key    : Dictionary key
//                len    : Length of the vector size
//                vector : Vector to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template<typename T>
int AddVectorNToDict(PyObject* dict, const char* key, const int len, const T* vector) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!PyDict_Check(dict)) {
        YT_ABORT("This is not a dict object (key = \"%s\")!\n", key);
    }

    Py_ssize_t vec_size = len;
    PyObject* tuple = PyTuple_New(vec_size);

    if (tuple != NULL) {
        if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {
            for (Py_ssize_t v = 0; v < vec_size; v++) {
                PyTuple_SET_ITEM(tuple, v, PyFloat_FromDouble((double)vector[v]));
            }
        } else if (typeid(T) == typeid(int) || typeid(T) == typeid(long) || typeid(T) == typeid(unsigned int) ||
                   typeid(T) == typeid(unsigned long)) {
            for (Py_ssize_t v = 0; v < vec_size; v++) {
                PyTuple_SET_ITEM(tuple, v, PyLong_FromLong((long)vector[v]));
            }
        } else if (typeid(T) == typeid(long long)) {
            for (Py_ssize_t v = 0; v < vec_size; v++) {
                PyTuple_SET_ITEM(tuple, v, PyLong_FromLongLong((long long)vector[v]));
            }
        } else {
            Py_DECREF(tuple);
            YT_ABORT("Unsupported data type (only support float, double, int, long, long long, unsigned int, unsigned "
                     "long)!\n");
        }
    } else {
        YT_ABORT("Creating a tuple object (key = \"%s\") ... failed!\n", key);
    }

    if (PyDict_SetItemString(dict, key, tuple) != 0) {
        Py_DECREF(tuple);
        YT_ABORT("Inserting a dictionary item with the key \"%s\" ... failed!\n", key);
    }

    Py_DECREF(tuple);

    return YT_SUCCESS;

}  // FUNCTION : add_dict_vector_n

//-------------------------------------------------------------------------------------------------------
// Function    :  AddStringToDict
// Description :  Auxiliary function for adding a string item to a Python dictionary
//
// Note        :  1. There is no function overloading here
//
// Parameter   :  dict   : Target Python dictionary
//                key    : Dictionary key
//                string : String to be inserted
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int AddStringToDict(PyObject* dict, const char* key, const char* string) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!PyDict_Check(dict)) {
        YT_ABORT("This is not a dict object (key = \"%s\", string = \"%s\")!\n", key, string);
    }

    PyObject* py_obj = PyUnicode_FromString(string);
    if (PyDict_SetItemString(dict, key, py_obj) != 0) {
        Py_DECREF(py_obj);
        YT_ABORT("Inserting a dictionary item with string \"%s\" and key \"%s\" ... failed!\n", string, key);
    }
    Py_DECREF(py_obj);

    return YT_SUCCESS;
}

// explicit template instantiation
template int AddScalarToDict<float>(PyObject* dict, const char* key, const float value);
template int AddScalarToDict<double>(PyObject* dict, const char* key, const double value);
template int AddScalarToDict<int>(PyObject* dict, const char* key, const int value);
template int AddScalarToDict<long>(PyObject* dict, const char* key, const long value);
template int AddScalarToDict<long long>(PyObject* dict, const char* key, const long long value);
template int AddScalarToDict<unsigned int>(PyObject* dict, const char* key, const unsigned int value);
template int AddScalarToDict<unsigned long>(PyObject* dict, const char* key, const unsigned long value);
template int AddVectorNToDict<float>(PyObject* dict, const char* key, const int len, const float* vector);
template int AddVectorNToDict<double>(PyObject* dict, const char* key, const int len, const double* vector);
template int AddVectorNToDict<int>(PyObject* dict, const char* key, const int len, const int* vector);
template int AddVectorNToDict<long>(PyObject* dict, const char* key, const int len, const long* vector);
template int AddVectorNToDict<long long>(PyObject* dict, const char* key, const int len, const long long* vector);
template int AddVectorNToDict<unsigned int>(PyObject* dict, const char* key, const int len, const unsigned int* vector);
template int AddVectorNToDict<unsigned long>(PyObject* dict, const char* key, const int len,
                                             const unsigned long* vector);
#endif  // #ifndef USE_PYBIND11
