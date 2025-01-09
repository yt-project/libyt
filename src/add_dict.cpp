#include <typeinfo>

#include "libyt_process_control.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

#ifndef USE_PYBIND11
//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_scalar
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
int add_dict_scalar(PyObject* dict, const char* key, const T value) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if "dict" is indeeed a dict object
    if (!PyDict_Check(dict))
        YT_ABORT("This is not a dict object (key = \"%s\", value = \"%.5g\")!\n", key, (double)value);

    // convert "value" to a Python object
    PyObject* py_obj;

    if (typeid(T) == typeid(float) || typeid(T) == typeid(double))
        py_obj = PyFloat_FromDouble((double)value);

    else if (typeid(T) == typeid(int) || typeid(T) == typeid(long) || typeid(T) == typeid(unsigned int) ||
             typeid(T) == typeid(unsigned long))
        py_obj = PyLong_FromLong((long)value);

    else if (typeid(T) == typeid(long long))
        py_obj = PyLong_FromLongLong((long long)value);

    else
        YT_ABORT(
            "Unsupported data type (only support float, double, int, long, long long, unsigned int, unsigned long)!\n");

    // insert "value" into "dict" with "key"
    if (PyDict_SetItemString(dict, key, py_obj) != 0)
        YT_ABORT("Inserting a dictionary item with value \"%.5g\" and key \"%s\" ... failed!\n", (double)value, key);

    // decrease the reference count
    Py_DECREF(py_obj);

    return YT_SUCCESS;

}  // FUNCTION : add_dict_scalar

//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_vector_n
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
int add_dict_vector_n(PyObject* dict, const char* key, const int len, const T* vector) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if "dict" is indeeed a dict object
    if (!PyDict_Check(dict)) YT_ABORT("This is not a dict object (key = \"%s\")!\n", key);

    // convert "vector" to a Python object (currently the size of vector is fixed to 3)
    Py_ssize_t VecSize = len;
    PyObject* tuple = PyTuple_New(VecSize);

    if (tuple != NULL) {
        if (typeid(T) == typeid(float) || typeid(T) == typeid(double)) {
            for (Py_ssize_t v = 0; v < VecSize; v++) {
                PyTuple_SET_ITEM(tuple, v, PyFloat_FromDouble((double)vector[v]));
            }
        } else if (typeid(T) == typeid(int) || typeid(T) == typeid(long) || typeid(T) == typeid(unsigned int) ||
                   typeid(T) == typeid(unsigned long)) {
            for (Py_ssize_t v = 0; v < VecSize; v++) {
                PyTuple_SET_ITEM(tuple, v, PyLong_FromLong((long)vector[v]));
            }
        } else if (typeid(T) == typeid(long long)) {
            for (Py_ssize_t v = 0; v < VecSize; v++) {
                PyTuple_SET_ITEM(tuple, v, PyLong_FromLongLong((long long)vector[v]));
            }
        } else {
            YT_ABORT("Unsupported data type (only support float, double, int, long, long long, unsigned int, unsigned "
                     "long)!\n");
        }
    } else {
        YT_ABORT("Creating a tuple object (key = \"%s\") ... failed!\n", key);
    }

    // insert "vector" into "dict" with "key"
    if (PyDict_SetItemString(dict, key, tuple) != 0)
        YT_ABORT("Inserting a dictionary item with the key \"%s\" ... failed!\n", key);

    // decrease the reference count
    Py_DECREF(tuple);

    return YT_SUCCESS;

}  // FUNCTION : add_dict_vector_n

//-------------------------------------------------------------------------------------------------------
// Function    :  add_dict_string
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
int add_dict_string(PyObject* dict, const char* key, const char* string) {
    SET_TIMER(__PRETTY_FUNCTION__);
    // check if "dict" is indeeed a dict object
    if (!PyDict_Check(dict)) YT_ABORT("This is not a dict object (key = \"%s\", string = \"%s\")!\n", key, string);

    // convert "string" to a Python object
    PyObject* py_obj = PyUnicode_FromString(string);

    // insert "string" into "dict" with "key"
    if (PyDict_SetItemString(dict, key, py_obj) != 0)
        YT_ABORT("Inserting a dictionary item with string \"%s\" and key \"%s\" ... failed!\n", string, key);

    // decrease the reference count
    Py_DECREF(py_obj);

    return YT_SUCCESS;

}  // FUNCTION : add_dict_string

// explicit template instantiation
template int add_dict_scalar<float>(PyObject* dict, const char* key, const float value);
template int add_dict_scalar<double>(PyObject* dict, const char* key, const double value);
template int add_dict_scalar<int>(PyObject* dict, const char* key, const int value);
template int add_dict_scalar<long>(PyObject* dict, const char* key, const long value);
template int add_dict_scalar<long long>(PyObject* dict, const char* key, const long long value);
template int add_dict_scalar<unsigned int>(PyObject* dict, const char* key, const unsigned int value);
template int add_dict_scalar<unsigned long>(PyObject* dict, const char* key, const unsigned long value);

template int add_dict_vector_n<float>(PyObject* dict, const char* key, const int len, const float* vector);
template int add_dict_vector_n<double>(PyObject* dict, const char* key, const int len, const double* vector);
template int add_dict_vector_n<int>(PyObject* dict, const char* key, const int len, const int* vector);
template int add_dict_vector_n<long>(PyObject* dict, const char* key, const int len, const long* vector);
template int add_dict_vector_n<long long>(PyObject* dict, const char* key, const int len, const long long* vector);
template int add_dict_vector_n<unsigned int>(PyObject* dict, const char* key, const int len,
                                             const unsigned int* vector);
template int add_dict_vector_n<unsigned long>(PyObject* dict, const char* key, const int len,
                                              const unsigned long* vector);
#endif  // #ifndef USE_PYBIND11
