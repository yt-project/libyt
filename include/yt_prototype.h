#ifndef LIBYT_PROJECT_INCLUDE_YT_PROTOTYPE_H_
#define LIBYT_PROJECT_INCLUDE_YT_PROTOTYPE_H_

// include relevant headers
#include <typeinfo>

#include "yt_type.h"

#ifdef USE_PYBIND11
#include "pybind11/numpy.h"
#else
#include <Python.h>
#endif

int check_yt_param_yt(const yt_param_yt& param_yt);
int print_yt_param_yt(const yt_param_yt& param_yt);
int print_yt_field(const yt_field& field);

#ifndef NO_PYTHON
template<typename T>
int add_dict_scalar(PyObject* dict, const char* key, const T value);
template<typename T>
int add_dict_vector_n(PyObject* dict, const char* key, const int len, const T* vector);
int add_dict_string(PyObject* dict, const char* key, const char* string);
#endif

#endif  // LIBYT_PROJECT_INCLUDE_YT_PROTOTYPE_H_
