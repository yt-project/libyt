#ifndef LIBYT_PROJECT_INCLUDE_PYTHON_CONTROLLER_H_
#define LIBYT_PROJECT_INCLUDE_PYTHON_CONTROLLER_H_

int CreateLibytModule();
int InitPython(int argc, char* argv[]);
int PreparePythonEnvForLibyt();
#ifndef USE_PYBIND11
template<typename T>
int add_dict_scalar(PyObject* dict, const char* key, const T value);
template<typename T>
int add_dict_vector_n(PyObject* dict, const char* key, const int len, const T* vector);
int add_dict_string(PyObject* dict, const char* key, const char* string);
#endif

#endif  // LIBYT_PROJECT_INCLUDE_PYTHON_CONTROLLER_H_
