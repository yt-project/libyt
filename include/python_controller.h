#ifndef LIBYT_PROJECT_INCLUDE_PYTHON_CONTROLLER_H_
#define LIBYT_PROJECT_INCLUDE_PYTHON_CONTROLLER_H_

int CreateLibytModule();
int InitPython(int argc, char* argv[]);
int PreparePythonEnvForLibyt();
#ifndef USE_PYBIND11
template<typename T>
int AddScalarToDict(PyObject* dict, const char* key, T value);
template<typename T>
int AddVectorNToDict(PyObject* dict, const char* key, int len, const T* vector);
int AddStringToDict(PyObject* dict, const char* key, const char* string);
#endif

#endif  // LIBYT_PROJECT_INCLUDE_PYTHON_CONTROLLER_H_
