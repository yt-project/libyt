#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

#ifndef USE_PYBIND11
template<typename T>
static int add_nonstring(const char* key, const int n, const T* input);
static int add_string(const char* key, const char* input);
#else
#define ADD_NONSTRING_TO_PARAM_USER()                                                                                  \
    {                                                                                                                  \
        pybind11::module_ libyt = pybind11::module_::import("libyt");                                                  \
        pybind11::dict py_param_user = libyt.attr("param_user");                                                       \
        if (n == 1) {                                                                                                  \
            py_param_user[key] = *input;                                                                               \
        } else {                                                                                                       \
            pybind11::tuple tuple(n);                                                                                  \
            for (int i = 0; i < n; i++) {                                                                              \
                tuple[i] = input[i];                                                                                   \
            }                                                                                                          \
            py_param_user[key] = tuple;                                                                                \
        }                                                                                                              \
        logging::LogDebug("Inserting code-specific parameter \"%-*s\" ... done\n", MaxParamNameWidth, key);            \
        return YT_SUCCESS;                                                                                             \
    }
#endif

// maximum string width of a key (for outputting debug information only)
static const int MaxParamNameWidth = 15;

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_set_UserParameter*
// Description :  Add code-specific parameters
//
// Note        :  1. All code-specific parameters are stored in "libyt.param_user"
//                2. Overloaded with various data types: float, double, int, long, long long, unsigned int,
//                   unsigned long, char*
//                   ==> But do not use c++ template since I don't know how to instantiate template
//                       without function name mangling ...
//
// Parameter   :  key   : Dictionary key
//                n     : Number of elements in the input array
//                input : Input array containing "n" elements or a single string
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------

int yt_set_UserParameterInt(const char* key, const int n, const int* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_nonstring(key, n, input);
#else
    ADD_NONSTRING_TO_PARAM_USER();
#endif
}

int yt_set_UserParameterLong(const char* key, const int n, const long* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_nonstring(key, n, input);
#else
    ADD_NONSTRING_TO_PARAM_USER();
#endif
}

int yt_set_UserParameterLongLong(const char* key, const int n, const long long* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_nonstring(key, n, input);
#else
    ADD_NONSTRING_TO_PARAM_USER();
#endif
}

int yt_set_UserParameterUint(const char* key, const int n, const unsigned int* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_nonstring(key, n, input);
#else
    ADD_NONSTRING_TO_PARAM_USER();
#endif
}

int yt_set_UserParameterUlong(const char* key, const int n, const unsigned long* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_nonstring(key, n, input);
#else
    ADD_NONSTRING_TO_PARAM_USER();
#endif
}

int yt_set_UserParameterFloat(const char* key, const int n, const float* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_nonstring(key, n, input);
#else
    ADD_NONSTRING_TO_PARAM_USER();
#endif
}

int yt_set_UserParameterDouble(const char* key, const int n, const double* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_nonstring(key, n, input);
#else
    ADD_NONSTRING_TO_PARAM_USER();
#endif
}

int yt_set_UserParameterString(const char* key, const char* input) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_)
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);

#ifndef USE_PYBIND11
    return add_string(key, input);
#else
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_param_user = libyt.attr("param_user");
    py_param_user[key] = input;
    return YT_SUCCESS;
#endif
}

#ifndef USE_PYBIND11
//***********************************************
// template for various input types except string
//***********************************************
template<typename T>
static int add_nonstring(const char* key, const int n, const T* input) {
    // export data to libyt.param_user
    if (typeid(T) == typeid(float) || typeid(T) == typeid(double) || typeid(T) == typeid(int) ||
        typeid(T) == typeid(long) || typeid(T) == typeid(unsigned int) || typeid(T) == typeid(unsigned long) ||
        typeid(T) == typeid(long long)) {
        //    scalar and 3-element array
        if (n == 1) {
            if (add_dict_scalar(LibytProcessControl::Get().py_param_user_, key, *input) == YT_FAIL) return YT_FAIL;
        } else {
            if (add_dict_vector_n(LibytProcessControl::Get().py_param_user_, key, n, input) == YT_FAIL) return YT_FAIL;
        }
    } else {
        YT_ABORT("Unsupported data type (only support char*, float*, double*, int*, long*, long long*, unsigned int*, "
                 "unsigned long*)!\n");
    }

    logging::LogDebug("Inserting code-specific parameter \"%-*s\" ... done\n", MaxParamNameWidth, key);

    return YT_SUCCESS;

}  // FUNCTION : add_nonstring

//***********************************************
// treat string input separately ...
//***********************************************
static int add_string(const char* key, const char* input) {
    // export data to libyt.param_user
    if (add_dict_string(LibytProcessControl::Get().py_param_user_, key, input) == YT_FAIL) return YT_FAIL;

    logging::LogDebug("Inserting code-specific parameter \"%-*s\" ... done\n", MaxParamNameWidth, key);

    return YT_SUCCESS;

}  // FUNCTION : add_string
#else
#undef ADD_NONSTRING_TO_PARAM_USER
#endif  // #ifndef USE_PYBIND11
