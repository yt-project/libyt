#include <typeinfo>

#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "python_controller.h"
#include "timer.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

#ifndef USE_PYBIND11
template<typename T>
static int add_nonstring(const char* key, const int n, const T* input);
static int add_string(const char* key, const char* input);
#else
#define ADD_NONSTRING_TO_PARAM_USER()                                                    \
  {                                                                                      \
    pybind11::module_ libyt = pybind11::module_::import("libyt");                        \
    pybind11::dict py_param_user = libyt.attr("param_user");                             \
    if (n == 1) {                                                                        \
      py_param_user[key] = *input;                                                       \
    } else {                                                                             \
      pybind11::tuple tuple(n);                                                          \
      for (int i = 0; i < n; i++) {                                                      \
        tuple[i] = input[i];                                                             \
      }                                                                                  \
      py_param_user[key] = tuple;                                                        \
    }                                                                                    \
    logging::LogDebug("Inserting code-specific parameter \"%-*s\" ... done\n",           \
                      MaxParamNameWidth,                                                 \
                      key);                                                              \
    return YT_SUCCESS;                                                                   \
  }
#endif

// maximum string width of a key (for outputting debug information only)
static const int MaxParamNameWidth = 15;

/**
 * \addtogroup api_yt_set_UserParameter
 * \name api_yt_set_UserParameter
 * Set user or code-specific parameters. All the parameters will be put under
 * @verbatim libyt.param_user["key"] = value @endverbatim.
 */

/**
 * \brief Set user parameter int
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param n[in] length of value array to be set in dictionary
 * @param input[in] value array
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
int yt_set_UserParameterInt(const char* key, const int n, const int* input) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // check if libyt has been initialized
  if (!LibytProcessControl::Get().libyt_initialized_) {
    YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
  }

#ifndef USE_PYBIND11
  return add_nonstring(key, n, input);
#else
  ADD_NONSTRING_TO_PARAM_USER();
#endif
}

/**
 * \brief Set user parameter long
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param n[in] length of value array to be set in dictionary
 * @param input[in] value array
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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

/**
 * \brief Set user parameter long long
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param n[in] length of value array to be set in dictionary
 * @param input[in] value array
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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

/**
 * \brief Set user parameter unsigned int
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param n[in] length of value array to be set in dictionary
 * @param input[in] value array
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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

/**
 * \brief Set user parameter unsigned long
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param n[in] length of value array to be set in dictionary
 * @param input[in] value array
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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

/**
 * \brief Set user parameter float
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param n[in] length of value array to be set in dictionary
 * @param input[in] value array
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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

/**
 * \brief Set user parameter double
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param n[in] length of value array to be set in dictionary
 * @param input[in] value array
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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

/**
 * \brief Set user parameter string
 * \details
 * 1. Key-value pair will be appended to @verbatim libyt.param_user @endverbatim.
 * 2. Data is copied.
 *
 * @param key[in] key
 * @param input[in] value
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
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
  if (typeid(T) == typeid(float) || typeid(T) == typeid(double) ||
      typeid(T) == typeid(int) || typeid(T) == typeid(long) ||
      typeid(T) == typeid(unsigned int) || typeid(T) == typeid(unsigned long) ||
      typeid(T) == typeid(long long)) {
    //    scalar and 3-element array
    if (n == 1) {
      if (python_controller::AddScalarToDict(
              LibytProcessControl::Get().py_param_user_, key, *input) == YT_FAIL)
        return YT_FAIL;
    } else {
      if (python_controller::AddVectorNToDict(
              LibytProcessControl::Get().py_param_user_, key, n, input) == YT_FAIL)
        return YT_FAIL;
    }
  } else {
    YT_ABORT("Unsupported data type (only support char*, float*, double*, int*, long*, "
             "long long*, unsigned int*, "
             "unsigned long*)!\n");
  }

  logging::LogDebug(
      "Inserting code-specific parameter \"%-*s\" ... done\n", MaxParamNameWidth, key);

  return YT_SUCCESS;

}  // FUNCTION : add_nonstring

//***********************************************
// treat string input separately ...
//***********************************************
static int add_string(const char* key, const char* input) {
  // export data to libyt.param_user
  if (python_controller::AddStringToDict(
          LibytProcessControl::Get().py_param_user_, key, input) == YT_FAIL)
    return YT_FAIL;

  logging::LogDebug(
      "Inserting code-specific parameter \"%-*s\" ... done\n", MaxParamNameWidth, key);

  return YT_SUCCESS;

}  // FUNCTION : add_string
#else
#undef ADD_NONSTRING_TO_PARAM_USER
#endif  // #ifndef USE_PYBIND11
