#include <cstdarg>
#include <string>

#include "function_info.h"
#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

/**
 * \addtogroup api_yt_run_Function libyt API: yt_run_FunctionArguments / yt_run_Function
 * \name api_yt_run_Function
 * Run a Python function defined in the inline script namespace. The inline script
 * namespace is passed in by \ref yt_initialize and \ref yt_param_libyt.
 */

/**
 * \brief Call Python function with args in in situ process
 * \fn int yt_run_FunctionArguments(const char* function_name, int argc, ...)
 * \details
 * 1. The function is run inside the inline script namespace, which is the script passed
 *    in \ref yt_initialize and \ref yt_param_libyt. Which means the script contains the
 *    function name you called.
 * 2. Must give argc (argument count), even if there are no arguments.
 * 3. libyt wraps function and its arguments using either \c """ or \c ''' triple quotes,
 *    so we must avoid using both of these triple quotes in function arguments.
 * 4. Under INTERACTIVE_MODE, function will be wrapped inside try/except.
 *    If an error occurred, it will be stored under
 *    \c libyt.interactive_mode.
 *
 * @param function_name[in] Python function name
 * @param argc[in] Number of arguments
 * @param ...[in] List of arguments, should be input as strings
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \rst
 * .. code-block:: c
 *
 *    // Equivalent to Python: function_name(1, 'string', var_name)
 *    yt_run_FunctionArguments("function_name", 3, "1", "'string'", "var_name");
 * \endrst
 */
int yt_run_FunctionArguments(const char* function_name, int argc, ...) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // check if libyt has been initialized
  if (!LibytProcessControl::Get().libyt_initialized_) {
    YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
  }

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  // always run m_Run = -1 function, and set 1.
  // always run unknown function and let Python generates function-not-defined error.
  int func_index =
      LibytProcessControl::Get().function_info_list_.GetFunctionIndex(function_name);
  if (func_index != -1) {
    if (LibytProcessControl::Get().function_info_list_[func_index].GetRun() ==
        FunctionInfo::RunStatus::kWillIdle) {
      logging::LogInfo("YT inline function \"%s\" was set to idle ... idle\n",
                       function_name);
      return YT_SUCCESS;
    } else if (LibytProcessControl::Get().function_info_list_[func_index].GetRun() ==
               FunctionInfo::RunStatus::kNotSetYet)
      LibytProcessControl::Get().function_info_list_[func_index].SetRun(
          FunctionInfo::RunStatus::kWillRun);
  } else {
    func_index = LibytProcessControl::Get().function_info_list_.AddNewFunction(
        function_name, FunctionInfo::RunStatus::kWillRun);
  }
  LibytProcessControl::Get().function_info_list_[func_index].SetStatus(
      FunctionInfo::ExecuteStatus::kNeedUpdate);
#endif

#ifndef SERIAL_MODE
  // start running inline function when every rank come to this stage.
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  // join function name and input arguments and
  // detect whether to use ''' or """ to wrap the function with arguments (default uses
  // """ to wrap)
  va_list Args;
  va_start(Args, argc);
  std::string str_wrapper("\"\"\"");
  bool wrapper_detected = false, unable_to_wrapped = false;

  std::string str_function(std::string(function_name) + std::string("("));
  for (int i = 0; i < argc; i++) {
    if (i != 0) str_function += std::string(",");

    // find what wrapper to use, raise error msg if it uses both """ and '''.
    std::string str_va_arg(va_arg(Args, const char*));
    if (!wrapper_detected) {
      if (str_va_arg.find("\"\"\"") != std::string::npos) {
        wrapper_detected = true;
        str_wrapper = std::string("'''");
      } else if (str_va_arg.find("'''") != std::string::npos) {
        wrapper_detected = true;
        str_wrapper = std::string("\"\"\"");
      }
    } else {
      // using both """ and ''' for triple quotes, unable to wrap
      if (str_va_arg.find(str_wrapper.c_str()) != std::string::npos) {
        unable_to_wrapped = true;
      }
    }

    // join arguments
    str_function += str_va_arg;
  }
  str_function += std::string(")");
  va_end(Args);

  if (unable_to_wrapped) {
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    // set error msg in interactive mode before returning YT_FAIL.
    std::string str_set_error =
        std::string("libyt.interactive_mode[\"func_err_msg\"][\"") +
        std::string(function_name) +
        std::string("\"] = \"LIBYT Error: Please avoid using both \\\"\\\"\\\" and "
                    "\'\'\' for triple quotes.\\n\"");
    LibytProcessControl::Get().function_info_list_[func_index].SetStatus(
        FunctionInfo::ExecuteStatus::kFailed);
    if (PyRun_SimpleString(str_set_error.c_str()) != 0) {
      logging::LogError("Unexpected error occurred when setting unable to wrap error "
                        "message in interactive mode.\n");
    }
#endif
    // return YT_FAIL
    logging::LogError("Please avoid using both \"\"\" and ''' for triple quotes.\n");
    YT_ABORT("Invoking %s ... failed\n", str_function.c_str());
  }

  // join function and input arguments into string wrapped by exec()
  std::string str_CallYT(std::string("exec(") + str_wrapper + str_function + str_wrapper +
                         std::string(", sys.modules[\"") +
                         std::string(LibytProcessControl::Get().param_libyt_.script) +
                         std::string("\"].__dict__)"));

  logging::LogInfo("Performing YT inline analysis %s ...\n", str_function.c_str());

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  std::string str_CallYT_TryExcept;
  str_CallYT_TryExcept =
      std::string("try:\n") + std::string("    ") + str_CallYT + std::string("\n") +
      std::string("except Exception as e:\n"
                  "    libyt.interactive_mode[\"func_err_msg\"][\"") +
      std::string(function_name) + std::string("\"] = traceback.format_exc()\n");
#endif

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  if (PyRun_SimpleString(str_CallYT_TryExcept.c_str()) != 0)
#else
  if (PyRun_SimpleString(str_CallYT.c_str()) != 0)
#endif
  {
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    LibytProcessControl::Get().function_info_list_[func_index].SetStatus(
        FunctionInfo::ExecuteStatus::kFailed);
#endif
    YT_ABORT("Unexpected error occurred while executing %s in script's namespace.\n",
             str_function.c_str());
  }

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  // update status in LibytProcessControl::Get().function_info_list_
  LibytProcessControl::Get().function_info_list_[func_index].SetStatusUsingPythonResult();
  FunctionInfo::ExecuteStatus all_status =
      LibytProcessControl::Get().function_info_list_[func_index].GetAllStatus();
#endif

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  logging::LogInfo(
      "Performing YT inline analysis %s ... %s.\n",
      str_function.c_str(),
      (all_status == FunctionInfo::ExecuteStatus::kSuccess) ? "done" : "failed");
#else
  logging::LogInfo("Performing YT inline analysis %s ... done.\n", str_function.c_str());
#endif

  return YT_SUCCESS;
}

/**
 * \brief Call Python function without args in in situ process
 * \fn int yt_run_Function(const char* function_name)
 * \details
 * 1. Route to \ref yt_run_FunctionArguments.
 *
 * @param function_name[in] Python function name
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \rst
 * .. code-block:: c
 *
 *    // Equivalent to Python: function_name()
 *    yt_run_Function("function_name");
 * \endrst
 */
int yt_run_Function(const char* function_name) {
  SET_TIMER(__PRETTY_FUNCTION__);

  int result = yt_run_FunctionArguments(function_name, 0);

  return result;
}