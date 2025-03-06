#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "python_controller.h"
#include "timer.h"

static void PrintLibytInfo();

/**
 * \defgroup api_yt_initialize libyt API: yt_initialize
 * \fn int yt_initialize(int argc, char* argv[], const yt_param_libyt* param_libyt)
 * \brief Initialize libyt
 * \details
 * 1. This function should not be called more than once.
 * 2. Initialize libyt workflow, Python interpreter, and import libyt module.
 *
 * @param argc[in]
 * @param argv[in]
 * @param param_libyt[in] libyt runtime parameters
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
int yt_initialize(int argc, char* argv[], const yt_param_libyt* param_libyt) {
  LibytProcessControl::Get().Initialize();

  SET_TIMER(__PRETTY_FUNCTION__);

  // yt_initialize should only be called once
  static int init_count = 0;
  init_count++;

  // still need to check "init_count" since yt_finalize() will set check point
  // libyt_initialized = false"
  if (LibytProcessControl::Get().libyt_initialized_ || init_count >= 2) {
    YT_ABORT("yt_initialize() should not be called more than once!\n");
  }

  // store user-provided parameters to a libyt internal variable
  // --> better do it **before** calling any log function since they will query
  // param_libyt.verbose
  LibytProcessControl::Get().param_libyt_.verbose = param_libyt->verbose;
  LibytProcessControl::Get().param_libyt_.script = param_libyt->script;
  LibytProcessControl::Get().param_libyt_.counter =
      param_libyt
          ->counter;  // useful during restart, where the initial counter can be non-zero
  LibytProcessControl::Get().param_libyt_.check_data = param_libyt->check_data;

  logging::LogInfo("******libyt version******\n");
  logging::LogInfo("         %d.%d.%d\n",
                   LIBYT_MAJOR_VERSION,
                   LIBYT_MINOR_VERSION,
                   LIBYT_MICRO_VERSION);
  PrintLibytInfo();
  logging::LogInfo("*************************\n");

  logging::LogInfo("Initializing libyt ...\n");
  logging::LogInfo("   verbose = %d\n", LibytProcessControl::Get().param_libyt_.verbose);
  logging::LogInfo("    script = %s\n", LibytProcessControl::Get().param_libyt_.script);
  logging::LogInfo("   counter = %ld\n", LibytProcessControl::Get().param_libyt_.counter);
  logging::LogInfo(
      "check_data = %s\n",
      (LibytProcessControl::Get().param_libyt_.check_data ? "true" : "false"));

#ifndef USE_PYBIND11
  // create libyt module, should be before init_python
  if (python_controller::CreateLibytModule() == YT_FAIL) return YT_FAIL;
#endif

  // initialize Python interpreter
  if (python_controller::InitPython(argc, argv) == YT_FAIL) {
    return YT_FAIL;
  }

  // import libyt and inline python script.
  if (python_controller::PreparePythonEnvForLibyt() == YT_FAIL) {
    return YT_FAIL;
  }

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  // set python exception hook and set not-yet-done error msg
  if (LibytPythonShell::SetExceptionHook() != YT_SUCCESS) {
    return YT_FAIL;
  }
  if (LibytPythonShell::InitializeNotDoneErrMsg() != YT_SUCCESS) {
    return YT_FAIL;
  }

  PyObject* exec_namespace = PyDict_GetItemString(
      LibytProcessControl::Get().py_interactive_mode_, "script_globals");
  PyObject* function_body_dict =
      PyDict_GetItemString(LibytProcessControl::Get().py_interactive_mode_, "func_body");
  if (LibytPythonShell::SetExecutionNamespace(exec_namespace) != YT_SUCCESS) {
    return YT_FAIL;
  }
  if (LibytPythonShell::SetFunctionBodyDict(function_body_dict) != YT_SUCCESS) {
    return YT_FAIL;
  }
#endif

  LibytProcessControl::Get().libyt_initialized_ = true;

  return YT_SUCCESS;

}  // FUNCTION : yt_initialize

static void PrintLibytInfo() {
#ifdef SERIAL_MODE
  logging::LogInfo("  SERIAL_MODE: ON\n");
#else
  logging::LogInfo("  SERIAL_MODE: OFF\n");
#endif

#ifdef INTERACTIVE_MODE
  logging::LogInfo("  INTERACTIVE_MODE: ON\n");
#else
  logging::LogInfo("  INTERACTIVE_MODE: OFF\n");
#endif

#ifdef JUPYTER_KERNEL
  logging::LogInfo("  JUPYTER_KERNEL: ON\n");
#else
  logging::LogInfo("  JUPYTER_KERNEL: OFF\n");
#endif

#ifdef SUPPORT_TIMER
  logging::LogInfo("  SUPPORT_TIMER: ON\n");
#else
  logging::LogInfo("  SUPPORT_TIMER: OFF\n");
#endif
}
