#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

static void PrintLibytInfo();

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_initialize
// Description :  Initialize libyt
//
// Note        :  1. Input "param_libyt" will be backed up to a libyt global variable
//                2. This function should not be called more than once (even if yt_finalize has been called)
//                   since some extensions (e.g., NumPy) may not work properly.
//                3. Initialize general info, user-defined MPI data type, and LibytProcessControl
//
// Parameter   :  argc        : Argument count
//                argv        : Argument vector
//                param_libyt : libyt runtime parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_initialize(int argc, char* argv[], const yt_param_libyt* param_libyt) {
    LibytProcessControl::Get().Initialize();

    SET_TIMER(__PRETTY_FUNCTION__);

    // yt_initialize should only be called once
    static int init_count = 0;
    init_count++;

    // still need to check "init_count" since yt_finalize() will set check point libyt_initialized = false"
    if (LibytProcessControl::Get().libyt_initialized_ || init_count >= 2)
        YT_ABORT("yt_initialize() should not be called more than once!\n");

    // store user-provided parameters to a libyt internal variable
    // --> better do it **before** calling any log function since they will query param_libyt.verbose
    LibytProcessControl::Get().param_libyt_.verbose = param_libyt->verbose;
    LibytProcessControl::Get().param_libyt_.script = param_libyt->script;
    LibytProcessControl::Get().param_libyt_.counter =
        param_libyt->counter;  // useful during restart, where the initial counter can be non-zero
    LibytProcessControl::Get().param_libyt_.check_data = param_libyt->check_data;

    log_info("******libyt version******\n");
    log_info("         %d.%d.%d\n", LIBYT_MAJOR_VERSION, LIBYT_MINOR_VERSION, LIBYT_MICRO_VERSION);
    PrintLibytInfo();
    log_info("*************************\n");

    log_info("Initializing libyt ...\n");
    log_info("   verbose = %d\n", LibytProcessControl::Get().param_libyt_.verbose);
    log_info("    script = %s\n", LibytProcessControl::Get().param_libyt_.script);
    log_info("   counter = %ld\n", LibytProcessControl::Get().param_libyt_.counter);
    log_info("check_data = %s\n", (LibytProcessControl::Get().param_libyt_.check_data ? "true" : "false"));

#ifndef USE_PYBIND11
    // create libyt module, should be before init_python
    if (CreateLibytModule() == YT_FAIL) return YT_FAIL;
#endif

    // initialize Python interpreter
    if (init_python(argc, argv) == YT_FAIL) return YT_FAIL;

    // import libyt and inline python script.
    if (init_libyt_module() == YT_FAIL) return YT_FAIL;

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    // set python exception hook and set not-yet-done error msg
    if (LibytPythonShell::SetExceptionHook() != YT_SUCCESS) return YT_FAIL;
    if (LibytPythonShell::InitializeNotDoneErrMsg() != YT_SUCCESS) return YT_FAIL;

    PyObject* exec_namespace = PyDict_GetItemString(LibytProcessControl::Get().py_interactive_mode_, "script_globals");
    PyObject* function_body_dict = PyDict_GetItemString(LibytProcessControl::Get().py_interactive_mode_, "func_body");
    if (LibytPythonShell::SetExecutionNamespace(exec_namespace) != YT_SUCCESS) return YT_FAIL;
    if (LibytPythonShell::SetFunctionBodyDict(function_body_dict) != YT_SUCCESS) return YT_FAIL;
#endif

    LibytProcessControl::Get().libyt_initialized_ = true;

    return YT_SUCCESS;

}  // FUNCTION : yt_initialize

static void PrintLibytInfo() {
#ifdef SERIAL_MODE
    log_info("  SERIAL_MODE: ON\n");
#else
    log_info("  SERIAL_MODE: OFF\n");
#endif

#ifdef INTERACTIVE_MODE
    log_info("  INTERACTIVE_MODE: ON\n");
#else
    log_info("  INTERACTIVE_MODE: OFF\n");
#endif

#ifdef JUPYTER_KERNEL
    log_info("  JUPYTER_KERNEL: ON\n");
#else
    log_info("  JUPYTER_KERNEL: OFF\n");
#endif

#ifdef SUPPORT_TIMER
    log_info("  SUPPORT_TIMER: ON\n");
#else
    log_info("  SUPPORT_TIMER: OFF\n");
#endif
}
