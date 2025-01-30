#include "numpy_controller.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  init_python
// Description :  Initialize Python interpreter
//
// Note        :  1. Called by yt_initialize()
//                2. Initialize Python interpreter and Numpy.
//                3. Set up identifier flag:
//                   sys._parallel = True --> run yt in inline mode
//                   sys._interactive_mode = True --> mpi does not abort when there is error (ABORT)
//
// Parameter   :  argc : Argument count
//                argv : Argument vector
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int init_python(int argc, char* argv[]) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef USE_PYBIND11
    // 0: skips the initialization registration of signal handlers
    Py_InitializeEx(0);
    if (Py_IsInitialized())
        log_debug("Initializing Python interpreter ... done\n");
    else {
        YT_ABORT("Initializing Python interpreter ... failed!\n");
    }
#else
    pybind11::initialize_interpreter();
#endif  // #ifndef USE_PYBIND11

    // import numpy
    if (numpy_controller::InitializeNumPy() == NumPyStatus::kNumPySuccess) {
        log_debug("Importing NumPy ... done\n");
    } else {
        YT_ABORT("Importing NumPy ... failed!\n");
    }

    // add the current location to the module search path
    if (PyRun_SimpleString("import sys; sys.path.insert(0, '.')") == 0) {
        log_debug("Adding search path for modules ... done\n");
    } else {
        YT_ABORT("Adding search path for modules ... failed!\n");
    }

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    // import traceback and inspect for interactive mode
    if (PyRun_SimpleString("import traceback, inspect") == 0) {
        log_debug("Import traceback and inspect ... done\n");
    } else {
        YT_ABORT("Import traceback and inspect ... failed!\n");
    }
#endif

    // set up identifier flag
    // (sys._parallel = True --> run yt in parallel in inline mode)
    if (PyRun_SimpleString("sys._parallel = True") == 0) {
        log_debug("Set sys._parallel=True ... done\n");
    } else {
        YT_ABORT("Set sys._parallel=True ... failed\n");
    }

    // (sys._interactive_mode = True --> mpi does not abort when there is error)
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    if (PyRun_SimpleString("sys._interactive_mode = True") == 0) {
        log_debug("Set sys._interactive_mode=True ... done\n");
    } else {
        YT_ABORT("Set sys._interactive_mode=True ... failed\n");
    }
#endif

    // import the garbage collector interface
    if (PyRun_SimpleString("import gc") == 0) {
        log_debug("Import Python garbage collector ... done\n");
    } else {
        YT_ABORT("Import Python garbage collector ... failed!\n");
    }

    return YT_SUCCESS;

}  // FUNCTION : init_python
