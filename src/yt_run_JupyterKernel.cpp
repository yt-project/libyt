#include "libyt.h"
#include "yt_combo.h"

#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL)
#include <iostream>

#include "LibytProcessControl.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_run_JupyterKernel
// Description :  Start libyt kernel for Jupyter Notebook access
//
// Note        :  1. Must enable -DINTERACTIVE_MODE and -DJUPYTER_KERNEL.
//                2. Must install libyt provisioner for jupyter client.
//                3. This API is like interactive mode, but with Jupyter Notebook access for better UI.
//
// Parameter   :  const char *flag_file_name : once this file is detected, it will activate libyt kernel.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_run_JupyterKernel(const char* flag_file_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

#if !defined(INTERACTIVE_MODE) || !defined(JUPYTER_KERNEL)
    log_error(
        "Cannot start libyt kernel for Jupyter. Please compile libyt with -DINTERACTIVE_MODE and -DJUPYTER_KERNEL.\n");
    return YT_FAIL;
#else

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // run new added functions
    if (g_func_status_list.run_func() != YT_SUCCESS)
        YT_ABORT("Something went wrong when running new added functions\n");

    // TODO: (LATER) check if we need to start libyt kernel by checking if file flag_file_name exist.

    return YT_SUCCESS;
#endif  // #if !defined(INTERACTIVE_MODE) && !defined(JUPYTER_KERNEL)
}
