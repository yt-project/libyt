#include "LibytProcessControl.h"
#include "libyt.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_finalize
// Description :  Undo all initializations done by yt_initialize()
//
// Note        :  1. Do not reinitialize libyt (i.e., calling yt_initialize()) after calling this function
//                   ==> Some extensions (e.g., NumPy) may not work properly
//                2. Make sure that the user has follow the full libyt workflow.
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_finalize() {
    SET_TIMER(__PRETTY_FUNCTION__);

    log_info("Exiting libyt ...\n");

    // check whether libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) YT_ABORT("Calling yt_finalize() before yt_initialize()!\n");

    // check if all the libyt allocated resource are freed
    if (!LibytProcessControl::Get().free_gridsPtr) YT_ABORT("Please invoke yt_free() before calling yt_finalize().\n");

    // free all libyt resources
    Py_Finalize();

    LibytProcessControl::Get().libyt_initialized = false;

    return YT_SUCCESS;

}  // FUNCTION : yt_finalize
