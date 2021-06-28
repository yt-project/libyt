#include "yt_combo.h"
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_finalize
// Description :  Undo all initializations done by yt_init()
//
// Note        :  1. Do not reinitialize libyt (i.e., calling yt_init()) after calling this function
//                   ==> Some extensions (e.g., NumPy) may not work properly
//                2. Make sure that the user has follow the full libyt workflow.
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_finalize()
{

   log_info( "Exiting libyt ...\n" );

// check whether libyt has been initialized
   if ( !g_param_libyt.libyt_initialized )   YT_ABORT( "Calling yt_finalize() before yt_init()!\n" );

// check if all the libyt allocated resource are freed
   if ( !g_param_libyt.free_gridsPtr ) YT_ABORT("Please invoke yt_free_gridsPtr() before calling yt_finalize().\n");

// free all libyt resources
   Py_Finalize();

   g_param_libyt.libyt_initialized = false;
   return YT_SUCCESS;

} // FUNCTION : yt_finalize
