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
#ifdef SUPPORT_TIMER
   g_timer->record_time("yt_finalize", 0);
#endif

   log_info( "Exiting libyt ...\n" );

// check whether libyt has been initialized
   if ( !g_param_libyt.libyt_initialized )   YT_ABORT( "Calling yt_finalize() before yt_init()!\n" );

// check if all the libyt allocated resource are freed
   if ( !g_param_libyt.free_gridsPtr ) YT_ABORT("Please invoke yt_free_gridsPtr() before calling yt_finalize().\n");

// free all libyt resources
   Py_Finalize();

   g_param_libyt.libyt_initialized = false;

#ifdef SUPPORT_TIMER
   // end timer and print.
   g_timer->record_time("yt_finalize", 1);
   g_timer->print_all_time();
   // destroy timer.
   delete g_timer;
#endif // #ifdef SUPPORT_TIMER

   return YT_SUCCESS;

} // FUNCTION : yt_finalize
