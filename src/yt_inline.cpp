#include "yt_combo.h"
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_inline
// Description :  Execute the YT inline analysis script
//
// Note        :  1. Script name is stored in "g_param_libyt.script"
//                2. This script must first import yt and then put all YT commands in a method named
//                   "yt_inline". Example:
//
//                   import yt
//
//                   def yt_inline():
//                      #your YT commands
//                      # ...
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_inline()
{

// check if libyt has been initialized
   if ( g_param_libyt.libyt_initialized )
      log_info( "Performing YT inline analysis ...\n" );
   else
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );

// check if YT parameters have been set
   if ( !g_param_libyt.param_yt_set )
      YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );

// check if user has call yt_get_gridsPtr(), so that libyt knows the local grids array ptr.
   if ( !g_param_libyt.get_gridsPtr )
      YT_ABORT( "Please invoke yt_get_gridsPtr() before calling %s()!\n", __FUNCTION__ );

// check if user has call yt_commit_grids(), so that grids are appended to YT.
   if ( !g_param_libyt.commit_grids )
      YT_ABORT( "Please invoke yt_commit_grids() before calling %s()!\n", __FUNCTION__ );

// Not sure if we need this MPI_Barrier
   MPI_Barrier(MPI_COMM_WORLD);

// execute YT script
// TODO: Change the execute function name to arbitrary
   int InlineFunctionWidth = strlen(g_param_yt.inline_function_name) + 4; // width = .<function_name>() + '\0'
   const int CallYT_CommandWidth = strlen( g_param_libyt.script ) + InlineFunctionWidth;
   char *CallYT = (char*) malloc( CallYT_CommandWidth*sizeof(char) );
   sprintf( CallYT, "%s.%s()", g_param_libyt.script, g_param_yt.inline_function_name );

   if ( PyRun_SimpleString( CallYT ) == 0 )
      log_debug( "Invoking \"%s\" ... done\n", CallYT );
   else
      YT_ABORT(  "Invoking \"%s\" ... failed\n", CallYT );

   free( CallYT );


// free resources to prepare for the next execution
   g_param_libyt.param_yt_set = false;
   g_param_libyt.get_gridsPtr = false;
   g_param_libyt.commit_grids    = false;
   g_param_libyt.counter ++;

   for (int i = 0; i < g_param_yt.num_grids_local; i = i+1){
      delete [] g_param_yt.grids_local[i].field_data;
   }
   delete [] g_param_yt.grids_local;
   delete [] g_param_yt.num_grids_local_MPI;
   g_param_yt.init();
   
   PyDict_Clear( g_py_grid_data  );
   PyDict_Clear( g_py_hierarchy  );
   PyDict_Clear( g_py_param_yt   );
   PyDict_Clear( g_py_param_user );

   PyRun_SimpleString( "gc.collect()" );

   return YT_SUCCESS;

} // FUNCTION : yt_inline
