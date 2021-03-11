#include "yt_combo.h"
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_inline
// Description :  Execute the YT inline analysis script
//
// Note        :  1. Python script name is stored in "g_param_libyt.script"
//                2. This python script must contain function of <function_name> you called.
//
// Parameter   :  char *function_name : the function inside python script that you want to execute.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_inline( char *function_name )
{

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized ){
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );
   }

// check if YT parameters have been set
   if ( !g_param_libyt.param_yt_set ){
      YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );
   }

// check if user has call yt_get_gridsPtr(), so that libyt knows the local grids array ptr.
   if ( !g_param_libyt.get_gridsPtr ){
      YT_ABORT( "Please invoke yt_get_gridsPtr() before calling %s()!\n", __FUNCTION__ );
   }

// check if user has call yt_commit_grids(), so that grids are appended to YT.
   if ( !g_param_libyt.commit_grids ){
      YT_ABORT( "Please invoke yt_commit_grids() before calling %s()!\n", __FUNCTION__ );
   }

// Not sure if we need this MPI_Barrier
   MPI_Barrier(MPI_COMM_WORLD);

   log_info( "Performing YT inline analysis ...\n" );

// execute function in python script
   int InlineFunctionWidth = strlen(function_name) + 4; // width = .<function_name>() + '\0'
   const int CallYT_CommandWidth = strlen( g_param_libyt.script ) + InlineFunctionWidth;
   char *CallYT = (char*) malloc( CallYT_CommandWidth*sizeof(char) );
   sprintf( CallYT, "%s.%s()", g_param_libyt.script, function_name );

   if ( PyRun_SimpleString( CallYT ) == 0 )
      log_debug( "Invoking \"%s\" ... done\n", CallYT );
   else
      YT_ABORT(  "Invoking \"%s\" ... failed\n", CallYT );

   free( CallYT );

   log_info( "Performing YT inline analysis ... done.\n" );

   return YT_SUCCESS;

} // FUNCTION : yt_inline
