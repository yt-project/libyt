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

// check that all grids are set correctly
// TODO: Maybe we can move this part to yt_add_grids()
   check_grids();

// execute YT script
   const int CallYT_CommandWidth = strlen( g_param_libyt.script ) + 13;   // 13 = ".yt_inline()" + '\0'
   char *CallYT = (char*) malloc( CallYT_CommandWidth*sizeof(char) );
   sprintf( CallYT, "%s.yt_inline()", g_param_libyt.script );

   if ( PyRun_SimpleString( CallYT ) == 0 )
      log_debug( "Invoking \"%s\" ... done\n", CallYT );
   else
      YT_ABORT(  "Invoking \"%s\" ... failed\n", CallYT );

   free( CallYT );


// TODO: Check the resources should be freed!!!
// free resources to prepare for the next execution
   g_param_yt.init();
   g_param_libyt.param_yt_set = false;
   g_param_libyt.get_gridsPtr = false;
   g_param_libyt.counter ++;

   delete [] g_param_libyt.grid_hierarchy_set;
   delete [] g_param_libyt.grid_data_set;
   g_param_libyt.grid_hierarchy_set = NULL;
   g_param_libyt.grid_data_set = NULL;
   
   PyDict_Clear( g_py_grid_data  );
   PyDict_Clear( g_py_hierarchy  );
   PyDict_Clear( g_py_param_yt   );
   PyDict_Clear( g_py_param_user );

   PyRun_SimpleString( "gc.collect()" );


   return YT_SUCCESS;

} // FUNCTION : yt_inline
