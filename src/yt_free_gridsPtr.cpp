#include "yt_combo.h"
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_free_gridsPtr()
// Description :  Refresh the python yt state after finish inline-analysis
//
// Note        :  1. Call and use by user, after they are done with all the inline-analysis in this 
//                   round.
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//
int yt_free_gridsPtr()
{
// free resources to prepare for the next round
   g_param_libyt.param_yt_set = false;
   g_param_libyt.get_gridsPtr = false;
   g_param_libyt.commit_grids = false;
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
} // FUNCTION: yt_free_gridsPtr()