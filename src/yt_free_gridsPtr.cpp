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
// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized ){
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );
   }

// check if YT parameters have been set
   if ( !g_param_libyt.param_yt_set ){
      YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );
   }

// check if user has call yt_get_fieldsPtr()
   if ( !g_param_libyt.get_fieldsPtr ){
      YT_ABORT( "Please invode yt_get_fieldsPtr() before calling %s()!\n", __FUNCTION__ );
   }

// check if user has call yt_get_gridsPtr(), so that libyt knows the local grids array ptr.
   if ( !g_param_libyt.get_gridsPtr ){
      YT_ABORT( "Please invoke yt_get_gridsPtr() before calling %s()!\n", __FUNCTION__ );
   }

// check if user has call yt_commit_grids(), so that grids are appended to YT.
   if ( !g_param_libyt.commit_grids ){
      YT_ABORT( "Please invoke yt_commit_grids() before calling %s()!\n", __FUNCTION__ );
   }

   // Make sure every rank has reach to this point
   MPI_Barrier( MPI_COMM_WORLD );

   // free resources to prepare for the next round
   g_param_libyt.param_yt_set  = false;
   g_param_libyt.get_fieldsPtr = false;
   g_param_libyt.get_gridsPtr  = false;
   g_param_libyt.commit_grids  = false;
   g_param_libyt.counter ++;

   // Free grids_local, num_grids_local_MPI, field_list
   for (int i = 0; i < g_param_yt.num_grids_local; i = i+1){
      if ( g_param_yt.grids_local[i].field_data != NULL ){
         delete [] g_param_yt.grids_local[i].field_data;
      }
      if ( g_param_yt.grids_local[i].particle_count_list != NULL ){
         delete [] g_param_yt.grids_local[i].particle_count_list;
      }
   }
   delete [] g_param_yt.grids_local;
   delete [] g_param_yt.num_grids_local_MPI;

   if ( g_param_yt.field_list != NULL ){
      delete [] g_param_yt.field_list;
   }
   if ( g_param_yt.particle_list != NULL ){
      for ( int i = 0; i < g_param_yt.num_species; i++ ){
         delete [] g_param_yt.particle_list[i].attr_list;
      }
      delete [] g_param_yt.particle_list;
   }

   // Reset g_param_yt
   g_param_yt.init();
   
   PyDict_Clear( g_py_grid_data  );
   PyDict_Clear( g_py_hierarchy  );
   PyDict_Clear( g_py_param_yt   );
   PyDict_Clear( g_py_param_user );

   PyRun_SimpleString( "gc.collect()" );

   g_param_libyt.free_gridsPtr = true;

   return YT_SUCCESS;
} // FUNCTION: yt_free_gridsPtr()