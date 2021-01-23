#include "yt_combo.h"
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_get_gridsPtr
// Description :  Get pointer of the array of struct yt_grid with length num_grids_local.
//
// Note        :  1. User should call this function after yt_set_parameter(), since we need grids_MPI or 
//                   num_grids_local.
//
// Parameter   :  yt_grid *grids_local : Initialize and store the grid structure array under this pointer
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//

int yt_get_gridsPtr( yt_grid *grids_local )
{
	// check if yt_set_parameter() have been called
   	if ( g_param_libyt.param_yt_set ) {
    	log_info( "Getting pointer to local grids array  ...\n" );
   	}
   	else {
   		YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );
   	}

	// Use g_param_yt.num_grids_local directly
	if ( g_param_yt.num_grids_local != INT_UNDEFINED ) {
		grids_local = new yt_grid [g_param_yt.num_grids_local];
	}

	// Count num_grids_local through grids_MPI
	else {
		int MyRank;
		int num_grids_local = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
		for ( int i = 0; i < g_param_yt.num_grids; i = i+1 ){
			if ( g_param_yt.grids_MPI[i] == MyRank ) {
				num_grids_local = num_grids_local + 1;
			}
		}
		g_param_yt.num_grids_local = num_grids_local;
	}

	// Initialize the grids_local array, and each fields data are set to NULL
	grids_local = new yt_grid [g_param_yt.num_grids_local];
	for ( int id = 0; id < g_param_yt.num_grids_local; id = id+1 ){
		grids_local[id].field_data = new void* [g_param_yt.num_fields];
		for ( int fid = 0; fid < g_param_yt.num_fields; fid = fid+1){
			grids_local[id].field_data[fid] = NULL;
		}
	}

	return YT_SUCCESS;
}