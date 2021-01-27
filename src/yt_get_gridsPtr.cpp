#include "yt_combo.h"
#include "libyt.h"

// TODO: Function name a little bit misleading..., name a new one later.


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_get_gridsPtr
// Description :  Get pointer of the array of struct yt_grid with length num_grids_local.
//
// Note        :  1. User should call this function after yt_set_parameter(), since we need grids_MPI or 
//                   num_grids_local.
//                2. Compute num_grids_local from grids_MPI if it doesn't exist. Should I take this part
//                   to yt_param_yt.h?
//
// Parameter   :  yt_grid *grids_local : Initialize and store the grid structure array under this pointer
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//

int yt_get_gridsPtr( yt_grid * &grids_local )
{
	// check if yt_set_parameter() have been called
   	if ( g_param_libyt.param_yt_set ) {
    	log_info( "Getting pointer to local grids array  ...\n" );
   	}
   	else {
   		YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );
   	}

   	// Get the MPI rank
   	int MyRank;
   	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

   	// TODO: Should I move this step to struct yt_param_yt
	// If g_param_yt.num_grids_local is set, do nothing
	if ( g_param_yt.num_grids_local != INT_UNDEFINED ) {
	}

	// If g_param_yt.num_grids_local not set, count num_grids_local through grids_MPI
	else {
		int num_grids_local = 0;
		for ( int i = 0; i < g_param_yt.num_grids; i = i+1 ){
			if ( g_param_yt.grids_MPI[i] == MyRank ) {
				num_grids_local = num_grids_local + 1;
			}
		}
		g_param_yt.num_grids_local = num_grids_local;
	}

	// Initialize the grids_local array.
	// Set the value if overlapped with g_param_yt,
	// and each fields data are set to NULL
	grids_local = new yt_grid [g_param_yt.num_grids_local];
	for ( int id = 0; id < g_param_yt.num_grids_local; id = id+1 ){
		grids_local[id].proc_num = MyRank;
		grids_local[id].num_fields = g_param_yt.num_fields;
		grids_local[id].field_labels = (const char **) g_param_yt.field_labels;
		grids_local[id].field_data = new void* [g_param_yt.num_fields];
		for ( int fid = 0; fid < g_param_yt.num_fields; fid = fid+1){
			grids_local[id].field_data[fid] = NULL;
		}
	}

	return YT_SUCCESS;
}