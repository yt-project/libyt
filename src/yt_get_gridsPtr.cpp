#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_get_gridsPtr
// Description :  Get pointer of the array of struct yt_grid with length num_grids_local.
//
// Note        :  1. User should call this function after yt_set_parameter(), 
//                   since we need num_grids_local, num_fields, and num_species.
//                2. Initialize field_data in one grid with
//                   (1) data_dim[3] = {0, 0, 0}
//                   (2) data_ptr    = NULL
//                   (3) data_dtype  = YT_DTYPE_UNKNOWN
//                3. If user call this function twice, then it will just return the previously initialized 
//                   and allocated array.
//
// Parameter   :  yt_grid **grids_local : Initialize and store the grid structure array under this 
//                                        pointer points to.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//
int yt_get_gridsPtr( yt_grid **grids_local )
{
	// check if libyt has been initialized
   	if ( !g_param_libyt.libyt_initialized ){
    	YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );     	
   	}

	// check if yt_set_parameter() have been called
   	if ( !g_param_libyt.param_yt_set ) {
    	YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );
    }

    // check if num_grids_local > 0, if not, grids_local won't be initialized
    if ( g_param_yt.num_grids_local <= 0 ){
    	YT_ABORT( "num_grids_local == %d <= 0, you don't need to input grids_local!\n",
    	           g_param_yt.num_grids_local);
    }    

   	log_info( "Getting pointer to local grids information ...\n" );

   	// If user call for the first time.
   	if ( !g_param_libyt.get_gridsPtr ){
	   	// Get the MPI rank
	   	int MyRank;
	   	MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

		// Initialize the grids_local array.
		// Set the value if overlapped with g_param_yt,
		// and each fields data are set to NULL, so that we can check if user input the data
		*grids_local = new yt_grid [g_param_yt.num_grids_local];
		for ( int id = 0; id < g_param_yt.num_grids_local; id = id+1 ){
			
			(*grids_local)[id].proc_num     = MyRank;
			
			// Dealing with individual field in one grid
			if ( g_param_yt.num_fields > 0 ){
				(*grids_local)[id].field_data   = new yt_data [g_param_yt.num_fields];
			}
			else{
				(*grids_local)[id].field_data   = NULL;
			}

			// Dealing with particle_count
			if ( g_param_yt.num_species > 0 ){
				(*grids_local)[id].particle_count_list = new long [g_param_yt.num_species];
				for ( int s = 0; s < g_param_yt.num_species; s++ ){
					(*grids_local)[id].particle_count_list[s] = 0;
				}
			}
			else{
				(*grids_local)[id].particle_count_list = NULL;
			}
		}

		// Store the grids_local to g_param_yt
		g_param_yt.grids_local = *grids_local;
   	}
   	else{
   		// If user already called this function before, we just return the initialized grids_local,
   		// to avoid memory leak.
   		*grids_local = g_param_yt.grids_local;
   	}


	// Above all works like charm
	g_param_libyt.get_gridsPtr = true;
	log_info( "Getting pointer to local grids information  ... done.\n" );
	
	return YT_SUCCESS;
}