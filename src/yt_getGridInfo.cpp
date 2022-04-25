#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  check_procedure
// Description :  Check if libyt is properly set.
//
// Note        :  1. This function will only be use in yt_getGridInfo_*().
//
// Parameter   :  const char *callFunc : function that calls check_procedure.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_procedure( const char *callFunc ){

	// check if libyt has been initialized
   	if ( !g_param_libyt.libyt_initialized ){
    	YT_ABORT( "Please follow the libyt procedure, forgot to invoke yt_init() before calling %s()!\n", callFunc );
   	}

	// check if YT parameters have been set
   	if ( !g_param_libyt.param_yt_set ){
    	YT_ABORT( "Please follow the libyt procedure, forgot to invoke yt_set_parameter() before calling %s()!\n", callFunc );
   	}

	// check if user sets field_list
   	if ( !g_param_libyt.get_fieldsPtr ){
      	YT_ABORT( "num_fields == %d, please invoke yt_get_fieldsPtr() before calling %s()!\n",
                   g_param_yt.num_fields, callFunc );
   	}

	// check if user sets particle_list
   	if ( !g_param_libyt.get_particlesPtr ){
      	YT_ABORT( "num_species == %d, please invoke yt_get_particlesPtr() before calling %s()!\n",
                   g_param_yt.num_species, callFunc );
   	}

	// check if user has call yt_get_gridsPtr(), so that libyt knows the local grids array ptr.
   	if ( !g_param_libyt.get_gridsPtr ){
      	YT_ABORT( "Please follow the libyt procedure, forgot to invoke yt_get_gridsPtr() before calling %s()!\n", callFunc );
   	}

	// check if user has call yt_commit_grids(), so that grids are checked and appended to YT.
   	if ( !g_param_libyt.commit_grids ){
      	YT_ABORT( "Please follow the libyt procedure, forgot to invoke yt_commit_grids() before calling %s()!\n", callFunc );
   	}

	return YT_SUCCESS;
}
//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_Dimensions
// Description :  Get dimension of the grid with grid id = gid.
//
// Note        :  1. This function will be called inside user's field derived_func or derived_func_with_name.
//                2. Return YT_FAIL if cannot find grid id = gid or grid dimension < 0.
//                3. grid_dimensions is defined in [x][y][z] <-> [0][1][2] coordinate.
//
// Parameter   :  const long  gid               : Target grid id
//                int         (*dimensions)[3]  : Write result to this pointer.
//                
// Example     :  int dim[3];
//                yt_getGridInfo_Dimensions( gid, &dim );
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_Dimensions( const long gid, int (*dimensions)[3] ){

	if ( check_procedure( __FUNCTION__ ) != YT_SUCCESS ){
		YT_ABORT( "Please follow the libyt procedure.\n" );
	}

	bool have_Grid = false;

	for ( int lid = 0; lid < g_param_yt.num_grids_local; lid++ ){
		if ( g_param_yt.grids_local[lid].id == gid ){
			have_Grid = true;
			for ( int d = 0; d < 3; d++ ){
                if (g_param_yt.grids_local[lid].grid_dimensions[d] < 0){
                    log_warning("In %s, GID [ %ld ] grid_dimensions[%d] = %d < 0.\n",
                                __FUNCTION__, gid, d, g_param_yt.grids_local[lid].grid_dimensions[d]);
                    return YT_FAIL;
                }
				(*dimensions)[d] = g_param_yt.grids_local[lid].grid_dimensions[d];
			}
			break;
		}
	}

	if ( !have_Grid ){
		int MyRank;
		MPI_Comm_rank( MPI_COMM_WORLD, &MyRank );
		log_warning("In %s, cannot find grid with GID [ %ld ] on MPI rank [%d].\n", 
			         __FUNCTION__, gid, MyRank);
		return YT_FAIL;
	}

	return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_FieldData
// Description :  Get field_data of field_name in the grid with grid id = gid .
//
// Note        :  1. This function will be called inside user's field derived_func or derived_func_with_name.
//                2. Return YT_FAIL if cannot find grid id = gid or if field_name is not in field_list.
//                3. User should cast to their own datatype after receiving the pointer.
//
// Parameter   :  const long   gid              : Target grid id.
//                const char  *field_name       : Target field name.
//                yt_data     *field_data       : Store the yt_data struct pointer that points to data here.
//                
// Example     :  yt_data Data;
//                yt_getGridInfo_FieldData( gid, "field_name", &Data );
//                double *FieldData = (double *) Data.data_ptr;
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_FieldData( const long gid, const char *field_name, yt_data *field_data){

	if ( check_procedure( __FUNCTION__ ) != YT_SUCCESS ){
		YT_ABORT( "Please follow the libyt procedure.\n" );
	}

	bool have_Grid  = false;
	bool have_Field = false;

	for ( int lid = 0; lid < g_param_yt.num_grids_local; lid++ ){
		if ( g_param_yt.grids_local[lid].id == gid ){
			have_Grid = true;
			for ( int v = 0; v < g_param_yt.num_fields; v++ ){
				if ( strcmp(g_param_yt.field_list[v].field_name, field_name) == 0 ){
					have_Field = true;
                    if ( g_param_yt.grids_local[lid].field_data[v].data_ptr == NULL ){
                        log_warning("In %s, GID [ %ld ] field [ %s ] data_ptr not set.\n",
                                    __FUNCTION__, gid, field_name);
                        return YT_FAIL;
                    }

					(*field_data).data_ptr = g_param_yt.grids_local[lid].field_data[v].data_ptr;
					for ( int d = 0; d < 3; d++ ){
						(*field_data).data_dimensions[d] = g_param_yt.grids_local[lid].field_data[v].data_dimensions[d];
					}
					(*field_data).data_dtype = g_param_yt.grids_local[lid].field_data[v].data_dtype;
					
					break;
				}
			}
			break;
		}
	}

	if ( !have_Field ){
		log_warning("In %s, cannot find field_name [ %s ] in field_list.\n", __FUNCTION__, field_name);
		return YT_FAIL;
	}

	if ( !have_Grid ){
		int MyRank;
		MPI_Comm_rank( MPI_COMM_WORLD, &MyRank );
		log_warning("In %s, cannot find grid with GID [ %ld ] on MPI rank [%d].\n", 
			         __FUNCTION__, gid, MyRank);
		return YT_FAIL;
	}

	return YT_SUCCESS;
}