#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_Dimensions
// Description :  Get dimension of the grid with grid id = gid.
//
// Note        :  1. This function will be called inside user's field derived_func.
//                2. Return YT_FAIL if cannot find grid id = gid.
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

	bool have_Grid = false;

	for ( int lid = 0; lid < g_param_yt.num_grids_local; lid++ ){
		if ( g_param_yt.grids_local[lid].id == gid ){
			have_Grid = true;
			for ( int d = 0; d < 3; d++ ){
				(*dimensions)[d] = g_param_yt.grids_local[lid].dimensions[d];
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
// Note        :  1. This function will be called inside user's field derived_func.
//                2. Return YT_FAIL if cannot find grid id = gid or if field_name is not in field_list.
//                3. User should cast to their own datatype after receiving the pointer.
//
// Parameter   :  const long   gid              : Target grid id.
//                const char  *field_name       : Target field name.
//                void       **field_data       : Store the field_data pointer to here.
//                
// Example     :  void *Data;
//                yt_getGridInfo_FieldData( gid, "field_name", &Data );
//                double *FieldData = (double *) Data;
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_FieldData( const long gid, const char *field_name, void **field_data){

	bool have_Grid  = false;
	bool have_Field = false;

	for ( int lid = 0; lid < g_param_yt.num_grids_local; lid++ ){
		if ( g_param_yt.grids_local[lid].id == gid ){
			have_Grid = true;
			for ( int v = 0; v < g_param_yt.num_fields; v++ ){
				if ( strcmp(g_param_yt.field_list[v].field_name, field_name) == 0 ){
					have_Field = true;
					*field_data = g_param_yt.grids_local[lid].field_data[v];
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