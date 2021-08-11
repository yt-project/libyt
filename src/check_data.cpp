#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  check_sum_num_grids_local_MPI
// Description :  Check sum of number of local grids in each MPI rank is equal to num_grids input by user.
//
// Note        :  1. Use inside yt_set_parameter()
//                2. Check sum of number of local grids in each MPI rank is equal to num_grids input by 
//                   user, which is equal to the number of total grids.
//                
// Parameter   :  int * &num_grids_local_MPI : Address to the int*, each element stores number of local 
//                                             grids in each MPI rank.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_sum_num_grids_local_MPI( int NRank, int * &num_grids_local_MPI ) {
    long num_grids = 0;
    for (int rid = 0; rid < NRank; rid = rid+1){
        num_grids = num_grids + (long)num_grids_local_MPI[rid];
    }
    if (num_grids != g_param_yt.num_grids){
        for(int rid = 0; rid < NRank; rid++){
            log_error("MPI rank [ %d ], num_grids_local = %d.\n", rid, num_grids_local_MPI[rid]);
        }
        YT_ABORT("Sum of local grids in each MPI rank [%ld] are not equal to input num_grids [%ld]!\n", 
                  num_grids, g_param_yt.num_grids );
    }

   return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  check_hierarchy
// Description :  Check that the hierarchy, parent-children relationships are correct
//
// Note        :  1. Use inside yt_commit_grids()
// 			      2. Check that the hierarchy is correct, even though we didn't build a parent-children 
//                   map.
// 				  
// Parameter   :  yt_hierarchy * &hierarchy : Contain full hierarchy
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_hierarchy(yt_hierarchy * &hierarchy) {
    
    // Create a search table for matching gid to hierarchy array index
	long *order = new long [g_param_yt.num_grids];
    for (long i = 0; i < g_param_yt.num_grids; i = i+1) {
        order[i] = -1;
    }
    
    // Check every grid id are unique, and also filled in the search table
    for (long i = 0; i < g_param_yt.num_grids; i = i+1) {
        if (order[ hierarchy[i].id ] == -1) {
            order[ hierarchy[i].id ] = i;
        }
        else {
            YT_ABORT("Grid ID [ %ld ] are not unique, both MPI rank %d and %d are using this grid id!\n", 
                      hierarchy[i].id, hierarchy[i].proc_num, hierarchy[ order[ hierarchy[i].id ] ].proc_num);
        }
    }

    // Check if all level > 0 have good parent id, and that children's edges don't exceed parent's
    for (long i = 0; i < g_param_yt.num_grids; i = i+1) {

        if ( hierarchy[i].level > 0 ) {
            
            // Check parent id
            if ( (hierarchy[i].parent_id < 0) || hierarchy[i].parent_id >= g_param_yt.num_grids ){
                YT_ABORT("Grid ID [%ld], Level %d, Parent ID [%ld], expect Parent ID to be 0 ~ %ld.\n", 
                          hierarchy[i].id, hierarchy[i].level, hierarchy[i].parent_id, g_param_yt.num_grids - 1);
            }
            else {
                // Check children's edges fall between parent's
                double *parent_left_edge = hierarchy[order[hierarchy[i].parent_id]].left_edge;
                double *parent_right_edge = hierarchy[order[hierarchy[i].parent_id]].right_edge;
                for (int d = 0; d < 3; d = d+1){
                    if ( !(parent_left_edge[d] <= hierarchy[i].left_edge[d]) ) {
                        YT_ABORT("Grid ID [%ld], Parent ID [%ld], grid_left_edge[%d] < parent_left_edge[%d].\n", 
                                  hierarchy[i].id, hierarchy[i].parent_id, d, d);
                    }
                    if ( !(hierarchy[i].right_edge[d] <= parent_right_edge[d]) ) {
                        YT_ABORT("Grid ID [%ld], Parent ID [%ld], parent_right_edge[%d] < grid_right_edge[%d].\n", 
                                  hierarchy[i].id, hierarchy[i].parent_id, d, d);
                    }
                }

                // Check parent's level = children level - 1
                int parent_level = hierarchy[order[hierarchy[i].parent_id]].level;
                if ( !(parent_level == hierarchy[i].level - 1) ){
                    YT_ABORT("Grid ID [%ld], Parent ID [%ld], parent level %d != children level %d - 1.\n", 
                              hierarchy[i].id, hierarchy[i].parent_id, parent_level, hierarchy[i].level);
                }
            }
        }
    }  

    // Free resource
    delete [] order;

	return YT_SUCCESS;
}