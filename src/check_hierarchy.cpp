#include "yt_combo.h"
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  check_hierarchy.cpp
// Description :  Check that the hierarchy, parent-children relationships are correct
//
// Note        :  1. Use inside yt_add_grids()
// 			      2. Check that the hierarchy is correct, even though we didn't build a parent-children 
//                   map.
// 				  
// Parameter   :  yt_hierarchy hierarchy : Contain full hierarchy
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
    for (int i = 0; i < g_param_yt.num_grids; i = i+1) {
        if (order[ hierarchy[i].id ] == -1) {
            order[ hierarchy[i].id ] = i;
        }
        else {
            YT_ABORT("Grid ID [ %d ] are not unique!\n", hierarchy[i].id);
        }
    }

    // Check if all level > 0 have parent id, and that children's edges don't exceed parent's
    for (int i = 0; i < g_param_yt.num_grids; i = i+1) {

        if ( hierarchy[i].level > 0 ) {
            
            // Check parent id
            if ( (hierarchy[i].parent_id < 0) || hierarchy[i].parent_id >= g_param_yt.num_grids ){
                YT_ABORT("Grid ID [%d], level %d, parent_id %d, expect parent_id be 0 ~ %d.\n", 
                          hierarchy[i].id, hierarchy[i].level, hierarchy[i].parent_id, g_param_yt.num_grids - 1);
            }
            else {
                // Check edges
                double *parent_left_edge = hierarchy[order[hierarchy[i].parent_id]].left_edge;
                double *parent_right_edge = hierarchy[order[hierarchy[i].parent_id]].right_edge;
                for (int d = 0; d < 3; d = d+1){
                    if ( !(parent_left_edge[d] <= hierarchy[i].left_edge[d]) ) {
                        YT_ABORT("Grid ID [%d], Parent ID [%d], grid_left_edge[%d] < parent_left_edge[%d].\n", 
                                  hierarchy[i].id, hierarchy[i].parent_id, d, d);
                    }
                    if ( !(hierarchy[i].right_edge[d] <= parent_right_edge[d]) ) {
                        YT_ABORT("Grid ID [%d], Parent ID [%d], parent_right_edge[%d] < grid_right_edge[%d].\n", 
                                  hierarchy[i].id, hierarchy[i].parent_id, d, d);
                    }
                }
            }
        }
    }  

    // Free resource
    delete [] order;

	return YT_SUCCESS;
}