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
// Function    :  check_field_list
// Description :  Check g_param_yt.field_list.
//
// Note        :  1. Use inside yt_commit_grids().
//                2. Check field_list
//                  (1) Validate each yt_field element in field_list.
//                  (2) Name of each field are unique.
//                
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_field_list(){
    // (1) Validate each yt_field element in field_list.
    for ( int v = 0; v < g_param_yt.num_fields; v++ ){
        yt_field field = g_param_yt.field_list[v];
        if ( !(field.validate()) ){
            YT_ABORT("Validating input field list element [%d] ... failed\n", v);
        }
    }

    // (2) Name of each field are unique.
    for ( int v1 = 0; v1 < g_param_yt.num_fields; v1++ ){
        for ( int v2 = v1+1; v2 < g_param_yt.num_fields; v2++ ){
            if ( strcmp(g_param_yt.field_list[v1].field_name, g_param_yt.field_list[v2].field_name) == 0 ){
                YT_ABORT("field_name in field_list[%d] and field_list[%d] are not unique!\n", v1, v2);
            }
        }
    }

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Function    :  check_particle_list
// Description :  Check g_param_yt.particle_list.
//
// Note        :  1. Use inside yt_commit_grids().
//                2. Check particle_list
//                  (1) Validate each yt_particle element in particle_list.
//                  (2) Species name (or ptype in YT-term) cannot be the same as g_param_yt.frontend.
//                  (3) Species names (or ptype in YT-term) are all unique.
//                
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_particle_list(){

    // (1) Validate each yt_particle element in particle_list.
    // (2) Check species name (or ptype in YT-term) cannot be the same as g_param_yt.frontend.
    for ( int p = 0; p < g_param_yt.num_species; p++ ){
        yt_particle particle = g_param_yt.particle_list[p];
        if ( !(particle.validate()) ){
            YT_ABORT("Validating input particle list element [%d] ... failed\n", p);
        }
        if ( strcmp(particle.species_name, g_param_yt.frontend) == 0 ){
            YT_ABORT("particle_list[%d], species_name == %s, frontend == %s, expect species_name different from the frontend!\n",
                     p, particle.species_name, g_param_yt.frontend);
        }
    }

    // (3) Species names (or ptype in YT-term) are all unique.
    for ( int p1 = 0; p1 < g_param_yt.num_species; p1++ ){
        for ( int p2 = p1+1; p2 < g_param_yt.num_species; p2++ ){
            if ( strcmp(g_param_yt.particle_list[p1].species_name, g_param_yt.particle_list[p2].species_name) == 0 ){
                YT_ABORT("species_name in particle_list[%d] and particle_list[%d] are not unique!\n", p1, p2);
            }
        }
    }

    return YT_SUCCESS;
}


//-------------------------------------------------------------------------------------------------------
// Function    :  check_grid
// Description :  Check g_param_yt.grids_local.
//
// Note        :  1. Use inside yt_commit_grids().
//                2. Check grids_local
//                  (1) Validate each yt_grid element in grids_local.
//                  (2) grid ID is between 0 ~ (num_grids-1).
//                  (3) parent ID is not bigger or equal to num_grids.
//                  (4) Root level starts at 0. So if level > 0, then parent ID >= 0.
//                  (5) domain left edge <= grid left edge.
//                  (6) grid right edge <= domain right edge.
//                  (7) grid left edge <= grid right edge. 
//                      (Not sure if this still holds for periodic condition.)
//                  (8) Raise warning if field_define_type = "cell-centered", and data_ptr == NULL.
//                  (9) Raise warning if field_define_type = "face-centered", and data_ptr == NULL.
//                  (10) If data_ptr != NULL, then data_dimensions > 0
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_grid(){
// Checking grids
// check each grids individually
    for (int i = 0; i < g_param_yt.num_grids_local; i = i+1) {

        yt_grid grid = g_param_yt.grids_local[i];

        // (1) Validate each yt_grid element in grids_local.
        if ( !(grid.validate()) )
            YT_ABORT(  "Validating input grid ID [%ld] ... failed\n", grid.id );

        // (2) grid ID is between 0 ~ (num_grids-1).
        if ((grid.id < 0) || grid.id >= g_param_yt.num_grids)
            YT_ABORT(  "Grid ID [%ld] not in the range between 0 ~ (number of grids [%ld] - 1)!\n",
                       grid.id, g_param_yt.num_grids );

        // (3) parent ID is not bigger or equal to num_grids.
        if (grid.parent_id >= g_param_yt.num_grids)
            YT_ABORT(  "Grid [%ld] parent ID [%ld] >= total number of grids [%ld]!\n",
                       grid.id, grid.parent_id, g_param_yt.num_grids );

        // (4) Root level starts at 0. So if level > 0, then parent ID >= 0.
        if (grid.level > 0 && grid.parent_id < 0)
            YT_ABORT(  "Grid [%ld] parent ID [%ld] < 0 at level [%d]!\n",
                       grid.id, grid.parent_id, grid.level );

        // edge
        for (int d = 0; d < 3; d = d+1) {

            // (5) Domain left edge <= grid left edge.
            if (grid.left_edge[d] < g_param_yt.domain_left_edge[d])
                YT_ABORT( "Grid [%ld] left edge [%13.7e] < domain left edge [%13.7e] along the dimension [%d]!\n",
                          grid.id, grid.left_edge[d], g_param_yt.domain_left_edge[d], d );

            // (6) grid right edge <= domain right edge.
            if (grid.right_edge[d] > g_param_yt.domain_right_edge[d])
                YT_ABORT( "Grid [%ld] right edge [%13.7e] > domain right edge [%13.7e] along the dimension [%d]!\n",
                          grid.id, grid.right_edge[d], g_param_yt.domain_right_edge[d], d );

            // (7) grid left edge <= grid right edge.
            if (grid.right_edge[d] < grid.left_edge[d])
                YT_ABORT( "Grid [%ld], right edge [%13.7e] < left edge [%13.7e]!\n",
                          grid.id, grid.right_edge[d], grid.left_edge[d]);
        }

        // check field_data in each individual grid
        for (int v = 0; v < g_param_yt.num_fields; v = v+1){

            // If field_define_type == "cell-centered"
            if ( strcmp(g_param_yt.field_list[v].field_define_type, "cell-centered") == 0 ) {

                // (8) Raise warning if field_define_type = "cell-centered", and data_ptr is not set == NULL.
                if ( grid.field_data[v].data_ptr == NULL ){
                    log_warning( "Grid [%ld], field_data [%s], field_define_type [%s], data_ptr is NULL, not set yet!",
                                 grid.id, g_param_yt.field_list[v].field_name, g_param_yt.field_list[v].field_define_type);
                }
            }

            // If field_define_type == "face-centered"
            if ( strcmp(g_param_yt.field_list[v].field_define_type, "face-centered") == 0 ) {

                // (9) Raise warning if field_define_type = "face-centered", and data_ptr is not set == NULL.
                if ( grid.field_data[v].data_ptr == NULL ){
                    log_warning( "Grid [%ld], field_data [%s], field_define_type [%s], data_ptr is NULL, not set yet!",
                                 grid.id, g_param_yt.field_list[v].field_name, g_param_yt.field_list[v].field_define_type);
                }
                else{
                    // (10) If data_ptr != NULL, then data_dimensions > 0
                    for ( int d = 0; d < 3; d++ ){
                        if ( grid.field_data[v].data_dimensions[d] <= 0 ){
                            YT_ABORT("Grid [%ld], field_data [%s], field_define_type [%s], data_dimensions[%d] == %d <= 0, should be > 0!\n",
                                     grid.id, g_param_yt.field_list[v].field_name, g_param_yt.field_list[v].field_define_type, d, grid.field_data[v].data_dimensions[d]);
                        }
                    }
                }
            }

            // If field_define_type == "derived_func"
            if ( strcmp(g_param_yt.field_list[v].field_define_type, "derived_func") == 0 ) {
                // (10) If data_ptr != NULL, then data_dimensions > 0
                if ( grid.field_data[v].data_ptr != NULL ){
                    for ( int d = 0; d < 3; d++ ){
                        if ( grid.field_data[v].data_dimensions[d] <= 0 ){
                            YT_ABORT("Grid [%ld], field_data [%s], field_define_type [%s], data_dimensions[%d] == %d <= 0, should be > 0!\n",
                                     grid.id, g_param_yt.field_list[v].field_name, g_param_yt.field_list[v].field_define_type, d, grid.field_data[v].data_dimensions[d]);
                        }
                    }
                }
            }
        }
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
//                  (1) Check every grid id are unique.
//                  (2) Check if all grids with level > 0, have a good parent id.
//                  (3) Check if children grids' edge fall between parent's.
//                  (4) Check parent's level = children level - 1.
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