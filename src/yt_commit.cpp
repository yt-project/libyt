#include "big_mpi.h"
#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_commit
// Description :  Add local grids, append field list and particle list info to the libyt Python module.
//
// Note        :  1. Must call yt_get_FieldsPtr (if num_fields>0), yt_get_ParticlesPtr (if num_par_types>0),
//                   yt_get_GridsPtr, which gets data info from user.
//                2. Check the local grids, field list, and particle list.
//                3. Append field_list info and particle_list info to libyt.param_yt['field_list'] and
//                   libyt.param_yt['particle_list'].
//                4. Gather hierarchy in different rank, and check hierarchy in check_hierarchy(), excluding
//                   particles.
//                5. If there is particle, we gather different particle type separately.
//                6. Pass the grids and hierarchy to YT in function append_grid().
//                7. We assume that one grid contains all the fields belong to that grid.
//                8. Free LibytProcessControl::Get().param_yt_.grids_local, after we have passed all grid info and data
//                in.
//                9. TODO: this can be more memory efficient when gathering hierarchy.
//
// Parameter   :
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_commit() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if YT parameters have been set
    if (!LibytProcessControl::Get().param_yt_set) {
        YT_ABORT("Please invoke yt_set_Parameters() before calling %s()!\n", __FUNCTION__);
    }

    // TODO: get_fieldsPtr is used here (need to have a new mechanism to make sure the amr structure
    //       is properly set before calling commit.
    // check if user sets field_list
    //    if (!LibytProcessControl::Get().get_fieldsPtr) {
    //        YT_ABORT("num_fields == %d, please invoke yt_get_FieldsPtr() before calling %s()!\n",
    //                 LibytProcessControl::Get().param_yt_.num_fields, __FUNCTION__);
    //    }

    // TODO: Same reason as above
    // check if user sets particle_list
    //    if (!LibytProcessControl::Get().get_particlesPtr) {
    //        YT_ABORT("num_par_types == %d, please invoke yt_get_ParticlesPtr() before calling %s()!\n",
    //                 LibytProcessControl::Get().param_yt_.num_par_types, __FUNCTION__);
    //    }

    // TODO: same reason as above
    // check if user has call yt_get_GridsPtr()
    //    if (!LibytProcessControl::Get().get_gridsPtr) {
    //        YT_ABORT("Please invoke yt_get_GridsPtr() before calling %s()!\n", __FUNCTION__);
    //    }

    log_info("Loading grids to yt ...\n");

    yt_param_yt& param_yt = LibytProcessControl::Get().param_yt_;

    // TODO: move check data process into data_structure_amr class
    // Check yt_field* field_list
    if (LibytProcessControl::Get().param_libyt_.check_data && param_yt.num_fields > 0) {
        if (check_field_list() != YT_SUCCESS) {
            YT_ABORT("Check field_list failed in %s!\n", __FUNCTION__);
        }
    }

    // Check yt_particle* particle_list
    if (LibytProcessControl::Get().param_libyt_.check_data && param_yt.num_par_types > 0) {
        if (check_particle_list() != YT_SUCCESS) {
            YT_ABORT("Check particle_list failed in %s!\n", __FUNCTION__);
        }
    }

    // Check yt_grid* grids_local
    if (LibytProcessControl::Get().param_libyt_.check_data && param_yt.num_grids_local > 0) {
        if (check_grid() != YT_SUCCESS) {
            YT_ABORT("Check grids_local failed in %s!\n", __FUNCTION__);
        }
    }

    // Add field_list to libyt.param_yt['field_list'] dictionary
    if (param_yt.num_fields > 0) {
        if (add_dict_field_list() != YT_SUCCESS) {
            YT_ABORT("Inserting dictionary libyt.param_yt['field_list'] failed!\n");
        }
    }

    // Add particle_list to libyt.param_yt['particle_list'] dictionary
    if (param_yt.num_par_types > 0) {
        if (add_dict_particle_list() != YT_SUCCESS) {
            YT_ABORT("Inserting dictionary libyt.param_yt['particle_list'] failed!\n");
        }
    }

    int root_rank = 0;

    // fill libyt.hierarchy with NumPy arrays or memoryviews
    if (allocate_hierarchy() == YT_SUCCESS)
        log_debug("Allocating libyt.hierarchy ... done\n");
    else
        YT_ABORT("Allocating libyt.hierarchy ... failed!\n");

#ifndef SERIAL_MODE
    // append grid to YT
    // We pass hierarchy to each rank as well.
    // Combine full hierarchy and the grid data that one rank has, otherwise fill in NULL in grid data.
    long start_block = 0;
    long end_block;
    for (int rank = 0; rank < LibytProcessControl::Get().mpi_rank_; rank++) {
        start_block += num_grids_local_MPI[rank];
    }
    end_block = start_block + param_yt.num_grids_local;

    yt_grid grid_combine;
    grid_combine.par_count_list = new long[param_yt.num_par_types];
    for (long i = 0; i < param_yt.num_grids; i = i + 1) {
        // Load from hierarchy_full
        for (int d = 0; d < 3; d = d + 1) {
            grid_combine.left_edge[d] = hierarchy_full[i].left_edge[d];
            grid_combine.right_edge[d] = hierarchy_full[i].right_edge[d];
            grid_combine.grid_dimensions[d] = hierarchy_full[i].dimensions[d];
        }
        for (int s = 0; s < param_yt.num_par_types; s++) {
            grid_combine.par_count_list[s] = particle_count_list_full[s][i];
        }
        grid_combine.id = hierarchy_full[i].id;
        grid_combine.parent_id = hierarchy_full[i].parent_id;
        grid_combine.level = hierarchy_full[i].level;
        grid_combine.proc_num = hierarchy_full[i].proc_num;

        // load from param_yt.grids_local
        if (start_block <= i && i < end_block) {
            // Get the pointer to data from grids_local
            grid_combine.field_data = grids_local[i - start_block].field_data;
            grid_combine.particle_data = grids_local[i - start_block].particle_data;
        } else {
            // Make it points to NULL
            grid_combine.field_data = nullptr;
            grid_combine.particle_data = nullptr;
        }

        // Append grid to YT
        if (append_grid(&grid_combine) != YT_SUCCESS) {
            delete[] hierarchy_full;
            delete[] hierarchy_local;
            for (int s = 0; s < param_yt.num_par_types; s++) {
                delete[] particle_count_list_full[s];
                delete[] particle_count_list_local[s];
            }
            delete[] particle_count_list_full;
            delete[] particle_count_list_local;

            delete[] grid_combine.par_count_list;
            YT_ABORT("Failed to append grid [ %ld ]!\n", grid_combine.id);
        }
    }
#else
    for (long i = 0; i < param_yt.num_grids; i = i + 1) {
        if (append_grid(&(grids_local[i])) != YT_SUCCESS) {
            YT_ABORT("Failed to append grid [%ld]!\n", grids_local[i].id);
        }
    }
#endif

    log_debug("Append grids to libyt.grid_data ... done!\n");

#ifndef SERIAL_MODE
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifndef SERIAL_MODE
    // Freed resource
    delete[] hierarchy_local;
    delete[] hierarchy_full;
    for (int s = 0; s < param_yt.num_par_types; s++) {
        delete[] particle_count_list_full[s];
        delete[] particle_count_list_local[s];
    }
    delete[] particle_count_list_full;
    delete[] particle_count_list_local;
    delete[] grid_combine.par_count_list;
#endif

    // Free grids_local
    if (LibytProcessControl::Get().get_gridsPtr && param_yt.num_grids_local > 0) {
        for (int i = 0; i < param_yt.num_grids_local; i = i + 1) {
            if (param_yt.num_fields > 0) {
                delete[] grids_local[i].field_data;
            }
            if (param_yt.num_par_types > 0) {
                delete[] grids_local[i].par_count_list;
                for (int p = 0; p < param_yt.num_par_types; p++) {
                    delete[] grids_local[i].particle_data[p];
                }
                delete[] grids_local[i].particle_data;
            }
        }
        delete[] grids_local;
    }

    // Above all works like charm
    LibytProcessControl::Get().commit_grids = true;
    LibytProcessControl::Get().get_gridsPtr = false;
    log_info("Loading grids to yt ... done.\n");

    return YT_SUCCESS;

}  // FUNCTION : yt_commit
