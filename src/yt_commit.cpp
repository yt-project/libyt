#include "LibytProcessControl.h"
#include "big_mpi.h"
#include "libyt.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_commit
// Description :  Add local grids, append field list and particle list info to the libyt Python module.
//
// Note        :  1. Must call yt_set_Parameters() in advance, which will preallocate memory for NumPy arrays.
//                2. Must call yt_get_FieldsPtr (if num_fields>0), yt_get_ParticlesPtr (if num_par_types>0),
//                   yt_get_GridsPtr, which gets data info from user.
//                3. Check the local grids, field list, and particle list.
//                4. Append field_list info and particle_list info to libyt.param_yt['field_list'] and
//                   libyt.param_yt['particle_list'].
//                5. Gather hierarchy in different rank, and check hierarchy in check_hierarchy(), excluding
//                   particles.
//                6. If there is particle, we gather different particle type separately.
//                7. Pass the grids and hierarchy to YT in function append_grid().
//                8. We assume that one grid contains all the fields belong to that grid.
//                9. Free g_param_yt.grids_local, after we have passed all grid info and data in.
//               10. TODO: this can be more memory efficient when gathering hierarchy.
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

    // check if user sets field_list
    if (!LibytProcessControl::Get().get_fieldsPtr) {
        YT_ABORT("num_fields == %d, please invoke yt_get_FieldsPtr() before calling %s()!\n", g_param_yt.num_fields,
                 __FUNCTION__);
    }

    // check if user sets particle_list
    if (!LibytProcessControl::Get().get_particlesPtr) {
        YT_ABORT("num_par_types == %d, please invoke yt_get_ParticlesPtr() before calling %s()!\n",
                 g_param_yt.num_par_types, __FUNCTION__);
    }

    // check if user has call yt_get_GridsPtr()
    if (!LibytProcessControl::Get().get_gridsPtr) {
        YT_ABORT("Please invoke yt_get_GridsPtr() before calling %s()!\n", __FUNCTION__);
    }

    log_info("Loading grids to yt ...\n");

    // Check yt_field* field_list
    if (g_param_libyt.check_data && g_param_yt.num_fields > 0) {
        if (check_field_list() != YT_SUCCESS) {
            YT_ABORT("Check field_list failed in %s!\n", __FUNCTION__);
        }
    }

    // Check yt_particle* particle_list
    if (g_param_libyt.check_data && g_param_yt.num_par_types > 0) {
        if (check_particle_list() != YT_SUCCESS) {
            YT_ABORT("Check particle_list failed in %s!\n", __FUNCTION__);
        }
    }

    // Check yt_grid* grids_local
    if (g_param_libyt.check_data && g_param_yt.num_grids_local > 0) {
        if (check_grid() != YT_SUCCESS) {
            YT_ABORT("Check grids_local failed in %s!\n", __FUNCTION__);
        }
    }

    // Add field_list to libyt.param_yt['field_list'] dictionary
    if (g_param_yt.num_fields > 0) {
        if (add_dict_field_list() != YT_SUCCESS) {
            YT_ABORT("Inserting dictionary libyt.param_yt['field_list'] failed!\n");
        }
    }

    // Add particle_list to libyt.param_yt['particle_list'] dictionary
    if (g_param_yt.num_par_types > 0) {
        if (add_dict_particle_list() != YT_SUCCESS) {
            YT_ABORT("Inserting dictionary libyt.param_yt['particle_list'] failed!\n");
        }
    }

    int RootRank = 0;

#ifndef SERIAL_MODE
    // initialize hierarchy array, prepare for collecting hierarchy in different ranks.
    yt_hierarchy *hierarchy_full, *hierarchy_local;

    // initialize hierarchy_full, hierarchy_local and to avoid new [0].
    if (g_param_yt.num_grids > 0) hierarchy_full = new yt_hierarchy[g_param_yt.num_grids];
    if (g_param_yt.num_grids_local > 0) hierarchy_local = new yt_hierarchy[g_param_yt.num_grids_local];

    // initialize particle_count_list[ptype_label][grid_id]
    long **particle_count_list_full, **particle_count_list_local;
    if (g_param_yt.num_par_types > 0) {
        particle_count_list_full = new long*[g_param_yt.num_par_types];
        particle_count_list_local = new long*[g_param_yt.num_par_types];
        for (int s = 0; s < g_param_yt.num_par_types; s++) {
            if (g_param_yt.num_grids > 0) particle_count_list_full[s] = new long[g_param_yt.num_grids];
            if (g_param_yt.num_grids_local > 0) particle_count_list_local[s] = new long[g_param_yt.num_grids_local];
        }
    }
#endif

    yt_grid* grids_local = LibytProcessControl::Get().grids_local;

#ifndef SERIAL_MODE
    // move user passed in data to hierarchy_local and particle_count_list_local for later MPI process
    for (int i = 0; i < g_param_yt.num_grids_local; i = i + 1) {
        yt_grid grid = grids_local[i];
        for (int d = 0; d < 3; d = d + 1) {
            hierarchy_local[i].left_edge[d] = grid.left_edge[d];
            hierarchy_local[i].right_edge[d] = grid.right_edge[d];
            hierarchy_local[i].dimensions[d] = grid.grid_dimensions[d];
        }
        for (int s = 0; s < g_param_yt.num_par_types; s = s + 1) {
            particle_count_list_local[s][i] = grid.par_count_list[s];
        }
        hierarchy_local[i].id = grid.id;
        hierarchy_local[i].parent_id = grid.parent_id;
        hierarchy_local[i].level = grid.level;
        hierarchy_local[i].proc_num = grid.proc_num;
    }

    // Big MPI_Gatherv, this is just a workaround method.
    int* num_grids_local_MPI = LibytProcessControl::Get().num_grids_local_MPI;
    big_MPI_Gatherv<yt_hierarchy>(RootRank, num_grids_local_MPI, (void*)hierarchy_local, &yt_hierarchy_mpi_type,
                                  (void*)hierarchy_full);
    for (int s = 0; s < g_param_yt.num_par_types; s++) {
        big_MPI_Gatherv<long>(RootRank, num_grids_local_MPI, (void*)particle_count_list_local[s], &yt_long_mpi_type,
                              (void*)particle_count_list_full[s]);
    }
#endif

    // Check that the hierarchy are correct, do the test on RootRank only
    if (g_param_libyt.check_data && g_myrank == RootRank) {
#ifndef SERIAL_MODE
        if (check_hierarchy(hierarchy_full) == YT_SUCCESS) {
#else
        if (check_hierarchy(grids_local) == YT_SUCCESS) {
#endif
            log_debug("Validating the parent-children relationship ... done!\n");
        } else {
#ifndef SERIAL_MODE
            if (g_param_yt.num_grids > 0) delete[] hierarchy_full;
            if (g_param_yt.num_grids_local > 0) delete[] hierarchy_local;
            if (g_param_yt.num_par_types > 0) {
                delete[] particle_count_list_full;
                delete[] particle_count_list_local;
                for (int s = 0; s < g_param_yt.num_par_types; s++) {
                    if (g_param_yt.num_grids > 0) delete[] particle_count_list_full[s];
                    if (g_param_yt.num_grids_local > 0) delete[] particle_count_list_local[s];
                }
            }
#endif
            YT_ABORT("Validating the parent-children relationship ... failed!\n")
        }
    }

#ifndef SERIAL_MODE
    // Not sure if we need this MPI_Barrier
    MPI_Barrier(MPI_COMM_WORLD);

    // broadcast hierarchy_full, particle_count_list_full to each rank as well.
    big_MPI_Bcast<yt_hierarchy>(RootRank, g_param_yt.num_grids, (void*)hierarchy_full, &yt_hierarchy_mpi_type);
    for (int s = 0; s < g_param_yt.num_par_types; s++) {
        big_MPI_Bcast<long>(RootRank, g_param_yt.num_grids, (void*)particle_count_list_full[s], &yt_long_mpi_type);
    }
#endif

#ifndef SERIAL_MODE
    // append grid to YT
    // We pass hierarchy to each rank as well.
    // Combine full hierarchy and the grid data that one rank has, otherwise fill in NULL in grid data.
    long start_block = 0;
    long end_block;
    for (int rank = 0; rank < g_myrank; rank++) {
        start_block += num_grids_local_MPI[rank];
    }
    end_block = start_block + g_param_yt.num_grids_local;

    yt_grid grid_combine;
    grid_combine.par_count_list = new long[g_param_yt.num_par_types];
    for (long i = 0; i < g_param_yt.num_grids; i = i + 1) {
        // Load from hierarchy_full
        for (int d = 0; d < 3; d = d + 1) {
            grid_combine.left_edge[d] = hierarchy_full[i].left_edge[d];
            grid_combine.right_edge[d] = hierarchy_full[i].right_edge[d];
            grid_combine.grid_dimensions[d] = hierarchy_full[i].dimensions[d];
        }
        for (int s = 0; s < g_param_yt.num_par_types; s++) {
            grid_combine.par_count_list[s] = particle_count_list_full[s][i];
        }
        grid_combine.id = hierarchy_full[i].id;
        grid_combine.parent_id = hierarchy_full[i].parent_id;
        grid_combine.level = hierarchy_full[i].level;
        grid_combine.proc_num = hierarchy_full[i].proc_num;

        // load from g_param_yt.grids_local
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
            if (g_param_yt.num_grids > 0) delete[] hierarchy_full;
            if (g_param_yt.num_grids_local > 0) delete[] hierarchy_local;
            if (g_param_yt.num_par_types > 0) {
                delete[] particle_count_list_full;
                delete[] particle_count_list_local;
                for (int s = 0; s < g_param_yt.num_par_types; s++) {
                    if (g_param_yt.num_grids > 0) delete[] particle_count_list_full[s];
                    if (g_param_yt.num_grids_local > 0) delete[] particle_count_list_local[s];
                }
            }
            delete[] grid_combine.par_count_list;
            YT_ABORT("Failed to append grid [ %ld ]!\n", grid_combine.id);
        }
    }
#else
    for (long i = 0; i < g_param_yt.num_grids; i = i + 1) {
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
    if (g_param_yt.num_grids_local > 0) delete[] hierarchy_local;
    if (g_param_yt.num_grids > 0) delete[] hierarchy_full;
    if (g_param_yt.num_par_types > 0) {
        for (int s = 0; s < g_param_yt.num_par_types; s++) {
            if (g_param_yt.num_grids > 0) delete[] particle_count_list_full[s];
            if (g_param_yt.num_grids_local > 0) delete[] particle_count_list_local[s];
        }
        delete[] particle_count_list_full;
        delete[] particle_count_list_local;
    }
    delete[] grid_combine.par_count_list;
#endif

    // Free grids_local
    if (LibytProcessControl::Get().get_gridsPtr && g_param_yt.num_grids_local > 0) {
        for (int i = 0; i < g_param_yt.num_grids_local; i = i + 1) {
            if (g_param_yt.num_fields > 0) {
                delete[] grids_local[i].field_data;
            }
            if (g_param_yt.num_par_types > 0) {
                delete[] grids_local[i].par_count_list;
                for (int p = 0; p < g_param_yt.num_par_types; p++) {
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
