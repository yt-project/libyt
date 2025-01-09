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

    log_info("Loading full hierarchy and local data to libyt ...\n");

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
    int root_rank = 0;
    LibytProcessControl::Get().data_structure_amr_.BindInfoToPython(LibytProcessControl::Get().py_param_yt_,
                                                                    "libyt.param_yt");

    LibytProcessControl::Get().data_structure_amr_.BindAllHierarchyToPython(root_rank);
    log_debug("Loading full hierarchy to libyt ... done!\n");

    LibytProcessControl::Get().data_structure_amr_.BindLocalDataToPython();
    log_debug("Loading local data to libyt ... done!\n");

    // Free grids_local
    LibytProcessControl::Get().data_structure_amr_.CleanUpGridsLocal();

    // Above all works like charm
    LibytProcessControl::Get().commit_grids = true;
    LibytProcessControl::Get().get_gridsPtr = false;
    log_info("Loading full hierarchy and local data ... done.\n");

    return YT_SUCCESS;

}  // FUNCTION : yt_commit
