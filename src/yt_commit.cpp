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
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_commit() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if YT parameters have been set
    if (!LibytProcessControl::Get().param_yt_set_) {
        YT_ABORT("Please invoke yt_set_Parameters() before calling %s()!\n", __FUNCTION__);
    }

    // Make sure data is set by user, which is to check if api is called.
    if (LibytProcessControl::Get().param_yt_.num_fields > 0 && !LibytProcessControl::Get().get_fields_ptr_) {
        YT_ABORT("num_fields == %d, please invoke yt_get_FieldsPtr() before calling %s()!\n",
                 LibytProcessControl::Get().param_yt_.num_fields, __FUNCTION__);
    }

    if (LibytProcessControl::Get().param_yt_.num_par_types > 0 && !LibytProcessControl::Get().get_particles_ptr_) {
        YT_ABORT("num_par_types == %d, please invoke yt_get_ParticlesPtr() before calling %s()!\n",
                 LibytProcessControl::Get().param_yt_.num_par_types, __FUNCTION__);
    }

    if (LibytProcessControl::Get().param_yt_.num_grids_local > 0 && !LibytProcessControl::Get().get_grids_ptr_) {
        YT_ABORT("Please invoke yt_get_GridsPtr() before calling %s()!\n", __FUNCTION__);
    }

    LogInfo("Loading full hierarchy and local data to libyt ...\n");

    // Add field_list to libyt.param_yt['field_list'] dictionary
    DataStructureOutput status;
    status = LibytProcessControl::Get().data_structure_amr_.BindInfoToPython("libyt.param_yt",
                                                                             LibytProcessControl::Get().py_param_yt_);
    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        log_debug("Loading field/particle info to libyt ... done!\n");
    } else {
        log_error(status.error.c_str());
        YT_ABORT("Loading field/particle info to libyt ... failed!\n");
    }

    int root_rank = 0;
    status = LibytProcessControl::Get().data_structure_amr_.BindAllHierarchyToPython(root_rank);
    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        log_debug("Loading full hierarchy to libyt ... done!\n");
    } else {
        log_error(status.error.c_str());
        YT_ABORT("Loading full hierarchy to libyt ... failed!\n");
    }

    status = LibytProcessControl::Get().data_structure_amr_.BindLocalDataToPython();
    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        log_debug("Loading local data to libyt ... done!\n");
    } else {
        log_error(status.error.c_str());
        YT_ABORT("Loading local data to libyt ... failed!\n");
    }

    // Free grids_local
    LibytProcessControl::Get().data_structure_amr_.CleanUpGridsLocal();

    // Above all works like charm
    LibytProcessControl::Get().commit_grids_ = true;
    LogInfo("Loading full hierarchy and local data ... done.\n");

    return YT_SUCCESS;

}  // FUNCTION : yt_commit
