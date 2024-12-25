#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_get_GridsPtr
// Description :  Get pointer of the array of struct yt_grid with length num_grids_local.
//
// Note        :  1. User should call this function after yt_set_Parameters(),
//                   since we initialize it there.
//
// Parameter   :  yt_grid **grids_local : Initialize and store the grid structure array under this
//                                        pointer points to.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_get_GridsPtr(yt_grid** grids_local) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if yt_set_Parameters() have been called
    if (!LibytProcessControl::Get().param_yt_set) {
        YT_ABORT("Please invoke yt_set_Parameters() before calling %s()!\n", __FUNCTION__);
    }

    // check if num_grids_local > 0, if not, grids_local won't be initialized
    if (LibytProcessControl::Get().param_yt_.num_grids_local <= 0) {
        YT_ABORT("num_grids_local == %d <= 0, you don't need to input grids_local!\n",
                 LibytProcessControl::Get().param_yt_.num_grids_local);
    }

    log_info("Getting pointer to local grids information ...\n");

    *grids_local = LibytProcessControl::Get().data_structure_amr_.grids_local_;

    LibytProcessControl::Get().get_gridsPtr = true;
    log_info("Getting pointer to local grids information  ... done.\n");

    return YT_SUCCESS;
}