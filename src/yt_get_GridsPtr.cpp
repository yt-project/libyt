#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

/**
 * \defgroup api_yt_get_GridsPtr libyt API: yt_get_GridsPtr
 * \fn int yt_get_GridsPtr(yt_grid** grids_local)
 * \brief Get pointer of local grids information array
 * \details
 * 1. User should call this function after \ref yt_set_Parameters,
 *    because the AMR structure is initialized there.
 *
 * \todo
 * 1. Loading grid hierarchy and data is so inefficient, should fix it in libyt-v1.0.
 *
 * @param grids_local[out] Pointer to the array of struct yt_grid is stored
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
int yt_get_GridsPtr(yt_grid** grids_local) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // check if libyt has been initialized
  if (!LibytProcessControl::Get().libyt_initialized_) {
    YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
  }

  // check if yt_set_Parameters() have been called
  if (!LibytProcessControl::Get().param_yt_set_) {
    YT_ABORT("Please invoke yt_set_Parameters() before calling %s()!\n", __FUNCTION__);
  }

  // check if num_grids_local > 0, if not, grids_local won't be initialized
  if (LibytProcessControl::Get().param_yt_.num_grids_local <= 0) {
    YT_ABORT("num_grids_local == %d <= 0, you don't need to input grids_local!\n",
             LibytProcessControl::Get().param_yt_.num_grids_local);
  }

  logging::LogInfo("Getting pointer to local grids information ...\n");

  *grids_local = LibytProcessControl::Get().data_structure_amr_.GetGridsLocal();

  LibytProcessControl::Get().get_grids_ptr_ = true;
  logging::LogInfo("Getting pointer to local grids information  ... done.\n");

  return YT_SUCCESS;
}