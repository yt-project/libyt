#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

/**
 * \defgroup api_yt_get_ParticlesPtr libyt API: yt_get_ParticlesPtr
 * \fn int yt_get_ParticlesPtr(yt_particle** particle_list)
 * \brief Get pointer of particle information array
 * \details
 * 1. User should call this function after \ref yt_set_Parameters,
 *    because the AMR structure is initialized there.
 *
 * \todo
 * 1. The setting up of particle info is just bad. Should fix it in libyt-v1.0.
 *
 * @param particle_list[out] Pointer to the array of struct yt_particle is stored
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
int yt_get_ParticlesPtr(yt_particle** particle_list) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // check if libyt has been initialized
  if (!LibytProcessControl::Get().libyt_initialized_) {
    YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
  }

  // check if yt_set_Parameters() have been called
  if (!LibytProcessControl::Get().param_yt_set_) {
    YT_ABORT("Please invoke yt_set_Parameters() before calling %s()!\n", __FUNCTION__);
  }

  // check if num_par_types > 0, if not, particle_list won't be initialized
  if (LibytProcessControl::Get().param_yt_.num_par_types <= 0) {
    YT_ABORT("num_par_types == %d <= 0, you don't need to input particle_list, and it is "
             "also not initialized!\n",
             LibytProcessControl::Get().param_yt_.num_par_types);
  }

  logging::LogInfo("Getting pointer to particle list information ...\n");

  *particle_list = LibytProcessControl::Get().data_structure_amr_.GetParticleList();

  LibytProcessControl::Get().get_particles_ptr_ = true;
  logging::LogInfo("Getting pointer to particle list information  ... done.\n");

  return YT_SUCCESS;
}