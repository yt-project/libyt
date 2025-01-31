#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_get_ParticlesPtr
// Description :  Get pointer of the array of struct yt_particle with length num_particles.
//
// Note        :  1. User should call this function after yt_set_Parameters(), since we initialize
//                   particle_list then.
//
// Parameter   :  yt_particle **particle_list  : Store the particle list array pointer.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//
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
        YT_ABORT("num_par_types == %d <= 0, you don't need to input particle_list, and it is also not initialized!\n",
                 LibytProcessControl::Get().param_yt_.num_par_types);
    }

    logging::LogInfo("Getting pointer to particle list information ...\n");

    *particle_list = LibytProcessControl::Get().data_structure_amr_.GetParticleList();

    LibytProcessControl::Get().get_particles_ptr_ = true;
    logging::LogInfo("Getting pointer to particle list information  ... done.\n");

    return YT_SUCCESS;
}