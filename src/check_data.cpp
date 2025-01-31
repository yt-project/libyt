#include "libyt_process_control.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  check_yt_param_yt
// Description :  Check yt_param_yt struct
//
// Note        :  1. Cosmological parameters are checked only if cosmological_simulation == 1
//                2. Only check if data are set properly, does not alter them.
//
// Parameter   :  const yt_param_yt &param_yt
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_yt_param_yt(const yt_param_yt& param_yt) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (param_yt.frontend == NULL) YT_ABORT("\"%s\" has not been set!\n", "frontend");
    for (int d = 0; d < 3; d++) {
        if (param_yt.domain_left_edge[d] == DBL_UNDEFINED)
            YT_ABORT("\"%s[%d]\" has not been set!\n", "domain_left_edge", d);
        if (param_yt.domain_right_edge[d] == DBL_UNDEFINED)
            YT_ABORT("\"%s[%d]\" has not been set!\n", "domain_right_edge", d);
    }
    if (param_yt.current_time == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "current_time");
    if (param_yt.cosmological_simulation == INT_UNDEFINED)
        YT_ABORT("\"%s\" has not been set!\n", "cosmological_simulation");
    if (param_yt.cosmological_simulation == 1) {
        if (param_yt.current_redshift == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "current_redshift");
        if (param_yt.omega_lambda == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "omega_lambda");
        if (param_yt.omega_matter == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "omega_matter");
        if (param_yt.hubble_constant == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "hubble_constant");
    }
    if (param_yt.length_unit == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "length_unit");
    if (param_yt.mass_unit == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "mass_unit");
    if (param_yt.time_unit == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "time_unit");
    if (param_yt.velocity_unit == DBL_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "velocity_unit");
    if (param_yt.magnetic_unit == DBL_UNDEFINED) LogWarning("\"%s\" has not been set!\n", "magnetic_unit");

    for (int d = 0; d < 3; d++) {
        if (param_yt.periodicity[d] == INT_UNDEFINED) YT_ABORT("\"%s[%d]\" has not been set!\n", "periodicity", d);
        if (param_yt.domain_dimensions[d] == INT_UNDEFINED)
            YT_ABORT("\"%s[%d]\" has not been set!\n", "domain_dimensions", d);
    }
    if (param_yt.dimensionality == INT_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "dimensionality");
    if (param_yt.refine_by == INT_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "refine_by");
    if (param_yt.num_grids == LNG_UNDEFINED) YT_ABORT("\"%s\" has not been set!\n", "num_grids");
    if (param_yt.num_par_types > 0 && param_yt.par_type_list == NULL)
        YT_ABORT("Particle type info par_type_list has not been set!\n");
    if (param_yt.num_par_types < 0 && param_yt.par_type_list != NULL)
        YT_ABORT("Particle type info num_par_types has not been set!\n");
    for (int s = 0; s < param_yt.num_par_types; s++) {
        if (param_yt.par_type_list[s].par_type == NULL || param_yt.par_type_list[s].num_attr < 0)
            YT_ABORT("par_type_list element [ %d ] is not set properly!\n", s);
    }

    return YT_SUCCESS;
}
