#include "libyt_process_control.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  print_yt_param_yt
// Description :  Print yt_param_yt struct if verbose level >= YT_VERBOSE_DEBUG
//
// Parameter   :  const yt_param_yt &param_yt
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int print_yt_param_yt(const yt_param_yt& param_yt) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (LibytProcessControl::Get().param_libyt_.verbose < YT_VERBOSE_DEBUG) return YT_SUCCESS;

    const int width_scalar = 25;
    const int width_vector = width_scalar - 3;

    if (param_yt.frontend != nullptr) LogDebug("   %-*s = %s\n", width_scalar, "frontend", param_yt.frontend);
    if (param_yt.fig_basename != nullptr)
        LogDebug("   %-*s = %s\n", width_scalar, "fig_basename", param_yt.fig_basename);
    for (int d = 0; d < 3; d++) {
        LogDebug("   %-*s[%d] = %13.7e\n", width_vector, "domain_left_edge", d, param_yt.domain_left_edge[d]);
    }
    for (int d = 0; d < 3; d++) {
        LogDebug("   %-*s[%d] = %13.7e\n", width_vector, "domain_right_edge", d, param_yt.domain_right_edge[d]);
    }
    LogDebug("   %-*s = %13.7e\n", width_scalar, "current_time", param_yt.current_time);
    LogDebug("   %-*s = %d\n", width_scalar, "cosmological_simulation", param_yt.cosmological_simulation);
    if (param_yt.cosmological_simulation) {
        LogDebug("   %-*s = %13.7e\n", width_scalar, "current_redshift", param_yt.current_redshift);
        LogDebug("   %-*s = %13.7e\n", width_scalar, "omega_lambda", param_yt.omega_lambda);
        LogDebug("   %-*s = %13.7e\n", width_scalar, "omega_matter", param_yt.omega_matter);
        LogDebug("   %-*s = %13.7e\n", width_scalar, "hubble_constant", param_yt.hubble_constant);
    }

    LogDebug("   %-*s = %13.7e\n", width_scalar, "length_unit", param_yt.length_unit);
    LogDebug("   %-*s = %13.7e\n", width_scalar, "mass_unit", param_yt.mass_unit);
    LogDebug("   %-*s = %13.7e\n", width_scalar, "time_unit", param_yt.time_unit);
    LogDebug("   %-*s = %13.7e\n", width_scalar, "velocity_unit", param_yt.velocity_unit);
    if (param_yt.magnetic_unit == DBL_UNDEFINED)
        LogDebug("   %-*s = %s\n", width_scalar, "magnetic_unit", "NOT SET, and will be set to 1.");
    else
        LogDebug("   %-*s = %13.7e\n", width_scalar, "magnetic_unit", param_yt.magnetic_unit);

    for (int d = 0; d < 3; d++) {
        LogDebug("   %-*s[%d] = %d\n", width_vector, "periodicity", d, param_yt.periodicity[d]);
    }
    for (int d = 0; d < 3; d++) {
        LogDebug("   %-*s[%d] = %d\n", width_vector, "domain_dimensions", d, param_yt.domain_dimensions[d]);
    }
    LogDebug("   %-*s = %d\n", width_scalar, "dimensionality", param_yt.dimensionality);
    LogDebug("   %-*s = %d\n", width_scalar, "refine_by", param_yt.refine_by);
    LogDebug("   %-*s = %d\n", width_scalar, "index_offset", param_yt.index_offset);
    LogDebug("   %-*s = %ld\n", width_scalar, "num_grids", param_yt.num_grids);

    LogDebug("   %-*s = %ld\n", width_scalar, "num_fields", param_yt.num_fields);
    LogDebug("   %-*s = %ld\n", width_scalar, "num_par_types", param_yt.num_par_types);
    for (int s = 0; s < param_yt.num_par_types; s++) {
        if (param_yt.par_type_list != nullptr)
            LogDebug("   %-*s[%d] = (type=\"%s\", num_attr=%d)\n", width_vector, "par_type_list", s,
                     param_yt.par_type_list[s].par_type, param_yt.par_type_list[s].num_attr);
        else
            LogDebug("   %-*s[%d] = (type=\"%s\", num_attr=%d)\n", width_vector, "par_type_list", s, "NULL",
                     param_yt.par_type_list[s].num_attr);
    }
    LogDebug("   %-*s = %ld\n", width_scalar, "num_grids_local", param_yt.num_grids_local);

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  print_yt_field
// Description :  Print yt_field struct if verbose level >= YT_VERBOSE_DEBUG
//
// Notes       :  1. TODO: Should also print field_dtype after updating the ugly yt_dtype system.
//
// Parameter   :  const yt_param_yt &param_yt
//
// Return      :  YT_SUCCESS
//-------------------------------------------------------------------------------------------------------
int print_yt_field(const yt_field& field) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (LibytProcessControl::Get().param_libyt_.verbose < YT_VERBOSE_DEBUG) return YT_SUCCESS;

    const int width_scalar = 25;

    LogDebug("   %-*s (%s):\n", width_scalar, field.field_name, field.field_type);
    LogDebug("   %-*s = \n", width_scalar + 2, "field_unit", field.field_unit);
    LogDebug("   %-*s = \n", width_scalar + 2, "contiguous_in_x", field.contiguous_in_x ? "True" : "False");
    LogDebug("   %-*s = (%d, %d, %d, %d, %d, %d)\n", width_scalar + 2, "field_ghost_cell", field.field_ghost_cell[0],
             field.field_ghost_cell[1], field.field_ghost_cell[2], field.field_ghost_cell[3], field.field_ghost_cell[4],
             field.field_ghost_cell[5]);

    return YT_SUCCESS;
}