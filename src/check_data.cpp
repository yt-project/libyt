#include "libyt_process_control.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  check_particle_list
// Description :  Check LibytProcessControl::Get().param_yt_.particle_list.
//
// Note        :  1. Use inside yt_commit().
//                2. Check particle_list
//                  (1) Validate each yt_particle element in particle_list.
//                  (2) Species name (or ptype in YT-term) cannot be the same as
//                  LibytProcessControl::Get().param_yt_.frontend. (3) Species names (or ptype in YT-term) are all
//                  unique.
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_particle_list() {
    SET_TIMER(__PRETTY_FUNCTION__);

    yt_particle* particle_list = LibytProcessControl::Get().data_structure_amr_.particle_list_;

    // (1) Validate each yt_particle element in particle_list.
    // (2) Check particle type name (or ptype in YT-term) cannot be the same as
    // LibytProcessControl::Get().param_yt_.frontend.
    for (int p = 0; p < LibytProcessControl::Get().param_yt_.num_par_types; p++) {
        yt_particle& particle = particle_list[p];
        if (!(check_yt_particle(particle))) {
            YT_ABORT("Validating input particle list element [%d] ... failed\n", p);
        }
        if (strcmp(particle.par_type, LibytProcessControl::Get().param_yt_.frontend) == 0) {
            YT_ABORT("particle_list[%d], par_type == %s, frontend == %s, expect particle type name different from the "
                     "frontend!\n",
                     p, particle.par_type, LibytProcessControl::Get().param_yt_.frontend);
        }
    }

    // (3) Particle type name (or ptype in YT-term) are all unique.
    for (int p1 = 0; p1 < LibytProcessControl::Get().param_yt_.num_par_types; p1++) {
        for (int p2 = p1 + 1; p2 < LibytProcessControl::Get().param_yt_.num_par_types; p2++) {
            if (strcmp(particle_list[p1].par_type, particle_list[p2].par_type) == 0) {
                YT_ABORT(
                    "par_type in particle_list[%d] and particle_list[%d] are the same, par_type should be unique!\n",
                    p1, p2);
            }
        }
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  check_grid
// Description :  Check LibytProcessControl::Get().param_yt_.grids_local.
//
// Note        :  1. Use inside yt_commit().
//                2. Check grids_local
//                  (1) Validate each yt_grid element in grids_local.
//                  (2) parent ID is not bigger or equal to num_grids.
//                  (3) Root level starts at 0. So if level > 0, then parent ID >= 0.
//                  (4) domain left edge <= grid left edge.
//                  (5) grid right edge <= domain right edge.
//                  (6) grid left edge <= grid right edge.
//                      (Not sure if this still holds for periodic condition.)
//                  (7) Abort if field_type = "cell-centered", and data_ptr == NULL.
//                  (8) Abort if field_type = "face-centered", and data_ptr == NULL.
//                  (9) If data_ptr != NULL, then data_dimensions > 0
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_grid() {
    SET_TIMER(__PRETTY_FUNCTION__);

    yt_grid* grids_local = LibytProcessControl::Get().data_structure_amr_.grids_local_;
    yt_field* field_list = LibytProcessControl::Get().data_structure_amr_.field_list_;

    // Checking grids
    // check each grids individually
    for (int i = 0; i < LibytProcessControl::Get().param_yt_.num_grids_local; i = i + 1) {
        yt_grid& grid = grids_local[i];

        // (1) Validate each yt_grid element in grids_local.
        if (check_yt_grid(grid) != YT_SUCCESS) YT_ABORT("Validating input grid ID [%ld] ... failed\n", grid.id);

        // (2) parent ID is not bigger or equal to num_grids.
        if (grid.parent_id >= LibytProcessControl::Get().param_yt_.num_grids)
            YT_ABORT("Grid [%ld] parent ID [%ld] >= total number of grids [%ld]!\n", grid.id, grid.parent_id,
                     LibytProcessControl::Get().param_yt_.num_grids);

        // (3) Root level starts at 0. So if level > 0, then parent ID >= 0.
        if (grid.level > 0 && grid.parent_id < 0)
            YT_ABORT("Grid [%ld] parent ID [%ld] < 0 at level [%d]!\n", grid.id, grid.parent_id, grid.level);

        // edge
        for (int d = 0; d < 3; d = d + 1) {
            // (4) Domain left edge <= grid left edge.
            if (grid.left_edge[d] < LibytProcessControl::Get().param_yt_.domain_left_edge[d])
                YT_ABORT("Grid [%ld] left edge [%13.7e] < domain left edge [%13.7e] along the dimension [%d]!\n",
                         grid.id, grid.left_edge[d], LibytProcessControl::Get().param_yt_.domain_left_edge[d], d);

            // (5) grid right edge <= domain right edge.
            if (grid.right_edge[d] > LibytProcessControl::Get().param_yt_.domain_right_edge[d])
                YT_ABORT("Grid [%ld] right edge [%13.7e] > domain right edge [%13.7e] along the dimension [%d]!\n",
                         grid.id, grid.right_edge[d], LibytProcessControl::Get().param_yt_.domain_right_edge[d], d);

            // (6) grid left edge <= grid right edge.
            if (grid.right_edge[d] < grid.left_edge[d])
                YT_ABORT("Grid [%ld], right edge [%13.7e] < left edge [%13.7e]!\n", grid.id, grid.right_edge[d],
                         grid.left_edge[d]);
        }

        // check field_data in each individual grid
        for (int v = 0; v < LibytProcessControl::Get().param_yt_.num_fields; v = v + 1) {
            // If field_type == "cell-centered"
            if (strcmp(field_list[v].field_type, "cell-centered") == 0) {
                // (7) Raise warning if field_type = "cell-centered", and data_ptr is not set == NULL.
                if (grid.field_data[v].data_ptr == NULL) {
                    YT_ABORT("Grid [%ld], field_data [%s], field_type [%s], data_ptr is NULL, not set yet!", grid.id,
                             field_list[v].field_name, field_list[v].field_type);
                }
            }

            // If field_type == "face-centered"
            if (strcmp(field_list[v].field_type, "face-centered") == 0) {
                // (8) Raise warning if field_type = "face-centered", and data_ptr is not set == NULL.
                if (grid.field_data[v].data_ptr == NULL) {
                    YT_ABORT("Grid [%ld], field_data [%s], field_type [%s], data_ptr is NULL, not set yet!", grid.id,
                             field_list[v].field_name, field_list[v].field_type);
                } else {
                    // (9) If data_ptr != NULL, then data_dimensions > 0
                    for (int d = 0; d < 3; d++) {
                        if (grid.field_data[v].data_dimensions[d] <= 0) {
                            YT_ABORT("Grid [%ld], field_data [%s], field_type [%s], data_dimensions[%d] == %d <= 0, "
                                     "should be > 0!\n",
                                     grid.id, field_list[v].field_name, field_list[v].field_type, d,
                                     grid.field_data[v].data_dimensions[d]);
                        }
                    }
                }
            }

            // If field_type == "derived_func"
            if (strcmp(field_list[v].field_type, "derived_func") == 0) {
                // (10) If data_ptr != NULL, then data_dimensions > 0
                if (grid.field_data[v].data_ptr != NULL) {
                    for (int d = 0; d < 3; d++) {
                        if (grid.field_data[v].data_dimensions[d] <= 0) {
                            YT_ABORT("Grid [%ld], field_data [%s], field_type [%s], data_dimensions[%d] == %d <= 0, "
                                     "should be > 0!\n",
                                     grid.id, field_list[v].field_name, field_list[v].field_type, d,
                                     grid.field_data[v].data_dimensions[d]);
                        }
                    }
                }
            }
        }
    }

    return YT_SUCCESS;
}

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
    if (param_yt.magnetic_unit == DBL_UNDEFINED) log_warning("\"%s\" has not been set!\n", "magnetic_unit");

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

//-------------------------------------------------------------------------------------------------------
// Function    :  check_yt_grid
// Description :  Check yt_grid struct
//
// Note        :  1. This function does not perform checks that depend on the input
//                   YT parameters (e.g., whether left_edge lies within the simulation domain)
//                   ==> These checks are performed in check_grid()
//                2. If check needs information other than this grid's info, it will be done in check_grid.
//
// Parameter   :  const yt_grid &grid
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_yt_grid(const yt_grid& grid) {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (int d = 0; d < 3; d++) {
        if (grid.left_edge[d] == DBL_UNDEFINED)
            YT_ABORT("\"%s[%d]\" has not been set for grid id [%ld]!\n", "left_edge", d, grid.id);
        if (grid.right_edge[d] == DBL_UNDEFINED)
            YT_ABORT("\"%s[%d]\" has not been set for grid id [%ld]!\n", "right_edge", d, grid.id);
    }
    for (int d = 0; d < 3; d++) {
        if (grid.grid_dimensions[d] == INT_UNDEFINED)
            YT_ABORT("\"%s[%d]\" has not been set for grid id [%ld]!\n", "grid_dimensions", d, grid.id);
    }
    if (grid.id == LNG_UNDEFINED) YT_ABORT("\"%s\" has not been set for grid id [%ld]!\n", "id", grid.id);
    if (grid.parent_id == LNG_UNDEFINED) YT_ABORT("\"%s\" has not been set for grid id [%ld]!\n", "parent_id", grid.id);
    if (grid.level == INT_UNDEFINED) YT_ABORT("\"%s\" has not been set for grid id [%ld]!\n", "level", grid.id);
    if (grid.proc_num == INT_UNDEFINED) YT_ABORT("\"%s\" has not been set for grid id [%ld]!\n", "proc_num", grid.id);
    for (int d = 0; d < 3; d++) {
        if (grid.grid_dimensions[d] <= 0)
            YT_ABORT("\"%s[%d]\" == %d <= 0 for grid [%ld]!\n", "grid_dimensions", d, grid.grid_dimensions[d], grid.id);
    }
    if (grid.level < 0) YT_ABORT("\"%s\" == %d < 0 for grid [%ld]!\n", "level", grid.level, grid.id);

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  check_yt_attribute
// Description :  Check yt_attribute struct
//
// Note        :  1. Validate data member value in one yt_attribute struct.
//                  (1) attr_name is set, and != nullptr.
//                  (2) attr_dtype is one of yt_dtype.
//
// Parameter   :  const yt_attribute &attr
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_yt_attribute(const yt_attribute& attr) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // attr_name is set
    if (attr.attr_name == nullptr) {
        YT_ABORT("attr_name is not set!\n");
    }

    // attr_dtype is one of yt_dtype
    bool valid = false;
    for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
        yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
        if (attr.attr_dtype == dtype) {
            valid = true;
            break;
        }
    }
    if (!valid) {
        YT_ABORT("In attr [%s], unknown attr_dtype!\n", attr.attr_name);
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  check_yt_particle
// Description :  Check yt_particle struct
//
// Note        :  1. Validate data member value in one yt_particle struct.
//                  (1) par_type is set != NULL
//                  (2) attr_list is set != NULL
//                  (3) num_attr should > 0
//                  (4) attr_name in attr_list should be unique
//                  (5) call yt_attribute validate for each attr_list elements.
//                  (6) raise error if coor_x, coor_y, coor_z is not set.
//                  (7) log_warning if get_par_attr not set.
//               2. Used inside check_particle_list().
//
// Parameter   :  const yt_particle &particle
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_yt_particle(const yt_particle& particle) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // par_type should be set
    if (particle.par_type == nullptr) {
        YT_ABORT("par_type is not set!\n");
    }

    // attr_list != NULL
    if (particle.attr_list == nullptr) {
        YT_ABORT("Particle type [ %s ], attr_list not set properly!\n", particle.par_type);
    }
    // num_attr should > 0
    if (particle.num_attr < 0) {
        YT_ABORT("Particle type [ %s ], num_attr not set properly!\n", particle.par_type);
    }

    // call yt_attribute validate for each attr_list elements.
    for (int i = 0; i < particle.num_attr; i++) {
        if (!(check_yt_attribute(particle.attr_list[i]))) {
            YT_ABORT("Particle type [ %s ], attr_list element [ %d ] not set properly!\n", particle.par_type, i);
        }
    }

    // attr_name in attr_list should be unique
    for (int i = 0; i < particle.num_attr; i++) {
        for (int j = i + 1; j < particle.num_attr; j++) {
            if (strcmp(particle.attr_list[i].attr_name, particle.attr_list[j].attr_name) == 0) {
                YT_ABORT("Particle type [ %s ], attr_list element [ %d ] and [ %d ] have same attr_name, expect them "
                         "to be unique!\n",
                         particle.par_type, i, j);
            }
        }
    }

    // if didn't input coor_x/y/z, yt cannot function properly for this particle.
    if (particle.coor_x == nullptr) {
        YT_ABORT("Particle type [ %s ], attribute name of coordinate x coor_x not set!\n", particle.par_type);
    }
    if (particle.coor_y == nullptr) {
        YT_ABORT("Particle type [ %s ], attribute name of coordinate y coor_y not set!\n", particle.par_type);
    }
    if (particle.coor_z == nullptr) {
        YT_ABORT("Particle type [ %s ], attribute name of coordinate z coor_z not set!\n", particle.par_type);
    }

    // if didn't input get_par_attr, yt cannot function properly for this particle.
    if (particle.get_par_attr == nullptr) {
        log_warning("Particle type [ %s ], function that gets particle attribute get_par_attr not set!\n",
                    particle.par_type);
    }

    return YT_SUCCESS;
}