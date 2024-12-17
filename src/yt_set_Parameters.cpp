#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_set_Parameters
// Description :  Set YT-specific parameters
//
// Note        :  1. Store yt relavent data in input "param_yt" to libyt.param_yt. Note that not all the
//                   data are passed in to python.
//                   To avoid user free the passed in array par_type_list, we initialize particle_list
//                   (needs info from par_type_list) right away. If num_par_types > 0.
//                   To make loading field_list and particle_list more systematic, we will allocate both
//                   field_list (if num_fields>0 ) and particle_list (if num_par_types>0) here.
//                2. Should be called after yt_initialize().
//                3. Check the validation of the data in param_yt.
//                4. Initialize python hierarchy allocate_hierarchy() and particle_list.
//                5. Gather each ranks number of local grids, we need this info in yt_commit().
//
// Parameter   :  param_yt : Structure storing YT-specific parameters that will later pass to YT, and
//                           other relavent data.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_set_Parameters(yt_param_yt* input_param_yt) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if libyt has free all the resource in previous inline-analysis
    if (!LibytProcessControl::Get().free_gridsPtr) {
        log_warning("Please invoke yt_free() before calling %s() for next iteration!\n", __FUNCTION__);
        YT_ABORT("Overwrite existing parameters may leads to memory leak, please called yt_free() first!\n");
    }

    log_info("Setting YT parameters ...\n");

    // reset all cosmological parameters to zero for non-cosmological datasets
    if (!input_param_yt->cosmological_simulation) {
        input_param_yt->current_redshift = 0.0;
        input_param_yt->omega_lambda = 0.0;
        input_param_yt->omega_matter = 0.0;
        input_param_yt->hubble_constant = 0.0;
    }

    // check if all parameters have been set properly
    if (check_yt_param_yt(*input_param_yt))
        log_debug("Validating YT parameters ... done\n");
    else
        YT_ABORT("Validating YT parameters ... failed\n");

    // print out all parameters
    log_debug("List of YT parameters:\n");
    print_yt_param_yt(*input_param_yt);

    // store user-provided parameters to a libyt internal variable
    // ==> must do this before calling allocate_hierarchy() since it will need "param_yt.num_grids"
    // ==> must do this before setting the default figure base name since it will overwrite param_yt.fig_basename
    LibytProcessControl::Get().param_yt_ = *input_param_yt;
    yt_param_yt& param_yt = LibytProcessControl::Get().param_yt_;

    // set the default figure base name if it's not set by users.
    // append LibytProcessControl::Get().param_libyt_.counter to prevent over-written
    char fig_basename[1000];
    if (input_param_yt->fig_basename == NULL) {
        sprintf(fig_basename, "Fig%09ld", LibytProcessControl::Get().param_libyt_.counter);
        param_yt.fig_basename = fig_basename;
    } else {
        sprintf(fig_basename, "%s%09ld", input_param_yt->fig_basename, LibytProcessControl::Get().param_libyt_.counter);
        param_yt.fig_basename = fig_basename;
    }

#ifdef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_param_yt = libyt.attr("param_yt");

    py_param_yt["frontend"] = param_yt.frontend;
    py_param_yt["fig_basename"] = param_yt.fig_basename;
    py_param_yt["current_time"] = param_yt.current_time;
    py_param_yt["current_redshift"] = param_yt.current_redshift;
    py_param_yt["omega_lambda"] = param_yt.omega_lambda;
    py_param_yt["omega_matter"] = param_yt.omega_matter;
    py_param_yt["hubble_constant"] = param_yt.hubble_constant;
    py_param_yt["length_unit"] = param_yt.length_unit;
    py_param_yt["mass_unit"] = param_yt.mass_unit;
    py_param_yt["time_unit"] = param_yt.time_unit;
    py_param_yt["velocity_unit"] = param_yt.velocity_unit;
    py_param_yt["cosmological_simulation"] = param_yt.cosmological_simulation;
    py_param_yt["dimensionality"] = param_yt.dimensionality;
    py_param_yt["refine_by"] = param_yt.refine_by;
    py_param_yt["index_offset"] = param_yt.index_offset;
    py_param_yt["num_grids"] = param_yt.num_grids;

    if (param_yt.magnetic_unit == DBL_UNDEFINED) {
        py_param_yt["magnetic_unit"] = 1.0;
    } else {
        py_param_yt["magnetic_unit"] = param_yt.magnetic_unit;
    }

    py_param_yt["domain_dimensions"] = pybind11::make_tuple(
        param_yt.domain_dimensions[0], param_yt.domain_dimensions[1], param_yt.domain_dimensions[2]);
    py_param_yt["domain_left_edge"] =
        pybind11::make_tuple(param_yt.domain_left_edge[0], param_yt.domain_left_edge[1], param_yt.domain_left_edge[2]);
    py_param_yt["domain_right_edge"] = pybind11::make_tuple(
        param_yt.domain_right_edge[0], param_yt.domain_right_edge[1], param_yt.domain_right_edge[2]);
    py_param_yt["periodicity"] =
        pybind11::make_tuple(param_yt.periodicity[0], param_yt.periodicity[1], param_yt.periodicity[2]);
#else
    // export data to libyt.param_yt
    // strings
    add_dict_string(LibytProcessControl::Get().py_param_yt_, "frontend", param_yt.frontend);
    add_dict_string(LibytProcessControl::Get().py_param_yt_, "fig_basename", param_yt.fig_basename);

    // scalars
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "current_time", param_yt.current_time);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "current_redshift", param_yt.current_redshift);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "omega_lambda", param_yt.omega_lambda);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "omega_matter", param_yt.omega_matter);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "hubble_constant", param_yt.hubble_constant);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "length_unit", param_yt.length_unit);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "mass_unit", param_yt.mass_unit);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "time_unit", param_yt.time_unit);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "velocity_unit", param_yt.velocity_unit);

    if (param_yt.magnetic_unit == DBL_UNDEFINED) {
        add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "magnetic_unit", 1);
    } else {
        add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "magnetic_unit", param_yt.magnetic_unit);
    }

    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "cosmological_simulation",
                    param_yt.cosmological_simulation);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "dimensionality", param_yt.dimensionality);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "refine_by", param_yt.refine_by);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "index_offset", param_yt.index_offset);
    add_dict_scalar(LibytProcessControl::Get().py_param_yt_, "num_grids", param_yt.num_grids);

    // vectors (stored as Python tuples)
    add_dict_vector_n(LibytProcessControl::Get().py_param_yt_, "domain_left_edge", 3, param_yt.domain_left_edge);
    add_dict_vector_n(LibytProcessControl::Get().py_param_yt_, "domain_right_edge", 3, param_yt.domain_right_edge);
    add_dict_vector_n(LibytProcessControl::Get().py_param_yt_, "periodicity", 3, param_yt.periodicity);
    add_dict_vector_n(LibytProcessControl::Get().py_param_yt_, "domain_dimensions", 3, param_yt.domain_dimensions);
#endif  // #ifdef USE_PYBIND11

    log_debug("Inserting YT parameters to libyt.param_yt ... done\n");

    // if num_fields > 0, which means we want to load fields
    if (param_yt.num_fields > 0) {
        LibytProcessControl::Get().data_structure_amr_.field_list_ = new yt_field[param_yt.num_fields];
    } else {
        LibytProcessControl::Get().data_structure_amr_.field_list_ = nullptr;
        LibytProcessControl::Get().get_fieldsPtr = true;
    }

    // if num_par_types > 0, which means want to load particle
    if (param_yt.num_par_types > 0) {
        // Initialize and setup yt_particle *particle_list in param_yt.particle_list,
        // to avoid user freeing yt_par_type *par_type_list.
        yt_particle* particle_list = new yt_particle[param_yt.num_par_types];
        for (int s = 0; s < param_yt.num_par_types; s++) {
            particle_list[s].par_type = param_yt.par_type_list[s].par_type;
            particle_list[s].num_attr = param_yt.par_type_list[s].num_attr;
            particle_list[s].attr_list = new yt_attribute[particle_list[s].num_attr];
        }
        LibytProcessControl::Get().data_structure_amr_.particle_list_ = particle_list;
    } else {
        // don't need to load particle, set as NULL.
        LibytProcessControl::Get().data_structure_amr_.particle_list_ = nullptr;
        LibytProcessControl::Get().get_particlesPtr = true;
    }

    // if num_grids_local <= 0, which means this rank doesn't need to load in grids_local info.
    if (param_yt.num_grids_local <= 0) {
        LibytProcessControl::Get().data_structure_amr_.grids_local_ = nullptr;
        LibytProcessControl::Get().get_gridsPtr = true;
    }

    // Make sure param_yt.num_grids_local is set,
    // and if param_yt.num_grids_local < 0, set it = 0
    if (param_yt.num_grids_local < 0) {
        // Prevent input long type and exceed int storage
        log_warning(
            "Number of local grids = %d at MPI rank %d, probably because of exceeding int storage or wrong input!\n",
            param_yt.num_grids_local, LibytProcessControl::Get().mpi_rank_);

        // if < 0, set it to 0, to avoid adding negative num_grids_local when checking num_grids.
        param_yt.num_grids_local = 0;
    }

#ifndef SERIAL_MODE
    // Gather number of local grids in each MPI rank
    LibytProcessControl::Get().data_structure_amr_.all_num_grids_local_ = new int[LibytProcessControl::Get().mpi_size_];
    CommMpi::SetAllNumGridsLocal(LibytProcessControl::Get().data_structure_amr_.all_num_grids_local_,
                                 param_yt.num_grids_local);

    // Check that sum of num_grids_local_MPI is equal to num_grids (total number of grids), abort if not.
    if (LibytProcessControl::Get().param_libyt_.check_data) {
        if (check_sum_num_grids_local_MPI(LibytProcessControl::Get().mpi_size_,
                                          LibytProcessControl::Get().data_structure_amr_.all_num_grids_local_) !=
            YT_SUCCESS) {
            YT_ABORT("Check sum of local grids in each MPI rank failed in %s!\n", __FUNCTION__);
        }
    }
#endif

    // If the above all works like charm.
    LibytProcessControl::Get().param_yt_set = true;
    LibytProcessControl::Get().free_gridsPtr = false;
    log_info("Setting YT parameters ... done.\n");

    return YT_SUCCESS;

}  // FUNCTION : yt_set_Parameters
