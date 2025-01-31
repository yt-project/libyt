#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "python_controller.h"
#include "timer.h"
#include "yt_prototype.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_set_Parameters
// Description :  Set YT-specific parameters
//
// Note        :  1. Store yt relevant data in input "param_yt" to libyt.param_yt. Note that not all the
//                   data are passed in to python.
//                2. To avoid user free the passed in array par_type_list, we initialize particle_list
//                   (needs info from par_type_list) right away if num_par_types > 0.
//                   The Amr data structure and storage is initialized here.
//                   TODO: The Api name is bad, should fix it in libyt-v1.0.
//                3. Should be called after yt_initialize().
//                4. Check the validation of the data in param_yt.
//                5. Initialize python hierarchy allocate_hierarchy() and particle_list.
//                6. Gather each ranks number of local grids, we need this info in yt_commit().
//
// Parameter   :  param_yt : Structure storing YT-specific parameters that will later pass to YT, and
//                           other relevant data.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_set_Parameters(yt_param_yt* input_param_yt) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized_) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if libyt has free all the resource in previous inline-analysis
    if (LibytProcessControl::Get().need_free_) {
        logging::LogWarning("Please invoke yt_free() before calling %s() for next iteration!\n", __FUNCTION__);
        YT_ABORT("Overwrite existing parameters may leads to memory leak, please called yt_free() first!\n");
    }

    logging::LogInfo("Setting YT parameters ...\n");

    // reset all cosmological parameters to zero for non-cosmological datasets
    if (!input_param_yt->cosmological_simulation) {
        input_param_yt->current_redshift = 0.0;
        input_param_yt->omega_lambda = 0.0;
        input_param_yt->omega_matter = 0.0;
        input_param_yt->hubble_constant = 0.0;
    }

    // check if all parameters have been set properly
    if (check_yt_param_yt(*input_param_yt))
        logging::LogDebug("Validating YT parameters ... done\n");
    else
        YT_ABORT("Validating YT parameters ... failed\n");

    // print out all parameters
    logging::LogDebug("List of YT parameters:\n");
    print_yt_param_yt(*input_param_yt);

    // store user-provided parameters to a libyt internal variable
    LibytProcessControl::Get().param_yt_ = *input_param_yt;
    yt_param_yt& param_yt = LibytProcessControl::Get().param_yt_;

    // Set up DataStructureAmr
    DataStructureOutput status;
    if (param_yt.num_par_types > 0) {
        status = LibytProcessControl::Get().data_structure_amr_.AllocateStorage(
            param_yt.num_grids, param_yt.num_grids_local, param_yt.num_fields, param_yt.num_par_types,
            param_yt.par_type_list, param_yt.index_offset, LibytProcessControl::Get().param_libyt_.check_data);
    } else {
        status = LibytProcessControl::Get().data_structure_amr_.AllocateStorage(
            param_yt.num_grids, param_yt.num_grids_local, param_yt.num_fields, 0, nullptr, param_yt.index_offset,
            LibytProcessControl::Get().param_libyt_.check_data);
    }

    if (status.status != DataStructureStatus::kDataStructureSuccess) {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    } else {
        logging::LogDebug("Allocate storage for amr data structure ... done\n");
    }

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
    AddStringToDict(LibytProcessControl::Get().py_param_yt_, "frontend", param_yt.frontend);
    AddStringToDict(LibytProcessControl::Get().py_param_yt_, "fig_basename", param_yt.fig_basename);

    // scalars
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "current_time", param_yt.current_time);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "current_redshift", param_yt.current_redshift);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "omega_lambda", param_yt.omega_lambda);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "omega_matter", param_yt.omega_matter);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "hubble_constant", param_yt.hubble_constant);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "length_unit", param_yt.length_unit);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "mass_unit", param_yt.mass_unit);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "time_unit", param_yt.time_unit);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "velocity_unit", param_yt.velocity_unit);

    if (param_yt.magnetic_unit == DBL_UNDEFINED) {
        AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "magnetic_unit", 1);
    } else {
        AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "magnetic_unit", param_yt.magnetic_unit);
    }

    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "cosmological_simulation",
                    param_yt.cosmological_simulation);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "dimensionality", param_yt.dimensionality);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "refine_by", param_yt.refine_by);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "index_offset", param_yt.index_offset);
    AddScalarToDict(LibytProcessControl::Get().py_param_yt_, "num_grids", param_yt.num_grids);

    // vectors (stored as Python tuples)
    AddVectorNToDict(LibytProcessControl::Get().py_param_yt_, "domain_left_edge", 3, param_yt.domain_left_edge);
    AddVectorNToDict(LibytProcessControl::Get().py_param_yt_, "domain_right_edge", 3, param_yt.domain_right_edge);
    AddVectorNToDict(LibytProcessControl::Get().py_param_yt_, "periodicity", 3, param_yt.periodicity);
    AddVectorNToDict(LibytProcessControl::Get().py_param_yt_, "domain_dimensions", 3, param_yt.domain_dimensions);
#endif  // #ifdef USE_PYBIND11

    logging::LogDebug("Inserting YT parameters to libyt.param_yt ... done\n");

    // If the above all works like charm.
    LibytProcessControl::Get().param_yt_set_ = true;
    LibytProcessControl::Get().need_free_ = true;
    logging::LogInfo("Setting YT parameters ... done.\n");

    return YT_SUCCESS;

}  // FUNCTION : yt_set_Parameters
