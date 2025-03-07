#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "python_controller.h"
#include "timer.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

static int CheckYtParamYt(const yt_param_yt& param_yt);
static int PrintYtParamYt(const yt_param_yt& param_yt);

/**
 * \defgroup api_yt_set_Parameters libyt API: yt_set_Parameters
 * \fn int yt_set_Parameters(yt_param_yt* input_param_yt)
 * \brief Set YT-specific parameters and parameters for initializing AMR data structure
 * \details
 * 1. Store yt relevant data to libyt Python module `libyt.param_yt`.
 * 2. Initialize AMR data structure and storage.
 * 3. Should be called after \ref yt_initialize.
 *
 * @param input_param_yt[in] YT-specific parameters and parameters for initializing AMR
 *                           data structure
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 */
int yt_set_Parameters(yt_param_yt* input_param_yt) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // check if libyt has been initialized
  if (!LibytProcessControl::Get().libyt_initialized_) {
    YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
  }

  // check if libyt has free all the resource in previous inline-analysis
  if (LibytProcessControl::Get().need_free_) {
    logging::LogWarning(
        "Please invoke yt_free() before calling %s() for next iteration!\n",
        __FUNCTION__);
    YT_ABORT("Overwrite existing parameters may leads to memory leak, please called "
             "yt_free() first!\n");
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
  if (CheckYtParamYt(*input_param_yt)) {
    logging::LogDebug("Validating YT parameters ... done\n");
  } else {
    YT_ABORT("Validating YT parameters ... failed\n");
  }

  // print out all parameters
  logging::LogDebug("List of YT parameters:\n");
  PrintYtParamYt(*input_param_yt);

  // store user-provided parameters to a libyt internal variable
  LibytProcessControl::Get().param_yt_ = *input_param_yt;
  yt_param_yt& param_yt = LibytProcessControl::Get().param_yt_;

  // Set up DataStructureAmr
  DataStructureOutput status;
  if (param_yt.num_par_types > 0) {
    status = LibytProcessControl::Get().data_structure_amr_.AllocateStorage(
        param_yt.num_grids,
        param_yt.num_grids_local,
        param_yt.num_fields,
        param_yt.num_par_types,
        param_yt.par_type_list,
        param_yt.index_offset,
        LibytProcessControl::Get().param_libyt_.check_data);
  } else {
    status = LibytProcessControl::Get().data_structure_amr_.AllocateStorage(
        param_yt.num_grids,
        param_yt.num_grids_local,
        param_yt.num_fields,
        0,
        nullptr,
        param_yt.index_offset,
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
    sprintf(fig_basename,
            "%s%09ld",
            input_param_yt->fig_basename,
            LibytProcessControl::Get().param_libyt_.counter);
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

  py_param_yt["domain_dimensions"] = pybind11::make_tuple(param_yt.domain_dimensions[0],
                                                          param_yt.domain_dimensions[1],
                                                          param_yt.domain_dimensions[2]);
  py_param_yt["domain_left_edge"] = pybind11::make_tuple(param_yt.domain_left_edge[0],
                                                         param_yt.domain_left_edge[1],
                                                         param_yt.domain_left_edge[2]);
  py_param_yt["domain_right_edge"] = pybind11::make_tuple(param_yt.domain_right_edge[0],
                                                          param_yt.domain_right_edge[1],
                                                          param_yt.domain_right_edge[2]);
  py_param_yt["periodicity"] = pybind11::make_tuple(
      param_yt.periodicity[0], param_yt.periodicity[1], param_yt.periodicity[2]);
#else
  // export data to libyt.param_yt
  // strings
  python_controller::AddStringToDict(
      LibytProcessControl::Get().py_param_yt_, "frontend", param_yt.frontend);
  python_controller::AddStringToDict(
      LibytProcessControl::Get().py_param_yt_, "fig_basename", param_yt.fig_basename);

  // scalars
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "current_time", param_yt.current_time);
  python_controller::AddScalarToDict(LibytProcessControl::Get().py_param_yt_,
                                     "current_redshift",
                                     param_yt.current_redshift);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "omega_lambda", param_yt.omega_lambda);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "omega_matter", param_yt.omega_matter);
  python_controller::AddScalarToDict(LibytProcessControl::Get().py_param_yt_,
                                     "hubble_constant",
                                     param_yt.hubble_constant);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "length_unit", param_yt.length_unit);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "mass_unit", param_yt.mass_unit);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "time_unit", param_yt.time_unit);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "velocity_unit", param_yt.velocity_unit);

  if (param_yt.magnetic_unit == DBL_UNDEFINED) {
    python_controller::AddScalarToDict(
        LibytProcessControl::Get().py_param_yt_, "magnetic_unit", 1);
  } else {
    python_controller::AddScalarToDict(
        LibytProcessControl::Get().py_param_yt_, "magnetic_unit", param_yt.magnetic_unit);
  }

  python_controller::AddScalarToDict(LibytProcessControl::Get().py_param_yt_,
                                     "cosmological_simulation",
                                     param_yt.cosmological_simulation);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "dimensionality", param_yt.dimensionality);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "refine_by", param_yt.refine_by);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "index_offset", param_yt.index_offset);
  python_controller::AddScalarToDict(
      LibytProcessControl::Get().py_param_yt_, "num_grids", param_yt.num_grids);

  // vectors (stored as Python tuples)
  python_controller::AddVectorNToDict(LibytProcessControl::Get().py_param_yt_,
                                      "domain_left_edge",
                                      3,
                                      param_yt.domain_left_edge);
  python_controller::AddVectorNToDict(LibytProcessControl::Get().py_param_yt_,
                                      "domain_right_edge",
                                      3,
                                      param_yt.domain_right_edge);
  python_controller::AddVectorNToDict(
      LibytProcessControl::Get().py_param_yt_, "periodicity", 3, param_yt.periodicity);
  python_controller::AddVectorNToDict(LibytProcessControl::Get().py_param_yt_,
                                      "domain_dimensions",
                                      3,
                                      param_yt.domain_dimensions);
#endif  // #ifdef USE_PYBIND11

  logging::LogDebug("Inserting YT parameters to libyt.param_yt ... done\n");

  // If the above all works like charm.
  LibytProcessControl::Get().param_yt_set_ = true;
  LibytProcessControl::Get().need_free_ = true;
  logging::LogInfo("Setting YT parameters ... done.\n");

  return YT_SUCCESS;

}  // FUNCTION : yt_set_Parameters

//-------------------------------------------------------------------------------------------------------
// Function    :  CheckYtParamYt
// Description :  Check yt_param_yt struct
//
// Note        :  1. Cosmological parameters are checked only if cosmological_simulation
// == 1
//                2. Only check if data are set properly, does not alter them.
//
// Parameter   :  const yt_param_yt &param_yt
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
static int CheckYtParamYt(const yt_param_yt& param_yt) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (param_yt.frontend == NULL) YT_ABORT("\"%s\" has not been set!\n", "frontend");
  for (int d = 0; d < 3; d++) {
    if (param_yt.domain_left_edge[d] == DBL_UNDEFINED)
      YT_ABORT("\"%s[%d]\" has not been set!\n", "domain_left_edge", d);
    if (param_yt.domain_right_edge[d] == DBL_UNDEFINED)
      YT_ABORT("\"%s[%d]\" has not been set!\n", "domain_right_edge", d);
  }
  if (param_yt.current_time == DBL_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "current_time");
  if (param_yt.cosmological_simulation == INT_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "cosmological_simulation");
  if (param_yt.cosmological_simulation == 1) {
    if (param_yt.current_redshift == DBL_UNDEFINED)
      YT_ABORT("\"%s\" has not been set!\n", "current_redshift");
    if (param_yt.omega_lambda == DBL_UNDEFINED)
      YT_ABORT("\"%s\" has not been set!\n", "omega_lambda");
    if (param_yt.omega_matter == DBL_UNDEFINED)
      YT_ABORT("\"%s\" has not been set!\n", "omega_matter");
    if (param_yt.hubble_constant == DBL_UNDEFINED)
      YT_ABORT("\"%s\" has not been set!\n", "hubble_constant");
  }
  if (param_yt.length_unit == DBL_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "length_unit");
  if (param_yt.mass_unit == DBL_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "mass_unit");
  if (param_yt.time_unit == DBL_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "time_unit");
  if (param_yt.velocity_unit == DBL_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "velocity_unit");
  if (param_yt.magnetic_unit == DBL_UNDEFINED)
    logging::LogWarning("\"%s\" has not been set!\n", "magnetic_unit");

  for (int d = 0; d < 3; d++) {
    if (param_yt.periodicity[d] == INT_UNDEFINED)
      YT_ABORT("\"%s[%d]\" has not been set!\n", "periodicity", d);
    if (param_yt.domain_dimensions[d] == INT_UNDEFINED)
      YT_ABORT("\"%s[%d]\" has not been set!\n", "domain_dimensions", d);
  }
  if (param_yt.dimensionality == INT_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "dimensionality");
  if (param_yt.refine_by == INT_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "refine_by");
  if (param_yt.num_grids == LNG_UNDEFINED)
    YT_ABORT("\"%s\" has not been set!\n", "num_grids");
  if (param_yt.num_par_types > 0 && param_yt.par_type_list == NULL)
    YT_ABORT("Particle type info par_type_list has not been set!\n");
  if (param_yt.num_par_types < 0 && param_yt.par_type_list != NULL)
    YT_ABORT("Particle type info num_par_types has not been set!\n");
  for (int s = 0; s < param_yt.num_par_types; s++) {
    if (param_yt.par_type_list[s].par_type == NULL ||
        param_yt.par_type_list[s].num_attr < 0)
      YT_ABORT("par_type_list element [ %d ] is not set properly!\n", s);
  }

  return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  PrintYtParamYt
// Description :  Print yt_param_yt struct if verbose level >= YT_VERBOSE_DEBUG
//
// Parameter   :  const yt_param_yt &param_yt
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
static int PrintYtParamYt(const yt_param_yt& param_yt) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (LibytProcessControl::Get().param_libyt_.verbose < YT_VERBOSE_DEBUG)
    return YT_SUCCESS;

  const int width_scalar = 25;
  const int width_vector = width_scalar - 3;

  if (param_yt.frontend != nullptr)
    logging::LogDebug("   %-*s = %s\n", width_scalar, "frontend", param_yt.frontend);
  if (param_yt.fig_basename != nullptr)
    logging::LogDebug(
        "   %-*s = %s\n", width_scalar, "fig_basename", param_yt.fig_basename);
  for (int d = 0; d < 3; d++) {
    logging::LogDebug("   %-*s[%d] = %13.7e\n",
                      width_vector,
                      "domain_left_edge",
                      d,
                      param_yt.domain_left_edge[d]);
  }
  for (int d = 0; d < 3; d++) {
    logging::LogDebug("   %-*s[%d] = %13.7e\n",
                      width_vector,
                      "domain_right_edge",
                      d,
                      param_yt.domain_right_edge[d]);
  }
  logging::LogDebug(
      "   %-*s = %13.7e\n", width_scalar, "current_time", param_yt.current_time);
  logging::LogDebug("   %-*s = %d\n",
                    width_scalar,
                    "cosmological_simulation",
                    param_yt.cosmological_simulation);
  if (param_yt.cosmological_simulation) {
    logging::LogDebug("   %-*s = %13.7e\n",
                      width_scalar,
                      "current_redshift",
                      param_yt.current_redshift);
    logging::LogDebug(
        "   %-*s = %13.7e\n", width_scalar, "omega_lambda", param_yt.omega_lambda);
    logging::LogDebug(
        "   %-*s = %13.7e\n", width_scalar, "omega_matter", param_yt.omega_matter);
    logging::LogDebug(
        "   %-*s = %13.7e\n", width_scalar, "hubble_constant", param_yt.hubble_constant);
  }

  logging::LogDebug(
      "   %-*s = %13.7e\n", width_scalar, "length_unit", param_yt.length_unit);
  logging::LogDebug("   %-*s = %13.7e\n", width_scalar, "mass_unit", param_yt.mass_unit);
  logging::LogDebug("   %-*s = %13.7e\n", width_scalar, "time_unit", param_yt.time_unit);
  logging::LogDebug(
      "   %-*s = %13.7e\n", width_scalar, "velocity_unit", param_yt.velocity_unit);
  if (param_yt.magnetic_unit == DBL_UNDEFINED)
    logging::LogDebug("   %-*s = %s\n",
                      width_scalar,
                      "magnetic_unit",
                      "NOT SET, and will be set to 1.");
  else
    logging::LogDebug(
        "   %-*s = %13.7e\n", width_scalar, "magnetic_unit", param_yt.magnetic_unit);

  for (int d = 0; d < 3; d++) {
    logging::LogDebug(
        "   %-*s[%d] = %d\n", width_vector, "periodicity", d, param_yt.periodicity[d]);
  }
  for (int d = 0; d < 3; d++) {
    logging::LogDebug("   %-*s[%d] = %d\n",
                      width_vector,
                      "domain_dimensions",
                      d,
                      param_yt.domain_dimensions[d]);
  }
  logging::LogDebug(
      "   %-*s = %d\n", width_scalar, "dimensionality", param_yt.dimensionality);
  logging::LogDebug("   %-*s = %d\n", width_scalar, "refine_by", param_yt.refine_by);
  logging::LogDebug(
      "   %-*s = %d\n", width_scalar, "index_offset", param_yt.index_offset);
  logging::LogDebug("   %-*s = %ld\n", width_scalar, "num_grids", param_yt.num_grids);

  logging::LogDebug("   %-*s = %ld\n", width_scalar, "num_fields", param_yt.num_fields);
  logging::LogDebug(
      "   %-*s = %ld\n", width_scalar, "num_par_types", param_yt.num_par_types);
  for (int s = 0; s < param_yt.num_par_types; s++) {
    if (param_yt.par_type_list != nullptr)
      logging::LogDebug("   %-*s[%d] = (type=\"%s\", num_attr=%d)\n",
                        width_vector,
                        "par_type_list",
                        s,
                        param_yt.par_type_list[s].par_type,
                        param_yt.par_type_list[s].num_attr);
    else
      logging::LogDebug("   %-*s[%d] = (type=\"%s\", num_attr=%d)\n",
                        width_vector,
                        "par_type_list",
                        s,
                        "NULL",
                        param_yt.par_type_list[s].num_attr);
  }
  logging::LogDebug(
      "   %-*s = %ld\n", width_scalar, "num_grids_local", param_yt.num_grids_local);

  return YT_SUCCESS;
}
