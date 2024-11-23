#include "LibytProcessControl.h"
#include "libyt.h"
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
int yt_set_Parameters(yt_param_yt* param_yt) {
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
    if (!param_yt->cosmological_simulation) {
        param_yt->current_redshift = param_yt->omega_lambda = param_yt->omega_matter = param_yt->hubble_constant = 0.0;
    }

    // check if all parameters have been set properly
    if (check_yt_param_yt(*param_yt))
        log_debug("Validating YT parameters ... done\n");
    else
        YT_ABORT("Validating YT parameters ... failed\n");

    // print out all parameters
    log_debug("List of YT parameters:\n");
    print_yt_param_yt(*param_yt);

    // store user-provided parameters to a libyt internal variable
    // ==> must do this before calling allocate_hierarchy() since it will need "g_param_yt.num_grids"
    // ==> must do this before setting the default figure base name since it will overwrite g_param_yt.fig_basename
    g_param_yt = *param_yt;

    // set the default figure base name if it's not set by users.
    // append g_param_libyt.counter to prevent over-written
    char fig_basename[1000];
    if (param_yt->fig_basename == NULL) {
        sprintf(fig_basename, "Fig%09ld", g_param_libyt.counter);
        g_param_yt.fig_basename = fig_basename;
    } else {
        sprintf(fig_basename, "%s%09ld", param_yt->fig_basename, g_param_libyt.counter);
        g_param_yt.fig_basename = fig_basename;
    }

#ifdef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_param_yt = libyt.attr("param_yt");

    py_param_yt["frontend"] = g_param_yt.frontend;
    py_param_yt["fig_basename"] = g_param_yt.fig_basename;
    py_param_yt["current_time"] = g_param_yt.current_time;
    py_param_yt["current_redshift"] = g_param_yt.current_redshift;
    py_param_yt["omega_lambda"] = g_param_yt.omega_lambda;
    py_param_yt["omega_matter"] = g_param_yt.omega_matter;
    py_param_yt["hubble_constant"] = g_param_yt.hubble_constant;
    py_param_yt["length_unit"] = g_param_yt.length_unit;
    py_param_yt["mass_unit"] = g_param_yt.mass_unit;
    py_param_yt["time_unit"] = g_param_yt.time_unit;
    py_param_yt["velocity_unit"] = g_param_yt.velocity_unit;
    py_param_yt["cosmological_simulation"] = g_param_yt.cosmological_simulation;
    py_param_yt["dimensionality"] = g_param_yt.dimensionality;
    py_param_yt["refine_by"] = g_param_yt.refine_by;
    py_param_yt["index_offset"] = g_param_yt.index_offset;
    py_param_yt["num_grids"] = g_param_yt.num_grids;

    if (g_param_yt.magnetic_unit == DBL_UNDEFINED) {
        py_param_yt["magnetic_unit"] = 1.0;
    } else {
        py_param_yt["magnetic_unit"] = g_param_yt.magnetic_unit;
    }

    py_param_yt["domain_dimensions"] = pybind11::make_tuple(
        g_param_yt.domain_dimensions[0], g_param_yt.domain_dimensions[1], g_param_yt.domain_dimensions[2]);
    py_param_yt["domain_left_edge"] = pybind11::make_tuple(
        g_param_yt.domain_left_edge[0], g_param_yt.domain_left_edge[1], g_param_yt.domain_left_edge[2]);
    py_param_yt["domain_right_edge"] = pybind11::make_tuple(
        g_param_yt.domain_right_edge[0], g_param_yt.domain_right_edge[1], g_param_yt.domain_right_edge[2]);
    py_param_yt["periodicity"] =
        pybind11::make_tuple(g_param_yt.periodicity[0], g_param_yt.periodicity[1], g_param_yt.periodicity[2]);
#else
    // export data to libyt.param_yt
    // strings
    add_dict_string(g_py_param_yt, "frontend", g_param_yt.frontend);
    add_dict_string(g_py_param_yt, "fig_basename", g_param_yt.fig_basename);

    // scalars
    add_dict_scalar(g_py_param_yt, "current_time", g_param_yt.current_time);
    add_dict_scalar(g_py_param_yt, "current_redshift", g_param_yt.current_redshift);
    add_dict_scalar(g_py_param_yt, "omega_lambda", g_param_yt.omega_lambda);
    add_dict_scalar(g_py_param_yt, "omega_matter", g_param_yt.omega_matter);
    add_dict_scalar(g_py_param_yt, "hubble_constant", g_param_yt.hubble_constant);
    add_dict_scalar(g_py_param_yt, "length_unit", g_param_yt.length_unit);
    add_dict_scalar(g_py_param_yt, "mass_unit", g_param_yt.mass_unit);
    add_dict_scalar(g_py_param_yt, "time_unit", g_param_yt.time_unit);
    add_dict_scalar(g_py_param_yt, "velocity_unit", g_param_yt.velocity_unit);

    if (g_param_yt.magnetic_unit == DBL_UNDEFINED) {
        add_dict_scalar(g_py_param_yt, "magnetic_unit", 1);
    } else {
        add_dict_scalar(g_py_param_yt, "magnetic_unit", g_param_yt.magnetic_unit);
    }

    add_dict_scalar(g_py_param_yt, "cosmological_simulation", g_param_yt.cosmological_simulation);
    add_dict_scalar(g_py_param_yt, "dimensionality", g_param_yt.dimensionality);
    add_dict_scalar(g_py_param_yt, "refine_by", g_param_yt.refine_by);
    add_dict_scalar(g_py_param_yt, "index_offset", g_param_yt.index_offset);
    add_dict_scalar(g_py_param_yt, "num_grids", g_param_yt.num_grids);

    // vectors (stored as Python tuples)
    add_dict_vector_n(g_py_param_yt, "domain_left_edge", 3, g_param_yt.domain_left_edge);
    add_dict_vector_n(g_py_param_yt, "domain_right_edge", 3, g_param_yt.domain_right_edge);
    add_dict_vector_n(g_py_param_yt, "periodicity", 3, g_param_yt.periodicity);
    add_dict_vector_n(g_py_param_yt, "domain_dimensions", 3, g_param_yt.domain_dimensions);
#endif  // #ifdef USE_PYBIND11

    log_debug("Inserting YT parameters to libyt.param_yt ... done\n");

    // if num_fields > 0, which means we want to load fields
    if (g_param_yt.num_fields > 0) {
        LibytProcessControl::Get().field_list = new yt_field[g_param_yt.num_fields];
    } else {
        LibytProcessControl::Get().field_list = nullptr;
        LibytProcessControl::Get().get_fieldsPtr = true;
    }

    // if num_par_types > 0, which means want to load particle
    if (g_param_yt.num_par_types > 0) {
        // Initialize and setup yt_particle *particle_list in g_param_yt.particle_list,
        // to avoid user freeing yt_par_type *par_type_list.
        yt_particle* particle_list = new yt_particle[g_param_yt.num_par_types];
        for (int s = 0; s < g_param_yt.num_par_types; s++) {
            particle_list[s].par_type = g_param_yt.par_type_list[s].par_type;
            particle_list[s].num_attr = g_param_yt.par_type_list[s].num_attr;
            particle_list[s].attr_list = new yt_attribute[particle_list[s].num_attr];
        }
        LibytProcessControl::Get().particle_list = particle_list;
    } else {
        // don't need to load particle, set as NULL.
        LibytProcessControl::Get().particle_list = nullptr;
        LibytProcessControl::Get().get_particlesPtr = true;
    }

    // if num_grids_local <= 0, which means this rank doesn't need to load in grids_local info.
    if (g_param_yt.num_grids_local <= 0) {
        LibytProcessControl::Get().grids_local = nullptr;
        LibytProcessControl::Get().get_gridsPtr = true;
    }

    // Make sure g_param_yt.num_grids_local is set,
    // and if g_param_yt.num_grids_local < 0, set it = 0
    if (g_param_yt.num_grids_local < 0) {
        // Prevent input long type and exceed int storage
        log_warning(
            "Number of local grids = %d at MPI rank %d, probably because of exceeding int storage or wrong input!\n",
            g_param_yt.num_grids_local, LibytProcessControl::Get().mpi_rank_);

        // if < 0, set it to 0, to avoid adding negative num_grids_local when checking num_grids.
        g_param_yt.num_grids_local = 0;
    }

#ifndef SERIAL_MODE
    // Gather num_grids_local in every rank and store at num_grids_local_MPI, with "MPI_Gather"
    // We need num_grids_local_MPI in MPI_Gatherv in yt_commit()
    int NRank;
    int RootRank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &NRank);
    int* num_grids_local_MPI = new int[NRank];
    LibytProcessControl::Get().num_grids_local_MPI = num_grids_local_MPI;

    MPI_Gather(&(g_param_yt.num_grids_local), 1, MPI_INT, num_grids_local_MPI, 1, MPI_INT, RootRank, MPI_COMM_WORLD);
    MPI_Bcast(num_grids_local_MPI, NRank, MPI_INT, RootRank, MPI_COMM_WORLD);

    // Check that sum of num_grids_local_MPI is equal to num_grids (total number of grids), abort if not.
    if (g_param_libyt.check_data) {
        if (check_sum_num_grids_local_MPI(NRank, num_grids_local_MPI) != YT_SUCCESS) {
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
