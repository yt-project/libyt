#include "LibytProcessControl.h"
#include "function_info.h"
#include "libyt.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_free()
// Description :  Refresh the python yt state after finish inline-analysis
//
// Note        :  1. Call and use by user, after they are done with all the inline-analysis in this
//                   round, or they want to freed everything allocated by libyt.
//                2. We also freed grids_local here, in case user didn't call yt_commit and cause memory
//                   leak.
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_free() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if user has run through all the routine.
    if (!LibytProcessControl::Get().commit_grids) {
        log_warning("You are going to free every libyt initialized and allocated array, "
                    "even though the inline-analysis procedure has not finished yet!\n");
    }

#ifndef SERIAL_MODE
    // Make sure every rank has reach to this point
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Free resource allocated in yt_set_Parameters():
    //    field_list, particle_list, attr_list, num_grids_local_MPI
    yt_param_yt& param_yt = LibytProcessControl::Get().param_yt_;
    yt_field* field_list = LibytProcessControl::Get().field_list;
    yt_particle* particle_list = LibytProcessControl::Get().particle_list;
    if (LibytProcessControl::Get().param_yt_set) {
        if (param_yt.num_fields > 0) delete[] field_list;
        if (param_yt.num_par_types > 0) {
            for (int i = 0; i < param_yt.num_par_types; i++) {
                delete[] particle_list[i].attr_list;
            }
            delete[] particle_list;
        }
        delete[] LibytProcessControl::Get().num_grids_local_MPI;
    }

    // Free resource allocated in yt_get_GridsPtr() in case it hasn't got freed yet:
    //    grids_local, field_data, particle_data, par_count_list
    yt_grid* grids_local = LibytProcessControl::Get().grids_local;
    if (LibytProcessControl::Get().get_gridsPtr && param_yt.num_grids_local > 0) {
        for (int i = 0; i < param_yt.num_grids_local; i = i + 1) {
            if (param_yt.num_fields > 0) {
                delete[] grids_local[i].field_data;
            }
            if (param_yt.num_par_types > 0) {
                delete[] grids_local[i].par_count_list;
                for (int p = 0; p < param_yt.num_par_types; p++) {
                    delete[] grids_local[i].particle_data[p];
                }
                delete[] grids_local[i].particle_data;
            }
        }
        delete[] grids_local;
    }

#ifdef USE_PYBIND11
    delete[] LibytProcessControl::Get().grid_left_edge;
    delete[] LibytProcessControl::Get().grid_right_edge;
    delete[] LibytProcessControl::Get().grid_dimensions;
    delete[] LibytProcessControl::Get().grid_parent_id;
    delete[] LibytProcessControl::Get().grid_levels;
    delete[] LibytProcessControl::Get().proc_num;
    if (param_yt.num_par_types > 0) {
        delete[] LibytProcessControl::Get().par_count_list;
    }
#endif

#ifndef USE_PYBIND11
    // Reset data in libyt module
    PyDict_Clear(g_py_grid_data);
    PyDict_Clear(g_py_particle_data);
    PyDict_Clear(g_py_hierarchy);
    PyDict_Clear(g_py_param_yt);
    PyDict_Clear(g_py_param_user);
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    PyDict_Clear(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"));
#endif
#else
    pybind11::module_ libyt = pybind11::module_::import("libyt");

    const char* keys_to_clear[] = {"grid_data", "particle_data", "hierarchy", "param_yt", "param_user"};
    const int keys_len = 5;
    for (int i = 0; i < keys_len; i++) {
        pybind11::dict py_dict = libyt.attr(keys_to_clear[i]);
        py_dict.clear();
    }
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    pybind11::dict py_interactive_mode = libyt.attr("interactive_mode");
    pybind11::dict py_func_err_msg = py_interactive_mode["func_err_msg"];
    py_func_err_msg.clear();
#endif
#endif

    PyRun_SimpleString("gc.collect()");

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    // Reset LibytProcessControl::Get().function_info_list_ status
    LibytProcessControl::Get().function_info_list_.ResetEveryFunctionStatus();
#endif
    // Reset check points
    LibytProcessControl::Get().param_yt_set = false;
    LibytProcessControl::Get().get_fieldsPtr = false;
    LibytProcessControl::Get().get_particlesPtr = false;
    LibytProcessControl::Get().get_gridsPtr = false;
    LibytProcessControl::Get().commit_grids = false;
    LibytProcessControl::Get().free_gridsPtr = true;
    LibytProcessControl::Get().param_libyt_.counter++;

    return YT_SUCCESS;
}  // FUNCTION: yt_free()
