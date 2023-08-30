#include "yt_combo.h"
#include "LibytProcessControl.h"
#include "libyt.h"


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
//
int yt_free() {
#ifdef SUPPORT_TIMER
    g_timer->record_time("yt_free", 0);
#endif

    // check if libyt has been initialized
    if (!LibytProcessControl::Get().libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if user has run through all the routine.
    if (!LibytProcessControl::Get().commit_grids) {
        log_warning("You are going to free every libyt initialized and allocated array, "
                    "even though the inline-analysis procedure has not finished yet!\n");
    }

    // Make sure every rank has reach to this point
    MPI_Barrier(MPI_COMM_WORLD);

    // Free resource allocated in yt_set_Parameters():
    //    field_list, particle_list, attr_list, num_grids_local_MPI
    if (LibytProcessControl::Get().param_yt_set) {
        if (g_param_yt.num_fields > 0) delete[] g_param_yt.field_list;
        if (g_param_yt.num_par_types > 0) {
            for (int i = 0; i < g_param_yt.num_par_types; i++) { delete[] g_param_yt.particle_list[i].attr_list; }
            delete[] g_param_yt.particle_list;
        }
        delete[] g_param_yt.num_grids_local_MPI;
    }

    // Free resource allocated in yt_get_GridsPtr() in case it hasn't got freed yet:
    //    grids_local, field_data, particle_data, par_count_list
    if (LibytProcessControl::Get().get_gridsPtr && g_param_yt.num_grids_local > 0) {
        for (int i = 0; i < g_param_yt.num_grids_local; i = i + 1) {
            if (g_param_yt.num_fields > 0) {
                delete[] g_param_yt.grids_local[i].field_data;
            }
            if (g_param_yt.num_par_types > 0) {
                delete[] g_param_yt.grids_local[i].par_count_list;
                for (int p = 0; p < g_param_yt.num_par_types; p++){
                    delete[] g_param_yt.grids_local[i].particle_data[p];
                }
                delete[] g_param_yt.grids_local[i].particle_data;
            }
        }
        delete[] g_param_yt.grids_local;
    }

    // Reset g_param_yt
    g_param_yt.init();

    // Reset data in libyt module
    PyDict_Clear(g_py_grid_data);
    PyDict_Clear(g_py_particle_data);
    PyDict_Clear(g_py_hierarchy);
    PyDict_Clear(g_py_param_yt);
    PyDict_Clear(g_py_param_user);
#ifdef INTERACTIVE_MODE
    PyDict_Clear(PyDict_GetItemString(g_py_interactive_mode, "func_err_msg"));
#endif
    PyRun_SimpleString("gc.collect()");

#ifdef INTERACTIVE_MODE
    // Reset g_func_status_list status
    g_func_status_list.reset();
#endif
    // Reset check points
    LibytProcessControl::Get().param_yt_set = false;
    LibytProcessControl::Get().get_fieldsPtr = false;
    LibytProcessControl::Get().get_particlesPtr = false;
    LibytProcessControl::Get().get_gridsPtr = false;
    LibytProcessControl::Get().commit_grids = false;
    LibytProcessControl::Get().free_gridsPtr = true;
    g_param_libyt.counter++;

#ifdef SUPPORT_TIMER
    // end timer.
    g_timer->record_time("yt_free", 1);
    // print out record time in this iteration.
    g_timer->print_all_time();
#endif

    return YT_SUCCESS;
} // FUNCTION: yt_free()
