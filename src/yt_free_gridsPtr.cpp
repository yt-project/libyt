#include "yt_combo.h"
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_free_gridsPtr()
// Description :  Refresh the python yt state after finish inline-analysis
//
// Note        :  1. Call and use by user, after they are done with all the inline-analysis in this 
//                   round or they want to freed everything allocated by libyt.
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//
int yt_free_gridsPtr() {
#ifdef SUPPORT_TIMER
    g_timer->record_time("yt_free_gridsPtr", 0);
#endif

    // check if libyt has been initialized
    if (!g_param_libyt.libyt_initialized) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if user has run through all the routine.
    if (!g_param_libyt.commit_grids) {
        log_warning("You are going to free every libyt initialized and allocated array, "
                    "even though the inline-analysis procedure has not finished yet!\n");
    }

    // Make sure every rank has reach to this point
    MPI_Barrier(MPI_COMM_WORLD);

    // Free resource allocated in yt_set_Parameters():
    //    field_list, particle_list, attr_list, num_grids_local_MPI
    if (g_param_libyt.param_yt_set) {
        if (g_param_yt.num_fields > 0) delete[] g_param_yt.field_list;
        if (g_param_yt.num_par_types > 0) {
            for (int i = 0; i < g_param_yt.num_par_types; i++) { delete[] g_param_yt.particle_list[i].attr_list; }
            delete[] g_param_yt.particle_list;
        }
        delete[] g_param_yt.num_grids_local_MPI;
    }

    // Free resource allocated in yt_get_gridsPtr():
    //    grids_local, field_data, par_count_list
    if (g_param_libyt.get_gridsPtr && g_param_yt.num_grids_local > 0) {
        for (int i = 0; i < g_param_yt.num_grids_local; i = i + 1) {
            if (g_param_yt.num_fields > 0)  delete[] g_param_yt.grids_local[i].field_data;
            if (g_param_yt.num_par_types > 0) delete[] g_param_yt.grids_local[i].par_count_list;
        }
        delete[] g_param_yt.grids_local;
    }

    // Reset g_param_yt
    g_param_yt.init();

    // Reset data in libyt module
    PyDict_Clear(g_py_grid_data);
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
    g_param_libyt.param_yt_set = false;
    g_param_libyt.get_fieldsPtr = false;
    g_param_libyt.get_particlesPtr = false;
    g_param_libyt.get_gridsPtr = false;
    g_param_libyt.commit_grids = false;
    g_param_libyt.counter++;

    g_param_libyt.free_gridsPtr = true;

#ifdef SUPPORT_TIMER
    // end timer.
    g_timer->record_time("yt_free_gridsPtr", 1);
    // print out record time in this iteration.
    g_timer->print_all_time();
#endif

    return YT_SUCCESS;
} // FUNCTION: yt_free_gridsPtr()
