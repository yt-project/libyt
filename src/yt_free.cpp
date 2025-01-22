#include "function_info.h"
#include "libyt.h"
#include "libyt_process_control.h"
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
    if (!LibytProcessControl::Get().libyt_initialized_) {
        YT_ABORT("Please invoke yt_initialize() before calling %s()!\n", __FUNCTION__);
    }

    // check if user has run through all the routine.
    if (!LibytProcessControl::Get().commit_grids_) {
        log_warning("You are going to free every libyt initialized and allocated array, "
                    "even though the inline-analysis procedure has not finished yet!\n");
    }

#ifndef SERIAL_MODE
    // Make sure every rank has reach to this point
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // Free resource allocated for data structure amr
    LibytProcessControl::Get().data_structure_amr_.CleanUp();

#ifndef USE_PYBIND11
    // Reset data in libyt module
    PyDict_Clear(LibytProcessControl::Get().py_param_yt_);
    PyDict_Clear(LibytProcessControl::Get().py_param_user_);
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    PyDict_Clear(PyDict_GetItemString(LibytProcessControl::Get().py_interactive_mode_, "func_err_msg"));
#endif
#else  // #ifndef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");

    const char* keys_to_clear[] = {"param_yt", "param_user"};
    const int keys_len = 2;
    for (int i = 0; i < keys_len; i++) {
        pybind11::dict py_dict = libyt.attr(keys_to_clear[i]);
        py_dict.clear();
    }
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    pybind11::dict py_interactive_mode = libyt.attr("interactive_mode");
    pybind11::dict py_func_err_msg = py_interactive_mode["func_err_msg"];
    py_func_err_msg.clear();
#endif
#endif  // #ifndef USE_PYBIND11

    PyRun_SimpleString("gc.collect()");

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    // Reset LibytProcessControl::Get().function_info_list_ status
    LibytProcessControl::Get().function_info_list_.ResetEveryFunctionStatus();
#endif
    // Reset check points
    LibytProcessControl::Get().param_yt_set_ = false;
    LibytProcessControl::Get().get_fields_ptr_ = false;
    LibytProcessControl::Get().get_particles_ptr_ = false;
    LibytProcessControl::Get().get_grids_ptr_ = false;
    LibytProcessControl::Get().commit_grids_ = false;
    LibytProcessControl::Get().need_free_ = false;
    LibytProcessControl::Get().param_libyt_.counter++;

    return YT_SUCCESS;
}  // FUNCTION: yt_free()
