#ifndef LIBYT_PROJECT_INCLUDE_LIBYT_PROCESS_CONTROL_H_
#define LIBYT_PROJECT_INCLUDE_LIBYT_PROCESS_CONTROL_H_

#include <Python.h>

#include "data_hub_amr.h"
#include "yt_type.h"

#ifndef SERIAL_MODE
#include "comm_mpi.h"
#endif
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
#include "function_info.h"
#include "libyt_python_shell.h"
#endif
#ifdef SUPPORT_TIMER
#include "timer_control.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Description :  Singleton to deal with libyt internal process. (not thread-safe)
//
// Notes       :  1. Initialize libyt process control.
//                2. TODO: probably should manage different modules into different classes,
//                   like timer and python shell.
//-------------------------------------------------------------------------------------------------------
class LibytProcessControl {
public:
    // MPI info
    int mpi_rank_;
    int mpi_size_;
    int mpi_root_;
#ifndef SERIAL_MODE
    CommMpi comm_mpi_;
#endif

#ifdef SUPPORT_TIMER
    // Timer Control
    TimerControl timer_control;
#endif

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    // Python shell
    LibytPythonShell python_shell_;
    FunctionInfoList function_info_list_;
    PyObject* py_interactive_mode_;
#endif

    // Process control check point and data management for libyt
    bool libyt_initialized;
    bool param_yt_set;
    bool get_fieldsPtr;
    bool get_particlesPtr;
    bool get_gridsPtr;
    bool commit_grids;
    bool free_gridsPtr;

    // libyt parameters
    yt_param_libyt param_libyt_;

    // AMR data structure and hierarchy
    yt_param_yt param_yt_;

    yt_field* field_list;
    yt_particle* particle_list;
    yt_grid* grids_local;
    int* all_num_grids_local_;

    PyObject* py_grid_data_;
    PyObject* py_particle_data_;
    PyObject* py_hierarchy_;
    PyObject* py_param_yt_;
    PyObject* py_param_user_;
    PyObject* py_libyt_info_;

    // Hierarchy
    double* grid_left_edge_;
    double* grid_right_edge_;
    int* grid_dimensions_;
    long* grid_parent_id_;
    int* grid_levels_;
    int* proc_num_;
    long* par_count_list_;

    // Singleton Methods
    LibytProcessControl(const LibytProcessControl& other) = delete;
    LibytProcessControl& operator=(const LibytProcessControl& other) = delete;
    static LibytProcessControl& Get();

    // Methods
    void Initialize();

private:
    LibytProcessControl();
    static LibytProcessControl s_Instance;
};

#endif  // LIBYT_PROJECT_INCLUDE_LIBYT_PROCESS_CONTROL_H_
