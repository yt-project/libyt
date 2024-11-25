#ifndef SERIAL_MODE
#include <mpi.h>
#endif

#include "LibytProcessControl.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Data member :  Static private LibytProcessControl instance.
// Description :  Singleton instance.
//-------------------------------------------------------------------------------------------------------
LibytProcessControl LibytProcessControl::s_Instance;

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Method      :  Private constructor
// Description :  Initialize the class.
//
// Notes       :  1  Allocating and freeing memories are manage by other function, this method only
//                   stores it.
//                2. This acts like a global.
//                3. This is helpful when there are other initialization processes added to libyt.
//                   (What I'm doing in this commit is trying to make libyt a C-library with minimal
//                    change.)
//                4. This singleton implementation is _not_ thread-safe. I didn't make it so, because
//                   the current libyt API is _not_ thread-safe.
//                5. Should only have one instance in each MPI process.
//                6. Initialize timer profile heading and file.
//
// Data Member : libyt_initialized : true ==> yt_initialize() has been called successfully
//               param_yt_set      : true ==> yt_set_Parameters() has been called successfully
//               get_fieldsPtr     : true ==> yt_get_FieldsPtr() has been called successfully
//               get_particlesPtr  : true ==> yt_get_ParticlesPtr() has been called successfully
//               get_gridsPtr      : true ==> yt_get_GridsPtr() has been called successfully
//               commit_grids      : true ==> yt_commit() has been called successfully
//               free_gridsPtr     : true ==> yt_free() has been called successfully, everything is reset and freed.
//
//               field_list          : field list, including field name, field data type, field definition ...
//               particle_list       : particle list, including particle name, data type ...
//               grids_local         : a data structure for storing local grids hierarchy and data memory
//                                     mapping temporary.
//               num_grids_local_MPI : for gathering different MPI processes hierarchy.
//
//               grid_left_edge      : AMR hierarchy for Pybind11.
//               grid_right_edge     :
//               grid_dimensions     :
//               grid_parent_id      :
//               grid_levels         :
//               proc_num            :
//               par_count_list      :
//-------------------------------------------------------------------------------------------------------
LibytProcessControl::LibytProcessControl() {
    // MPI info
    mpi_rank_ = 0;
    mpi_size_ = 1;
    mpi_root_ = 0;

    // Check points for libyt process
    libyt_initialized = false;
    param_yt_set = false;
    get_fieldsPtr = false;
    get_particlesPtr = false;
    get_gridsPtr = false;
    commit_grids = false;
    free_gridsPtr = true;

    field_list = nullptr;
    particle_list = nullptr;
    grids_local = nullptr;
    num_grids_local_MPI = nullptr;

    py_grid_data_ = nullptr;
    py_particle_data_ = nullptr;
    py_hierarchy_ = nullptr;
    py_param_yt_ = nullptr;
    py_param_user_ = nullptr;
    py_libyt_info_ = nullptr;
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    py_interactive_mode_ = nullptr;
#endif

#ifdef USE_PYBIND11
    grid_left_edge = nullptr;
    grid_right_edge = nullptr;
    grid_dimensions = nullptr;
    grid_parent_id = nullptr;
    grid_levels = nullptr;
    proc_num = nullptr;
    par_count_list = nullptr;
#endif
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Method      :  Public method
// Description :  Initialize libyt process control; every operation should happen after this step.
//
// Notes       :  1. It is called in yt_initialize().
//                2. Initialize MPI rank, MPI size. (if not in SERIAL_MODE)
//                3. Initialize and create libyt profile file. (if SUPPORT_TIMER is set)
//-------------------------------------------------------------------------------------------------------
void LibytProcessControl::Initialize() {
#ifndef SERIAL_MODE
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
#endif

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    LibytPythonShell::SetMPIInfo(mpi_size_, mpi_root_, mpi_rank_);
#endif

#ifdef SUPPORT_TIMER
    // Set time profile controller
    std::string filename = "libytTimeProfile_MPI";
    filename += std::to_string(mpi_rank_);
    filename += ".json";
    timer_control.CreateFile(filename.c_str(), mpi_rank_);
#endif
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Method      :  Public static method
// Description :  Get the singleton instance.
//
// Notes       :  1. Return the reference of LibytProcessControl instance.
//-------------------------------------------------------------------------------------------------------
LibytProcessControl& LibytProcessControl::Get() { return s_Instance; }
