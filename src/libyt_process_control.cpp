#ifndef SERIAL_MODE
#include <mpi.h>
#endif

#include "libyt_process_control.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Data member :  Static private LibytProcessControl instance.
// Description :  Singleton instance.
//-------------------------------------------------------------------------------------------------------
LibytProcessControl LibytProcessControl::instance_;

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
//-------------------------------------------------------------------------------------------------------
LibytProcessControl::LibytProcessControl() {
    // MPI info
    mpi_rank_ = 0;
    mpi_size_ = 1;
    mpi_root_ = 0;

    // Check points for libyt process
    libyt_initialized_ = false;
    param_yt_set_ = false;
    get_fields_ptr_ = false;
    get_particles_ptr_ = false;
    get_grids_ptr_ = false;
    commit_grids_ = false;
    free_grids_ptr_ = true;

    py_param_yt_ = nullptr;
    py_param_user_ = nullptr;
    py_libyt_info_ = nullptr;
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    py_interactive_mode_ = nullptr;
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

    CommMpi::InitializeInfo(0);
    CommMpi::InitializeYtLongMpiDataType();
    CommMpi::InitializeYtHierarchyMpiDataType();
    CommMpi::InitializeMpiRmaAddressMpiDataType();
    CommMpi::InitializeAmrDataArray3DMpiDataType();
    CommMpi::InitializeAmrDataArray1DMpiDataType();
    CommMpi::InitializeYtRmaGridInfoMpiDataType();
    CommMpi::InitializeYtRmaParticleInfoMpiDataType();
#endif

#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    LibytPythonShell::SetMPIInfo(mpi_size_, mpi_root_, mpi_rank_);  // TODO: rename
#endif
    DataStructureAmr::SetMpiInfo(mpi_size_, mpi_root_, mpi_rank_);

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
LibytProcessControl& LibytProcessControl::Get() { return instance_; }
