#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)
#include "libyt_worker.h"

#include <mpi.h>

#include <iostream>

#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytWorker
// Method      :  Constructor
//
// Notes       :  1. Initialize m_py_global which is script's namespace.
//
// Arguments   :  int myrank : my MPI process num
//                int mysize : my MPI size in MPI_COMM_WORLD
//-------------------------------------------------------------------------------------------------------
LibytWorker::LibytWorker(int myrank, int mysize, int root)
    : m_py_global(NULL), m_mpi_rank(myrank), m_mpi_size(mysize), m_mpi_root(root) {
    SET_TIMER(__PRETTY_FUNCTION__);

    m_py_global = PyDict_GetItemString(g_py_interactive_mode, "script_globals");
}

LibytWorker::~LibytWorker() {}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytWorker
// Method      :  start
// Description :  Start the listener and dispatch execute and shutdown jobs to methods.
//
// Notes       :  1. Get indicator broadcast by root rank:
//                   (1) -1 : shutdown
//                   (2)  1 : execute code
//
// Arguments   :  (None)
//-------------------------------------------------------------------------------------------------------
void LibytWorker::start() {
    SET_TIMER(__PRETTY_FUNCTION__);

    bool done = false;
    while (!done) {
        int indicator;
        MPI_Bcast(&indicator, 1, MPI_INT, m_mpi_root, MPI_COMM_WORLD);

        // Dispatch jobs
        switch (indicator) {
            case -1:
                done = true;
                break;
            case 1:
                execute_code();
            default:
                done = false;
        }
    }

    log_debug("Leaving libyt worker on MPI process %d\n", m_mpi_rank);
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytWorker
// Method      :  execute_code
// Description :  Prepare to get code from root rank and execute code
//
// Notes       :  1. This is a collective operation.
//                2.
//
// Arguments   :  (None)
//-------------------------------------------------------------------------------------------------------
void LibytWorker::execute_code() {
    if (m_mpi_rank == m_mpi_root) {
    }
}

#endif  // #if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)