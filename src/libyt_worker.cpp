#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)
#include "libyt_worker.h"

#include <mpi.h>

#include <iostream>

#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytWorker
// Method      :  Constructor
//
// Notes       :  1.
//
// Arguments   :  int myrank : my MPI process num
//                int mysize : my MPI size in MPI_COMM_WORLD
//                int root   : MPI root rank
//-------------------------------------------------------------------------------------------------------
LibytWorker::LibytWorker(int myrank, int mysize, int root) : m_mpi_rank(myrank), m_mpi_size(mysize), m_mpi_root(root) {
    SET_TIMER(__PRETTY_FUNCTION__);
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
            case -1: {
                done = true;
                break;
            }
            case 1: {
                std::array<std::string, 2> temp_string = execute_code();
            }
            default: {
                done = false;
            }
        }
    }

    log_debug("Leaving libyt worker on MPI process %d\n", m_mpi_rank);
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytWorker
// Method      :  execute_code
// Description :  Get code and execute code
//
// Notes       :  1. This is a collective operation, requires every rank to call this function.
//                2. Root rank will gather stdout and stderr from non-root rank, so the string returned
//                   contains each ranks dumped output in root, and non-root rank only returns output from
//                   itself.
//                3. This method is called by LibytWorker::start and LibytKernel::execute_request_impl.
//
// Arguments   :
//
// Return      :  std::array<std::string, 2> output[0] : stdout
//                                           output[1] : stderr
//-------------------------------------------------------------------------------------------------------
std::array<std::string, 2> LibytWorker::execute_code() {
    SET_TIMER(__PRETTY_FUNCTION__);

    return {"", ""};
}

#endif  // #if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)