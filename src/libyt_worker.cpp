#if defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)
#include "libyt_worker.h"

#include <mpi.h>

#include "libyt_process_control.h"
#include "magic_command.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytWorker
// Method      :  Constructor
//
// Notes       :  1. Assume only non-root rank will call this class.
//
// Arguments   :  int myrank : my MPI process num
//                int mysize : my MPI size in MPI_COMM_WORLD
//                int root   : MPI root rank
//-------------------------------------------------------------------------------------------------------
LibytWorker::LibytWorker(int myrank, int mysize, int root) : m_mpi_rank(myrank), m_mpi_size(mysize), m_mpi_root(root) {
    SET_TIMER(__PRETTY_FUNCTION__);
}

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
                LibytProcessControl::Get().python_shell_.ClearHistory();
                break;
            }
            case 1: {
                std::vector<PythonOutput> output;
                PythonStatus all_execute_status =
                    LibytProcessControl::Get().python_shell_.AllExecuteCell("", "", m_mpi_root, output, m_mpi_root);
                break;
            }
            case 2: {
                MagicCommand command(MagicCommand::EntryPoint::kLibytJupyterKernel);
                MagicCommandOutput temp = command.Run();
                break;
            }
            default: {
                done = true;
                LogError("Unknown job indicator '%d'\n", indicator);
            }
        }
    }

    LogDebug("Leaving libyt worker on MPI process %d\n", m_mpi_rank);
}

#endif  // #if defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)