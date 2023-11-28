#if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)
#include "libyt_worker.h"

#include <iostream>

#include "yt_combo.h"

LibytWorker::LibytWorker(int myrank, int mysize) : m_MPI_Rank(myrank), m_MPI_Size(mysize) {
    m_py_global = PyDict_GetItemString(g_py_interactive_mode, "script_globals");
}

LibytWorker::~LibytWorker() {}

void LibytWorker::start() {}

#endif  // #if defined(INTERACTIVE_MODE) && defined(JUPYTER_KERNEL) && !defined(SERIAL_MODE)