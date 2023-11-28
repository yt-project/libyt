#ifndef __LIBYT_WORKER_H__
#define __LIBYT_WORKER_H__

#include <Python.h>

class LibytWorker {
public:
    LibytWorker(int myrank, int mysize);
    ~LibytWorker();

    void start();

private:
    PyObject* m_py_global;

    int m_MPI_Rank;
    int m_MPI_Size;
};

#endif  // __LIBYT_WORKER_H__
