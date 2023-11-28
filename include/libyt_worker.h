#ifndef __LIBYT_WORKER_H__
#define __LIBYT_WORKER_H__

#include <Python.h>

class LibytWorker {
public:
    LibytWorker(int myrank, int mysize, int root);
    ~LibytWorker();

    void start();

private:
    PyObject* m_py_global;

    int m_mpi_rank;
    int m_mpi_size;
    int m_mpi_root;

    void execute_code();
};

#endif  // __LIBYT_WORKER_H__
