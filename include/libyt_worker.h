#ifndef __LIBYT_WORKER_H__
#define __LIBYT_WORKER_H__

#include <Python.h>

#include <array>
#include <string>

class LibytWorker {
public:
    LibytWorker(int myrank, int mysize, int root);
    ~LibytWorker();

    void start();
    std::array<std::string, 2> execute_code();

private:
    int m_mpi_rank;
    int m_mpi_size;
    int m_mpi_root;
};

#endif  // __LIBYT_WORKER_H__
