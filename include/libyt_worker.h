#ifndef LIBYT_PROJECT_INCLUDE_LIBYT_WORKER_H_
#define LIBYT_PROJECT_INCLUDE_LIBYT_WORKER_H_

#include <Python.h>

#include <array>
#include <string>

class LibytWorker {
 public:
  LibytWorker(int myrank, int mysize, int root);

  void start();

 private:
  int m_mpi_rank;
  int m_mpi_size;
  int m_mpi_root;
};

#endif  // LIBYT_PROJECT_INCLUDE_LIBYT_WORKER_H_
