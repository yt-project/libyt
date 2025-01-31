#ifndef __YT_PROTOTYPE_H__
#define __YT_PROTOTYPE_H__

// include relevant headers
#include <typeinfo>

#include "yt_type.h"

#ifdef USE_PYBIND11
#include "pybind11/numpy.h"
#else
#include <Python.h>
#endif

#ifndef SERIAL_MODE
#include <mpi.h>

#include "data_structure_amr.h"
int big_MPI_Get_dtype(void* recv_buff, long data_len, yt_dtype* data_dtype, MPI_Datatype* mpi_dtype, int get_rank,
                      MPI_Aint base_address, MPI_Win* window);
int check_sum_num_grids_local_MPI(int mpi_size, int* num_grids_local_MPI);
#endif

void log_info(const char* Format, ...);
void log_warning(const char* format, ...);
void log_debug(const char* Format, ...);
void log_error(const char* format, ...);

int check_yt_param_yt(const yt_param_yt& param_yt);
int print_yt_param_yt(const yt_param_yt& param_yt);
int print_yt_field(const yt_field& field);

#ifndef NO_PYTHON
template<typename T>
int add_dict_scalar(PyObject* dict, const char* key, const T value);
template<typename T>
int add_dict_vector_n(PyObject* dict, const char* key, const int len, const T* vector);
int add_dict_string(PyObject* dict, const char* key, const char* string);
#endif

#endif  // #ifndef __YT_PROTOTYPE_H__
