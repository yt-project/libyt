#ifndef __YT_PROTOTYPE_H__
#define __YT_PROTOTYPE_H__

// include relevant headers
#include <typeinfo>

#include "yt_type.h"

#ifdef USE_PYBIND11
#include "pybind11/numpy.h"
#endif

#ifndef SERIAL_MODE
#include "amr_grid.h"

int big_MPI_Get_dtype(void* recv_buff, long data_len, yt_dtype* data_dtype, MPI_Datatype* mpi_dtype, int get_rank,
                      MPI_Aint base_address, MPI_Win* window);
int get_mpi_dtype(yt_dtype data_type, MPI_Datatype* mpi_dtype);
int check_hierarchy(yt_hierarchy*& hierarchy);
int check_sum_num_grids_local_MPI(int NRank, int*& num_grids_local_MPI);
#else
int check_hierarchy(yt_grid*& hierarchy);
#endif

void log_info(const char* Format, ...);
void log_warning(const char* format, ...);
void log_debug(const char* Format, ...);
void log_error(const char* format, ...);

int create_libyt_module();
int init_python(int argc, char* argv[]);
int init_libyt_module();
int allocate_hierarchy();
int get_npy_dtype(yt_dtype data_type, int* npy_dtype);
int get_dtype_size(yt_dtype data_type, int* dtype_size);
int get_dtype_typeid(yt_dtype data_type, const std::type_info** dtype_id);
int get_dtype_allocation(yt_dtype data_type, unsigned long length, void** data_ptr);
int get_yt_dtype_from_npy(int npy_dtype, yt_dtype* data_dtype);
#ifdef USE_PYBIND11
pybind11::array get_pybind11_allocate_array_dtype(yt_dtype data_type, const std::vector<long>& shape,
                                                  const std::vector<long>& stride);
#endif
int append_grid(yt_grid* grid);

int check_field_list();
int check_particle_list();
int check_grid();
int check_yt_param_yt(const yt_param_yt& param_yt);
int check_yt_grid(const yt_grid& grid);
int check_yt_field(const yt_field& field);
int check_yt_attribute(const yt_attribute& attr);
int check_yt_particle(const yt_particle& particle);
int print_yt_param_yt(const yt_param_yt& param_yt);
int print_yt_field(const yt_field& field);

#ifndef NO_PYTHON
template<typename T>
int add_dict_scalar(PyObject* dict, const char* key, const T value);
template<typename T>
int add_dict_vector_n(PyObject* dict, const char* key, const int len, const T* vector);
int add_dict_string(PyObject* dict, const char* key, const char* string);
int add_dict_field_list();
int add_dict_particle_list();
#endif

#endif  // #ifndef __YT_PROTOTYPE_H__
