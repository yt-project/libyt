#ifndef __YT_PROTOTYPE_H__
#define __YT_PROTOTYPE_H__



// include relevant headers
#include "yt_type.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_hierarchy
// Description :  Data structure for pass hierarchy of the grid in MPI process, it is meant to be temporary.
//       Notes :  1. We don't deal with particle count in each ptype here.
//
// Data Member :  dimensions     : Number of cells along each direction
//                left_edge      : Grid left  edge in code units
//                right_edge     : Grid right edge in code units
//                level          : AMR level (0 for the root level)
//                proc_num       : An array of MPI rank that the grid belongs
//                id             : Grid ID
//                parent_id      : Parent grid ID
//                proc_num       : Process number, grid belong to which MPI rank
//-------------------------------------------------------------------------------------------------------
struct yt_hierarchy{
      double left_edge[3];
      double right_edge[3];

      long   id;
      long   parent_id;

      int    dimensions[3];
      int    level;
      int    proc_num;
};

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_rma_grid_info
// Description :  Data structure for getting remote grids, it's meant for temporary used.
//
// Data Member :  long     id         : Grid id.
//                MPI_Aint address    : Window address at which this data buffer attaches to.
//                int      rank       : Rank that contains the data buffer.
//                yt_dtype data_dtype : Data type of the array.
//                int      data_dim[3]: Data array's dimension, the actual dimension, which include ghost cell.
//-------------------------------------------------------------------------------------------------------
struct yt_rma_grid_info
{
    long     id;
    MPI_Aint address;
    int      rank;
    yt_dtype data_dtype;
    int      data_dim[3];  // Is in the view of the data array.
};

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_rma_particle_info
// Description :  Data structure for getting remote particle attribute, it's meant for temporary used.
//
// Notes       :  1. I change the order of the data member, in order to make creating mpi user data type
//                   more efficient.
//
// Data Member :  long     id         : Grid id.
//                MPI_Aint address    : Window address at which this data buffer attaches to.
//                long     data_len   : Data array's length.
//                int      rank       : Rank that contains the data buffer.
//-------------------------------------------------------------------------------------------------------
struct yt_rma_particle_info
{
    long     id;
    MPI_Aint address;
    long     data_len;  // Is in the view of the data array.
    int      rank;
};

void log_info   ( const char *Format, ... );
void log_warning( const char *format, ... );
void log_debug  ( const char *Format, ... );
void log_error  ( const char *format, ... );
int  create_libyt_module();
int  init_python( int argc, char *argv[] );
int  init_libyt_module();
int  allocate_hierarchy();
int  big_MPI_Gatherv(int RootRank, int *sendcounts, void *sendbuffer, MPI_Datatype *mpi_datatype, void *buffer, int cast_type);
int  big_MPI_Bcast(int RootRank, long sendcount, void *buffer, MPI_Datatype *mpi_datatype, int cast_type);
int  big_MPI_Get(void *recv_buff, long data_len, yt_dtype *data_dtype, MPI_Datatype *mpi_dtype, int get_rank, MPI_Aint base_address, MPI_Win *window);
int  get_npy_dtype( yt_dtype data_type, int *npy_dtype );
int  get_mpi_dtype( yt_dtype data_type, MPI_Datatype *mpi_dtype );
int  get_dtype_size( yt_dtype data_type, int *dtype_size );
int  append_grid( yt_grid *grid );
int  check_sum_num_grids_local_MPI( int NRank, int * &num_grids_local_MPI );
int  check_field_list();
int  check_particle_list();
int  check_grid();
int  check_hierarchy(yt_hierarchy * &hierarchy);
int  check_yt_param_yt(const yt_param_yt &param_yt);
int  check_yt_grid(const yt_grid &grid);
int  check_yt_field(const yt_field &field);
int  check_yt_attribute(const yt_attribute &attr);
int  check_yt_particle(const yt_particle &particle);
int  print_yt_param_yt(const yt_param_yt &param_yt);
int  print_yt_field(const yt_field &field);
#ifndef NO_PYTHON
template <typename T>
int  add_dict_scalar( PyObject *dict, const char *key, const T value );
template <typename T>
int  add_dict_vector_n( PyObject *dict, const char *key, const int len, const T *vector );
int  add_dict_string( PyObject *dict, const char *key, const char *string );

int  add_dict_field_list( );
int  add_dict_particle_list( );
#endif



#endif // #ifndef __YT_PROTOTYPE_H__
