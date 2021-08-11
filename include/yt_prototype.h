#ifndef __YT_PROTOTYPE_H__
#define __YT_PROTOTYPE_H__



// include relevant headers
#include "yt_type.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_hierarchy
// Description :  Data structure for pass hierarchy of the grid in MPI process, it is meant to be temperary.
//
// Data Member :  dimensions     : Number of cells along each direction
//                left_edge      : Grid left  edge in code units
//                right_edge     : Grid right edge in code units
//                grid_particle_count : Nunber of total particles in this grid
//                level          : AMR level (0 for the root level)
//                proc_num       : An array of MPI rank that the grid belongs
//                id             : Grid ID (0-indexed ==> must be in the range 0 <= id < total number of grids)
//                parent_id      : Parent grid ID (0-indexed, -1 for grids on the root level)
//                proc_num       : Process number, grid belong to which MPI rank
//-------------------------------------------------------------------------------------------------------
struct yt_hierarchy{
      double left_edge[3];
      double right_edge[3];

      long   grid_particle_count;
      long   id;
      long   parent_id;

      int    dimensions[3];
      int    level;
      int    proc_num;
};

void log_info   ( const char *Format, ... );
void log_warning( const char *format, ... );
void log_debug  ( const char *Format, ... );
void log_error  ( const char *format, ... );
int  create_libyt_module();
int  init_python( int argc, char *argv[] );
int  init_libyt_module();
int  allocate_hierarchy();
int  append_grid( yt_grid *grid );
int  check_sum_num_grids_local_MPI( int NRank, int * &num_grids_local_MPI );
int  check_field_list();
int  check_particle_list();
int  check_hierarchy(yt_hierarchy * &hierarchy);
#ifndef NO_PYTHON
template <typename T>
int  add_dict_scalar( PyObject *dict, const char *key, const T value );
template <typename T>
int  add_dict_vector3( PyObject *dict, const char *key, const T *vector );
int  add_dict_string( PyObject *dict, const char *key, const char *string );

int  add_dict_field_list( );
int  add_dict_particle_list( );
#endif



#endif // #ifndef __YT_PROTOTYPE_H__
