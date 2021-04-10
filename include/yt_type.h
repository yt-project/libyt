#ifndef __YT_TYPE_H__
#define __YT_TYPE_H__



/*******************************************************************************
/
/  Data types used by libyt
/
********************************************************************************/


// short names for unsigned types
typedef unsigned int       uint;
typedef unsigned long int  ulong;


// enumerate types
enum yt_verbose { YT_VERBOSE_OFF=0, YT_VERBOSE_INFO=1, YT_VERBOSE_WARNING=2, YT_VERBOSE_DEBUG=3 };
enum yt_ftype   { YT_FTYPE_UNKNOWN=0, YT_FLOAT=1, YT_DOUBLE=2 };


// structures
#include "yt_type_param_libyt.h"
#include "yt_type_param_yt.h"
#include "yt_type_grid.h"
#include "yt_type_field.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_hierarchy
// Description :  Data structure for pass hierarchy of the grid in MPI process, it is meant to be temperary.
//
// Data Member :  dimensions     : Number of cells along each direction
//                left_edge      : Grid left  edge in code units
//                right_edge     : Grid right edge in code units
//                particle_count : Nunber of particles in this grid
//                level          : AMR level (0 for the root level)
//                proc_num       : An array of MPI rank that the grid belongs
//                id             : Grid ID (0-indexed ==> must be in the range 0 <= id < total number of grids)
//                parent_id      : Parent grid ID (0-indexed, -1 for grids on the root level)
//                proc_num       : Process number, grid belong to which MPI rank
//-------------------------------------------------------------------------------------------------------
struct yt_hierarchy{
      double left_edge[3];
      double right_edge[3];

      long   particle_count;
      long   id;
      long   parent_id;

      int    dimensions[3];
      int    level;
      int    proc_num;
};

#endif // #ifndef __YT_TYPE_H__
