#ifndef __YT_TYPE_GRID_H__
#define __YT_TYPE_GRID_H__

#include "yt_macro.h"

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_data
// Description :  Data structure to store a field data's pointer and its array dimensions.
//
// Notes       :  1. This struct will be use in yt_grid data member field_data and particle_data.
//
// Data Member :  data_ptr           : data pointer
//                data_dimensions[3] : dimension of the data to be passed to python for wrapping, which is the actual
//                                     size of the array.
//                                     Def => fieldData[ dim[0] ][ dim[1] ][ dim[2] ]
//                data_dtype         : Data type of the field in specific grid. If this is set as YT_DTYPE_UNKNOWN,
//                                     then we will use field_dtype define in field_list as input field data type.
//
// Methods     : yt_data            : Constructor
//-------------------------------------------------------------------------------------------------------
typedef struct yt_data {
    void* data_ptr;
    int data_dimensions[3];
    yt_dtype data_dtype;

#ifdef __cplusplus
    //===================================================================================
    // Method      :  yt_data
    // Description :  Constructor of the structure "yt_data"
    //
    // Note        :  Initialize all data members
    //
    // Parameter   :  None
    //===================================================================================
    yt_data() {
        data_ptr = nullptr;
        for (int d = 0; d < 3; d++) {
            data_dimensions[d] = 0;
        }
        data_dtype = YT_DTYPE_UNKNOWN;
    }
#endif  // #ifdef __cplusplus
} yt_data;

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_grid
// Description :  Data structure to store a full single grid with data pointer
//
// Notes       :  1. We assume that each element in array[3] are all in use, which is we only support
//                   dim 3 for now, though it can be [len][0][0].
//
// Data Member :  grid_dimensions : Number of cells along each direction in [x][y][z] coordinate.
//                left_edge       : Grid left  edge in code units
//                right_edge      : Grid right edge in code units
//                par_count_list  : Array that records number of particles in each species, the input order
//                                  should be the same as the input particle_list.
//                level           : AMR level (0 for the root level)
//                id              : Grid ID
//                parent_id       : Parent grid ID
//                proc_num        : Process number, grid belong to which MPI rank
//                field_data      : Pointer pointing to yt_data array, each element stores an info for
//                                  field array to be wrapped.
//                particle_data   : Pointer to pointer pointing to yt_data array (yt_data** array),
//                                  Ex: particle_data[0][1] represents particle data for
//                                  particle_list[0].attr_list[1].
//
// Method      :  yt_grid  : Constructor
//-------------------------------------------------------------------------------------------------------
typedef struct yt_grid {
    double left_edge[3];
    double right_edge[3];
    long* par_count_list;
    long id;
    long parent_id;
    int grid_dimensions[3];
    int level;
    int proc_num;
    yt_data* field_data;
    yt_data** particle_data;

#ifdef __cplusplus
    //===================================================================================
    // Method      :  yt_grid
    // Description :  Constructor of the structure "yt_grid"
    //
    // Note        :  Initialize all data members
    //
    // Parameter   :  None
    //===================================================================================
    yt_grid() {
        for (int d = 0; d < 3; d++) {
            left_edge[d] = DBL_UNDEFINED;
            right_edge[d] = DBL_UNDEFINED;
        }
        for (int d = 0; d < 3; d++) {
            grid_dimensions[d] = INT_UNDEFINED;
        }
        par_count_list = nullptr;
        id = LNG_UNDEFINED;
        parent_id = LNG_UNDEFINED;
        level = INT_UNDEFINED;
        proc_num = INT_UNDEFINED;
        field_data = nullptr;
        particle_data = nullptr;
    }
#endif  // #ifdef __cplusplus

} yt_grid;

#endif  // #ifndef __YT_TYPE_GRID_H__
