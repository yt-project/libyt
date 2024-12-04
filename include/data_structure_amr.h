#ifndef LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
#define LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_

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
struct yt_hierarchy {
    double left_edge[3]{-1.0, -1.0, -1.0};
    double right_edge[3]{-1.0, -1.0, -1.0};
    long id = -1;
    long parent_id = -2;
    int dimensions[3]{-1, -1, -1};
    int level = -1;
    int proc_num = -1;
};

// TODO: explore if I can std::move
struct AmrDataArray3DInfo {
    long id;
    yt_dtype data_dtype;
    int data_dim[3];
    bool swap_axes;  // TODO: rename to continuous_in_x
};

struct AmrDataArray3D {
    long id;
    yt_dtype data_dtype;
    int data_dim[3];
    bool swap_axes;  // TODO: rename to continuous_in_x
    void* data_ptr;
};

#endif  // LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
