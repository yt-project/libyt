#ifndef AMR_GRID_H
#define AMR_GRID_H

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
    int dimensions[3]{0, 0, 0};
    int level = -1;
    int proc_num = -1;
};

#endif  // AMR_GRID_H
