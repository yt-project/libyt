#ifndef LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
#define LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_

#include <Python.h>

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
struct yt_hierarchy {
    double left_edge[3]{-1.0, -1.0, -1.0};
    double right_edge[3]{-1.0, -1.0, -1.0};
    long id = -1;
    long parent_id = -2;
    int dimensions[3]{-1, -1, -1};
    int level = -1;
    int proc_num = -1;
};

class DataStructureAmr {
public:
    int* all_num_grids_local_;

    // MPI
    static int mpi_size_;
    static int mpi_root_;
    static int mpi_rank_;

    // Hierarchy -- set by yt_param_yt passed in
    long num_grids_;
    int num_fields_;
    int num_par_types_;
    int num_grids_local_;

    double* grid_left_edge_;
    double* grid_right_edge_;
    int* grid_dimensions_;
    long* grid_parent_id_;
    int* grid_levels_;
    int* proc_num_;
    long* par_count_list_;

    // AMR data structure to data
    yt_field* field_list_;
    yt_particle* particle_list_;
    yt_grid* grids_local_;

    // Python bindings
    PyObject* py_grid_data_;
    PyObject* py_particle_data_;
    PyObject* py_hierarchy_;

public:
    DataStructureAmr();
    void SetUp(long num_grids, int num_grids_local, int num_fields, int num_par_types = 0);
    static void SetMpiInfo(const int mpi_size, const int mpi_root, const int mpi_rank) {
        mpi_size_ = mpi_size;
        mpi_root_ = mpi_root;
        mpi_rank_ = mpi_rank;
    }

    DataStructureAmr();
    void SetUp(long num_grids, int num_grids_local, int num_fields, int num_par_types = 0,
               yt_par_type* par_type_list = nullptr);
};

#endif  // LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
