#ifndef LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
#define LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_

#include <Python.h>

#include "yt_type.h"

class DataStructureAmr {
public:
    int* all_num_grids_local_;

    // Hierarchy
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
};

#endif  // LIBYT_PROJECT_INCLUDE_DATA_STRUCTURE_AMR_H_
