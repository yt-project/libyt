#include "data_structure_amr.h"

DataStructureAmr::DataStructureAmr()
    : all_num_grids_local_(nullptr),
      grid_left_edge_(nullptr),
      grid_right_edge_(nullptr),
      grid_dimensions_(nullptr),
      grid_parent_id_(nullptr),
      grid_levels_(nullptr),
      proc_num_(nullptr),
      par_count_list_(nullptr),
      field_list_(nullptr),
      particle_list_(nullptr),
      grids_local_(nullptr),
      py_grid_data_(nullptr),
      py_particle_data_(nullptr),
      py_hierarchy_(nullptr) {}