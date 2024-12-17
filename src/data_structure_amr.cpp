#include "data_structure_amr.h"

int DataStructureAmr::mpi_size_;
int DataStructureAmr::mpi_root_;
int DataStructureAmr::mpi_rank_;

DataStructureAmr::DataStructureAmr()
    : all_num_grids_local_(nullptr),
      num_grids_(0),
      num_fields_(0),
      num_par_types_(0),
      num_grids_local_(0),
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

void DataStructureAmr::SetUp(long num_grids, int num_grids_local, int num_fields, int num_par_types) {
    num_grids_ = num_grids;
    num_fields_ = num_fields;
    num_par_types_ = num_par_types;
    num_grids_local_ = num_grids_local;
}
