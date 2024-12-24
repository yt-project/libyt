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

//-------------------------------------------------------------------------------------------------------
// Class         :  DataStructureAmr
// Public Method :  SetUp
//
// Notes       :  1. Initialize the Amr structure storage.
//                   (1) Field list
//                   (2) Particle list
//                   (3) Local grid list
//                   (4) Hierarchy bindings at C-side
//                2. Particles and its type list are optional input. If num_par_types > 0, then par_type_list
//                   is read.
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::SetUp(long num_grids, int num_grids_local, int num_fields, int num_par_types,
                             yt_par_type* par_type_list) {
    num_grids_ = num_grids;
    num_fields_ = num_fields;
    num_par_types_ = num_par_types;
    num_grids_local_ = num_grids_local;
}
