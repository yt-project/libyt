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

    // Initialize the data structure
    AllocateFieldList();
    AllocateParticleList(par_type_list);
    AllocateGridsLocal();
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  AllocateFieldList
//
// Notes       :  1. Allocate and initialize storage for field list.
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::AllocateFieldList() {
    if (num_fields_ > 0) {
        field_list_ = new yt_field[num_fields_];
    } else {
        field_list_ = nullptr;
    }
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  AllocateParticleList
//
// Notes       :  1. Allocate and initialize storage for particle list.
//                2. Since we need particle type and its number of attributes, we need to pass in the
//                   particle type list.
//                   (TODO: this part can be better, when designing libyt-v1.0)
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::AllocateParticleList(yt_par_type* par_type_list) {
    if (num_par_types_ > 0) {
        particle_list_ = new yt_particle[num_par_types_];
        for (int s = 0; s < num_par_types_; s++) {
            particle_list_[s].par_type = par_type_list[s].par_type;
            particle_list_[s].num_attr = par_type_list[s].num_attr;
            particle_list_[s].attr_list = new yt_attribute[particle_list_[s].num_attr];
        }
    } else {
        particle_list_ = nullptr;
    }
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  AllocateGridsLocal
//
// Notes       :  1. Allocate and initialize storage for grid list.
//                2. Initialize field_data and particle_data in one grid with
//                   (1) data_dimensions[3] = {0, 0, 0}
//                   (2) data_ptr           = NULL
//                   (3) data_dtype         = YT_DTYPE_UNKNOWN
//                   field_data[0]       represents field_list[0]
//                   particle_data[0][1] represents particle_list[0].attr_list[1]
//                3. This is only for user to pass in hierarchy and data fields to wrap,
//                   should be removed in the future.
//                   (TODO: this part can be better, when designing libyt-v1.0)
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::AllocateGridsLocal() {
    if (num_grids_local_ > 0) {
        grids_local_ = new yt_grid[num_grids_local_];

        for (int lid = 0; lid < num_grids_local_; lid++) {
            grids_local_[lid].proc_num = mpi_rank_;

            // Array for storing pointers for fields in a grid
            if (num_fields_ > 0) {
                grids_local_[lid].field_data = new yt_data[num_fields_];
            } else {
                grids_local_[lid].field_data = nullptr;
            }

            // Array for storing pointers for different particle data attributes in a grid.
            // Ex: particle_data[0][1] represents particle_list[0].attr_list[1] data
            if (num_par_types_ > 0) {
                grids_local_[lid].particle_data = new yt_data*[num_par_types_];
                for (int p = 0; p < num_par_types_; p++) {
                    grids_local_[lid].particle_data[p] = new yt_data[particle_list_[p].num_attr];
                }

                grids_local_[lid].par_count_list = new long[num_par_types_];
                for (int s = 0; s < num_par_types_; s++) {
                    grids_local_[lid].par_count_list[s] = 0;
                }
            } else {
                grids_local_[lid].particle_data = nullptr;
                grids_local_[lid].par_count_list = nullptr;
            }
        }
    } else {
        grids_local_ = nullptr;
    }
}

void DataStructureAmr::AllocateHierarchy() {}
