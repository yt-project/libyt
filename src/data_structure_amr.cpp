#include "data_structure_amr.h"
#ifndef SERIAL_MODE
#include "big_mpi.h"
#include "comm_mpi.h"
#endif

#include "yt_combo.h"
#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

int DataStructureAmr::mpi_size_;
int DataStructureAmr::mpi_root_;
int DataStructureAmr::mpi_rank_;

static PyObject* WrapToNumPyArray(int dim, npy_intp* npy_dim, yt_dtype data_dtype, void* data_ptr) {
    int npy_dtype;
    get_npy_dtype(data_dtype, &npy_dtype);
    PyObject* py_data = PyArray_SimpleNewFromData(dim, npy_dim, npy_dtype, data_ptr);
    return py_data;
}

DataStructureAmr::DataStructureAmr()
    : num_grids_(0),
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
    // TODO: return error msg if < 0
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

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  AllocateAllHierarchyStorageForPython
//
// Notes       :  1. Allocate hierarchy for Python bindings
//                2. TODO: need to create another function to free it.
//                3. TODO: acutally, I'm not sure if data structure contains python code is a good idea.
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::AllocateAllHierarchyStorageForPython() {
    // Allocate storage
    grid_left_edge_ = new double[num_grids_ * 3];
    grid_right_edge_ = new double[num_grids_ * 3];
    grid_dimensions_ = new int[num_grids_ * 3];
    grid_parent_id_ = new long[num_grids_];
    grid_levels_ = new int[num_grids_];
    proc_num_ = new int[num_grids_];
    if (num_par_types_ > 0) {
        par_count_list_ = new long[num_grids_ * num_par_types_];
    } else {
        par_count_list_ = nullptr;
    }

    // Bind to Python
    npy_intp np_dim[2];
    np_dim[0] = num_grids_;

    np_dim[1] = 3;
    PyObject* py_grid_left_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, grid_left_edge_);
    PyObject* py_grid_right_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, grid_right_edge_);
    PyObject* py_grid_dimensions = WrapToNumPyArray(2, np_dim, YT_INT, grid_dimensions_);

    np_dim[1] = 1;
    PyObject* py_grid_parent_id = WrapToNumPyArray(2, np_dim, YT_LONG, grid_parent_id_);
    PyObject* py_grid_levels = WrapToNumPyArray(2, np_dim, YT_INT, grid_levels_);
    PyObject* py_proc_num = WrapToNumPyArray(2, np_dim, YT_INT, proc_num_);
    PyObject* py_par_count_list;
    if (num_par_types_ > 0) {
        np_dim[1] = num_par_types_;
        py_par_count_list = WrapToNumPyArray(2, np_dim, YT_LONG, par_count_list_);
    }

    // Bind them to libyt.hierarchy
    // Even though the pointer is de-referenced, still need to freed it in the memory ourselves at freed
#ifndef USE_PYBIND11
    PyDict_SetItemString(py_hierarchy_, "grid_left_edge", py_grid_left_edge);
    PyDict_SetItemString(py_hierarchy_, "grid_right_edge", py_grid_right_edge);
    PyDict_SetItemString(py_hierarchy_, "grid_dimensions", py_grid_dimensions);
    PyDict_SetItemString(py_hierarchy_, "grid_parent_id", py_grid_parent_id);
    PyDict_SetItemString(py_hierarchy_, "grid_levels", py_grid_levels);
    PyDict_SetItemString(py_hierarchy_, "proc_num", py_proc_num);
    if (param_yt.num_par_types > 0) {
        PyDict_SetItemString(py_hierarchy_, "par_count_list", py_par_count_list);
    }
#else   // #ifndef USE_PYBIND11
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_hierarchy = libyt.attr("hierarchy");

    py_hierarchy["grid_left_edge"] = py_grid_left_edge;
    py_hierarchy["grid_right_edge"] = py_grid_right_edge;
    py_hierarchy["grid_dimensions"] = py_grid_dimensions;
    py_hierarchy["grid_parent_id"] = py_grid_parent_id;
    py_hierarchy["grid_levels"] = py_grid_levels;
    py_hierarchy["proc_num"] = py_proc_num;
    if (num_par_types_ > 0) {
        py_hierarchy["par_count_list"] = py_par_count_list;
    }
#endif  // #ifndef USE_PYBIND11

    // Deref, since we didn't make the array owned by python
    Py_DECREF(py_grid_left_edge);
    Py_DECREF(py_grid_right_edge);
    Py_DECREF(py_grid_dimensions);
    Py_DECREF(py_grid_parent_id);
    Py_DECREF(py_grid_levels);
    Py_DECREF(py_proc_num);
    if (num_par_types_ > 0) {
        Py_DECREF(py_par_count_list);
    }
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  GatherAllHierarchy
//
// Notes       :  1. Gather hierarchy from different ranks to root rank, and then broadcast it to all ranks.
//                2. It stores the output in pointer passed in by the client, and it needs to be freed once
//                   it's done.
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::GatherAllHierarchy(int mpi_root, yt_hierarchy** full_hierarchy_ptr,
                                          long*** full_particle_count_ptr) {
#ifndef SERIAL_MODE
    // Get num_grids_local in different ranks
    int* all_num_grids_local = new int[mpi_size_];
    CommMpi::SetAllNumGridsLocal(all_num_grids_local, num_grids_local_);  // TODO: call check sum of all_num_grids_local

    // Prepare storage for Mpi
    yt_hierarchy* hierarchy_full = new yt_hierarchy[num_grids_];
    yt_hierarchy* hierarchy_local = new yt_hierarchy[num_grids_local_];
    long** particle_count_list_full = new long*[num_par_types_];
    long** particle_count_list_local = new long*[num_par_types_];
    for (int s = 0; s < num_par_types_; s++) {
        particle_count_list_full[s] = new long[num_grids_];
        particle_count_list_local[s] = new long[num_grids_local_];
    }

    // Copy and prepare data for Mpi (TODO: Can I not use hierarchy_local/particle_local?
    for (int i = 0; i < num_grids_local_; i = i + 1) {
        yt_grid& grid = grids_local_[i];
        for (int d = 0; d < 3; d = d + 1) {
            hierarchy_local[i].left_edge[d] = grid.left_edge[d];
            hierarchy_local[i].right_edge[d] = grid.right_edge[d];
            hierarchy_local[i].dimensions[d] = grid.grid_dimensions[d];
        }
        for (int s = 0; s < num_par_types_; s = s + 1) {
            particle_count_list_local[s][i] = grid.par_count_list[s];
        }
        hierarchy_local[i].id = grid.id;
        hierarchy_local[i].parent_id = grid.parent_id;
        hierarchy_local[i].level = grid.level;
        hierarchy_local[i].proc_num = grid.proc_num;
    }

    // TODO: create big_MPI_AllGatherv, since we are going to check data in every rank
    big_MPI_Gatherv<yt_hierarchy>(mpi_root, all_num_grids_local, (void*)hierarchy_local,
                                  &CommMpi::yt_hierarchy_mpi_type_, (void*)hierarchy_full);
    for (int s = 0; s < num_par_types_; s++) {
        big_MPI_Gatherv<long>(mpi_root, all_num_grids_local, (void*)particle_count_list_local[s],
                              &CommMpi::yt_long_mpi_type_, (void*)particle_count_list_full[s]);
    }
    // broadcast hierarchy_full, particle_count_list_full to each rank as well.
    big_MPI_Bcast<yt_hierarchy>(mpi_root, num_grids_, (void*)hierarchy_full, &CommMpi::yt_hierarchy_mpi_type_);
    for (int s = 0; s < num_par_types_; s++) {
        big_MPI_Bcast<long>(mpi_root, num_grids_, (void*)particle_count_list_full[s], &CommMpi::yt_long_mpi_type_);
    }

    // Return the full hierarchy and particle count list
    *full_hierarchy_ptr = hierarchy_full;
    *full_particle_count_ptr = particle_count_list_full;

    // Clean up
    delete[] all_num_grids_local;
    delete[] hierarchy_local;
    for (int s = 0; s < num_par_types_; s++) {
        delete[] particle_count_list_local[s];
    }
    delete[] particle_count_list_local;
#endif
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  BindAllHierarchyToPython
//
// Notes       :  1. Check data if check_data is true.
//                2. If it is under Mpi mode, we need to gather hierarchy from different ranks to all ranks.
//                3. TODO: Need to check if py_hierarchy contains data, not sure if I should put it here or
//                         in AllocateAllHierarchyStorageForPython. (Do this when checking Public API)
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::BindAllHierarchyToPython(int mpi_root, bool check_data) {
#ifndef SERIAL_MODE
    // Gather hierarchy from different ranks to root rank.
    yt_hierarchy* hierarchy_full = nullptr;
    long** particle_count_list_full = nullptr;
    GatherAllHierarchy(mpi_root, &hierarchy_full, &particle_count_list_full);  // TODO: check the hierarchy
#endif

    // Allocate memory for full hierarchy and bind it to Python
    AllocateAllHierarchyStorageForPython();

    // Bind hierarchy to Python
    // TODO: start here

    // Clean up
#ifndef SERIAL_MODE
    delete[] hierarchy_full;
    for (int s = 0; s < num_par_types_; s++) {
        delete[] particle_count_list_full[s];
    }
    delete[] particle_count_list_full;
#endif
}

void DataStructureAmr::BindLocalDataToPython() {}
void DataStructureAmr::CleanUp() {}
