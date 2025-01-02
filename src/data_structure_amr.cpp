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

//-------------------------------------------------------------------------------------------------------
// Helper Function : WrapToNumPyArray
//
// Notes           :  1. The array wrapped is still owned by us.
//                    2. If it is read-only, then we need to clear the NPY_ARRAY_WRITEABLE flag.
//-------------------------------------------------------------------------------------------------------
static PyObject* WrapToNumPyArray(int dim, npy_intp* npy_dim, yt_dtype data_dtype, void* data_ptr, bool readonly) {
    int npy_dtype;
    get_npy_dtype(data_dtype, &npy_dtype);
    PyObject* py_data = PyArray_SimpleNewFromData(dim, npy_dim, npy_dtype, data_ptr);

    if (readonly) {
        PyArray_CLEARFLAGS((PyArrayObject*)py_data, NPY_ARRAY_WRITEABLE);
    }

    return py_data;
}

DataStructureAmr::DataStructureAmr()
    : num_grids_(0),
      num_fields_(0),
      num_par_types_(0),
      num_grids_local_(0),
      index_offset_(0),
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
// Public Method :  SetPythonBindings
//
// Notes       :  1. Set the PyObject* pointer for libyt.hierarchy, libyt.grid_data, libyt.particle_data
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::SetPythonBindings(PyObject* py_hierarchy, PyObject* py_grid_data, PyObject* py_particle_data) {
    py_hierarchy_ = py_hierarchy;
    py_grid_data_ = py_grid_data;
    py_particle_data_ = py_particle_data;
}

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
DataStructureOutput DataStructureAmr::SetUp(long num_grids, int num_grids_local, int num_fields, int num_par_types,
                                            yt_par_type* par_type_list, int index_offset) {
    if (num_grids < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of grids should not be negative."};
    }
    if (num_grids_local < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of local grids should not be negative."};
    }
    if (num_fields < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of fields should not be negative."};
    }
    if (num_par_types < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of particle types should not be negative."};
    } else if (num_par_types > 0 && par_type_list == nullptr) {
        return {DataStructureStatus::kDataStructureFailed, "Particle type list is not set."};
    }

    num_grids_ = num_grids;
    num_grids_local_ = num_grids_local;
    num_fields_ = num_fields;
    num_par_types_ = num_par_types;
    index_offset_ = index_offset;
    has_particle_ = (num_par_types_ > 0);

    // Initialize the data structure
    AllocateFieldList();
    AllocateParticleList(par_type_list);
    AllocateGridsLocal();

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
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
//                2. TODO: I'm not sure if data structure contains python code is a good idea.
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
    PyObject* py_grid_left_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, grid_left_edge_, false);
    PyObject* py_grid_right_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, grid_right_edge_, false);
    PyObject* py_grid_dimensions = WrapToNumPyArray(2, np_dim, YT_INT, grid_dimensions_, false);

    np_dim[1] = 1;
    PyObject* py_grid_parent_id = WrapToNumPyArray(2, np_dim, YT_LONG, grid_parent_id_, false);
    PyObject* py_grid_levels = WrapToNumPyArray(2, np_dim, YT_INT, grid_levels_, false);
    PyObject* py_proc_num = WrapToNumPyArray(2, np_dim, YT_INT, proc_num_, false);
    PyObject* py_par_count_list;
    if (num_par_types_ > 0) {
        np_dim[1] = num_par_types_;
        py_par_count_list = WrapToNumPyArray(2, np_dim, YT_LONG, par_count_list_, false);
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
    if (num_par_types_ > 0) {
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
//                4. TODO: Do I need to move data twice, which is gathering data, and then move it to Python
//                         storage?
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
#ifndef SERIAL_MODE
    for (long i = 0; i < num_grids_; i++) {
        long index = hierarchy_full[i].id - index_offset_;
        for (int d = 0; d < 3; d++) {
            grid_left_edge_[index * 3 + d] = hierarchy_full[i].left_edge[d];
            grid_right_edge_[index * 3 + d] = hierarchy_full[i].right_edge[d];
            grid_dimensions_[index * 3 + d] = hierarchy_full[i].dimensions[d];
        }
        grid_parent_id_[index] = hierarchy_full[i].parent_id;
        grid_levels_[index] = hierarchy_full[i].level;
        proc_num_[index] = hierarchy_full[i].proc_num;
        if (num_par_types_ > 0) {
            for (int p = 0; p < num_par_types_; p++) {
                par_count_list_[index * num_par_types_ + p] = particle_count_list_full[p][i];
            }
        }
    }
#else
    for (long i = 0; i < num_grids_local_; i++) {
        long index = grids_local_[i].id - index_offset_;
        for (int d = 0; d < 3; d++) {
            grid_left_edge_[index * 3 + d] = grids_local_[i].left_edge[d];
            grid_right_edge_[index * 3 + d] = grids_local_[i].right_edge[d];
            grid_dimensions_[index * 3 + d] = grids_local_[i].grid_dimensions[d];
        }
        grid_parent_id_[index] = grids_local_[i].parent_id;
        grid_levels_[index] = grids_local_[i].level;
        proc_num_[index] = grids_local_[i].proc_num;
        if (num_par_types_ > 0) {
            for (int p = 0; p < num_par_types_; p++) {
                par_count_list_[index * num_par_types_ + p] = grids_local_[i].par_count_list[p];
            }
        }
    }
#endif

    // Clean up
#ifndef SERIAL_MODE
    delete[] hierarchy_full;
    for (int s = 0; s < num_par_types_; s++) {
        delete[] particle_count_list_full[s];
    }
    delete[] particle_count_list_full;
#endif
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  BindLocalFieldDataToPython
//
// Notes       :  1. Wrap and build field data to a dictionary in libyt.grid_data[gid][fname].
//                2. The key (gid, fname) will only be inside the dictionary only if the data is not nullptr.
//                3. TODO: Assume all field data under same grid id is passed in and wrapped at once.
//                         Maybe put building to a dictionary part at the end.
//                4. TODO: Currently, the API forces this function to bind and build all the data
//                         inside the grids_local_ array at once. Might change it in the future libyt v1.0.
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::BindLocalFieldDataToPython(const yt_grid& grid) const {
    PyObject *py_grid_id, *py_field_labels, *py_field_data;
    py_grid_id = PyLong_FromLong(grid.id);
    py_field_labels = PyDict_New();
    for (int v = 0; v < num_fields_; v++) {
        // append data to dict only if data is not NULL.
        if ((grid.field_data)[v].data_ptr == nullptr) continue;

        // check if dictionary exists, if no add new dict under key gid
        if (PyDict_Contains(py_grid_data_, py_grid_id) != 1) {
            PyDict_SetItem(py_grid_data_, py_grid_id, py_field_labels);
        }

        // insert data under py_field_labels dict
        // (1) Grab NumPy Enumerate Type in order: (1)data_dtype (2)field_dtype
        int grid_dtype;
        if (get_npy_dtype((grid.field_data)[v].data_dtype, &grid_dtype) == YT_SUCCESS) {
        } else if (get_npy_dtype(field_list_[v].field_dtype, &grid_dtype) == YT_SUCCESS) {
            (grid.field_data)[v].data_dtype = field_list_[v].field_dtype;
        } else {
            Py_DECREF(py_grid_id);
            Py_DECREF(py_field_labels);
            std::string error = "(grid id, field) = (" + std::to_string(grid.id) + ", " +
                                std::string(field_list_[v].field_name) + ") cannot get NumPy enumerate type properly.";
            return {DataStructureStatus::kDataStructureFailed, error};
        }

        // (2) Get the dimension of the input array
        // Only "cell-centered" will be set to grid_dimensions + ghost cell, else should be set in data_dimensions.
        if (strcmp(field_list_[v].field_type, "cell-centered") == 0) {
            // Get grid_dimensions and consider contiguous_in_x or not, since grid_dimensions is defined as [x][y][z].
            if (field_list_[v].contiguous_in_x) {
                for (int d = 0; d < 3; d++) {
                    (grid.field_data)[v].data_dimensions[d] = (grid.grid_dimensions)[2 - d];
                }
            } else {
                for (int d = 0; d < 3; d++) {
                    (grid.field_data)[v].data_dimensions[d] = (grid.grid_dimensions)[d];
                }
            }
            // Plus the ghost cell to get the actual array dimensions.
            for (int d = 0; d < 6; d++) {
                (grid.field_data)[v].data_dimensions[d / 2] += field_list_[v].field_ghost_cell[d];
            }
        }
        // See if all data_dimensions > 0, abort if not.
        for (int d = 0; d < 3; d++) {
            if ((grid.field_data)[v].data_dimensions[d] <= 0) {
                Py_DECREF(py_grid_id);
                Py_DECREF(py_field_labels);
                std::string error = "(grid id, field) = (" + std::to_string(grid.id) + ", " +
                                    std::string(field_list_[v].field_name) + ") data dimension " + std::to_string(d) +
                                    " is " + std::to_string((grid.field_data)[v].data_dimensions[d]) + " <= 0.";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
        }

        npy_intp grid_dims[3] = {(grid.field_data)[v].data_dimensions[0], (grid.field_data)[v].data_dimensions[1],
                                 (grid.field_data)[v].data_dimensions[2]};

        // (3) Insert data to dict
        // PyArray_SimpleNewFromData simply creates an array wrapper and does not allocate and own the array
        py_field_data = PyArray_SimpleNewFromData(3, grid_dims, grid_dtype, (grid.field_data)[v].data_ptr);

        // Mark this memory (NumPy array) read-only
        PyArray_CLEARFLAGS((PyArrayObject*)py_field_data, NPY_ARRAY_WRITEABLE);

        // add the field data to dict "libyt.grid_data[grid_id][field_list.field_name]"
        PyDict_SetItemString(py_field_labels, field_list_[v].field_name, py_field_data);

        // call decref since PyDict_SetItemString() returns a new reference
        Py_DECREF(py_field_data);
    }

    // call decref since both PyLong_FromLong() and PyDict_New() return a new reference
    Py_DECREF(py_grid_id);
    Py_DECREF(py_field_labels);
    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  BindLocalParticleDataToPython
//
// Notes       :  1. Wrap and build particle data to a dictionary in libyt.particle_data[gid][ptype][attr].
//                2. The key (gid, ptype, attr) will only be inside the dictionary only if the data is not nullptr.
//                3. TODO: Currently, the API forces this function to bind and build all the data
//                         inside the grids_local_ array at once. Might change it in the future libyt v1.0.
//                4. TODO: Future Api shouldn't make hierarchy and data to closely related, so that we can have
//                         more flexibility. Like Enzo contains particle data with tuple values.
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::BindLocalParticleDataToPython(const yt_grid& grid) const {
    PyObject *py_grid_id, *py_ptype_labels, *py_attributes, *py_data;
    py_grid_id = PyLong_FromLong(grid.id);
    py_ptype_labels = PyDict_New();
    for (int p = 0; p < num_par_types_; p++) {
        py_attributes = PyDict_New();
        for (int a = 0; a < particle_list_[p].num_attr; a++) {
            // skip if particle attribute pointer is NULL
            if ((grid.particle_data)[p][a].data_ptr == nullptr) continue;

            // Wrap the data array if pointer exist
            int data_dtype;
            if (get_npy_dtype(particle_list_[p].attr_list[a].attr_dtype, &data_dtype) != YT_SUCCESS) {
                Py_DECREF(py_attributes);
                Py_DECREF(py_ptype_labels);
                Py_DECREF(py_grid_id);
                std::string error = "(particle type, attribute) = (" + std::string(particle_list_[p].par_type) + ", " +
                                    std::string(particle_list_[p].attr_list[a].attr_name) +
                                    ") cannot get NumPy enumerate type properly.";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            if ((grid.par_count_list)[p] <= 0) {
                Py_DECREF(py_attributes);
                Py_DECREF(py_ptype_labels);
                Py_DECREF(py_grid_id);
                std::string error = "(particle type, grid id) = (" + std::string(particle_list_[p].par_type) + ", " +
                                    std::to_string(grid.id) + ") particle count is " +
                                    std::to_string((grid.par_count_list)[p]) + " <= 0.";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            npy_intp array_dims[1] = {(grid.par_count_list)[p]};
            py_data = PyArray_SimpleNewFromData(1, array_dims, data_dtype, (grid.particle_data)[p][a].data_ptr);
            PyArray_CLEARFLAGS((PyArrayObject*)py_data, NPY_ARRAY_WRITEABLE);

            // Get the dictionary and append py_data
            if (PyDict_Contains(py_particle_data_, py_grid_id) != 1) {
                // 1st time append, nothing exist under libyt.particle_data[gid]
                PyDict_SetItem(py_particle_data_, py_grid_id,
                               py_ptype_labels);  // libyt.particle_data[gid] = dict()
                PyDict_SetItemString(py_ptype_labels, particle_list_[p].par_type, py_attributes);
            } else {
                // libyt.particle_data[gid] exist, check if libyt.particle_data[gid][ptype] exist
                PyObject* py_ptype_name = PyUnicode_FromString(particle_list_[p].par_type);
                if (PyDict_Contains(py_ptype_labels, py_ptype_name) != 1) {
                    PyDict_SetItemString(py_ptype_labels, particle_list_[p].par_type, py_attributes);
                }
                Py_DECREF(py_ptype_name);
            }
            PyDict_SetItemString(py_attributes, particle_list_[p].attr_list[a].attr_name, py_data);
            Py_DECREF(py_data);
        }
        Py_DECREF(py_attributes);
    }

    Py_DECREF(py_ptype_labels);
    Py_DECREF(py_grid_id);

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  BindLocalDataToPython
//
// Notes       :  1. Wrap local data and build it to a dictionary.
//                2. TODO: Currently, the API forces this function to bind and build all the data
//                         inside the grids_local_ array at once. Might change it in the future libyt v1.0.
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::BindLocalDataToPython() {
    for (int i = 0; i < num_grids_local_; i++) {
        if (num_fields_ > 0) {
            BindLocalFieldDataToPython(grids_local_[i]);
        }
        if (num_par_types_ > 0) {
            BindLocalParticleDataToPython(grids_local_[i]);
        }
    }
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  CleanUpFieldList
//
// Notes       :  1. Clean up field_list_
//                2. Reset num_fields_ = 0 and field_list_ = nullptr.
//                3. Counterpart of AllocateFieldList().
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUpFieldList() {
    if (num_fields_ > 0) {
        delete[] field_list_;
    }

    num_fields_ = 0;
    field_list_ = nullptr;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  CleanUpFieldList
//
// Notes       :  1. Clean up particle_list_
//                2. Reset num_par_types_ = 0 and particle_list_ = nullptr.
//                3. Counterpart of AllocateParticleList().
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUpParticleList() {
    if (num_par_types_ > 0) {
        for (int i = 0; i < num_par_types_; i++) {
            delete[] particle_list_[i].attr_list;
        }
        delete[] particle_list_;
    }

    num_par_types_ = 0;
    particle_list_ = nullptr;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  CleanUpGridsLocal
//
// Notes       :  1. Clean up grids_local_, since it's only for user to pass in hierarchy and data.
//                2. Reset num_grids_local_ = 0 and grids_local_ = nullptr.
//                3. Counterpart of AllocateGridsLocal().
//                4. This method is separate from the rest of the clean up methods is because it cleans up
//                   the data for holding user input, which is not needed after committing everything.
//                   TODO: bad Api design
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUpGridsLocal() {
    if (num_grids_local_ > 0) {
        for (int i = 0; i < num_grids_local_; i = i + 1) {
            if (num_fields_ > 0) {
                delete[] grids_local_[i].field_data;
            }
            if (num_par_types_ > 0) {
                delete[] grids_local_[i].par_count_list;
                for (int p = 0; p < num_par_types_; p++) {
                    delete[] grids_local_[i].particle_data[p];
                }
                delete[] grids_local_[i].particle_data;
            }
        }
        delete[] grids_local_;
    }

    num_grids_local_ = 0;
    grids_local_ = nullptr;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  CleanUpGridsLocal
//
// Notes       :  1. Clean all hierarchy Python bindings
//                2. Counterpart for AllocateAllHierarchyStorageForPython().
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUpAllHierarchyStorageForPython() {
    // C storage
    delete[] grid_left_edge_;
    delete[] grid_right_edge_;
    delete[] grid_dimensions_;
    delete[] grid_parent_id_;
    delete[] grid_levels_;
    delete[] proc_num_;
    if (has_particle_) {
        delete[] par_count_list_;
    }
    grid_left_edge_ = nullptr;
    grid_right_edge_ = nullptr;
    grid_dimensions_ = nullptr;
    grid_parent_id_ = nullptr;
    grid_levels_ = nullptr;
    proc_num_ = nullptr;
    par_count_list_ = nullptr;

    // Python bindings
#ifndef USE_PYBIND11
    // Reset data in libyt module
    PyDict_Clear(py_hierarchy_);
#else
    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_dict = libyt.attr("hierarchy");
    py_dict.clear();
#endif
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  CleanUpLocalDataPythonBindings
//
// Notes       :  1. Clean local data Python bindings
//                2. Counterpart for BindLocalDataToPython().
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUpLocalDataPythonBindings() {
#ifndef USE_PYBIND11
    // Reset data in libyt module
    PyDict_Clear(py_grid_data_);
    PyDict_Clear(py_particle_data_);
#else
    pybind11::module_ libyt = pybind11::module_::import("libyt");

    const char* keys_to_clear[] = {"grid_data", "particle_data"};
    const int keys_len = 2;
    for (int i = 0; i < keys_len; i++) {
        pybind11::dict py_dict = libyt.attr(keys_to_clear[i]);
        py_dict.clear();
    }
#endif
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  CleanUp
//
// Notes       :  1. Clean up all the data structure and bindings to Python.
//                2. TODO: should I separate Python bindings into a new class?
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUp() {
    CleanUpFieldList();
    CleanUpParticleList();
    CleanUpGridsLocal();
    CleanUpAllHierarchyStorageForPython();
    CleanUpLocalDataPythonBindings();

    has_particle_ = false;
    num_grids_ = 0;
    index_offset_ = 0;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFullHierarchyGridDimensions
//
// Notes       :  1. Read the full hierarchy grid dimensions loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetFullHierarchyGridDimensions(long gid, int* dimensions) const {
    if (grid_dimensions_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    for (int d = 0; d < 3; d++) {
        dimensions[d] = grid_dimensions_[(gid - index_offset_) * 3 + d];
    }

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFullHierarchyGridLeftEdge
//
// Notes       :  1. Read the full hierarchy grid left edge loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetFullHierarchyGridLeftEdge(long gid, double* left_edge) const {
    if (grid_left_edge_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    for (int d = 0; d < 3; d++) {
        left_edge[d] = grid_left_edge_[(gid - index_offset_) * 3 + d];
    }

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFullHierarchyGridRightEdge
//
// Notes       :  1. Read the full hierarchy grid right edge loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetFullHierarchyGridRightEdge(long gid, double* right_edge) const {
    if (grid_right_edge_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    for (int d = 0; d < 3; d++) {
        right_edge[d] = grid_right_edge_[(gid - index_offset_) * 3 + d];
    }

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFullHierarchyGridParentId
//
// Notes       :  1. Read the full hierarchy grid parent id loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetFullHierarchyGridParentId(long gid, long* parent_id) const {
    if (grid_parent_id_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    *parent_id = grid_parent_id_[gid - index_offset_];

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFullHierarchyGridLevel
//
// Notes       :  1. Read the full hierarchy grid level loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetFullHierarchyGridLevel(long gid, int* level) const {
    if (grid_levels_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    *level = grid_levels_[gid - index_offset_];

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFullHierarchyGridProcNum
//
// Notes       :  1. Read the full hierarchy grid proc number (mpi rank) loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetFullHierarchyGridProcNum(long gid, int* proc_num) const {
    if (proc_num_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    *proc_num = proc_num_[gid - index_offset_];

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFullHierarchyGridParticleCount
//
// Notes       :  1. Read the full hierarchy grid particle count for a ptype loaded in Python.
//                2. This method is only valid if the data structure contains particle data.
//                3. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetFullHierarchyGridParticleCount(long gid, const char* ptype,
                                                                        long* par_count) const {
    if (!has_particle_) {
        std::string error = "Doesn't contain particle data.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if (par_count_list_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    // Find index of ptype
    if (particle_list_ == nullptr) {
        std::string error = "Particle list is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    int label = -1;
    for (int s = 0; s < num_par_types_; s++) {
        if (strcmp(particle_list_[s].par_type, ptype) == 0) {
            label = s;
            break;
        }
    }
    if (label == -1) {
        std::string error = "Cannot find (particle type) = " + std::string(ptype) + " in particle_list.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    *par_count = par_count_list_[(gid - index_offset_) * num_par_types_ + label];

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetLocalFieldData
//
// Notes       :  1. Read the local field data bind to Python libyt.grid_data[gid][fname].
//                2. Counterpart of BindLocalFieldDataToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetLocalFieldData(long gid, const char* field_name, yt_data* field_data) const {
    // Get dictionary libyt.grid_data[gid][fname]
    PyObject* py_grid_id = PyLong_FromLong(gid);
    PyObject* py_field = PyUnicode_FromString(field_name);

    if (PyDict_Contains(py_grid_data_, py_grid_id) != 1 ||
        PyDict_Contains(PyDict_GetItem(py_grid_data_, py_grid_id), py_field) != 1) {
        std::string error = "Cannot find field data (grid id, field) = " + std::to_string(gid) + ", " + field_name +
                            " on MPI rank " + std::to_string(mpi_rank_) + ".\n";
        Py_DECREF(py_grid_id);
        Py_DECREF(py_field);
        return {DataStructureStatus::kDataStructureFailed, error};
    }
    PyArrayObject* py_array_obj = (PyArrayObject*)PyDict_GetItem(PyDict_GetItem(py_grid_data_, py_grid_id), py_field);

    Py_DECREF(py_grid_id);
    Py_DECREF(py_field);

    // Get NumPy array dimensions/data pointer/dtype
    npy_intp* py_array_dims = PyArray_DIMS(py_array_obj);
    for (int d = 0; d < 3; d++) {
        (*field_data).data_dimensions[d] = (int)py_array_dims[d];
    }
    (*field_data).data_ptr = PyArray_DATA(py_array_obj);
    PyArray_Descr* py_array_info = PyArray_DESCR(py_array_obj);
    if (get_yt_dtype_from_npy(py_array_info->type_num, &(*field_data).data_dtype) != YT_SUCCESS) {
        std::string error =
            "No matching yt_dtype for NumPy data type num " + std::to_string(py_array_info->type_num) + ".\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetLocalParticleData
//
// Notes       :  1. Read the local field data bind to Python libyt.particle_data[gid][ptype][attr].
//                2. Counterpart of BindLocalParticleDataToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetLocalParticleData(long gid, const char* ptype, const char* attr,
                                                           yt_data* par_data) const {
    // Get dictionary libyt.particle_data[gid][ptype]
    PyObject* py_grid_id = PyLong_FromLong(gid);
    PyObject* py_ptype = PyUnicode_FromString(ptype);
    PyObject* py_attr = PyUnicode_FromString(attr);

    if (PyDict_Contains(py_particle_data_, py_grid_id) != 1 ||
        PyDict_Contains(PyDict_GetItem(py_particle_data_, py_grid_id), py_ptype) != 1 ||
        PyDict_Contains(PyDict_GetItem(PyDict_GetItem(py_particle_data_, py_grid_id), py_ptype), py_attr) != 1) {
        Py_DECREF(py_grid_id);
        Py_DECREF(py_ptype);
        Py_DECREF(py_attr);

        std::string error = "Cannot find particle data (grid id, particle type, attribute) = " + std::to_string(gid) +
                            ", " + ptype + ", " + attr + " on MPI rank " + std::to_string(mpi_rank_) + ".\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }
    PyArrayObject* py_data = (PyArrayObject*)PyDict_GetItem(
        PyDict_GetItem(PyDict_GetItem(py_particle_data_, py_grid_id), py_ptype), py_attr);

    Py_DECREF(py_grid_id);
    Py_DECREF(py_ptype);
    Py_DECREF(py_attr);

    // Get NumPy array dimensions/data pointer/dtype
    npy_intp* py_data_dims = PyArray_DIMS(py_data);
    (*par_data).data_dimensions[0] = (int)py_data_dims[0];
    (*par_data).data_dimensions[1] = 0;
    (*par_data).data_dimensions[2] = 0;
    (*par_data).data_ptr = PyArray_DATA(py_data);
    PyArray_Descr* py_data_info = PyArray_DESCR(py_data);
    if (get_yt_dtype_from_npy(py_data_info->type_num, &(*par_data).data_dtype) != YT_SUCCESS) {
        std::string error =
            "No matching yt_dtype for NumPy data type num " + std::to_string(py_data_info->type_num) + ".\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}
