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

//-------------------------------------------------------------------------------------------------------
// Class         :  DataStructureAmr
// Public Method :  Constructor
//
// Notes       :  1. Doesn't contain necessary information to initialize the data structure.
//-------------------------------------------------------------------------------------------------------
DataStructureAmr::DataStructureAmr()
    : check_data_(false),
      num_grids_(0),
      num_fields_(0),
      num_par_types_(0),
      num_grids_local_(0),
      num_grids_local_field_data_(0),
      num_grids_local_par_data_(0),
      has_particle_(false),
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
      py_hierarchy_(nullptr),
      py_grid_data_(nullptr),
      py_particle_data_(nullptr) {}

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
// Public Method :  AllocateStorage
//
// Notes       :  1. Initialize the Amr structure storage.
//                   (1) Field list
//                   (2) Particle list
//                   (3) Local grid list
//                   (4) Hierarchy bindings at C-side
//                2. Make sure it is cleaned up before calling this.
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::AllocateStorage(long num_grids, int num_grids_local, int num_fields,
                                                      int num_par_types, yt_par_type* par_type_list, int index_offset,
                                                      bool check_data) {
    // Initialize the data structure
    DataStructureOutput status;

    status = AllocateFieldList(num_fields);
    if (status.status != DataStructureStatus::kDataStructureSuccess) {
        return {DataStructureStatus::kDataStructureFailed, status.error};
    }

    status = AllocateParticleList(num_par_types, par_type_list);
    if (status.status != DataStructureStatus::kDataStructureSuccess) {
        return {DataStructureStatus::kDataStructureFailed, status.error};
    }

    status = AllocateGridsLocal(num_grids_local, num_fields, num_par_types, par_type_list);
    if (status.status != DataStructureStatus::kDataStructureSuccess) {
        return {DataStructureStatus::kDataStructureFailed, status.error};
    }

    status = AllocateFullHierarchyStorageForPython(num_grids, num_par_types);
    if (status.status != DataStructureStatus::kDataStructureSuccess) {
        return {DataStructureStatus::kDataStructureFailed, status.error};
    }

    index_offset_ = index_offset;
    check_data_ = check_data;

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  AllocateFieldList
//
// Notes       :  1. Allocate and initialize storage for field list.
//                2. num_fields_ tracks the array length of field_list_.
//                3. Make sure field_list_ is properly freed before new allocation.
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::AllocateFieldList(int num_fields) {
    if (num_fields < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of fields should not be negative."};
    }

    if (field_list_ != nullptr) {
        CleanUpFieldList();
    }

    if (num_fields > 0) {
        field_list_ = new yt_field[num_fields];
        num_fields_ = num_fields;
    }

    return {DataStructureStatus::kDataStructureSuccess, ""};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  AllocateParticleList
//
// Notes       :  1. Allocate and initialize storage for particle list.
//                2. Since we need particle type and its number of attributes, we need to pass in the
//                   particle type list. And num_par_types_ tracks the array length of particle_list_.
//                3. Make sure particle_list_ is properly freed before new allocation.
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::AllocateParticleList(int num_par_types, yt_par_type* par_type_list) {
    if (num_par_types < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of particle types should not be negative."};
    }
    if (num_par_types > 0 && par_type_list == nullptr) {
        return {DataStructureStatus::kDataStructureFailed, "Particle type list is not set."};
    }

    if (particle_list_ != nullptr) {
        CleanUpParticleList();
    }

    if (num_par_types > 0) {
        particle_list_ = new yt_particle[num_par_types];
        for (int s = 0; s < num_par_types; s++) {
            particle_list_[s].par_type = par_type_list[s].par_type;
            particle_list_[s].num_attr = par_type_list[s].num_attr;
            particle_list_[s].attr_list = new yt_attribute[par_type_list[s].num_attr];
        }
        num_par_types_ = num_par_types;
    }

    return {DataStructureStatus::kDataStructureSuccess, ""};
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
//                4. num_grids_locals_/num_grids_local_field_data_/num_grids_local_par_data_ tracks the array length
//                   of grids_local. (TODO: bad Api design)
//                   Make sure grids_local_ is properly freed before new allocation.
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::AllocateGridsLocal(int num_grids_local, int num_fields, int num_par_types,
                                                         yt_par_type* par_type_list) {
    if (num_grids_local < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of local grids should not be negative."};
    }
    if (num_fields < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of fields should not be negative."};
    }
    if (num_par_types < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of particle types should not be negative."};
    }
    if (num_par_types > 0 && par_type_list == nullptr) {
        return {DataStructureStatus::kDataStructureFailed, "Particle type list is not set."};
    }

    if (grids_local_ != nullptr) {
        CleanUpGridsLocal();
    }

    if (num_grids_local > 0) {
        grids_local_ = new yt_grid[num_grids_local];

        for (int lid = 0; lid < num_grids_local; lid++) {
            grids_local_[lid].proc_num = mpi_rank_;

            // Array for storing pointers for fields in a grid
            if (num_fields > 0) {
                grids_local_[lid].field_data = new yt_data[num_fields];
            } else {
                grids_local_[lid].field_data = nullptr;
            }

            // Array for storing pointers for different particle data attributes in a grid.
            // Ex: particle_data[0][1] represents particle_list[0].attr_list[1] data
            if (num_par_types > 0) {
                grids_local_[lid].particle_data = new yt_data*[num_par_types];
                for (int p = 0; p < num_par_types; p++) {
                    grids_local_[lid].particle_data[p] = new yt_data[par_type_list[p].num_attr];
                }

                grids_local_[lid].par_count_list = new long[num_par_types];
                for (int s = 0; s < num_par_types; s++) {
                    grids_local_[lid].par_count_list[s] = 0;
                }
            } else {
                grids_local_[lid].particle_data = nullptr;
                grids_local_[lid].par_count_list = nullptr;
            }
        }

        num_grids_local_ = num_grids_local;
        num_grids_local_field_data_ = num_fields;
        num_grids_local_par_data_ = num_par_types;
    }

    return {DataStructureStatus::kDataStructureSuccess, ""};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  AllocateFullHierarchyStorageForPython
//
// Notes       :  1. Allocate full hierarchy storage for Python bindings.
//                2. Make sure it is empty before creating a new allocation.
//                   If it is not empty, over-write the existing one.
//                3. num_grids_/has_particle_ are used to track the allocation status of the hierarchy.
//                   has_particle_ is set through num_par_types.
//                   Make sure hierarchy is properly freed before new allocation.
//                4. I'm not sure if data structure contains python code is a good idea.
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::AllocateFullHierarchyStorageForPython(long num_grids, int num_par_types) {
    if (num_grids < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of grids should not be negative."};
    }
    if (num_par_types < 0) {
        return {DataStructureStatus::kDataStructureFailed, "Number of particle types should not be negative."};
    }

    if (grid_left_edge_ != nullptr) {
        CleanUpFullHierarchyStorageForPython();
    }

    // Allocate storage
    grid_left_edge_ = new double[num_grids * 3];
    grid_right_edge_ = new double[num_grids * 3];
    grid_dimensions_ = new int[num_grids * 3];
    grid_parent_id_ = new long[num_grids];
    grid_levels_ = new int[num_grids];
    proc_num_ = new int[num_grids];
    if (num_par_types > 0) {
        par_count_list_ = new long[num_grids * num_par_types];
    } else {
        par_count_list_ = nullptr;
    }

    // Bind to Python
    npy_intp np_dim[2];
    np_dim[0] = num_grids;

    np_dim[1] = 3;
    PyObject* py_grid_left_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, grid_left_edge_, false);
    PyObject* py_grid_right_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, grid_right_edge_, false);
    PyObject* py_grid_dimensions = WrapToNumPyArray(2, np_dim, YT_INT, grid_dimensions_, false);

    np_dim[1] = 1;
    PyObject* py_grid_parent_id = WrapToNumPyArray(2, np_dim, YT_LONG, grid_parent_id_, false);
    PyObject* py_grid_levels = WrapToNumPyArray(2, np_dim, YT_INT, grid_levels_, false);
    PyObject* py_proc_num = WrapToNumPyArray(2, np_dim, YT_INT, proc_num_, false);
    PyObject* py_par_count_list;
    if (num_par_types > 0) {
        np_dim[1] = num_par_types;
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
    if (num_par_types > 0) {
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
    if (num_par_types > 0) {
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
    if (num_par_types > 0) {
        Py_DECREF(py_par_count_list);
    }

    num_grids_ = num_grids;
    has_particle_ = (num_par_types > 0);

    return {DataStructureStatus::kDataStructureSuccess, ""};
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
                                          long*** full_particle_count_ptr) const {
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
// Private Method :  BindFieldListToPython
//
// Notes       :  1. Bind field_list_ to Python dictionary passed in.
//                   The data is not directly link, the data is copied.
//                2. TODO: Due to bad api design, field_list_ is created under libyt.param_yt["field_list"]
//                         as a dictionary.
//                3. If using Python C API, PyUnicode_FromString is Python-API >= 3.5, and it returns a new reference.
//                4. When creating key-value pair under this dictionary:
//                   (1) assume that we have all the field name unique.
//                   (2) field_display_name is NULL, set it to Py_None.
//                   (3) Structure of each element in libyt.param_yt["field_list"] dictionary:
//            field_list_dict    field_info_dict        info_list     name_alias_list
//                   |               |                      |               |
//                   { <field_name>: {"attribute"       : ( <field_unit>, (<field_name_alias>, ), <field_display_name>)
//                                    "field_type"      :  <field_type>,
//                                    "contiguous_in_x" :  true / false
//                                    "ghost_cell"      : ( beginning of 0-dim, ending of 0-dim,
//                                                          beginning of 1-dim, ending of 1-dim,
//                                                          beginning of 2-dim, ending of 2-dim  ) },
//                   }
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::BindFieldListToPython(PyObject* py_dict, const std::string& py_dict_name) const {
#ifndef USE_PYBIND11
    PyObject* field_list_dict = PyDict_New();
    PyObject *key, *val;

    for (int i = 0; i < num_fields_; i++) {
        PyObject* field_info_dict = PyDict_New();
        PyObject* info_list = PyList_New(0);

        // Append "field_unit" to "info_list"
        val = PyUnicode_FromString((field_list_)[i].field_unit);
        if (PyList_Append(info_list, val) != 0) {
            Py_DECREF(field_list_dict);
            Py_DECREF(field_info_dict);
            Py_DECREF(info_list);
            Py_XDECREF(val);
            std::string error = "(field, field_unit) = (" + std::string((field_list_)[i].field_name) + ", " +
                                std::string((field_list_)[i].field_unit) + "), failed to append field_unit to list!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(val);

        // Create "name_alias_list" and append to "info_list"
        PyObject* name_alias_list = PyList_New(0);
        for (int j = 0; j < (field_list_)[i].num_field_name_alias; j++) {
            val = PyUnicode_FromString((field_list_)[i].field_name_alias[j]);
            if (PyList_Append(name_alias_list, val) != 0) {
                Py_DECREF(field_list_dict);
                Py_DECREF(field_info_dict);
                Py_DECREF(info_list);
                Py_DECREF(name_alias_list);
                Py_XDECREF(val);
                std::string error = "(field, field_name_alias) = (" + std::string((field_list_)[i].field_name) + ", " +
                                    std::string((field_list_)[i].field_name_alias[j]) +
                                    "), failed to append alias name to field_name_alias list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(val);
        }
        if (PyList_Append(info_list, name_alias_list) != 0) {
            Py_DECREF(field_list_dict);
            Py_DECREF(field_info_dict);
            Py_DECREF(info_list);
            Py_DECREF(name_alias_list);
            std::string error = "(field) = (" + std::string((field_list_)[i].field_name) +
                                "), failed to append field_name_alias to list!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(name_alias_list);

        // Load "field_display_name" to "info_list"
        // If field_display_name == NULL, load Py_None.
        if ((field_list_)[i].field_display_name == nullptr) {
            if (PyList_Append(info_list, Py_None) != 0) {
                Py_DECREF(field_list_dict);
                Py_DECREF(field_info_dict);
                Py_DECREF(info_list);
                std::string error = "(field, field_display_name) = (" + std::string((field_list_)[i].field_name) +
                                    ", nullptr), failed to append Py_None to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
        } else {
            val = PyUnicode_FromString((field_list_)[i].field_display_name);
            if (PyList_Append(info_list, val) != 0) {
                Py_DECREF(field_list_dict);
                Py_DECREF(field_info_dict);
                Py_DECREF(info_list);
                Py_XDECREF(val);
                std::string error = "(field, field_display_name) = (" + std::string((field_list_)[i].field_name) +
                                    ", " + std::string((field_list_)[i].field_display_name) +
                                    "), failed to append field_display_name to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(val);
        }

        // Insert "info_list" to "field_info_dict" with key "attribute"
        if (PyDict_SetItemString(field_info_dict, "attribute", info_list) != 0) {
            Py_DECREF(field_list_dict);
            Py_DECREF(field_info_dict);
            Py_DECREF(info_list);
            std::string error =
                "(field) = (" + std::string((field_list_)[i].field_name) +
                "), failed to add key-value pair 'attribute' and info list (alias name, display name)!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(info_list);

        // Load "field_type" to "field_info_dict".
        val = PyUnicode_FromString((field_list_)[i].field_type);
        if (PyDict_SetItemString(field_info_dict, "field_type", val) != 0) {
            Py_DECREF(field_list_dict);
            Py_DECREF(field_info_dict);
            Py_XDECREF(val);
            std::string error = "(field, field_type) = (" + std::string((field_list_)[i].field_name) + ", " +
                                std::string((field_list_)[i].field_type) +
                                "), failed to add key-value pair 'field_type'!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(val);

        // Load "contiguous_in_x" to "field_info_dict".
        if ((field_list_)[i].contiguous_in_x) {
            if (PyDict_SetItemString(field_info_dict, "contiguous_in_x", Py_True) != 0) {
                Py_DECREF(field_list_dict);
                Py_DECREF(field_info_dict);
                std::string contiguous_in_x_str = ((field_list_)[i].contiguous_in_x) ? "true" : "false";
                std::string error = "(field, contiguous_in_x) = (" + std::string((field_list_)[i].field_name) + ", " +
                                    contiguous_in_x_str + "), failed to add key-value pair 'contiguous_in_x'!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
        } else {
            if (PyDict_SetItemString(field_info_dict, "contiguous_in_x", Py_False) != 0) {
                Py_DECREF(field_list_dict);
                Py_DECREF(field_info_dict);
                std::string contiguous_in_x_str = ((field_list_)[i].contiguous_in_x) ? "true" : "false";
                std::string error = "(field, contiguous_in_x) = (" + std::string((field_list_)[i].field_name) + ", " +
                                    contiguous_in_x_str + "), failed to add key-value pair 'contiguous_in_x'!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
        }

        // Load "ghost_cell" to "field_info_dict"
        PyObject* ghost_cell_list = PyList_New(0);
        for (int d = 0; d < 6; d++) {
            val = PyLong_FromLong((long)(field_list_)[i].field_ghost_cell[d]);
            if (PyList_Append(ghost_cell_list, val) != 0) {
                Py_DECREF(field_list_dict);
                Py_DECREF(field_info_dict);
                Py_DECREF(ghost_cell_list);
                Py_XDECREF(val);
                std::string error = "(field) = (" + std::string((field_list_)[i].field_name) +
                                    "), failed to append ghost cell to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(val);
        }
        if (PyDict_SetItemString(field_info_dict, "ghost_cell", ghost_cell_list) != 0) {
            Py_DECREF(field_list_dict);
            Py_DECREF(field_info_dict);
            Py_DECREF(ghost_cell_list);
            std::string error = "(field) = (" + std::string((field_list_)[i].field_name) +
                                "), failed to add key-value pair 'ghost_cell'!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(ghost_cell_list);

        // Load "field_info_dict" to "field_list_dict", with key = field_name
        key = PyUnicode_FromString((field_list_)[i].field_name);
        if (PyDict_SetItem(field_list_dict, key, field_info_dict) != 0) {
            Py_DECREF(field_list_dict);
            Py_DECREF(field_info_dict);
            Py_XDECREF(key);
            std::string error = "(field) = (" + std::string((field_list_)[i].field_name) +
                                "), failed to attach dictionary under key!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(key);

        Py_DECREF(field_info_dict);
    }

    if (PyDict_SetItemString(py_dict, "field_list", field_list_dict) != 0) {
        Py_DECREF(field_list_dict);
        std::string error = "Inserting dictionary 'field_list' to '" + py_dict_name + "' failed!\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    Py_DECREF(field_list_dict);
#else
    pybind11::dict py_param_yt = pybind11::cast<pybind11::dict>(py_dict);
    pybind11::dict py_field_list = pybind11::dict();
    py_param_yt["field_list"] = py_field_list;

    for (int i = 0; i < num_fields_; i++) {
        py_field_list[field_list_[i].field_name] = pybind11::dict();

        pybind11::tuple py_name_alias = pybind11::tuple(field_list_[i].num_field_name_alias);
        for (int a = 0; a < field_list_[i].num_field_name_alias; a++) {
            py_name_alias[a] = field_list_[i].field_name_alias[a];
        }
        py_field_list[field_list_[i].field_name]["attribute"] =
            pybind11::make_tuple(field_list_[i].field_unit, py_name_alias, field_list_[i].field_display_name);
        py_field_list[field_list_[i].field_name]["field_type"] = field_list_[i].field_type;
        py_field_list[field_list_[i].field_name]["contiguous_in_x"] = pybind11::bool_(field_list_[i].contiguous_in_x);
        py_field_list[field_list_[i].field_name]["ghost_cell"] = pybind11::make_tuple(
            field_list_[i].field_ghost_cell[0], field_list_[i].field_ghost_cell[1], field_list_[i].field_ghost_cell[2],
            field_list_[i].field_ghost_cell[3], field_list_[i].field_ghost_cell[4], field_list_[i].field_ghost_cell[5]);
    }
#endif  // #ifndef USE_PYBIND11

    return {DataStructureStatus::kDataStructureSuccess, ""};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  BindFieldListToPython
//
// Notes       :  1. Bind particle_list_ to Python dictionary passed in.
//                   The data is not directly link, the data is copied.
//                2. TODO: Due to bad api design, particle_list_ is created under
//                         libyt.param_yt["particle_list"]
//                3. If using Python C API, PyUnicode_FromString is Python-API >= 3.5, and it returns a new reference.
//                4. When creating key-value pair under this dictionary:
//                   (1) Assume that we have all the particle name "par_type" unique.
//                       And in each species, they have unique "attr_name".
//                   (2) If attr_display_name is NULL, set it to Py_None.
//      particle_list_dict   species_dict     attr_dict        attr_list  name_alias_list
//              |                 |               |                |              |
//              { <par_type>: { "attribute" : { <attr_name1> : ( <attr_unit>, (<attr_name_alias>), <attr_display_name>),
//                                              <attr_name2> : ( <attr_unit>, (<attr_name_alias>),
//                                              <attr_display_name>)},
//                              "particle_coor_label" : (<coor_x>, <coor_y>, <coor_z>),
//                                                      |
//                                                      |
//                                                   coor_list
//                              "label": <index in particle_list>}
//               }
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::BindParticleListToPython(PyObject* py_dict,
                                                               const std::string& py_dict_name) const {
#ifndef USE_PYBIND11
    PyObject* particle_list_dict = PyDict_New();
    PyObject *key, *val;
    for (int s = 0; s < num_par_types_; s++) {
        PyObject* species_dict = PyDict_New();

        // Insert a series of attr_list to attr_dict with key <attr_name>
        PyObject* attr_dict = PyDict_New();
        for (int a = 0; a < particle_list_[s].num_attr; a++) {
            PyObject* attr_list = PyList_New(0);
            yt_attribute attr = particle_list_[s].attr_list[a];

            // Append attr_unit to attr_list
            val = PyUnicode_FromString(attr.attr_unit);
            if (PyList_Append(attr_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(attr_dict);
                Py_DECREF(attr_list);
                Py_XDECREF(val);
                std::string error = "(par_type, attr_unit) = (" + std::string(particle_list_[s].par_type) + ", " +
                                    std::string(attr.attr_unit) + "), failed to append attr_unit to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(val);

            // Create name_alias_list and append to attr_list
            PyObject* name_alias_list = PyList_New(0);
            for (int i = 0; i < attr.num_attr_name_alias; i++) {
                val = PyUnicode_FromString(attr.attr_name_alias[i]);
                if (PyList_Append(name_alias_list, val) != 0) {
                    Py_DECREF(particle_list_dict);
                    Py_DECREF(species_dict);
                    Py_DECREF(attr_dict);
                    Py_DECREF(attr_list);
                    Py_DECREF(name_alias_list);
                    Py_XDECREF(val);
                    std::string error = "(par_type, attr_name, attr_name_alias) = (" +
                                        std::string(particle_list_[s].par_type) + ", " + std::string(attr.attr_name) +
                                        ", " + std::string(attr.attr_name_alias[i]) +
                                        "), failed to append attr_name_alias to list!\n";
                    return {DataStructureStatus::kDataStructureFailed, error};
                }
                Py_DECREF(val);
            }
            if (PyList_Append(attr_list, name_alias_list) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(attr_dict);
                Py_DECREF(attr_list);
                Py_DECREF(name_alias_list);
                std::string error = "(par_type, attr_name) = (" + std::string(particle_list_[s].par_type) + ", " +
                                    std::string(attr.attr_name) + "), failed to append name_alias_list to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(name_alias_list);

            // Append attr_display_name to attr_list if != NULL, otherwise append None.
            if (attr.attr_display_name == nullptr) {
                if (PyList_Append(attr_list, Py_None) != 0) {
                    Py_DECREF(particle_list_dict);
                    Py_DECREF(species_dict);
                    Py_DECREF(attr_dict);
                    Py_DECREF(attr_list);
                    std::string error = "(par_type, attr_name, attr_display_name) = (" +
                                        std::string(particle_list_[s].par_type) + ", " + std::string(attr.attr_name) +
                                        ", nullptr), failed to append Py_None to list!\n";
                    return {DataStructureStatus::kDataStructureFailed, error};
                }
            } else {
                val = PyUnicode_FromString(attr.attr_display_name);
                if (PyList_Append(attr_list, val) != 0) {
                    Py_DECREF(particle_list_dict);
                    Py_DECREF(species_dict);
                    Py_DECREF(attr_dict);
                    Py_DECREF(attr_list);
                    Py_XDECREF(val);
                    std::string error = "(par_type, attr_name, attr_display_name) = (" +
                                        std::string(particle_list_[s].par_type) + ", " + std::string(attr.attr_name) +
                                        ", " + std::string(attr.attr_display_name) +
                                        "), failed to append attr_display_name to list!\n";
                    return {DataStructureStatus::kDataStructureFailed, error};
                }
                Py_DECREF(val);
            }

            // Insert attr_list to attr_dict with key = <attr_name>
            key = PyUnicode_FromString(attr.attr_name);
            if (PyDict_SetItem(attr_dict, key, attr_list) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(attr_dict);
                Py_DECREF(attr_list);
                Py_XDECREF(key);
                std::string error = "(par_type, attr_name) = (" + std::string(particle_list_[s].par_type) + ", " +
                                    std::string(attr.attr_name) + "), failed to append attribute list to dict!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(key);

            Py_DECREF(attr_list);
        }

        // Insert attr_dict to species_dict with key = "attribute"
        if (PyDict_SetItemString(species_dict, "attribute", attr_dict) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_DECREF(attr_dict);
            std::string error = "(par_type) = (" + std::string(particle_list_[s].par_type) +
                                "), failed to add key-value pair 'attribute' and attribute dict!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(attr_dict);

        // Create coor_list and insert it to species_dict with key = "particle_coor_label"
        PyObject* coor_list = PyList_New(0);

        if (particle_list_[s].coor_x == nullptr) {
            if (PyList_Append(coor_list, Py_None) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                std::string error = "(par_type, coor_x) = (" + std::string(particle_list_[s].par_type) +
                                    ", nullptr), failed to append Py_None to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
        } else {
            val = PyUnicode_FromString(particle_list_[s].coor_x);
            if (PyList_Append(coor_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                Py_XDECREF(val);
                std::string error = "(par_type, coor_x) = (" + std::string(particle_list_[s].par_type) + ", " +
                                    std::string(particle_list_[s].coor_x) + "), failed to append coor_x to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(val);
        }

        if (particle_list_[s].coor_y == nullptr) {
            if (PyList_Append(coor_list, Py_None) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                std::string error = "(par_type, coor_y) = (" + std::string(particle_list_[s].par_type) +
                                    ", nullptr), failed to append Py_None to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
        } else {
            val = PyUnicode_FromString(particle_list_[s].coor_y);
            if (PyList_Append(coor_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                Py_XDECREF(val);
                std::string error = "(par_type, coor_y) = (" + std::string(particle_list_[s].par_type) + ", " +
                                    std::string(particle_list_[s].coor_y) + "), failed to append coor_y to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(val);
        }

        if (particle_list_[s].coor_z == nullptr) {
            if (PyList_Append(coor_list, Py_None) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                std::string error = "(par_type, coor_z) = (" + std::string(particle_list_[s].par_type) +
                                    ", nullptr), failed to append Py_None to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
        } else {
            val = PyUnicode_FromString(particle_list_[s].coor_z);
            if (PyList_Append(coor_list, val) != 0) {
                Py_DECREF(particle_list_dict);
                Py_DECREF(species_dict);
                Py_DECREF(coor_list);
                Py_XDECREF(val);
                std::string error = "(par_type, coor_z) = (" + std::string(particle_list_[s].par_type) + ", " +
                                    std::string(particle_list_[s].coor_z) + "), failed to append coor_z to list!\n";
                return {DataStructureStatus::kDataStructureFailed, error};
            }
            Py_DECREF(val);
        }

        // Insert coor_list to species_dict with key = "particle_coor_label"
        if (PyDict_SetItemString(species_dict, "particle_coor_label", coor_list) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_DECREF(coor_list);
            std::string error = "(par_type) = (" + std::string(particle_list_[s].par_type) +
                                "), failed to add key-value pair 'particle_coor_label' and coordinate list!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(coor_list);

        // Insert label s to species_dict, with key = "label"
        key = PyLong_FromLong((long)s);
        if (PyDict_SetItemString(species_dict, "label", key) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_XDECREF(key);
            std::string error = "(par_type) = (" + std::string(particle_list_[s].par_type) +
                                "), failed to add key-value pair 'label' and " + std::to_string(s) + "\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(key);

        // Insert species_dict to particle_list_dict with key = <par_type>
        key = PyUnicode_FromString(particle_list_[s].par_type);
        if (PyDict_SetItem(particle_list_dict, key, species_dict) != 0) {
            Py_DECREF(particle_list_dict);
            Py_DECREF(species_dict);
            Py_XDECREF(key);
            std::string error = "(par_type) = (" + std::string(particle_list_[s].par_type) +
                                "), failed to attach dictionary under key!\n";
            return {DataStructureStatus::kDataStructureFailed, error};
        }
        Py_DECREF(key);

        Py_DECREF(species_dict);
    }

    // Insert particle_list_dict to libyt.param_yt["particle_list"]
    if (PyDict_SetItemString(py_dict, "particle_list", particle_list_dict) != 0) {
        Py_DECREF(particle_list_dict);
        std::string error = "Inserting dictionary 'particle_list' to '" + py_dict_name + "' failed!\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }
    Py_DECREF(particle_list_dict);
#else
    pybind11::dict py_param_yt = pybind11::cast<pybind11::dict>(py_dict);
    pybind11::dict py_particle_list = pybind11::dict();
    py_param_yt["particle_list"] = py_particle_list;

    for (int i = 0; i < num_par_types_; i++) {
        py_particle_list[particle_list_[i].par_type] = pybind11::dict();

        pybind11::dict py_attr_dict = pybind11::dict();
        py_particle_list[particle_list_[i].par_type]["attribute"] = py_attr_dict;
        for (int v = 0; v < particle_list_[i].num_attr; v++) {
            pybind11::tuple py_name_alias = pybind11::tuple(particle_list_[i].attr_list[v].num_attr_name_alias);
            for (int a = 0; a < particle_list_[i].attr_list[v].num_attr_name_alias; a++) {
                py_name_alias[a] = particle_list_[i].attr_list[v].attr_name_alias[a];
            }

            py_attr_dict[particle_list_[i].attr_list[v].attr_name] =
                pybind11::make_tuple(particle_list_[i].attr_list[v].attr_unit, py_name_alias,
                                     particle_list_[i].attr_list[v].attr_display_name);
        }

        py_particle_list[particle_list_[i].par_type]["particle_coor_label"] =
            pybind11::make_tuple(particle_list_[i].coor_x, particle_list_[i].coor_y, particle_list_[i].coor_z);
        py_particle_list[particle_list_[i].par_type]["label"] = i;
    }
#endif  // #ifndef USE_PYBIND11

    return {DataStructureStatus::kDataStructureSuccess, ""};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  BindInfoToPython
//
// Notes       :  1. Bind field_list_ and particle_list_ info to Python dictionary.
//                2. The current structure will not know if the data is set or not.
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::BindInfoToPython(PyObject* py_dict, const std::string& py_dict_name) {
    if (num_fields_ > 0) {
        BindFieldListToPython(py_dict, py_dict_name);
    }
    if (num_par_types_ > 0) {
        BindParticleListToPython(py_dict, py_dict_name);
    }
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  BindAllHierarchyToPython
//
// Notes       :  1. Check data if check_data is true.
//                2. If it is under Mpi mode, we need to gather hierarchy from different ranks to all ranks.
//                3. The allocation of full hierarchy is done at AllocateStorage.
//                4. The current structure will not know if the data is set or not.
//                5. TODO: Do I need to move data twice, which is gathering data, and then move it to Python
//                         storage?
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::BindAllHierarchyToPython(int mpi_root) {
#ifndef SERIAL_MODE
    // Gather hierarchy from different ranks to root rank.
    yt_hierarchy* hierarchy_full = nullptr;
    long** particle_count_list_full = nullptr;
    GatherAllHierarchy(mpi_root, &hierarchy_full, &particle_count_list_full);

    // Bind hierarchy to Python
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
//                2. The current structure will not know if the data is set or not.
//                3. TODO: Currently, the API forces this function to bind and build all the data
//                         inside the grids_local_ array at once. Might change it in the future libyt v1.0.
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::BindLocalDataToPython() const {
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
//                2. Reset num_grids_local_/num_grids_local_field_data_/num_grids_local_par_data_ = 0 and
//                   grids_local_ = nullptr.
//                3. Counterpart of AllocateGridsLocal().
//                4. This method is separate from the rest of the clean up methods is because it cleans up
//                   the data for holding user input, which is not needed after committing everything.
//                   TODO: bad Api design
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUpGridsLocal() {
    if (num_grids_local_ > 0) {
        for (int i = 0; i < num_grids_local_; i = i + 1) {
            if (num_grids_local_field_data_ > 0) {
                delete[] grids_local_[i].field_data;
            }
            if (num_grids_local_par_data_ > 0) {
                delete[] grids_local_[i].par_count_list;
                for (int p = 0; p < num_grids_local_par_data_; p++) {
                    delete[] grids_local_[i].particle_data[p];
                }
                delete[] grids_local_[i].particle_data;
            }
        }
        delete[] grids_local_;
    }

    num_grids_local_ = 0;
    num_grids_local_field_data_ = 0;
    num_grids_local_par_data_ = 0;
    grids_local_ = nullptr;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Private Method :  CleanUpFullHierarchyStorageForPython
//
// Notes       :  1. Clean full hierarchy Python bindings, and reset hierarchy pointer to nullptr and num_grids_ = 0.
//                2. Counterpart for AllocateAllHierarchyStorageForPython().
//-------------------------------------------------------------------------------------------------------
void DataStructureAmr::CleanUpFullHierarchyStorageForPython() {
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

    num_grids_ = 0;
    has_particle_ = false;

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
void DataStructureAmr::CleanUpLocalDataPythonBindings() const {
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
    CleanUpFullHierarchyStorageForPython();
    CleanUpLocalDataPythonBindings();

    index_offset_ = 0;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetFieldIndex
//
// Notes       :  1. Get the field index in the field_list_ based on field_name.
//                2. Return -1 if the field_name is not found.
//-------------------------------------------------------------------------------------------------------
int DataStructureAmr::GetFieldIndex(const char* field_name) const {
    int field_id = -1;
    for (int v = 0; v < num_fields_; v++) {
        if (strcmp(field_name, field_list_[v].field_name) == 0) {
            field_id = v;
            break;
        }
    }
    return field_id;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetParticleIndex
//
// Notes       :  1. Get the particle index in the particle_list_ based on particle_type.
//                2. Return -1 if the particle_type is not found.
//-------------------------------------------------------------------------------------------------------
int DataStructureAmr::GetParticleIndex(const char* particle_type) const {
    int ptype_index = -1;
    for (int v = 0; v < num_par_types_; v++) {
        if (strcmp(particle_type, particle_list_[v].par_type) == 0) {
            ptype_index = v;
            break;
        }
    }
    return ptype_index;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetParticleAttributeIndex
//
// Notes       :  1. Get the particle attribute index in the particle_list_ based on particle index and
//                   attribute name.
//                2. Return -1 if the particle_type or attribute name is not found.
//-------------------------------------------------------------------------------------------------------
int DataStructureAmr::GetParticleAttributeIndex(int particle_type_index, const char* attr_name) const {
    int pattr_index = -1;
    if (particle_type_index < 0 || particle_type_index >= num_par_types_) {
        return pattr_index;
    }

    for (int a = 0; a < particle_list_[particle_type_index].num_attr; a++) {
        if (strcmp(attr_name, particle_list_[particle_type_index].attr_list[a].attr_name) == 0) {
            pattr_index = a;
            break;
        }
    }
    return pattr_index;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetPythonBoundFullHierarchyGridDimensions
//
// Notes       :  1. Read the full hierarchy grid dimensions loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundFullHierarchyGridDimensions(long gid, int* dimensions) const {
    if (grid_dimensions_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) < 0 || (gid - index_offset_) >= num_grids_) {
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
// Public Method  :  GetPythonBoundFullHierarchyGridLeftEdge
//
// Notes       :  1. Read the full hierarchy grid left edge loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundFullHierarchyGridLeftEdge(long gid, double* left_edge) const {
    if (grid_left_edge_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) < 0 || (gid - index_offset_) >= num_grids_) {
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
// Public Method  :  GetPythonBoundFullHierarchyGridRightEdge
//
// Notes       :  1. Read the full hierarchy grid right edge loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundFullHierarchyGridRightEdge(long gid, double* right_edge) const {
    if (grid_right_edge_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) < 0 || (gid - index_offset_) >= num_grids_) {
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
// Public Method  :  GetPythonBoundFullHierarchyGridParentId
//
// Notes       :  1. Read the full hierarchy grid parent id loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundFullHierarchyGridParentId(long gid, long* parent_id) const {
    if (grid_parent_id_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) < 0 || (gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    *parent_id = grid_parent_id_[gid - index_offset_];

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetPythonBoundFullHierarchyGridLevel
//
// Notes       :  1. Read the full hierarchy grid level loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundFullHierarchyGridLevel(long gid, int* level) const {
    if (grid_levels_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) < 0 || (gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    *level = grid_levels_[gid - index_offset_];

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetPythonBoundFullHierarchyGridProcNum
//
// Notes       :  1. Read the full hierarchy grid proc number (mpi rank) loaded in Python.
//                2. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundFullHierarchyGridProcNum(long gid, int* proc_num) const {
    if (proc_num_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) < 0 || (gid - index_offset_) >= num_grids_) {
        std::string error = "(grid id) = " + std::to_string(gid) + " is out of range.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    *proc_num = proc_num_[gid - index_offset_];

    return {DataStructureStatus::kDataStructureSuccess, std::string()};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  DataStructureAmr
// Public Method  :  GetPythonBoundFullHierarchyGridParticleCount
//
// Notes       :  1. Read the full hierarchy grid particle count for a ptype loaded in Python.
//                2. This method is only valid if the data structure contains particle data.
//                3. Counterpart of BindAllHierarchyToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundFullHierarchyGridParticleCount(long gid, const char* ptype,
                                                                                   long* par_count) const {
    if (!has_particle_) {
        std::string error = "Doesn't contain particle data.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if (par_count_list_ == nullptr) {
        std::string error = "Full hierarchy is not initialized yet.\n";
        return {DataStructureStatus::kDataStructureFailed, error};
    }

    if ((gid - index_offset_) < 0 || (gid - index_offset_) >= num_grids_) {
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
// Public Method  :  GetPythonBoundLocalFieldData
//
// Notes       :  1. Read the local field data bind to Python libyt.grid_data[gid][fname].
//                2. Counterpart of BindLocalFieldDataToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundLocalFieldData(long gid, const char* field_name,
                                                                   yt_data* field_data) const {
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
// Public Method  :  GetPythonBoundLocalParticleData
//
// Notes       :  1. Read the local field data bind to Python libyt.particle_data[gid][ptype][attr].
//                2. Counterpart of BindLocalParticleDataToPython().
//-------------------------------------------------------------------------------------------------------
DataStructureOutput DataStructureAmr::GetPythonBoundLocalParticleData(long gid, const char* ptype, const char* attr,
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
