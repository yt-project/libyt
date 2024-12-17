#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

static int set_field_data(yt_grid* grid);
static int set_particle_data(yt_grid* grid);

//-------------------------------------------------------------------------------------------------------
// Function    :  append_grid
// Description :  Add a single full grid to the libyt Python module
//
// Note        :  1. Store the input "grid" to libyt.hierarchy, libyt.grid_data, libyt.particle_data.
//                2. When setting libyt.hierarchy:
//                   since grid id doesn't have to be 0-indexed (set LibytProcessControl::Get().param_yt_.index_offset),
//                   but the hierarchy array starts at 0, we need to minus index_offset when setting hierarchy.
//                3. When setting libyt.grid_data and libyt.particle_data:
//                   always maintain the same grid passed in by simulation, which means it doesn't have
//                   to start from 0.
//                4. Called and use by yt_commit().
//
// Parameter   :  yt_grid *grid
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int append_grid(yt_grid* grid) {
    // export grid info to libyt.hierarchy
    long index = grid->id - LibytProcessControl::Get().param_yt_.index_offset;
    for (int d = 0; d < 3; d++) {
        LibytProcessControl::Get().data_structure_amr_.grid_left_edge_[index * 3 + d] = grid->left_edge[d];
        LibytProcessControl::Get().data_structure_amr_.grid_right_edge_[index * 3 + d] = grid->right_edge[d];
        LibytProcessControl::Get().data_structure_amr_.grid_dimensions_[index * 3 + d] = grid->grid_dimensions[d];
    }
    LibytProcessControl::Get().data_structure_amr_.grid_parent_id_[index] = grid->parent_id;
    LibytProcessControl::Get().data_structure_amr_.grid_levels_[index] = grid->level;
    LibytProcessControl::Get().data_structure_amr_.proc_num_[index] = grid->proc_num;
    if (LibytProcessControl::Get().param_yt_.num_par_types > 0) {
        for (int p = 0; p < LibytProcessControl::Get().param_yt_.num_par_types; p++) {
            LibytProcessControl::Get()
                .data_structure_amr_.par_count_list_[index * LibytProcessControl::Get().param_yt_.num_par_types + p] =
                grid->par_count_list[p];
        }
    }

    log_debug("Inserting grid [%ld] info to libyt.hierarchy ... done\n", grid->id);

    // export grid data and particle data to libyt.grid_data and libyt.particle_data
    if (grid->field_data != nullptr && set_field_data(grid) != YT_SUCCESS) {
        YT_ABORT("Failed to append grid [%ld] field data.\n", grid->id);
    }
    if (grid->particle_data != nullptr && set_particle_data(grid) != YT_SUCCESS) {
        YT_ABORT("Failed to append grid [%ld] particle data.\n", grid->id);
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  set_field_data
// Description :  Wrap data pointer and added under libyt.grid_data
//
// Note        :  1. libyt.grid_data[grid_id][field_list.field_name] = NumPy array from field pointer.
//                2. Append to dictionary only when there is data pointer passed in.
//
// Parameter   :  yt_grid *grid
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
static int set_field_data(yt_grid* grid) {
    yt_field* field_list = LibytProcessControl::Get().data_structure_amr_.field_list_;

    PyObject *py_grid_id, *py_field_labels, *py_field_data;
    py_grid_id = PyLong_FromLong(grid->id);
    py_field_labels = PyDict_New();
    for (int v = 0; v < LibytProcessControl::Get().param_yt_.num_fields; v++) {
        // append data to dict only if data is not NULL.
        if ((grid->field_data)[v].data_ptr == nullptr) continue;

        // check if dictionary exists, if no add new dict under key gid
        if (PyDict_Contains(LibytProcessControl::Get().data_structure_amr_.py_grid_data_, py_grid_id) != 1) {
            PyDict_SetItem(LibytProcessControl::Get().data_structure_amr_.py_grid_data_, py_grid_id, py_field_labels);
        }

        // insert data under py_field_labels dict
        // (1) Grab NumPy Enumerate Type in order: (1)data_dtype (2)field_dtype
        int grid_dtype;
        if (get_npy_dtype((grid->field_data)[v].data_dtype, &grid_dtype) == YT_SUCCESS) {
            log_debug("Grid ID [ %ld ], field data [ %s ], grab NumPy enumerate type from data_dtype.\n", grid->id,
                      field_list[v].field_name);
        } else if (get_npy_dtype(field_list[v].field_dtype, &grid_dtype) == YT_SUCCESS) {
            (grid->field_data)[v].data_dtype = field_list[v].field_dtype;
            log_debug("Grid ID [ %ld ], field data [ %s ], grab NumPy enumerate type from field_dtype.\n", grid->id,
                      field_list[v].field_name);
        } else {
            Py_DECREF(py_grid_id);
            Py_DECREF(py_field_labels);
            YT_ABORT("Grid ID [ %ld ], field data [ %s ], cannot get the NumPy enumerate type properly.\n", grid->id,
                     field_list[v].field_name);
        }

        // (2) Get the dimension of the input array
        // Only "cell-centered" will be set to grid_dimensions + ghost cell, else should be set in data_dimensions.
        if (strcmp(field_list[v].field_type, "cell-centered") == 0) {
            // Get grid_dimensions and consider contiguous_in_x or not, since grid_dimensions is defined as [x][y][z].
            if (field_list[v].contiguous_in_x) {
                for (int d = 0; d < 3; d++) {
                    (grid->field_data)[v].data_dimensions[d] = (grid->grid_dimensions)[2 - d];
                }
            } else {
                for (int d = 0; d < 3; d++) {
                    (grid->field_data)[v].data_dimensions[d] = (grid->grid_dimensions)[d];
                }
            }
            // Plus the ghost cell to get the actual array dimensions.
            for (int d = 0; d < 6; d++) {
                (grid->field_data)[v].data_dimensions[d / 2] += field_list[v].field_ghost_cell[d];
            }
        }
        // See if all data_dimensions > 0, abort if not.
        for (int d = 0; d < 3; d++) {
            if ((grid->field_data)[v].data_dimensions[d] <= 0) {
                Py_DECREF(py_grid_id);
                Py_DECREF(py_field_labels);
                YT_ABORT("Grid ID [ %ld ], field data [ %s ], data_dimensions[%d] = %d <= 0.\n", grid->id,
                         field_list[v].field_name, d, (grid->field_data)[v].data_dimensions[d]);
            }
        }

        npy_intp grid_dims[3] = {(grid->field_data)[v].data_dimensions[0], (grid->field_data)[v].data_dimensions[1],
                                 (grid->field_data)[v].data_dimensions[2]};

        // (3) Insert data to dict
        // PyArray_SimpleNewFromData simply creates an array wrapper and does not allocate and own the array
        py_field_data = PyArray_SimpleNewFromData(3, grid_dims, grid_dtype, (grid->field_data)[v].data_ptr);

        // Mark this memory (NumPy array) read-only
        PyArray_CLEARFLAGS((PyArrayObject*)py_field_data, NPY_ARRAY_WRITEABLE);

        // add the field data to dict "libyt.grid_data[grid_id][field_list.field_name]"
        PyDict_SetItemString(py_field_labels, field_list[v].field_name, py_field_data);

        // call decref since PyDict_SetItemString() returns a new reference
        Py_DECREF(py_field_data);

        log_debug("Inserting grid [%ld] field data [%s] to libyt.grid_data ... done\n", grid->id,
                  field_list[v].field_name);
    }

    // call decref since both PyLong_FromLong() and PyDict_New() return a new reference
    Py_DECREF(py_grid_id);
    Py_DECREF(py_field_labels);

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  set_particle_data
// Description :  Wrap data pointer and added under libyt.particle_data
//
// Note        :  1. libyt.particle_data[grid_id][particle_list.par_type][attr_name]
//                   = NumPy array created through wrapping data pointer.
//                2. Append to dictionary only when there is data pointer passed in.
//                3. The logistic is we create dictionary no matter what, and only append under
//                   py_particle_data if there is data to wrap, so that ref count increases.
//
// Parameter   :  yt_grid *grid
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
static int set_particle_data(yt_grid* grid) {
    yt_particle* particle_list = LibytProcessControl::Get().data_structure_amr_.particle_list_;

    PyObject *py_grid_id, *py_ptype_labels, *py_attributes, *py_data;
    py_grid_id = PyLong_FromLong(grid->id);
    py_ptype_labels = PyDict_New();
    for (int p = 0; p < LibytProcessControl::Get().param_yt_.num_par_types; p++) {
        py_attributes = PyDict_New();
        for (int a = 0; a < particle_list[p].num_attr; a++) {
            // skip if particle attribute pointer is NULL
            if ((grid->particle_data)[p][a].data_ptr == nullptr) continue;

            // Wrap the data array if pointer exist
            int data_dtype;
            if (get_npy_dtype(particle_list[p].attr_list[a].attr_dtype, &data_dtype) != YT_SUCCESS) {
                log_error("Cannot get particle type [%s] attribute [%s] data type. Unable to wrap particle array\n",
                          particle_list[p].par_type, particle_list[p].attr_list[a].attr_name);
                continue;
            }
            if ((grid->par_count_list)[p] <= 0) {
                log_error("Cannot wrapped particle array with length %ld <= 0\n", (grid->par_count_list)[p]);
                continue;
            }
            npy_intp array_dims[1] = {(grid->par_count_list)[p]};
            py_data = PyArray_SimpleNewFromData(1, array_dims, data_dtype, (grid->particle_data)[p][a].data_ptr);
            PyArray_CLEARFLAGS((PyArrayObject*)py_data, NPY_ARRAY_WRITEABLE);

            // Get the dictionary and append py_data
            if (PyDict_Contains(LibytProcessControl::Get().data_structure_amr_.py_particle_data_, py_grid_id) != 1) {
                // 1st time append, nothing exist under libyt.particle_data[gid]
                PyDict_SetItem(LibytProcessControl::Get().data_structure_amr_.py_particle_data_, py_grid_id,
                               py_ptype_labels);  // libyt.particle_data[gid] = dict()
                PyDict_SetItemString(py_ptype_labels, particle_list[p].par_type, py_attributes);
            } else {
                // libyt.particle_data[gid] exist, check if libyt.particle_data[gid][ptype] exist
                PyObject* py_ptype_name = PyUnicode_FromString(particle_list[p].par_type);
                if (PyDict_Contains(py_ptype_labels, py_ptype_name) != 1) {
                    PyDict_SetItemString(py_ptype_labels, particle_list[p].par_type, py_attributes);
                }
                Py_DECREF(py_ptype_name);
            }
            PyDict_SetItemString(py_attributes, particle_list[p].attr_list[a].attr_name, py_data);
            Py_DECREF(py_data);

            // debug message
            log_debug("Inserting grid [%ld] particle [%s] attribute [%s] data to libyt.particle_data ... done\n",
                      grid->id, particle_list[p].par_type, particle_list[p].attr_list[a].attr_name);
        }
        Py_DECREF(py_attributes);
    }

    Py_DECREF(py_ptype_labels);
    Py_DECREF(py_grid_id);

    return YT_SUCCESS;
}