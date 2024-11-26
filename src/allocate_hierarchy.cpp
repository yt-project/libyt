#include "libyt_process_control.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  allocate_hierarchy
// Description :  Fill the libyt.hierarchy dictionary with NumPy arrays or memoryviews
//
// Note        :  1. Called by yt_commit(), since it needs param_yt.num_grids.
//                2. These NumPy array will be set when calling yt_commit().
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int allocate_hierarchy() {
    SET_TIMER(__PRETTY_FUNCTION__);

    yt_param_yt& param_yt = LibytProcessControl::Get().param_yt_;

#ifndef USE_PYBIND11
    // remove all key-value pairs if one wants to overwrite the existing dictionary
    // ==> it should happen only if one calls yt_set_Parameters() more than once
    if (PyDict_Size(LibytProcessControl::Get().py_hierarchy_) > 0) {
        PyDict_Clear(LibytProcessControl::Get().py_hierarchy_);
        log_warning("Removing existing key-value pairs in libyt.hierarchy ... done\n");
    }

    // allocate NumPy arrays and attach them to libyt.hierarchy
    npy_intp np_dim[2];
    PyObject* py_obj;

    np_dim[0] = (npy_intp)param_yt.num_grids;

// convenient macro
#define ADD_DICT(DIM1, KEY, TYPE)                                                                                      \
    {                                                                                                                  \
        np_dim[1] = (npy_intp)DIM1;                                                                                    \
        py_obj = PyArray_SimpleNew(2, np_dim, TYPE);                                                                   \
                                                                                                                       \
        if (PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, KEY, py_obj) != 0) {                        \
            Py_XDECREF(py_obj);                                                                                        \
            YT_ABORT("Inserting the key \"%s\" to libyt.hierarchy ... failed!\n", KEY);                                \
        }                                                                                                              \
                                                                                                                       \
        Py_DECREF(py_obj);                                                                                             \
    }

    ADD_DICT(3, "grid_left_edge", NPY_DOUBLE)
    ADD_DICT(3, "grid_right_edge", NPY_DOUBLE)
    ADD_DICT(3, "grid_dimensions", NPY_INT)
    ADD_DICT(1, "grid_parent_id", NPY_LONG)
    ADD_DICT(1, "grid_levels", NPY_INT)
    ADD_DICT(1, "proc_num", NPY_INT)
    if (param_yt.num_par_types > 0) {
        ADD_DICT(param_yt.num_par_types, "par_count_list", NPY_LONG)
    }
#undef ADD_DICT
#else
    LibytProcessControl::Get().grid_left_edge = new double[param_yt.num_grids * 3];
    LibytProcessControl::Get().grid_right_edge = new double[param_yt.num_grids * 3];
    LibytProcessControl::Get().grid_dimensions = new int[param_yt.num_grids * 3];
    LibytProcessControl::Get().grid_parent_id = new long[param_yt.num_grids];
    LibytProcessControl::Get().grid_levels = new int[param_yt.num_grids];
    LibytProcessControl::Get().proc_num = new int[param_yt.num_grids];
    if (param_yt.num_par_types > 0) {
        LibytProcessControl::Get().par_count_list = new long[param_yt.num_grids * param_yt.num_par_types];
    } else {
        LibytProcessControl::Get().par_count_list = nullptr;
    }

    pybind11::module_ libyt = pybind11::module_::import("libyt");
    pybind11::dict py_hierarchy = libyt.attr("hierarchy");

    py_hierarchy["grid_left_edge"] =
        pybind11::memoryview::from_buffer(LibytProcessControl::Get().grid_left_edge,   // buffer pointer
                                          std::vector<long>({param_yt.num_grids, 3}),  // shape (rows, cols)
                                          {sizeof(double) * 3, sizeof(double)}         // strides in bytes
        );
    py_hierarchy["grid_right_edge"] =
        pybind11::memoryview::from_buffer(LibytProcessControl::Get().grid_right_edge,  // buffer pointer
                                          std::vector<long>({param_yt.num_grids, 3}),  // shape (rows, cols)
                                          {sizeof(double) * 3, sizeof(double)}         // strides in bytes
        );
    py_hierarchy["grid_dimensions"] =
        pybind11::memoryview::from_buffer(LibytProcessControl::Get().grid_dimensions,  // buffer pointer
                                          std::vector<long>({param_yt.num_grids, 3}),  // shape (rows, cols)
                                          {sizeof(int) * 3, sizeof(int)}               // strides in bytes
        );
    py_hierarchy["grid_parent_id"] =
        pybind11::memoryview::from_buffer(LibytProcessControl::Get().grid_parent_id,   // buffer pointer
                                          std::vector<long>({param_yt.num_grids, 1}),  // shape (rows)
                                          {sizeof(long), sizeof(long)}                 // strides in bytes
        );
    py_hierarchy["grid_levels"] =
        pybind11::memoryview::from_buffer(LibytProcessControl::Get().grid_levels,      // buffer pointer
                                          std::vector<long>({param_yt.num_grids, 1}),  // shape (rows)
                                          {sizeof(int), sizeof(int)}                   // strides in bytes
        );
    py_hierarchy["proc_num"] =
        pybind11::memoryview::from_buffer(LibytProcessControl::Get().proc_num,         // buffer pointer
                                          std::vector<long>({param_yt.num_grids, 1}),  // shape (rows)
                                          {sizeof(int), sizeof(int)}                   // strides in bytes
        );
    if (param_yt.num_par_types > 0) {
        py_hierarchy["par_count_list"] = pybind11::memoryview::from_buffer(
            LibytProcessControl::Get().par_count_list,                        // buffer pointer
            std::vector<long>({param_yt.num_grids, param_yt.num_par_types}),  // shape (rows, cols)
            {sizeof(long) * param_yt.num_par_types, sizeof(long)}             // strides in bytes
        );
    }
#endif  // #ifndef USE_PYBIND11

    return YT_SUCCESS;

}  // FUNCTION : allocate_hierarchy
