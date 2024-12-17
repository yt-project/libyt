#include "libyt_process_control.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#endif

static PyObject* WrapToNumPyArray(int dim, npy_intp* npy_dim, yt_dtype data_dtype, void* data_ptr) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int npy_dtype;
    get_npy_dtype(data_dtype, &npy_dtype);

    PyObject* py_data = PyArray_SimpleNewFromData(dim, npy_dim, npy_dtype, data_ptr);

    return py_data;
}

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

    PyObject* py_data;
    npy_intp np_dim[2];

    // Even though the pointer is de-referenced, still need to freed it in the memory ourselves at freed
    np_dim[0] = param_yt.num_grids;
    np_dim[1] = 3;
    py_data = WrapToNumPyArray(2, np_dim, YT_DOUBLE, LibytProcessControl::Get().grid_left_edge);
    py_hierarchy["grid_left_edge"] = py_data;
    Py_DECREF(py_data);

    py_data = WrapToNumPyArray(2, np_dim, YT_DOUBLE, LibytProcessControl::Get().grid_right_edge);
    py_hierarchy["grid_right_edge"] = py_data;
    Py_DECREF(py_data);

    py_data = WrapToNumPyArray(2, np_dim, YT_INT, LibytProcessControl::Get().grid_dimensions);
    py_hierarchy["grid_dimensions"] = py_data;
    Py_DECREF(py_data);

    np_dim[1] = 1;
    py_data = WrapToNumPyArray(2, np_dim, YT_LONG, LibytProcessControl::Get().grid_parent_id);
    py_hierarchy["grid_parent_id"] = py_data;
    Py_DECREF(py_data);

    py_data = WrapToNumPyArray(2, np_dim, YT_INT, LibytProcessControl::Get().grid_levels);
    py_hierarchy["grid_levels"] = py_data;
    Py_DECREF(py_data);

    py_data = WrapToNumPyArray(2, np_dim, YT_INT, LibytProcessControl::Get().proc_num);
    py_hierarchy["proc_num"] = py_data;
    Py_DECREF(py_data);

    if (param_yt.num_par_types > 0) {
        np_dim[1] = param_yt.num_par_types;
        py_data = WrapToNumPyArray(2, np_dim, YT_LONG, LibytProcessControl::Get().par_count_list);
        py_hierarchy["par_count_list"] = py_data;
        Py_DECREF(py_data);
    }
#endif  // #ifndef USE_PYBIND11

    return YT_SUCCESS;

}  // FUNCTION : allocate_hierarchy
