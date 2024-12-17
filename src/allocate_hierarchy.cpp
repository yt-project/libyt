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

    // TODO: check this warning when doing DataStructureAmr (it should have mechanism to allocate and free this)
    // remove all key-value pairs if one wants to overwrite the existing dictionary
    // ==> it should happen only if one calls yt_set_Parameters() more than once
    //    if (PyDict_Size(LibytProcessControl::Get().py_hierarchy_) > 0) {
    //        PyDict_Clear(LibytProcessControl::Get().py_hierarchy_);
    //        log_warning("Removing existing key-value pairs in libyt.hierarchy ... done\n");
    //    }

    yt_param_yt& param_yt = LibytProcessControl::Get().param_yt_;
    LibytProcessControl::Get().grid_left_edge_ = new double[param_yt.num_grids * 3];
    LibytProcessControl::Get().grid_right_edge_ = new double[param_yt.num_grids * 3];
    LibytProcessControl::Get().grid_dimensions_ = new int[param_yt.num_grids * 3];
    LibytProcessControl::Get().grid_parent_id_ = new long[param_yt.num_grids];
    LibytProcessControl::Get().grid_levels_ = new int[param_yt.num_grids];
    LibytProcessControl::Get().proc_num_ = new int[param_yt.num_grids];
    if (param_yt.num_par_types > 0) {
        LibytProcessControl::Get().par_count_list_ = new long[param_yt.num_grids * param_yt.num_par_types];
    } else {
        LibytProcessControl::Get().par_count_list_ = nullptr;
    }

    npy_intp np_dim[2];
    np_dim[0] = param_yt.num_grids;

    np_dim[1] = 3;
    PyObject* py_grid_left_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, LibytProcessControl::Get().grid_left_edge_);
    PyObject* py_grid_right_edge = WrapToNumPyArray(2, np_dim, YT_DOUBLE, LibytProcessControl::Get().grid_right_edge_);
    PyObject* py_grid_dimensions = WrapToNumPyArray(2, np_dim, YT_INT, LibytProcessControl::Get().grid_dimensions_);

    np_dim[1] = 1;
    PyObject* py_grid_parent_id = WrapToNumPyArray(2, np_dim, YT_LONG, LibytProcessControl::Get().grid_parent_id_);
    PyObject* py_grid_levels = WrapToNumPyArray(2, np_dim, YT_INT, LibytProcessControl::Get().grid_levels_);
    PyObject* py_proc_num = WrapToNumPyArray(2, np_dim, YT_INT, LibytProcessControl::Get().proc_num_);
    PyObject* py_par_count_list;
    if (param_yt.num_par_types > 0) {
        np_dim[1] = param_yt.num_par_types;
        py_par_count_list = WrapToNumPyArray(2, np_dim, YT_LONG, LibytProcessControl::Get().par_count_list_);
    }

    // Bind them to libyt.hierarchy
    // Even though the pointer is de-referenced, still need to freed it in the memory ourselves at freed
    // (TODO: should I make it owned by python?)
#ifndef USE_PYBIND11
    PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, "grid_left_edge", py_grid_left_edge);
    PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, "grid_right_edge", py_grid_right_edge);
    PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, "grid_dimensions", py_grid_dimensions);
    PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, "grid_parent_id", py_grid_parent_id);
    PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, "grid_levels", py_grid_levels);
    PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, "proc_num", py_proc_num);
    if (param_yt.num_par_types > 0) {
        PyDict_SetItemString(LibytProcessControl::Get().py_hierarchy_, "par_count_list", py_par_count_list);
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
    if (param_yt.num_par_types > 0) {
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
    if (param_yt.num_par_types > 0) {
        Py_DECREF(py_par_count_list);
    }

    return YT_SUCCESS;
}  // FUNCTION : allocate_hierarchy
