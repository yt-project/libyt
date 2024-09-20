#ifdef USE_PYBIND11

#include "LibytProcessControl.h"
#include "libyt.h"
#include "pybind11/embed.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Description :  List of libyt C extension python methods built using Pybind11 API
//
// Note        :  1. List of python C extension methods functions.
//                2. These function will be called in python, so the parameters indicate python
//                   input type.
//
// Lists       :       Python Method           C Extension Function
//              .............................................................
//                     derived_func(int, str)  derived_func(long, const char*)
//                     get_particle          libyt_particle_get_particle
//                     get_field_remote      libyt_field_get_field_remote
//                     get_particle_remote   libyt_particle_get_particle_remote
//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------
// Function    :  derived_func
// Description :  Use the derived function inside yt_field struct to generate the field, then pass back
//                to Python.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. This function only needs to deal with the local grids.
//                3. The returned numpy array data type is according to field's field_dtype defined at
//                   yt_field.
//                4. Now, input from Python only contains gid and field name. In the future, when we
//                   support hybrid OpenMP/MPI, it can accept list and a string.
//
// Python Parameter     :          int : GID of the grid
//                                 str : field name
// C Function Parameter :         long : GID of the grid
//                         const char* : field name
//
// Return      :  numpy.3darray
//-------------------------------------------------------------------------------------------------------
pybind11::array derived_func(long gid, const char* field_name) {
#ifdef SUPPORT_TIMER
    SET_TIMER(__PRETTY_FUNCTION__, &timer_control);
#endif

    // Get field info and catch error
    void (*derived_func)(const int, const long*, const char*, yt_array*) = nullptr;
    yt_field* field_list = LibytProcessControl::Get().field_list;
    int field_id = -1;
    yt_dtype field_dtype = YT_DTYPE_UNKNOWN;

    for (int v = 0; v < g_param_yt.num_fields; v++) {
        if (strcmp(field_list[v].field_name, field_name) == 0) {
            field_id = v;
            field_dtype = field_list[v].field_dtype;
            if (field_list[v].derived_func != nullptr) {
                derived_func = field_list[v].derived_func;
            }
            break;
        }
    }

    if (field_id == -1) {
        std::string error_msg = "Cannot find field_name [ " + std::string(field_name) + " ] in field_list.\n";
        throw pybind11::value_error(error_msg.c_str());
    }
    if (derived_func == nullptr) {
        // TODO: should test this part.
        std::string error_msg = "In field_list, field_name [ " + std::string(field_list[field_id].field_name) +
                                " ], derived_func did not set properly.\n";
        // TODO: should use pybind11::set_error, but it's not working.
        PyErr_SetString(PyExc_NotImplementedError, error_msg.c_str());
        throw pybind11::error_already_set();
    }
    if (field_dtype == YT_DTYPE_UNKNOWN) {
        std::string error_msg = "In field_list, field_name [ " + std::string(field_list[field_id].field_name) +
                                " ], field_dtype did not set properly.\n";
        throw pybind11::value_error(error_msg.c_str());
    }

    // Get grid info and catch error
    int grid_dimensions[3], proc_num;
    if (yt_getGridInfo_ProcNum(gid, &proc_num) != YT_SUCCESS ||
        yt_getGridInfo_Dimensions(gid, &grid_dimensions) != YT_SUCCESS) {
        std::string error_msg = "Cannot get grid [ " + std::to_string(gid) + " ] dimensions or MPI process rank.\n";
        throw pybind11::value_error(error_msg.c_str());
    }

    if (proc_num != g_myrank) {
        std::string error_msg = "Trying to prepare nonlocal grid. Grid [ " + std::to_string(gid) +
                                " ] is on MPI rank [ " + std::to_string(proc_num) + " ].\n";
        throw pybind11::value_error(error_msg.c_str());
    }
    for (int d = 0; d < 3; d++) {
        if (grid_dimensions[d] < 0) {
            std::string error_msg = "Trying to prepare grid [ " + std::to_string(gid) + " ] that has grid_dimensions[" +
                                    std::to_string(d) + "] = " + std::to_string(grid_dimensions[d]) + " < 0.\n";
            throw pybind11::value_error(error_msg.c_str());
        }
    }

    // Generate derived field data
    std::vector<int> shape, stride;
    int dtype_size;
    get_dtype_size(field_dtype, &dtype_size);
    if (field_list[field_id].contiguous_in_x) {
        shape = {grid_dimensions[2], grid_dimensions[1], grid_dimensions[0]};
    } else {
        shape = {grid_dimensions[0], grid_dimensions[1], grid_dimensions[2]};
    }
    stride = {dtype_size * shape[1] * shape[2], dtype_size * shape[2], dtype_size};
    pybind11::array output = get_pybind11_allocate_array_dtype(field_dtype, shape, stride);

    // Call derived function
    yt_array data_array[1];
    data_array[0].data_ptr = static_cast<void*>(output.mutable_data());
    data_array[0].data_length = shape[0] * shape[1] * shape[2];
    data_array[0].gid = gid;
    long list_gid[1] = {gid};

    derived_func(1, list_gid, field_name, data_array);

    return static_cast<pybind11::array>(output);
}

PYBIND11_EMBEDDED_MODULE(libyt, m) {
#ifdef SUPPORT_TIMER
    SET_TIMER(__PRETTY_FUNCTION__, &timer_control);
#endif
    m.attr("param_yt") = pybind11::dict();
    m.attr("param_user") = pybind11::dict();
    m.attr("hierarchy") = pybind11::dict();
    m.attr("grid_data") = pybind11::dict();
    m.attr("particle_data") = pybind11::dict();
    m.attr("libyt_info") = pybind11::dict();
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
    m.attr("interactive_mode") = pybind11::dict();
#endif

    m.attr("libyt_info")["version"] =
        pybind11::make_tuple(LIBYT_MAJOR_VERSION, LIBYT_MINOR_VERSION, LIBYT_MICRO_VERSION);
    m.attr("libyt_info")["SERIAL_MODE"] = pybind11::bool_(true);
    m.attr("libyt_info")["INTERACTIVE_MODE"] = pybind11::bool_(false);
    m.attr("libyt_info")["JUPYTER_KERNEL"] = pybind11::bool_(false);
#ifdef SUPPORT_TIMER
    m.attr("libyt_info")["SUPPORT_TIMER"] = pybind11::bool_(true);
#else
    m.attr("libyt_info")["SUPPORT_TIMER"] = pybind11::bool_(false);
#endif

    m.def("derived_func", &derived_func, pybind11::return_value_policy::take_ownership);
}

#endif  // #ifdef USE_PYBIND11