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
    SET_TIMER(__PRETTY_FUNCTION__);

    // Get field info and catch error
    void (*derived_func)(const int, const long*, const char*, yt_array*) = nullptr;
    yt_field* field_list = LibytProcessControl::Get().field_list;
    int field_id = -1;
    yt_dtype field_dtype = YT_DTYPE_UNKNOWN;

    for (int v = 0; v < g_param_yt.num_fields; v++) {
        if (strcmp(field_list[v].field_name, field_name) == 0) {
            field_id = v;
            field_dtype = field_list[v].field_dtype;
            derived_func = field_list[v].derived_func;
            break;
        }
    }

    if (field_id == -1) {
        std::string error_msg = "Cannot find field_name [ " + std::string(field_name) + " ] in field_list.\n";
        throw pybind11::value_error(error_msg.c_str());
    }
    if (derived_func == nullptr) {
        // TODO: should test this part.
        std::string error_msg =
            "In field_list, field_name [ " + std::string(field_name) + " ], derived_func did not set properly.\n";
        // TODO: should use pybind11::set_error, but it's not working.
        PyErr_SetString(PyExc_NotImplementedError, error_msg.c_str());
        throw pybind11::error_already_set();
    }
    if (field_dtype == YT_DTYPE_UNKNOWN) {
        std::string error_msg =
            "In field_list, field_name [ " + std::string(field_name) + " ], field_dtype did not set properly.\n";
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
    std::vector<long> shape, stride;
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

//-------------------------------------------------------------------------------------------------------
// Function    :  get_particle
// Description :  Use the get_par_attr defined inside yt_particle struct to get the particle attributes.
//
// Note        :  1. Support only grid dimension = 3 for now, which is "coor_x", "coor_y", "coor_z" in
//                   yt_particle must be set.
//                2. Deal with local particles only.
//                3. The returned numpy array data type well be set by attr_dtype.
//                4. We will always return 1D numpy array, with length equal particle count of the species
//                   in that grid.
//                5. Return Py_None if number of ptype particle == 0.
//
// Python Parameter     :          int : GID of the grid
//                                 str : ptype, particle type
//                                 str : attr_name, attribute name
// C Function Parameter :         long : GID of the grid
//                         const char* : ptype, particle type
//                         const char* : attr_name, attribute name
//
// Return      :  numpy.1darray
//-------------------------------------------------------------------------------------------------------
pybind11::array get_particle(long gid, const char* ptype, const char* attr_name) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Get particle info and catch error
    void (*get_par_attr)(const int, const long*, const char*, const char*, yt_array*) = nullptr;
    yt_particle* particle_list = LibytProcessControl::Get().particle_list;
    int particle_id = -1, attr_id = -1;
    yt_dtype attr_dtype = YT_DTYPE_UNKNOWN;

    for (int v = 0; v < g_param_yt.num_par_types; v++) {
        if (strcmp(particle_list[v].par_type, ptype) == 0) {
            particle_id = v;
            get_par_attr = particle_list[v].get_par_attr;
            for (int w = 0; w < particle_list[v].num_attr; w++) {
                if (strcmp(particle_list[v].attr_list[w].attr_name, attr_name) == 0) {
                    attr_id = w;
                    attr_dtype = particle_list[v].attr_list[w].attr_dtype;
                    break;
                }
            }
            break;
        }
    }

    if (particle_id == -1) {
        std::string error_msg = "Cannot find particle type [ " + std::string(ptype) + " ] in particle_list.\n";
        throw pybind11::value_error(error_msg.c_str());
    }
    if (attr_id == -1) {
        std::string error_msg = "Cannot find attribute [ " + std::string(attr_name) + " ] in particle type [ " +
                                std::string(ptype) + " ] in particle_list.\n";
        throw pybind11::value_error(error_msg.c_str());
    }
    if (get_par_attr == nullptr) {
        std::string error_msg =
            "In particle_list, par_type [ " + std::string(ptype) + " ], get_par_attr did not set properly.\n";
        PyErr_SetString(PyExc_NotImplementedError, error_msg.c_str());
        throw pybind11::error_already_set();
    }
    if (attr_dtype == YT_DTYPE_UNKNOWN) {
        std::string error_msg = "In particle_list, par_type [ " + std::string(ptype) + " ] attr_name [ " +
                                std::string(attr_name) + " ], attr_dtype did not set properly.\n";
        throw pybind11::value_error(error_msg.c_str());
    }

    // Get particle info and catch error
    long array_length;
    int proc_num;
    if (yt_getGridInfo_ProcNum(gid, &proc_num) != YT_SUCCESS ||
        yt_getGridInfo_ParticleCount(gid, ptype, &array_length) != YT_SUCCESS) {
        std::string error_msg = "Cannot get particle number in grid [ " + std::to_string(gid) + " ] or MPI rank.\n";
        throw pybind11::value_error(error_msg.c_str());
    }

    if (proc_num != g_myrank) {
        std::string error_msg = "Trying to prepare nonlocal particles. Grid [ " + std::to_string(gid) +
                                " ] is on MPI rank [ " + std::to_string(proc_num) + " ].\n";
        throw pybind11::value_error(error_msg.c_str());
    }
    if (array_length == 0) {
        // TODO: can I do this??? Even if I cannot, just return a dummy array, yt_libyt filters particle 0 too.
        // but should find a better solution too.
        return pybind11::none();
    }
    if (array_length < 0) {
        std::string error_msg = "Trying to prepare particle type [ " + std::string(ptype) + " ] in grid [ " +
                                std::to_string(gid) + " ] that has particle count = " + std::to_string(array_length) +
                                " < 0.\n";
        throw pybind11::value_error(error_msg.c_str());
    }

    // Generate particle data
    int dtype_size;
    get_dtype_size(attr_dtype, &dtype_size);
    std::vector<long> shape({array_length});
    std::vector<long> stride({dtype_size});
    pybind11::array output = get_pybind11_allocate_array_dtype(attr_dtype, shape, stride);

    // Call get particle attribute function
    yt_array data_array[1];
    data_array[0].data_ptr = static_cast<void*>(output.mutable_data());
    data_array[0].data_length = array_length;
    data_array[0].gid = gid;
    long list_gid[1] = {gid};

    get_par_attr(1, list_gid, ptype, attr_name, data_array);

    return static_cast<pybind11::array>(output);
}

PYBIND11_EMBEDDED_MODULE(libyt, m) {
    SET_TIMER(__PRETTY_FUNCTION__);

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
    m.def("get_particle", &get_particle, pybind11::return_value_policy::take_ownership);
}

#endif  // #ifdef USE_PYBIND11