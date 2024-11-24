#ifdef USE_PYBIND11

#include <iostream>

#include "LibytProcessControl.h"
#include "libyt.h"
#include "pybind11/embed.h"
#include "pybind11/numpy.h"
#include "pybind11/pytypes.h"
#include "yt_combo.h"
#include "yt_rma_field.h"
#include "yt_rma_particle.h"
#include "yt_type_array.h"

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

    for (int v = 0; v < LibytProcessControl::Get().param_yt_.num_fields; v++) {
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
        std::string error_msg =
            "In field_list, field_name [ " + std::string(field_name) + " ], derived_func did not set properly.";
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

    if (proc_num != LibytProcessControl::Get().mpi_rank_) {
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

    for (int v = 0; v < LibytProcessControl::Get().param_yt_.num_par_types; v++) {
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
            "In particle_list, par_type [ " + std::string(ptype) + " ], get_par_attr did not set properly.";
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

    if (proc_num != LibytProcessControl::Get().mpi_rank_) {
        std::string error_msg = "Trying to prepare nonlocal particles. Grid [ " + std::to_string(gid) +
                                " ] is on MPI rank [ " + std::to_string(proc_num) + " ].\n";
        throw pybind11::value_error(error_msg.c_str());
    }
    if (array_length == 0) {
        // TODO: can I do this??? Even if I cannot, just return a dummy array, yt_libyt filters particle 0 too.
        // but should find a better solution too.
        // It returns a numpy.ndarray with () object
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

//-------------------------------------------------------------------------------------------------------
// Function    :  get_field_remote
// Description :  Get non-local field data from remote ranks.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. We return in dictionary objects.
//                3. We assume that the fname_list passed in has the same fname order in each rank.
//                4. This function will get all the desired fields and grids.
//                5. Directly return None if it is in SERIAL_MODE.
//                   TODO: Not sure if this would affect the performance. And do I even need this?
//
// Parameter   :  list obj : fname_list    : list of field name to get.
//                     int : len_fname_list: length of fname_list.
//                list obj : to_prepare   : list of grid ids you need to prepare.
//                list obj : nonlocal_id  : nonlocal grid id that you want to get.
//                list obj : nonlocal_rank: where to get those nonlocal grid.
//
// Return      :  dict obj data[grid id][field_name][:,:,:] or None if it is serial mode
//-------------------------------------------------------------------------------------------------------
pybind11::object get_field_remote(const pybind11::list& py_fname_list, int len_fname_list,
                                  const pybind11::list& py_to_prepare, int len_to_prepare,
                                  const pybind11::list& py_nonlocal_id, const pybind11::list& py_nonlocal_rank,
                                  int len_nonlocal) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
    pybind11::dict py_output = pybind11::dict();
    pybind11::dict py_field;
    for (auto py_fname : py_fname_list) {
        // initialize RMA
        yt_rma_field RMAOperation = yt_rma_field(py_fname.cast<std::string>().c_str(), len_to_prepare, len_nonlocal);

        // prepare data
        for (auto py_gid : py_to_prepare) {
            long gid = py_gid.cast<long>();
            if (RMAOperation.prepare_data(gid) != YT_SUCCESS) {
                std::string error_msg = "Something went wrong in yt_rma_field when preparing data.";
                PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                throw pybind11::error_already_set();
            }
        }

        // Gather all prepared data and using rank 0 as root
        RMAOperation.gather_all_prepare_data(0);

        // Fetch remote data
        for (int i = 0; i < len_nonlocal; i++) {
            long get_gid = py_nonlocal_id[i].cast<long>();
            int get_rank = py_nonlocal_rank[i].cast<int>();
            if (RMAOperation.fetch_remote_data(get_gid, get_rank) != YT_SUCCESS) {
                std::string error_msg = "Something went wrong in yt_rma_field when fetching remote data.";
                PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                throw pybind11::error_already_set();
            }
        }

        // Clean up prepared data
        RMAOperation.clean_up();

        // Get fetched data, wrap them, and bind to Python dictionary
        PyObject* py_data;
        for (int i = 0; i < len_nonlocal; i++) {
            // (1) Get fetched data
            long gid;
            const char* fname = nullptr;
            yt_dtype data_dtype;
            int data_dim[3];
            void* data_ptr = nullptr;
            if (RMAOperation.get_fetched_data(&gid, &fname, &data_dtype, &data_dim, &data_ptr) != YT_SUCCESS) {
                break;
            }

            // (2) Wrap data_ptr to numpy array and make it owned by Python
            npy_intp npy_dim[3] = {data_dim[0], data_dim[1], data_dim[2]};
            int npy_dtype;
            get_npy_dtype(data_dtype, &npy_dtype);
            py_data = PyArray_SimpleNewFromData(3, npy_dim, npy_dtype, data_ptr);
            PyArray_ENABLEFLAGS((PyArrayObject*)py_data, NPY_ARRAY_OWNDATA);

            // (3) Build Python dictionary data[grid id][field_name][:,:,:]
            if (!py_output.contains(pybind11::int_(gid))) {
                py_field = pybind11::dict();
                py_output[pybind11::int_(gid)] = py_field;
            } else {
                py_field = py_output[pybind11::int_(gid)];
            }
            py_field[fname] = py_data;
            Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we don't care it anymore.
        }
    }
    return py_output;
#else   // #ifndef SERIAL_MODE
    return pybind11::none();
#endif  // #ifndef SERIAL_MODE
}

//-------------------------------------------------------------------------------------------------------
// Function    :  get_particle_remote
// Description :  Get non-local particle data from remote ranks.
//
// Note        :  1. We return in dictionary objects.
//                2. We assume that the list of to-get attribute has the same ptype and attr order in each
//                   rank.
//                3. If there are no particles in one grid, then we write Py_None to it.
//                4. Directly return None if it is in SERIAL_MODE
//                   TODO: Not sure if this would affect the performance. And do I even need this?
//                         Wondering if pybind11 do dynamic casting back to a dictionary.
//                5. TODO: Some of the passed in arguments are not used, will fix it and define new API in the future.
//                         get_field_remote and get_particle_remote should be merged.
//
// Parameter   :  dict obj : ptf          : {<ptype>: [<attr1>, <attr2>, ...]} particle type and attributes
//                                          to read.
//                iterable obj : ptf_keys : list of ptype keys.
//                list obj : to_prepare   : list of grid ids you need to prepare.
//                list obj : nonlocal_id  : nonlocal grid id that you want to get.
//                list obj : nonlocal_rank: where to get those nonlocal grid.
//
// Return      :  dict obj data[grid id][ptype][attribute]
//-------------------------------------------------------------------------------------------------------
pybind11::object get_particle_remote(const pybind11::dict& py_ptf, const pybind11::iterable& py_ptf_keys,
                                     const pybind11::list& py_to_prepare, int len_to_prepare,
                                     const pybind11::list& py_nonlocal_id, const pybind11::list& py_nonlocal_rank,
                                     int len_nonlocal) {
    SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
    pybind11::dict py_output = pybind11::dict();
    for (auto& py_ptype : py_ptf_keys) {
        for (auto& py_attr : py_ptf[py_ptype]) {
            // initialize RMA
            yt_rma_particle RMAOperation =
                yt_rma_particle(py_ptype.cast<std::string>().c_str(), py_attr.cast<std::string>().c_str(),
                                len_to_prepare, len_nonlocal);

            // prepare data
            for (auto& py_gid : py_to_prepare) {
                long gid = py_gid.cast<long>();
                if (RMAOperation.prepare_data(gid) != YT_SUCCESS) {
                    std::string error_msg = "Something went wrong in yt_rma_particle when preparing data.";
                    PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    throw pybind11::error_already_set();
                }
            }

            // Gather all prepared data and using rank 0 as root
            RMAOperation.gather_all_prepare_data(0);

            // Fetch remote data
            for (int i = 0; i < len_nonlocal; i++) {
                long get_gid = py_nonlocal_id[i].cast<long>();
                int get_rank = py_nonlocal_rank[i].cast<int>();
                if (RMAOperation.fetch_remote_data(get_gid, get_rank) != YT_SUCCESS) {
                    std::string error_msg = "Something went wrong in yt_rma_particle when fetching remote data.";
                    PyErr_SetString(PyExc_RuntimeError, error_msg.c_str());
                    throw pybind11::error_already_set();
                }
            }

            // Clean up
            RMAOperation.clean_up();

            // Get fetched data, wrap them, and bind to Python dictionary
            PyObject* py_data = nullptr;
            for (int i = 0; i < len_nonlocal; i++) {
                // (1) Get fetched data
                long gid;
                const char* ptype = nullptr;
                const char* attr_name = nullptr;
                yt_dtype data_dtype;
                long data_length;
                void* data_ptr = nullptr;
                if (RMAOperation.get_fetched_data(&gid, &ptype, &attr_name, &data_dtype, &data_length, &data_ptr) !=
                    YT_SUCCESS) {
                    break;
                }

                // (2) Wrap data_ptr to numpy array and make it owned by Python
                if (data_length > 0) {
                    npy_intp npy_dim[1] = {data_length};
                    int npy_dtype;
                    get_npy_dtype(data_dtype, &npy_dtype);
                    py_data = PyArray_SimpleNewFromData(1, npy_dim, npy_dtype, data_ptr);
                    PyArray_ENABLEFLAGS((PyArrayObject*)py_data, NPY_ARRAY_OWNDATA);
                }

                // (3) Build Python dictionary data[grid id][ptype][attr] = py_data or None
                if (!py_output.contains(pybind11::int_(gid))) {
                    py_output[pybind11::int_(gid)] = pybind11::dict();
                }
                if (!py_output[pybind11::int_(gid)].contains(ptype)) {
                    py_output[pybind11::int_(gid)][ptype] = pybind11::dict();
                }
                if (data_length > 0) {
                    py_output[pybind11::int_(gid)][ptype][attr_name] = py_data;
                    Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we don't care it anymore.}
                } else {
                    py_output[pybind11::int_(gid)][ptype][attr_name] = pybind11::none();
                }
            }
        }
    }
    return py_output;
#else   // #ifndef SERIAL_MODE
    return pybind11::none();
#endif  // #ifndef SERIAL_MODE
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

#ifdef SERIAL_MODE
    m.attr("libyt_info")["SERIAL_MODE"] = pybind11::bool_(true);
#else
    m.attr("libyt_info")["SERIAL_MODE"] = pybind11::bool_(false);
#endif

#ifdef INTERACTIVE_MODE
    m.attr("libyt_info")["INTERACTIVE_MODE"] = pybind11::bool_(true);
#else
    m.attr("libyt_info")["INTERACTIVE_MODE"] = pybind11::bool_(false);
#endif

#ifdef JUPYTER_KERNEL
    m.attr("libyt_info")["JUPYTER_KERNEL"] = pybind11::bool_(true);
#else
    m.attr("libyt_info")["JUPYTER_KERNEL"] = pybind11::bool_(false);
#endif

#ifdef SUPPORT_TIMER
    m.attr("libyt_info")["SUPPORT_TIMER"] = pybind11::bool_(true);
#else
    m.attr("libyt_info")["SUPPORT_TIMER"] = pybind11::bool_(false);
#endif

    m.def("derived_func", &derived_func, pybind11::return_value_policy::take_ownership);
    m.def("get_particle", &get_particle, pybind11::return_value_policy::take_ownership);
    m.def("get_field_remote", &get_field_remote, pybind11::return_value_policy::take_ownership);
    m.def("get_particle_remote", &get_particle_remote, pybind11::return_value_policy::take_ownership);
}

#endif  // #ifdef USE_PYBIND11