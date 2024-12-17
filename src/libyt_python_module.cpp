#ifdef USE_PYBIND11

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include <iostream>

#include "comm_mpi_rma.h"
#include "libyt.h"
#include "libyt_process_control.h"
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
    yt_field* field_list = LibytProcessControl::Get().data_structure_amr_.field_list_;
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
    yt_particle* particle_list = LibytProcessControl::Get().data_structure_amr_.particle_list_;
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
//                4. This function will get all the fields and grids in combination.
//                   So the total returned data get is len(fname_list) * len(nonlocal_id).
//                5. Directly return None if it is in SERIAL_MODE.
//                   TODO: Not sure if this would affect the performance. And do I even need this?
//                6. In Python, it is called like:
//                   libyt.get_field_remote( fname_list,
//                                           len(fname_list),
//                                           to_prepare,
//                                           len(to_prepare),
//                                           nonlocal_id,
//                                           nonlocal_rank,
//                                           len(nonlocal_id))
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

    // Create fetch data list
    std::vector<CommMpiRmaQueryInfo> fetch_data_list;
    fetch_data_list.reserve(len_nonlocal);
    for (int i = 0; i < len_nonlocal; i++) {
        fetch_data_list.emplace_back(
            CommMpiRmaQueryInfo{py_nonlocal_rank[i].cast<int>(), py_nonlocal_id[i].cast<long>()});
    }

    // Create prepare id list
    std::vector<long> prepare_id_list;
    for (auto& py_gid : py_to_prepare) {
        prepare_id_list.emplace_back(py_gid.cast<long>());
    }

    // Initialize one CommMpiRma at a time for a field.
    // TODO: Will support distributing multiple types of field after dealing with labeling for each type of field.
    for (auto& py_fname : py_fname_list) {
        // Prepare data for each field on each MPI rank.
        std::string fname = py_fname.cast<std::string>();

        DataHubAmr local_amr_data;
        DataHubReturn<AmrDataArray3D> prepared_data =
            local_amr_data.GetLocalFieldData(LibytProcessControl::Get().data_structure_amr_, fname, prepare_id_list);

        // Make sure every process can get the local field data correctly, otherwise fail fast.
        DataHubStatus all_status = static_cast<DataHubStatus>(CommMpi::CheckAllStates(
            static_cast<int>(prepared_data.status), static_cast<int>(DataHubStatus::kDataHubSuccess),
            static_cast<int>(DataHubStatus::kDataHubSuccess), static_cast<int>(DataHubStatus::kDataHubFailed)));
        if (all_status != DataHubStatus::kDataHubSuccess) {
            if (prepared_data.status == DataHubStatus::kDataHubFailed) {
                PyErr_SetString(PyExc_RuntimeError, local_amr_data.GetErrorStr().c_str());
            } else {
                PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
            }
            // local_amr_data.ClearCache();
            throw pybind11::error_already_set();
        }

        // Call MPI RMA operation
        CommMpiRmaAmrDataArray3D comm_mpi_rma(fname, "amr_grid");
        CommMpiRmaReturn<AmrDataArray3D> rma_return =
            comm_mpi_rma.GetRemoteData(prepared_data.data_list, fetch_data_list);
        if (rma_return.all_status != CommMpiRmaStatus::kMpiSuccess) {
            if (rma_return.status != CommMpiRmaStatus::kMpiSuccess) {
                PyErr_SetString(PyExc_RuntimeError, comm_mpi_rma.GetErrorStr().c_str());
            } else {
                PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
            }
            // local_amr_data.ClearCache();
            throw pybind11::error_already_set();
        }

        // Wrap to Python dictionary
        for (const AmrDataArray3D& fetched_data : rma_return.data_list) {
            npy_intp npy_dim[3];
            if (fetched_data.contiguous_in_x) {
                npy_dim[0] = fetched_data.data_dim[2];
                npy_dim[1] = fetched_data.data_dim[1];
                npy_dim[2] = fetched_data.data_dim[0];
            } else {
                npy_dim[0] = fetched_data.data_dim[0];
                npy_dim[1] = fetched_data.data_dim[1];
                npy_dim[2] = fetched_data.data_dim[2];
            }
            int npy_dtype;
            get_npy_dtype(fetched_data.data_dtype, &npy_dtype);
            PyObject* py_data = PyArray_SimpleNewFromData(3, npy_dim, npy_dtype, fetched_data.data_ptr);
            PyArray_ENABLEFLAGS((PyArrayObject*)py_data, NPY_ARRAY_OWNDATA);

            if (!py_output.contains(pybind11::int_(fetched_data.id))) {
                py_field = pybind11::dict();
                py_output[pybind11::int_(fetched_data.id)] = py_field;
            } else {
                py_field = py_output[pybind11::int_(fetched_data.id)];
            }
            py_field[fname.c_str()] = py_data;
            Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we don't care it anymore.
        }

        // local_amr_data.ClearCache();
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
//                3. We first filter out data len <= 0 in prepared data and fetched data. So that we don't
//                   pass nullptr around in rma.
//                4. If there are no particles in one grid, then we write Py_None to it.
//                5. Directly return None if it is in SERIAL_MODE
//                   TODO: Not sure if this would affect the performance. And do I even need this?
//                         Wondering if pybind11 do dynamic casting back to a dictionary.
//                6. TODO: Some of the passed in arguments are not used, will fix it and define new API in the future.
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

    // Initialize one CommMpiRma at a time for a particle attribute.
    // TODO: Will support distributing multiple types of field/particle after dealing with labeling for each of them
    //       And also, get_field_remote/get_particle_remote can be merged once the API to yt_libyt has changed.
    for (auto& py_ptype : py_ptf_keys) {
        for (auto& py_attr : py_ptf[py_ptype]) {
            // Prepare data for particle count > 0
            std::string ptype = py_ptype.cast<std::string>();
            std::string attr = py_attr.cast<std::string>();

            std::vector<long> prepare_id_list;
            for (auto& py_gid : py_to_prepare) {
                long count;
                yt_getGridInfo_ParticleCount(py_gid.cast<long>(), ptype.c_str(), &count);
                if (count > 0) {
                    prepare_id_list.emplace_back(py_gid.cast<long>());
                }
            }

            DataHubAmr local_particle_data;
            DataHubReturn<AmrDataArray1D> prepared_data = local_particle_data.GetLocalParticleData(
                LibytProcessControl::Get().data_structure_amr_, ptype, attr, prepare_id_list);
            DataHubStatus all_status = static_cast<DataHubStatus>(CommMpi::CheckAllStates(
                static_cast<int>(prepared_data.status), static_cast<int>(DataHubStatus::kDataHubSuccess),
                static_cast<int>(DataHubStatus::kDataHubSuccess), static_cast<int>(DataHubStatus::kDataHubFailed)));

            if (all_status != DataHubStatus::kDataHubSuccess) {
                if (prepared_data.status == DataHubStatus::kDataHubFailed) {
                    PyErr_SetString(PyExc_RuntimeError, local_particle_data.GetErrorStr().c_str());
                } else {
                    PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
                }
                // local_particle_data.ClearCache();
                throw pybind11::error_already_set();
            }

            // Separate particle count > 0 and create fetch data list
            std::vector<CommMpiRmaQueryInfo> fetch_data_list;
            std::vector<long> fetch_particle_count0_list;
            for (int i = 0; i < len_nonlocal; i++) {
                long count;
                yt_getGridInfo_ParticleCount(py_nonlocal_id[i].cast<long>(), ptype.c_str(), &count);
                if (count > 0) {
                    fetch_data_list.emplace_back(
                        CommMpiRmaQueryInfo{py_nonlocal_rank[i].cast<int>(), py_nonlocal_id[i].cast<long>()});
                } else {
                    fetch_particle_count0_list.emplace_back(py_nonlocal_id[i].cast<long>());
                }
            }

            // Call MPI RMA operation
            CommMpiRmaAmrDataArray1D comm_mpi_rma(ptype + "-" + attr, "amr_particle");
            CommMpiRmaReturn<AmrDataArray1D> rma_return =
                comm_mpi_rma.GetRemoteData(prepared_data.data_list, fetch_data_list);
            if (rma_return.all_status != CommMpiRmaStatus::kMpiSuccess) {
                if (rma_return.status != CommMpiRmaStatus::kMpiSuccess) {
                    PyErr_SetString(PyExc_RuntimeError, comm_mpi_rma.GetErrorStr().c_str());
                } else {
                    PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
                }
                // local_particle_data.ClearCache();
                throw pybind11::error_already_set();
            }

            // Wrap fetched data to a Python dictionary
            for (const AmrDataArray1D& fetched_data : rma_return.data_list) {
                // Create dictionary data[grid id][ptype]
                long gid = fetched_data.id;
                if (!py_output.contains(pybind11::int_(gid))) {
                    py_output[pybind11::int_(gid)] = pybind11::dict();
                }
                if (!py_output[pybind11::int_(gid)].contains(ptype)) {
                    py_output[pybind11::int_(gid)][ptype.c_str()] = pybind11::dict();
                }

                if (fetched_data.data_len > 0) {
                    PyObject* py_data;
                    npy_intp npy_dim[1] = {fetched_data.data_len};
                    int npy_dtype;
                    get_npy_dtype(fetched_data.data_dtype, &npy_dtype);
                    py_data = PyArray_SimpleNewFromData(1, npy_dim, npy_dtype, fetched_data.data_ptr);
                    PyArray_ENABLEFLAGS((PyArrayObject*)py_data, NPY_ARRAY_OWNDATA);

                    py_output[pybind11::int_(gid)][ptype.c_str()][attr.c_str()] = py_data;
                    Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we don't care it anymore.
                } else {
                    py_output[pybind11::int_(gid)][ptype.c_str()][attr.c_str()] = pybind11::none();
                }
            }

            // Wrap particle count = 0 to a Python dictionary
            for (const long& gid : fetch_particle_count0_list) {
                if (!py_output.contains(pybind11::int_(gid))) {
                    py_output[pybind11::int_(gid)] = pybind11::dict();
                }
                if (!py_output[pybind11::int_(gid)].contains(ptype)) {
                    py_output[pybind11::int_(gid)][ptype.c_str()] = pybind11::dict();
                }
                py_output[pybind11::int_(gid)][ptype.c_str()][attr.c_str()] = pybind11::none();
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