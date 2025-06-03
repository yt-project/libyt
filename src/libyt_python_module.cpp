#include <iostream>

#include "comm_mpi_rma.h"
#include "dtype_utilities.h"
#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "numpy_controller.h"
#include "python_controller.h"
#include "timer.h"

#ifdef USE_PYBIND11
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#endif

#ifndef SERIAL_MODE
template<typename DataClass, typename RmaDataClass>
static std::string CallFieldRma(const std::string& fname,
                                const std::vector<long>& prepare_id_list,
                                const std::vector<CommMpiRmaQueryInfo>& fetch_data_list,
                                RmaDataClass& rma) {
  // Prepare data for each field on each MPI rank, fail fast if any process fails.
  DataHubAmrField<DataClass> local_amr_data(false);
  DataHubReturn<DataClass> prepared_data = local_amr_data.GetLocalFieldData(
      LibytProcessControl::Get().data_structure_amr_, fname, prepare_id_list);

  DataHubStatus all_status = static_cast<DataHubStatus>(
      CommMpi::CheckAllStates(static_cast<int>(prepared_data.status),
                              static_cast<int>(DataHubStatus::kDataHubSuccess),
                              static_cast<int>(DataHubStatus::kDataHubSuccess),
                              static_cast<int>(DataHubStatus::kDataHubFailed)));
  if (all_status != DataHubStatus::kDataHubSuccess) {
    if (prepared_data.status == DataHubStatus::kDataHubFailed) {
      return local_amr_data.GetErrorStr();
    } else {
      return std::string("Error occurred in other MPI process.");
    }
  }

  // Call MPI RMA operation
  CommMpiRmaReturn<DataClass> rma_return =
      rma.GetRemoteData(prepared_data.data_list, fetch_data_list);
  if (rma_return.all_status != CommMpiRmaStatus::kMpiSuccess) {
    if (rma_return.status != CommMpiRmaStatus::kMpiSuccess) {
      return rma.GetErrorStr();
    } else {
      return std::string("Error occurred in other MPI process.");
    }
  }

  return std::string("success");
}
#endif

//-------------------------------------------------------------------------------------------------------
// Description :  List of libyt C extension python methods built using Pybind11 API
//
// Note        :  1. List of python C extension methods functions.
//                2. These function will be called in python, so the parameters indicate
//                python
//                   input type.
//
// Lists       :       Python Method           C Extension Function
//              .............................................................
//                     derived_func(int, str)  derived_func(long, const char*)
//                     get_particle            libyt_particle_get_particle
//                     get_field_remote        libyt_field_get_field_remote
//                     get_particle_remote     libyt_particle_get_particle_remote
//-------------------------------------------------------------------------------------------------------

#ifdef USE_PYBIND11
//-------------------------------------------------------------------------------------------------------
// Helper function : AllocatePybind11Array
// Description     : Allocate pybind11::array_t array based on yt_dtype, shape, and
// stride.
//
// Notes           : 1. If the yt_dtype is not found, it will return pybind11::array().
//-------------------------------------------------------------------------------------------------------
static pybind11::array AllocatePybind11Array(yt_dtype data_type,
                                             const std::vector<long>& shape,
                                             const std::vector<long>& stride) {
  switch (data_type) {
    case YT_FLOAT:
      return pybind11::array_t<float>(shape, stride);
    case YT_DOUBLE:
      return pybind11::array_t<double>(shape, stride);
    case YT_LONGDOUBLE:
      return pybind11::array_t<long double>(shape, stride);
    case YT_CHAR:
      return pybind11::array_t<char>(shape, stride);
    case YT_UCHAR:
      return pybind11::array_t<unsigned char>(shape, stride);
    case YT_SHORT:
      return pybind11::array_t<short>(shape, stride);
    case YT_USHORT:
      return pybind11::array_t<unsigned short>(shape, stride);
    case YT_INT:
      return pybind11::array_t<int>(shape, stride);
    case YT_UINT:
      return pybind11::array_t<unsigned int>(shape, stride);
    case YT_LONG:
      return pybind11::array_t<long>(shape, stride);
    case YT_ULONG:
      return pybind11::array_t<unsigned long>(shape, stride);
    case YT_LONGLONG:
      return pybind11::array_t<long long>(shape, stride);
    case YT_ULONGLONG:
      return pybind11::array_t<unsigned long long>(shape, stride);
    case YT_DTYPE_UNKNOWN:
      return pybind11::array();
    default:
      return pybind11::array();
  }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  DerivedFunc
// Description :  Use the derived function inside yt_field struct to generate the field,
// then pass back
//                to Python.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. This function only needs to deal with the local grids.
//                3. The returned numpy array data type is according to field's
//                field_dtype defined at
//                   yt_field.
//                4. Now, input from Python only contains gid and field name. In the
//                future, when we
//                   support hybrid OpenMP/MPI, it can accept list and a string.
//                5. Due to the creation of pybind11::array_t using an existing pointer
//                makes Python not owning
//                   an array, we have to initialize a new buffer in array_t and copy the
//                   generated data into the new buffer. (ref:
//                   https://stackoverflow.com/a/44682603/20413451)
//
// Python Parameter     :          int : GID of the grid
//                                 str : field name
// C Function Parameter :         long : GID of the grid
//                         const char* : field name
//
// Return      :  numpy.3darray
//-------------------------------------------------------------------------------------------------------
pybind11::array DerivedFunc(long gid, const char* field_name) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // Generate data
  std::vector<long> gid_list = {gid};
  std::vector<AmrDataArray3D> storage;
  DataStructureOutput status =
      LibytProcessControl::Get().data_structure_amr_.GenerateLocalFieldData(
          gid_list, field_name, storage);
  if (status.status != DataStructureStatus::kDataStructureSuccess) {
    for (const AmrDataArray3D& kData : storage) {
      free(kData.data_ptr);
    }
    if (status.status == DataStructureStatus::kDataStructureNotImplemented) {
      PyErr_SetString(PyExc_NotImplementedError, status.error.c_str());
      throw pybind11::error_already_set();
    } else {
      throw pybind11::value_error(status.error.c_str());
    }
  }

  // Allocate pybind11
  // (We need this otherwise the array is not owned by python :()
  std::vector<long> shape, stride;
  int dtype_size = dtype_utilities::GetYtDtypeSize(storage[0].data_dtype);
  if (storage[0].contiguous_in_x) {
    shape = {storage[0].data_dim[2], storage[0].data_dim[1], storage[0].data_dim[0]};
  } else {
    shape = {storage[0].data_dim[0], storage[0].data_dim[1], storage[0].data_dim[2]};
  }
  stride = {dtype_size * shape[1] * shape[2], dtype_size * shape[2], dtype_size};
  pybind11::array output = AllocatePybind11Array(storage[0].data_dtype, shape, stride);

  // Copy data from storage to output and then free storage
  memcpy(output.mutable_data(),
         storage[0].data_ptr,
         dtype_size * shape[0] * shape[1] * shape[2]);
  for (const AmrDataArray3D& kData : storage) {
    free(kData.data_ptr);
  }

  return static_cast<pybind11::array>(output);
}

//-------------------------------------------------------------------------------------------------------
// Function    :  GetParticle
// Description :  Use the get_par_attr defined inside yt_particle struct to get the
// particle attributes.
//
// Note        :  1. Support only grid dimension = 3 for now, which is "coor_x", "coor_y",
// "coor_z" in
//                   yt_particle must be set.
//                2. Deal with local particles only.
//                3. The returned numpy array data type well be set by attr_dtype.
//                4. We will always return 1D numpy array, with length equal particle
//                count of the species
//                   in that grid.
//                5. Return Py_None if number of ptype particle == 0.
//                6. Due to the creation of pybind11::array_t using an existing pointer
//                makes Python not owning
//                   an array, we have to initialize a new buffer in array_t and copy the
//                   generated data into the new buffer. (ref:
//                   https://stackoverflow.com/a/44682603/20413451)
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
pybind11::array GetParticle(long gid, const char* ptype, const char* attr_name) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // Generate data
  std::vector<long> gid_list = {gid};
  std::vector<AmrDataArray1D> storage;
  DataStructureOutput status =
      LibytProcessControl::Get().data_structure_amr_.GenerateLocalParticleData(
          gid_list, ptype, attr_name, storage);
  if (status.status != DataStructureStatus::kDataStructureSuccess) {
    for (const AmrDataArray1D& kData : storage) {
      free(kData.data_ptr);
    }
    if (status.status == DataStructureStatus::kDataStructureNotImplemented) {
      PyErr_SetString(PyExc_NotImplementedError, status.error.c_str());
      throw pybind11::error_already_set();
    } else {
      throw pybind11::value_error(status.error.c_str());
    }
  }

  // Return None if there is no particle
  if (storage[0].data_dim[0] == 0) {
    // TODO: can I do this??? Even if I cannot, just return a dummy array, yt_libyt
    // filters particle 0 too. but should find a better solution too. It returns a
    // numpy.ndarray with () object
    return pybind11::none();
  } else {
    // Allocate pybind11
    // (We need this otherwise the array is not owned by python :()
    int dtype_size = dtype_utilities::GetYtDtypeSize(storage[0].data_dtype);
    std::vector<long> shape({storage[0].data_dim[0]});
    std::vector<long> stride({dtype_size});
    pybind11::array output = AllocatePybind11Array(storage[0].data_dtype, shape, stride);

    // Copy data from storage to output and then free storage
    memcpy(output.mutable_data(), storage[0].data_ptr, dtype_size * shape[0]);
    for (const AmrDataArray1D& kData : storage) {
      free(kData.data_ptr);
    }

    return static_cast<pybind11::array>(output);
  }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  GetFieldRemote
// Description :  Get non-local field data from remote ranks.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. We return in dictionary objects.
//                3. We assume that the fname_list passed in has the same fname order in
//                each rank.
//                4. This function will get all the fields and grids in combination.
//                   So the total returned data get is len(fname_list) * len(nonlocal_id).
//                5. Directly return None if it is in SERIAL_MODE.
//                   TODO: Not sure if this would affect the performance. And do I even
//                   need this?
//                6. The duplicated code also shows that I show singled out the method to
//                   construct and bind the data under libyt.
//                7. In Python, it is called like:
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
pybind11::object GetFieldRemote(const pybind11::list& py_fname_list, int len_fname_list,
                                const pybind11::list& py_to_prepare, int len_to_prepare,
                                const pybind11::list& py_nonlocal_id,
                                const pybind11::list& py_nonlocal_rank,
                                int len_nonlocal) {
  SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
  pybind11::dict py_output = pybind11::dict();
  pybind11::dict py_field;

  // Create fetch data list
  std::vector<CommMpiRmaQueryInfo> fetch_data_list;
  fetch_data_list.reserve(len_nonlocal);
  for (int i = 0; i < len_nonlocal; i++) {
    fetch_data_list.emplace_back(CommMpiRmaQueryInfo{py_nonlocal_rank[i].cast<int>(),
                                                     py_nonlocal_id[i].cast<long>()});
  }

  // Create prepare id list
  std::vector<long> prepare_id_list;
  for (auto& py_gid : py_to_prepare) {
    prepare_id_list.emplace_back(py_gid.cast<long>());
  }

  int dimensionality = LibytProcessControl::Get().data_structure_amr_.GetDimensionality();

  // Initialize one CommMpiRma at a time for a field.
  // TODO: Will support distributing multiple types of field after dealing with labeling
  // for each type of field.
  for (auto& py_fname : py_fname_list) {
    std::string fname = py_fname.cast<std::string>();

    if (dimensionality == 3) {
      // RMA
      CommMpiRmaAmrDataArray3D rma(fname, "amr_grid");
      std::string rma_result_msg = CallFieldRma<AmrDataArray3D, CommMpiRmaAmrDataArray3D>(
          fname, prepare_id_list, fetch_data_list, rma);
      if (rma_result_msg != "success") {
        PyErr_SetString(PyExc_RuntimeError, rma_result_msg.c_str());
        throw pybind11::error_already_set();
      }

      // Wrap to Python dictionary
      for (const AmrDataArray3D& fetched_data : rma.GetFetchedData()) {
        npy_intp npy_dim[3];
        for (int d = 0; d < 3; d++) {
          npy_dim[d] = fetched_data.data_dim[d];
        }
        PyObject* py_data = numpy_controller::ArrayToNumPyArray(
            3, npy_dim, fetched_data.data_dtype, fetched_data.data_ptr, false, true);

        if (!py_output.contains(pybind11::int_(fetched_data.id))) {
          py_field = pybind11::dict();
          py_output[pybind11::int_(fetched_data.id)] = py_field;
        } else {
          py_field = py_output[pybind11::int_(fetched_data.id)];
        }
        py_field[fname.c_str()] = py_data;
        Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we don't
                             // care it anymore.
      }
    } else if (dimensionality == 2) {
      // RMA
      CommMpiRmaAmrDataArray2D rma(fname, "amr_grid");
      std::string rma_result_msg = CallFieldRma<AmrDataArray2D, CommMpiRmaAmrDataArray2D>(
          fname, prepare_id_list, fetch_data_list, rma);
      if (rma_result_msg != "success") {
        PyErr_SetString(PyExc_RuntimeError, rma_result_msg.c_str());
        throw pybind11::error_already_set();
      }

      for (const AmrDataArray2D& fetched_data : rma.GetFetchedData()) {
        npy_intp npy_dim[2] = {fetched_data.data_dim[0], fetched_data.data_dim[1]};
        PyObject* py_data = numpy_controller::ArrayToNumPyArray(
            2, npy_dim, fetched_data.data_dtype, fetched_data.data_ptr, false, true);

        if (!py_output.contains(pybind11::int_(fetched_data.id))) {
          py_field = pybind11::dict();
          py_output[pybind11::int_(fetched_data.id)] = py_field;
        } else {
          py_field = py_output[pybind11::int_(fetched_data.id)];
        }
        py_field[fname.c_str()] = py_data;
        Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we don't
                             // care it anymore.
      }
    } else {
      // RMA
      CommMpiRmaAmrDataArray1D rma(fname, "amr_grid");
      std::string rma_result_msg = CallFieldRma<AmrDataArray1D, CommMpiRmaAmrDataArray1D>(
          fname, prepare_id_list, fetch_data_list, rma);
      if (rma_result_msg != "success") {
        PyErr_SetString(PyExc_RuntimeError, rma_result_msg.c_str());
        throw pybind11::error_already_set();
      }

      for (const AmrDataArray1D& fetched_data : rma.GetFetchedData()) {
        npy_intp npy_dim[1] = {fetched_data.data_dim[0]};
        PyObject* py_data = numpy_controller::ArrayToNumPyArray(
            1, npy_dim, fetched_data.data_dtype, fetched_data.data_ptr, false, true);

        if (!py_output.contains(pybind11::int_(fetched_data.id))) {
          py_field = pybind11::dict();
          py_output[pybind11::int_(fetched_data.id)] = py_field;
        } else {
          py_field = py_output[pybind11::int_(fetched_data.id)];
        }
        py_field[fname.c_str()] = py_data;
        Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we don't
                             // care it anymore.
      }
    }
  }

  return py_output;
#else   // #ifndef SERIAL_MODE
  return pybind11::none();
#endif  // #ifndef SERIAL_MODE
}

//-------------------------------------------------------------------------------------------------------
// Function    :  GetParticleRemote
// Description :  Get non-local particle data from remote ranks.
//
// Note        :  1. We return in dictionary objects.
//                2. We assume that the list of to-get attribute has the same ptype and
//                attr order in each
//                   rank.
//                3. We first filter out data len <= 0 in prepared data and fetched data.
//                So that we don't
//                   pass nullptr around in rma.
//                4. If there are no particles in one grid, then we write Py_None to it.
//                5. Directly return None if it is in SERIAL_MODE
//                   TODO: Not sure if this would affect the performance. And do I even
//                   need this?
//                         Wondering if pybind11 do dynamic casting back to a dictionary.
//                6. TODO: Some of the passed in arguments are not used, will fix it and
//                define new API in the future.
//                         get_field_remote and get_particle_remote should be merged.
//
// Parameter   :  dict obj : ptf          : {<ptype>: [<attr1>, <attr2>, ...]} particle
// type and attributes
//                                          to read.
//                iterable obj : ptf_keys : list of ptype keys.
//                list obj : to_prepare   : list of grid ids you need to prepare.
//                list obj : nonlocal_id  : nonlocal grid id that you want to get.
//                list obj : nonlocal_rank: where to get those nonlocal grid.
//
// Return      :  dict obj data[grid id][ptype][attribute]
//-------------------------------------------------------------------------------------------------------
pybind11::object GetParticleRemote(const pybind11::dict& py_ptf,
                                   const pybind11::iterable& py_ptf_keys,
                                   const pybind11::list& py_to_prepare,
                                   int len_to_prepare,
                                   const pybind11::list& py_nonlocal_id,
                                   const pybind11::list& py_nonlocal_rank,
                                   int len_nonlocal) {
  SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
  pybind11::dict py_output = pybind11::dict();

  // Initialize one CommMpiRma at a time for a particle attribute.
  // TODO: Will support distributing multiple types of field/particle after dealing with
  // labeling for each of them
  //       And also, get_field_remote/get_particle_remote can be merged once the API to
  //       yt_libyt has changed.
  for (auto& py_ptype : py_ptf_keys) {
    for (auto& py_attr : py_ptf[py_ptype]) {
      // Prepare data for particle count > 0
      std::string ptype = py_ptype.cast<std::string>();
      std::string attr = py_attr.cast<std::string>();

      std::vector<long> prepare_id_list;
      for (auto& py_gid : py_to_prepare) {
        long count;
        LibytProcessControl::Get()
            .data_structure_amr_.GetPythonBoundFullHierarchyGridParticleCount(
                py_gid.cast<long>(), ptype.c_str(), &count);
        if (count > 0) {
          prepare_id_list.emplace_back(py_gid.cast<long>());
        }
      }

      DataHubAmrParticle local_particle_data(false);
      DataHubReturn<AmrDataArray1D> prepared_data =
          local_particle_data.GetLocalParticleData(
              LibytProcessControl::Get().data_structure_amr_,
              ptype,
              attr,
              prepare_id_list);
      DataHubStatus all_status = static_cast<DataHubStatus>(
          CommMpi::CheckAllStates(static_cast<int>(prepared_data.status),
                                  static_cast<int>(DataHubStatus::kDataHubSuccess),
                                  static_cast<int>(DataHubStatus::kDataHubSuccess),
                                  static_cast<int>(DataHubStatus::kDataHubFailed)));

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
        LibytProcessControl::Get()
            .data_structure_amr_.GetPythonBoundFullHierarchyGridParticleCount(
                py_nonlocal_id[i].cast<long>(), ptype.c_str(), &count);
        if (count > 0) {
          fetch_data_list.emplace_back(CommMpiRmaQueryInfo{
              py_nonlocal_rank[i].cast<int>(), py_nonlocal_id[i].cast<long>()});
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

        if (fetched_data.data_dim[0] > 0) {
          PyObject* py_data;
          npy_intp npy_dim[1] = {fetched_data.data_dim[0]};
          py_data = numpy_controller::ArrayToNumPyArray(
              1, npy_dim, fetched_data.data_dtype, fetched_data.data_ptr, false, true);

          py_output[pybind11::int_(gid)][ptype.c_str()][attr.c_str()] = py_data;
          Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we
                               // don't care it anymore.
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

  m.def("derived_func", &DerivedFunc, pybind11::return_value_policy::take_ownership);
  m.def("get_particle", &GetParticle, pybind11::return_value_policy::take_ownership);
  m.def(
      "get_field_remote", &GetFieldRemote, pybind11::return_value_policy::take_ownership);
  m.def("get_particle_remote",
        &GetParticleRemote,
        pybind11::return_value_policy::take_ownership);
}

#else  // #ifdef USE_PYBIND11

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_field_derived_func
// Description :  Use the derived function inside yt_field struct to generate the field,
// then pass back
//                to Python.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. This function only needs to deal with the local grids.
//                3. The returned numpy array data type is according to field's
//                field_dtype defined at
//                   yt_field.
//                4. grid_dimensions[3] is in [x][y][z] coordinate.
//                5. Now, input from Python only contains gid and field name. In the
//                future, when we
//                   support hybrid OpenMP/MPI, it can accept list and a string.
//
// Parameter   :  int : GID of the grid
//                str : field name
//
// Return      :  numpy.3darray
//-------------------------------------------------------------------------------------------------------
static PyObject* LibytFieldDerivedFunc(PyObject* self, PyObject* args) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // Parse the input arguments input by python.
  // If not in the format libyt.derived_func( int , str ), raise an error
  long gid;
  char* field_name;
  if (!PyArg_ParseTuple(args, "ls", &gid, &field_name)) {
    PyErr_SetString(PyExc_TypeError,
                    "Wrong input type, expect to be libyt.derived_func(int, str).");
    return NULL;
  }

  // Generate data
  std::vector<long> gid_list = {gid};
  std::vector<AmrDataArray3D> storage;
  DataStructureOutput status =
      LibytProcessControl::Get().data_structure_amr_.GenerateLocalFieldData(
          gid_list, field_name, storage);
  if (status.status != DataStructureStatus::kDataStructureSuccess) {
    if (status.status == DataStructureStatus::kDataStructureNotImplemented) {
      PyErr_Format(PyExc_NotImplementedError, status.error.c_str());
      return NULL;
    } else {
      PyErr_Format(PyExc_ValueError, status.error.c_str());
      return NULL;
    }
  }

  // Wrapping the data to numpy array, for now, storage only contains one data
  int nd = 3;
  npy_intp dims[3];
  if (storage[0].contiguous_in_x) {
    dims[0] = storage[0].data_dim[2];
    dims[1] = storage[0].data_dim[1];
    dims[2] = storage[0].data_dim[0];
  } else {
    dims[0] = storage[0].data_dim[0];
    dims[1] = storage[0].data_dim[1];
    dims[2] = storage[0].data_dim[2];
  }

  PyObject* py_data = numpy_controller::ArrayToNumPyArray(
      nd, dims, storage[0].data_dtype, storage[0].data_ptr, false, true);

  return py_data;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_particle_get_particle
// Description :  Use the get_par_attr defined inside yt_particle struct to get the
// particle attributes.
//
// Note        :  1. Support only grid dimension = 3 for now, which is "coor_x", "coor_y",
// "coor_z" in
//                   yt_particle must be set.
//                2. Deal with local particles only.
//                3. The returned numpy array data type well be set by attr_dtype.
//                4. We will always return 1D numpy array, with length equal particle
//                count of the species
//                   in that grid.
//                5. Return Py_None if number of ptype particle == 0.
//
// Parameter   :  int : GID of the grid
//                str : ptype, particle species, ex:"io"
//                str : attribute, or in terms in yt, which is particle.
//
// Return      :  numpy.1darray
//-------------------------------------------------------------------------------------------------------
static PyObject* LibytParticleGetParticle(PyObject* self, PyObject* args) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // Parse the input arguments input by python.
  // If not in the format libyt.get_particle( int , str , str ), raise an error
  long gid;
  char* ptype;
  char* attr_name;

  if (!PyArg_ParseTuple(args, "lss", &gid, &ptype, &attr_name)) {
    PyErr_SetString(PyExc_TypeError,
                    "Wrong input type, expect to be libyt.get_particle(int, str, str).");
    return NULL;
  }

  // Generate data
  std::vector<long> gid_list = {gid};
  std::vector<AmrDataArray1D> storage;
  DataStructureOutput status =
      LibytProcessControl::Get().data_structure_amr_.GenerateLocalParticleData(
          gid_list, ptype, attr_name, storage);
  if (status.status != DataStructureStatus::kDataStructureSuccess) {
    if (status.status == DataStructureStatus::kDataStructureNotImplemented) {
      PyErr_Format(PyExc_NotImplementedError, status.error.c_str());
      return NULL;
    } else {
      PyErr_Format(PyExc_ValueError, status.error.c_str());
      return NULL;
    }
  }

  // Wrapping the data to numpy array, for now, storage only contains one data
  if (storage[0].data_dim[0] == 0) {
    Py_INCREF(Py_None);
    return Py_None;
  } else {
    int nd = 1;
    npy_intp dims[1] = {storage[0].data_dim[0]};
    PyObject* py_data = numpy_controller::ArrayToNumPyArray(
        nd, dims, storage[0].data_dtype, storage[0].data_ptr, false, true);

    return py_data;
  }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_field_get_field_remote
// Description :  Get non-local field data from remote ranks.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. We return in dictionary objects.
//                3. We assume that the fname_list passed in has the same fname order in
//                each rank.
//                4. This function will get all the fields and grids in combination.
//                   So the total returned data get is len(fname_list) * len(nonlocal_id).
//                5. Directly return None if it is in SERIAL_MODE.
//                6. In Python, it is called like:
//                   libyt.get_field_remote( fname_list,
//                                           len(fname_list),
//                                           to_prepare,
//                                           len(to_prepare),
//                                           nonlocal_id,
//                                           nonlocal_rank,
//                                           len(nonlocal_id))
//
// Parameter   :  list obj : fname_list   : list of field name to get.
//                list obj : to_prepare   : list of grid ids you need to prepare.
//                list obj : nonlocal_id  : nonlocal grid id that you want to get.
//                list obj : nonlocal_rank: where to get those nonlocal grid.
//
// Return      :  dict obj data[grid id][field_name][:,:,:]
//-------------------------------------------------------------------------------------------------------
static PyObject* LibytFieldGetFieldRemote(PyObject* self, PyObject* args) {
  SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
  // Parse the input list arguments by python
  PyObject* arg1;  // fname_list, we will make it an iterable object.
  PyObject* py_prepare_grid_id_list;
  PyObject* py_get_grid_id_list;
  PyObject* py_get_grid_rank_list;

  int len_fname_list;  // Max number of field is INT_MAX
  int len_prepare;     // Since maximum number of local grid is INT_MAX
  long len_get_grid;   // Max of total grid number is LNG_MAX

  if (!PyArg_ParseTuple(args,
                        "OiOiOOl",
                        &arg1,
                        &len_fname_list,
                        &py_prepare_grid_id_list,
                        &len_prepare,
                        &py_get_grid_id_list,
                        &py_get_grid_rank_list,
                        &len_get_grid)) {
    PyErr_SetString(
        PyExc_TypeError,
        "Wrong input type, "
        "expect to be libyt.get_field_remote(list, int, list, int, list, list, long).\n");
    return NULL;
  }

  // Make these input lists iterators.
  PyObject* py_fname_list = PyObject_GetIter(arg1);
  if (py_fname_list == NULL) {
    PyErr_SetString(PyExc_TypeError, "fname_list is not an iterable object!\n");
    return NULL;
  }

  std::vector<PyObject*>
      py_deref_list;  // Dereference these PyObjects for early return when error occurs.
  py_deref_list.push_back(py_fname_list);

  // Create prepare data id list
  std::vector<long> prepare_id_list;
  prepare_id_list.reserve(len_prepare);
  for (int i = 0; i < len_prepare; i++) {
    PyObject* py_prepare_grid_id = PyList_GetItem(py_prepare_grid_id_list, i);
    prepare_id_list.push_back(PyLong_AsLong(py_prepare_grid_id));
  }

  // Create fetch data list
  std::vector<CommMpiRmaQueryInfo> fetch_data_list;
  fetch_data_list.reserve(len_get_grid);
  for (int i = 0; i < len_get_grid; i++) {
    PyObject* py_get_grid_id = PyList_GetItem(py_get_grid_id_list, i);
    PyObject* py_get_grid_rank = PyList_GetItem(py_get_grid_rank_list, i);
    fetch_data_list.push_back(
        CommMpiRmaQueryInfo{static_cast<int>(PyLong_AsLong(py_get_grid_rank)),
                            PyLong_AsLong(py_get_grid_id)});
  }

  // Create Python dictionary for storing remote data.
  PyObject* py_output = PyDict_New();

  // Get all remote grid id in field name fname, get one field at a time.
  PyObject* py_fname;
  while ((py_fname = PyIter_Next(py_fname_list))) {
    // Prepare local data
    py_deref_list.push_back(py_fname);
    char* fname = PyBytes_AsString(py_fname);
    DataHubAmrField<AmrDataArray3D> local_amr_data(false);
    DataHubReturn<AmrDataArray3D> prepared_data = local_amr_data.GetLocalFieldData(
        LibytProcessControl::Get().data_structure_amr_, fname, prepare_id_list);
    DataHubStatus all_status = static_cast<DataHubStatus>(
        CommMpi::CheckAllStates(static_cast<int>(prepared_data.status),
                                static_cast<int>(DataHubStatus::kDataHubSuccess),
                                static_cast<int>(DataHubStatus::kDataHubSuccess),
                                static_cast<int>(DataHubStatus::kDataHubFailed)));
    if (all_status != DataHubStatus::kDataHubSuccess) {
      if (prepared_data.status == DataHubStatus::kDataHubFailed) {
        PyErr_SetString(PyExc_RuntimeError, local_amr_data.GetErrorStr().c_str());
      } else {
        PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
      }
      for (auto& py_item : py_deref_list) {
        Py_DECREF(py_item);
      }
      return NULL;
    }

    // Call Mpi RMA operation
    CommMpiRmaAmrDataArray3D comm_mpi_rma(fname, "amr_grid");
    CommMpiRmaReturn<AmrDataArray3D> rma_return =
        comm_mpi_rma.GetRemoteData(prepared_data.data_list, fetch_data_list);
    if (rma_return.all_status != CommMpiRmaStatus::kMpiSuccess) {
      if (rma_return.status != CommMpiRmaStatus::kMpiSuccess) {
        PyErr_SetString(PyExc_RuntimeError, comm_mpi_rma.GetErrorStr().c_str());
      } else {
        PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
      }
      for (auto& py_item : py_deref_list) {
        Py_DECREF(py_item);
      }
      return NULL;
    }

    // Wrap to Python dictionary
    for (const AmrDataArray3D& fetched_data : rma_return.data_list) {
      // Create dictionary output[grid id][field_name]
      PyObject* py_grid_id = PyLong_FromLong(fetched_data.id);
      PyObject* py_field_label;
      if (PyDict_Contains(py_output, py_grid_id) == 0) {
        py_field_label = PyDict_New();
        PyDict_SetItem(py_output, py_grid_id, py_field_label);
        Py_DECREF(py_field_label);
      }
      py_field_label = PyDict_GetItem(py_output, py_grid_id);
      Py_DECREF(py_grid_id);

      // Wrap the data to NumPy array
      npy_intp npy_dim[3];
      for (int d = 0; d < 3; d++) {
        npy_dim[d] = fetched_data.data_dim[d];
      }
      PyObject* py_field_data = numpy_controller::ArrayToNumPyArray(
          3, npy_dim, fetched_data.data_dtype, fetched_data.data_ptr, false, true);
      PyDict_SetItemString(py_field_label, fname, py_field_data);
      Py_DECREF(py_field_data);
    }
    Py_DECREF(py_fname);
    py_deref_list.pop_back();
  }

  // Dereference Python objects
  Py_DECREF(py_fname_list);
  py_deref_list.pop_back();

  // Return to Python
  return py_output;
#else   // #ifndef SERIAL_MODE
  Py_RETURN_NONE;
#endif  // #ifndef SERIAL_MODE
}

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_particle_get_particle_remote
// Description :  Get non-local particle data from remote ranks.
//
// Note        :  1. We return in dictionary objects.
//                2. We assume that the list of to-get attribute has the same ptype and
//                attr order in each
//                   rank.
//                3. We first filter out data len <= 0 in prepared data and fetched data.
//                So that we don't
//                   pass nullptr around in rma. (TODO: there must be a better way, ex: a
//                   better Api)
//                4. If there are no particles in one grid, then we write Py_None to it.
//                5. Directly return None if it is in SERIAL_MODE
//
// Parameter   :  dict obj : ptf          : {<ptype>: [<attr1>, <attr2>, ...]} particle
// type and attributes
//                                          to read.
//                iterable obj : ptf_keys : list of ptype keys.
//                list obj : to_prepare   : list of grid ids you need to prepare.
//                list obj : nonlocal_id  : nonlocal grid id that you want to get.
//                list obj : nonlocal_rank: where to get those nonlocal grid.
//
// Return      :  dict obj data[grid id][ptype][attribute]
//-------------------------------------------------------------------------------------------------------
static PyObject* LibytParticleGetParticleRemote(PyObject* self, PyObject* args) {
  SET_TIMER(__PRETTY_FUNCTION__);

#ifndef SERIAL_MODE
  // Parse the input list arguments by Python
  PyObject* py_ptf_dict;
  PyObject* arg2;
  PyObject* py_prepare_list;
  PyObject* py_to_get_list;
  PyObject* py_get_rank_list;

  int len_prepare;
  long len_to_get;

  if (!PyArg_ParseTuple(args,
                        "OOOiOOl",
                        &py_ptf_dict,
                        &arg2,
                        &py_prepare_list,
                        &len_prepare,
                        &py_to_get_list,
                        &py_get_rank_list,
                        &len_to_get)) {
    PyErr_SetString(PyExc_TypeError,
                    "Wrong input type, "
                    "expect to be libyt.get_particle_remote(dict, iter, list, int, list, "
                    "list, long).\n");
    return NULL;
  }

  PyObject* py_ptf_keys = PyObject_GetIter(arg2);
  if (py_ptf_keys == NULL) {
    PyErr_SetString(PyExc_TypeError, "py_ptf_keys is not an iterable object!\n");
    return NULL;
  }

  PyObject* py_output = PyDict_New();  // Variables for creating output.
  std::vector<PyObject*>
      py_deref_list;  // Dereference these PyObjects for early return when error occurs.
  py_deref_list.push_back(py_ptf_keys);

  // Run through all the py_ptf_dict and its value.
  PyObject* py_ptype;
  while ((py_ptype = PyIter_Next(py_ptf_keys))) {
    py_deref_list.push_back(py_ptype);
    char* ptype = PyBytes_AsString(py_ptype);

    // Get attribute list inside key ptype in py_ptf_dict.
    PyObject* py_value = PyDict_GetItem(py_ptf_dict, py_ptype);
    PyObject* py_attr_iter = PyObject_GetIter(py_value);
    py_deref_list.push_back(py_attr_iter);

    // Iterate through attribute list, and perform RMA operation.
    PyObject* py_attribute;
    while ((py_attribute = PyIter_Next(py_attr_iter))) {
      py_deref_list.push_back(py_attribute);
      char* attr = PyBytes_AsString(py_attribute);

      // Prepare data for particle count > 0
      std::vector<long> prepare_id_list;
      prepare_id_list.reserve(len_prepare);
      for (int i = 0; i < len_prepare; i++) {
        PyObject* py_prepare_id = PyList_GetItem(py_prepare_list, i);
        long gid = PyLong_AsLong(py_prepare_id);
        long count;
        LibytProcessControl::Get()
            .data_structure_amr_.GetPythonBoundFullHierarchyGridParticleCount(
                gid, ptype, &count);
        if (count > 0) {
          prepare_id_list.push_back(gid);
        }
      }
      DataHubAmrParticle local_particle_data(false);
      DataHubReturn<AmrDataArray1D> prepared_data =
          local_particle_data.GetLocalParticleData(
              LibytProcessControl::Get().data_structure_amr_,
              ptype,
              attr,
              prepare_id_list);
      DataHubStatus all_status = static_cast<DataHubStatus>(
          CommMpi::CheckAllStates(static_cast<int>(prepared_data.status),
                                  static_cast<int>(DataHubStatus::kDataHubSuccess),
                                  static_cast<int>(DataHubStatus::kDataHubSuccess),
                                  static_cast<int>(DataHubStatus::kDataHubFailed)));
      if (all_status != DataHubStatus::kDataHubSuccess) {
        if (prepared_data.status == DataHubStatus::kDataHubFailed) {
          PyErr_SetString(PyExc_RuntimeError, local_particle_data.GetErrorStr().c_str());
        } else {
          PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
        }
        for (auto& item : py_deref_list) {
          Py_DECREF(item);
        }
        return NULL;
      }

      // Create fetch data list and separate particle count > 0
      std::vector<CommMpiRmaQueryInfo> fetch_data_list;
      std::vector<long> fetch_particle_count0_list;
      for (int i = 0; i < len_to_get; i++) {
        PyObject* py_get_id = PyList_GetItem(py_to_get_list, i);
        PyObject* py_get_rank = PyList_GetItem(py_get_rank_list, i);
        long get_gid = PyLong_AsLong(py_get_id);
        int get_rank = (int)PyLong_AsLong(py_get_rank);
        long count;
        LibytProcessControl::Get()
            .data_structure_amr_.GetPythonBoundFullHierarchyGridParticleCount(
                get_gid, ptype, &count);
        if (count > 0) {
          fetch_data_list.emplace_back(CommMpiRmaQueryInfo{get_rank, get_gid});
        } else {
          fetch_particle_count0_list.emplace_back(get_gid);
        }
      }

      // Call Mpi RMA operation
      std::string rma_name = std::string(ptype) + "-" + std::string(attr);
      CommMpiRmaAmrDataArray1D comm_mpi_rma(rma_name, "amr_particle");
      CommMpiRmaReturn<AmrDataArray1D> rma_return =
          comm_mpi_rma.GetRemoteData(prepared_data.data_list, fetch_data_list);
      if (rma_return.all_status != CommMpiRmaStatus::kMpiSuccess) {
        if (rma_return.status != CommMpiRmaStatus::kMpiSuccess) {
          PyErr_SetString(PyExc_RuntimeError, comm_mpi_rma.GetErrorStr().c_str());
        } else {
          PyErr_SetString(PyExc_RuntimeError, "Error occurred in other MPI process.");
        }
        for (auto& item : py_deref_list) {
          Py_DECREF(item);
        }
        return NULL;
      }

      // Wrap data to a Python dictionary
      for (const AmrDataArray1D& fetched_data : rma_return.data_list) {
        // Create dictionary data[grid id][ptype][attribute]
        long gid = fetched_data.id;
        PyObject* py_grid_id = PyLong_FromLong(gid);
        PyObject* py_ptype_dict;
        if (PyDict_Contains(py_output, py_grid_id) == 0) {
          py_ptype_dict = PyDict_New();
          PyDict_SetItem(py_output, py_grid_id, py_ptype_dict);
          Py_DECREF(py_ptype_dict);
        }
        py_ptype_dict = PyDict_GetItem(py_output, py_grid_id);
        Py_DECREF(py_grid_id);

        PyObject* py_ptype_key = PyUnicode_FromString(ptype);
        PyObject* py_attribute_dict;
        if (PyDict_Contains(py_ptype_dict, py_ptype_key) == 0) {
          py_attribute_dict = PyDict_New();
          PyDict_SetItem(py_ptype_dict, py_ptype_key, py_attribute_dict);
          Py_DECREF(py_attribute_dict);
        }
        py_attribute_dict = PyDict_GetItem(py_ptype_dict, py_ptype_key);
        Py_DECREF(py_ptype_key);

        // Wrap and bind to py_attribute_dict
        if (fetched_data.data_dim[0] > 0) {
          npy_intp npy_dim[1] = {fetched_data.data_dim[0]};
          PyObject* py_data = numpy_controller::ArrayToNumPyArray(
              1, npy_dim, fetched_data.data_dtype, fetched_data.data_ptr, false, true);
          PyDict_SetItemString(py_attribute_dict, attr, py_data);
          Py_DECREF(py_data);  // Need to deref it, since it's owned by Python, and we
                               // don't care it anymore.
        } else {
          PyDict_SetItemString(py_attribute_dict, attr, Py_None);
        }
      }

      // Wrap particle count = 0 to a Python dictionary
      for (const long& gid : fetch_particle_count0_list) {
        // Create dictionary data[grid id][ptype][attribute]
        PyObject* py_grid_id = PyLong_FromLong(gid);
        PyObject* py_ptype_dict;
        if (PyDict_Contains(py_output, py_grid_id) == 0) {
          py_ptype_dict = PyDict_New();
          PyDict_SetItem(py_output, py_grid_id, py_ptype_dict);
          Py_DECREF(py_ptype_dict);
        }
        py_ptype_dict = PyDict_GetItem(py_output, py_grid_id);
        Py_DECREF(py_grid_id);

        PyObject* py_ptype_key = PyUnicode_FromString(ptype);
        PyObject* py_attribute_dict;
        if (PyDict_Contains(py_ptype_dict, py_ptype_key) == 0) {
          py_attribute_dict = PyDict_New();
          PyDict_SetItem(py_ptype_dict, py_ptype_key, py_attribute_dict);
          Py_DECREF(py_attribute_dict);
        }
        py_attribute_dict = PyDict_GetItem(py_ptype_dict, py_ptype_key);
        Py_DECREF(py_ptype_key);

        // set data[grid id][ptype][attribute] = None
        PyDict_SetItemString(py_attribute_dict, attr, Py_None);
      }

      // Free unused resource
      Py_DECREF(py_attribute);
      py_deref_list.pop_back();
    }

    // Free unused resource.
    Py_DECREF(py_attr_iter);
    Py_DECREF(py_ptype);
    py_deref_list.pop_back();
    py_deref_list.pop_back();
  }

  // Free unneeded resource.
  Py_DECREF(py_ptf_keys);
  py_deref_list.pop_back();

  // Return.
  return py_output;
#else   // #ifndef SERIAL_MODE
  Py_RETURN_NONE;
#endif  // #ifndef SERIAL_MODE
}

//-------------------------------------------------------------------------------------------------------
// Description :  Preparation for creating libyt python module
//
// Note        :  1. Contains data blocks for creating libyt python module.
//                2. Only initialize libyt python module, not import to system yet.
//
// Lists:      :  libyt_method_list       : Declare libyt C extension python methods.
//                libyt_module_definition : Definition to libyt python module.
//                PyInit_libyt            : Create libyt python module, and append python
//                objects,
//                                          ex: dictionary.
//-------------------------------------------------------------------------------------------------------

// Define functions in module, list all libyt module methods here
static PyMethodDef libyt_method_list[] = {
    // { "method_name", c_function_name, METH_VARARGS, "Description"},
    {"derived_func",
     LibytFieldDerivedFunc,
     METH_VARARGS,
     "Get local derived field data."},
    {"get_particle",
     LibytParticleGetParticle,
     METH_VARARGS,
     "Get local particle attribute data."},
    {"get_field_remote",
     LibytFieldGetFieldRemote,
     METH_VARARGS,
     "Get remote field data."},
    {"get_particle_remote",
     LibytParticleGetParticleRemote,
     METH_VARARGS,
     "Get remote particle attribute data."},
    {NULL, NULL, 0, NULL}  // sentinel
};

// Declare the definition of libyt_module
static struct PyModuleDef libyt_module_definition = {
    PyModuleDef_HEAD_INIT, "libyt", "libyt documentation", -1, libyt_method_list};

// Create libyt python module
static PyObject* PyInitLibyt(void) {
  SET_TIMER(__PRETTY_FUNCTION__);

  // Create libyt module
  PyObject* libyt_module = PyModule_Create(&libyt_module_definition);
  if (libyt_module != nullptr) {
    logging::LogDebug("Creating libyt module ... done\n");
  } else {
    YT_ABORT("Creating libyt module ... failed!\n");
  }

  // Add objects dictionary
  PyObject* py_grid_data = PyDict_New();
  PyObject* py_particle_data = PyDict_New();
  PyObject* py_hierarchy = PyDict_New();
  LibytProcessControl::Get().data_structure_amr_.SetPythonBindings(
      py_hierarchy, py_grid_data, py_particle_data);
  LibytProcessControl::Get().py_param_yt_ = PyDict_New();
  LibytProcessControl::Get().py_param_user_ = PyDict_New();
  LibytProcessControl::Get().py_libyt_info_ = PyDict_New();
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  LibytProcessControl::Get().py_interactive_mode_ = PyDict_New();
#endif

  // set libyt info
  PyObject* py_version = Py_BuildValue(
      "(iii)", LIBYT_MAJOR_VERSION, LIBYT_MINOR_VERSION, LIBYT_MICRO_VERSION);
  PyDict_SetItemString(LibytProcessControl::Get().py_libyt_info_, "version", py_version);
  Py_DECREF(py_version);
#ifdef SERIAL_MODE
  PyDict_SetItemString(LibytProcessControl::Get().py_libyt_info_, "SERIAL_MODE", Py_True);
#else
  PyDict_SetItemString(
      LibytProcessControl::Get().py_libyt_info_, "SERIAL_MODE", Py_False);
#endif
#ifdef INTERACTIVE_MODE
  PyDict_SetItemString(
      LibytProcessControl::Get().py_libyt_info_, "INTERACTIVE_MODE", Py_True);
#else
  PyDict_SetItemString(
      LibytProcessControl::Get().py_libyt_info_, "INTERACTIVE_MODE", Py_False);
#endif
#ifdef JUPYTER_KERNEL
  PyDict_SetItemString(
      LibytProcessControl::Get().py_libyt_info_, "JUPYTER_KERNEL", Py_True);
#else
  PyDict_SetItemString(
      LibytProcessControl::Get().py_libyt_info_, "JUPYTER_KERNEL", Py_False);
#endif
#ifdef SUPPORT_TIMER
  PyDict_SetItemString(
      LibytProcessControl::Get().py_libyt_info_, "SUPPORT_TIMER", Py_True);
#else
  PyDict_SetItemString(
      LibytProcessControl::Get().py_libyt_info_, "SUPPORT_TIMER", Py_False);
#endif

  // add dict object to libyt python module
  PyModule_AddObject(libyt_module, "grid_data", py_grid_data);
  PyModule_AddObject(libyt_module, "particle_data", py_particle_data);
  PyModule_AddObject(libyt_module, "hierarchy", py_hierarchy);
  PyModule_AddObject(libyt_module, "param_yt", LibytProcessControl::Get().py_param_yt_);
  PyModule_AddObject(
      libyt_module, "param_user", LibytProcessControl::Get().py_param_user_);
  PyModule_AddObject(
      libyt_module, "libyt_info", LibytProcessControl::Get().py_libyt_info_);
#if defined(INTERACTIVE_MODE) || defined(JUPYTER_KERNEL)
  PyModule_AddObject(
      libyt_module, "interactive_mode", LibytProcessControl::Get().py_interactive_mode_);
#endif

  logging::LogDebug("Attaching empty dictionaries to libyt module ... done\n");

  return libyt_module;
}

#endif  // #ifdef USE_PYBIND11

//-------------------------------------------------------------------------------------------------------
// Namespace   :  python_controller
// Function    :  CreateLibytModule
// Description :  Create the libyt module
//
// Note        :  1. Create libyt module, should be called before Py_Initialize().
//                2. Only has effect when in pure Python C API (-DUSE_PYBIND11=OFF)
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int python_controller::CreateLibytModule() {
  SET_TIMER(__PRETTY_FUNCTION__);

#ifndef USE_PYBIND11
  PyImport_AppendInittab("libyt", &PyInitLibyt);
#endif

  return YT_SUCCESS;
}
