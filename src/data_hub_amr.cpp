#include "data_hub_amr.h"

#include "dtype_utilities.h"

//-------------------------------------------------------------------------------------------------------
// Class         :  DataHub
// Public Method :  ClearCache
//
// Notes       :  1. Clear the cache, and decide if new allocation needs to be freed.
//                2. Assuming DataClass struct has data pointer called data_ptr.
//-------------------------------------------------------------------------------------------------------
template<typename DataClass>
void DataHub<DataClass>::ClearCache() {
  if (!take_ownership_) {
    for (size_t i = 0; i < data_array_list_.size(); i++) {
      if (is_new_allocation_list_[i]) {
        free(data_array_list_[i].data_ptr);
      }
    }
  }

  is_new_allocation_list_.clear();
  data_array_list_.clear();
  error_str_ = std::string("");
}

template class DataHub<AmrDataArray3D>;
template class DataHub<AmrDataArray1D>;

//-------------------------------------------------------------------------------------------------------
// Class         :  DataHubAmrDataArray3D
// Public Method :  GetLocalFieldData
//
// Notes       :  1. Get local field data and cache it.
//                2. The method is specific for AMR data defined in libyt.
//                3. The method is designed to be called many times by user and fails
//                fast.
//                   Every time it is called, it will clear the cache.
//                4. Can retrieve data type derived_func/cell-centered/face-centered.
//                   And assume that the data type is the same.
//                5. If the data retrival requires new allocation of data buffer (ex:
//                derived function),
//                   then it will be marked in is_new_allocation_list_. It will later be
//                   freed by ClearCache.
//                6. TODO: not test it yet, only need this when doing memory leaking test.
//-------------------------------------------------------------------------------------------------------
DataHubReturn<AmrDataArray3D> DataHubAmrDataArray3D::GetLocalFieldData(
    const DataStructureAmr& ds_amr, const std::string& field_name,
    const std::vector<long>& grid_id_list) {
  // Free cache before doing new query
  ClearCache();

  yt_field* field_list = ds_amr.GetFieldList();
  int field_id = ds_amr.GetFieldIndex(field_name.c_str());
  if (field_id == -1) {
    error_str_ = std::string("Cannot find field_name [ ") + field_name +
                 std::string(" ] in field_list on MPI rank ") +
                 std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
    return {DataHubStatus::kDataHubFailed, data_array_list_};
  }

  if (strcmp(field_list[field_id].field_type, "derived_func") == 0) {
    DataStructureOutput status =
        ds_amr.GenerateLocalFieldData(grid_id_list, field_name.c_str(), data_array_list_);
    is_new_allocation_list_.assign(data_array_list_.size(), true);
    if (status.status != DataStructureStatus::kDataStructureSuccess) {
      error_str_ = std::move(status.error);
      return {DataHubStatus::kDataHubFailed, data_array_list_};
    }
  } else if (strcmp(field_list[field_id].field_type, "cell-centered") == 0 ||
             strcmp(field_list[field_id].field_type, "face-centered") == 0) {
    for (const long& gid : grid_id_list) {
      yt_data field_data;
      DataStructureOutput status =
          ds_amr.GetPythonBoundLocalFieldData(gid, field_name.c_str(), &field_data);
      if (status.status != DataStructureStatus::kDataStructureSuccess) {
        error_str_ = status.error;
        error_str_ += std::string("Failed to get data (field_name, gid) = (") +
                      field_name + std::string(", ") + std::to_string(gid) +
                      std::string(") on MPI rank ") +
                      std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
        return {DataHubStatus::kDataHubFailed, data_array_list_};
      }
      AmrDataArray3D amr_data{};
      amr_data.id = gid;
      amr_data.contiguous_in_x = field_list[field_id].contiguous_in_x;
      amr_data.data_dtype = field_data.data_dtype;
      for (int d = 0; d < 3; d++) {
        amr_data.data_dim[d] = field_data.data_dimensions[d];
      }
      amr_data.data_ptr = field_data.data_ptr;

      is_new_allocation_list_.emplace_back(false);
      data_array_list_.emplace_back(amr_data);
    }
  } else {
    error_str_ = std::string("Unknown field type [ ") +
                 std::string(field_list[field_id].field_type) +
                 std::string(" ] in field [ ") + field_name +
                 std::string(" ] on MPI rank ") +
                 std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
    return {DataHubStatus::kDataHubFailed, data_array_list_};
  }

  return {DataHubStatus::kDataHubSuccess, data_array_list_};
}

//-------------------------------------------------------------------------------------------------------
// Class         :  DataHubAmrDataArray1D
// Public Method :  GetLocalParticleData
//
// Notes       :  1. Get local particle data and cache it.
//                2. The method is specific for AMR data defined in libyt.
//                3. The method is designed to be called many times by user and fails
//                fast.
//                4. If the data retrival requires new allocation of data buffer (ex:
//                get_par_attr),
//                   then it will be marked in is_new_allocation_list_. It will later be
//                   freed by ClearCache.
//                5. Faithfully return the data even if it has length 0.
//                6. The order of grid_id_list passed in and the data_array_list_ is not
//                necessarily the same.
//                7. It first look for data in libyt.particle_data, if not found, it will
//                call get_par_attr.
//                   (TODO: this also shows it is a bad Api design.)
//                8. TODO: not test it yet, only need this when doing memory leaking test.
//-------------------------------------------------------------------------------------------------------
DataHubReturn<AmrDataArray1D> DataHubAmrDataArray1D::GetLocalParticleData(
    const DataStructureAmr& ds_amr, const std::string& ptype, const std::string& pattr,
    const std::vector<long>& grid_id_list) {
  // Free cache before doing new query
  ClearCache();

  // Get particle type index and attribute index
  int ptype_index = ds_amr.GetParticleIndex(ptype.c_str());
  int pattr_index = ds_amr.GetParticleAttributeIndex(ptype_index, pattr.c_str());
  if (ptype_index == -1 || pattr_index == -1) {
    error_str_ = std::string("(particle type, attribute) = (") + ptype +
                 std::string(", ") + pattr + std::string(") not found.\n");
    return {DataHubStatus::kDataHubFailed, data_array_list_};
  }

  // Loop through grid_id_list and try to get data in libyt.particle_data
  // If failed, which means the length may be 0 or it needs to call get_par_attr (TODO:
  // should I separate 0 length?)
  std::vector<long> generate_gid_list;
  for (const long& kGid : grid_id_list) {
    // Try to retrieve the particle data in libyt.particle_data
    yt_data par_array;
    DataStructureOutput status = ds_amr.GetPythonBoundLocalParticleData(
        kGid, ptype.c_str(), pattr.c_str(), &par_array);
    if (status.status == DataStructureStatus::kDataStructureSuccess) {
      // Read from libyt.particle_data
      AmrDataArray1D amr_1d_data{};
      amr_1d_data.id = kGid;
      amr_1d_data.data_dtype = par_array.data_dtype;
      amr_1d_data.data_ptr = par_array.data_ptr;
      amr_1d_data.data_len = par_array.data_dimensions[0];
      is_new_allocation_list_.emplace_back(false);
      data_array_list_.emplace_back(amr_1d_data);
    } else {
      generate_gid_list.push_back(kGid);
    }
  }

  // Get data from get particle attribute function
  DataStructureOutput status = ds_amr.GenerateLocalParticleData(
      generate_gid_list, ptype.c_str(), pattr.c_str(), data_array_list_);
  is_new_allocation_list_.assign(data_array_list_.size(), true);
  if (status.status != DataStructureStatus::kDataStructureSuccess) {
    error_str_ = std::move(status.error);
    return {DataHubStatus::kDataHubFailed, data_array_list_};
  }

  return {DataHubStatus::kDataHubSuccess, data_array_list_};
}
