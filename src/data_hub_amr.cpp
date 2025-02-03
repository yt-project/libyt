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
//                3. The method is designed to be called many times by user and fails fast.
//                   Every time it is called, it will clear the cache.
//                4. Can retrieve data type derived_func/cell-centered/face-centered.
//                   And assume that the data type is the same.
//                5. If the data retrival requires new allocation of data buffer (ex: derived function),
//                   then it will be marked in is_new_allocation_list_. It will later be freed by ClearCache.
//                   (TODO: do I need is_new_allocation_list_ to be a list?)
//                6. TODO: The function is not tested yet
//-------------------------------------------------------------------------------------------------------
DataHubReturn<AmrDataArray3D> DataHubAmrDataArray3D::GetLocalFieldData(const DataStructureAmr& ds_amr,
                                                                       const std::string& field_name,
                                                                       const std::vector<long>& grid_id_list) {
    // Free cache before doing new query
    ClearCache();

    yt_field* field_list = ds_amr.GetFieldList();
    int field_id = ds_amr.GetFieldIndex(field_name.c_str());
    if (field_id == -1) {
        error_str_ = std::string("Cannot find field_name [ ") + field_name +
                     std::string(" ] in field_list on MPI rank ") + std::to_string(DataStructureAmr::mpi_rank_) +
                     std::string(".\n");
        return {DataHubStatus::kDataHubFailed, data_array_list_};
    }

    if (strcmp(field_list[field_id].field_type, "derived_func") == 0) {
        DataStructureOutput status = ds_amr.GenerateFieldData(grid_id_list, field_name.c_str(), data_array_list_);
        is_new_allocation_list_.assign(data_array_list_.size(), true);
        if (status.status != DataStructureStatus::kDataStructureSuccess) {
            error_str_ = status.error;
            return {DataHubStatus::kDataHubFailed, data_array_list_};
        }
    } else if (strcmp(field_list[field_id].field_type, "cell-centered") == 0 ||
               strcmp(field_list[field_id].field_type, "face-centered") == 0) {
        for (const long& gid : grid_id_list) {
            yt_data field_data;
            DataStructureOutput status = ds_amr.GetPythonBoundLocalFieldData(gid, field_name.c_str(), &field_data);
            if (status.status != DataStructureStatus::kDataStructureSuccess) {
                error_str_ = status.error;
                error_str_ += std::string("Failed to get data (field_name, gid) = (") + field_name + std::string(", ") +
                              std::to_string(gid) + std::string(") on MPI rank ") +
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
        error_str_ = std::string("Unknown field type [ ") + std::string(field_list[field_id].field_type) +
                     std::string(" ] in field [ ") + field_name + std::string(" ] on MPI rank ") +
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
//                3. The method is designed to be called many times by user and fails fast.
//                4. If the data retrival requires new allocation of data buffer (ex: get_par_attr),
//                   then it will be marked in is_new_allocation_list_. It will later be freed by ClearCache.
//                5. It first look for data in libyt.particle_data, if not found, it will call get_par_attr.
//                6. TODO: The function is not tested yet
//-------------------------------------------------------------------------------------------------------
DataHubReturn<AmrDataArray1D> DataHubAmrDataArray1D::GetLocalParticleData(const DataStructureAmr& ds_amr,
                                                                          const std::string& ptype,
                                                                          const std::string& pattr,
                                                                          const std::vector<long>& grid_id_list) {
    // Free cache before doing new query
    ClearCache();

    yt_particle* particle_list = ds_amr.GetParticleList();
    int ptype_index = ds_amr.GetParticleIndex(ptype.c_str());
    int pattr_index = ds_amr.GetParticleAttributeIndex(ptype_index, pattr.c_str());
    if (ptype_index == -1 || pattr_index == -1) {
        error_str_ = std::string("Cannot find (particle type, attribute) = (") + ptype + std::string(", ") + pattr +
                     std::string(") in particle_list on MPI rank ") + std::to_string(DataStructureAmr::mpi_rank_) +
                     std::string(".\n");
        return {DataHubStatus::kDataHubFailed, data_array_list_};
    }

    for (const long& gid : grid_id_list) {
        AmrDataArray1D amr_1d_data{};

        // Get particle info
        amr_1d_data.id = gid;
        amr_1d_data.data_dtype = particle_list[ptype_index].attr_list[pattr_index].attr_dtype;
        DataStructureOutput ds_status =
            ds_amr.GetPythonBoundFullHierarchyGridParticleCount(gid, ptype.c_str(), &amr_1d_data.data_len);
        if (ds_status.status != DataStructureStatus::kDataStructureSuccess) {
            error_str_ = ds_status.error;
            error_str_ += std::string("Failed to get particle count for (particle type, gid) = (") + ptype +
                          std::string(", ") + std::to_string(gid) + std::string(") on MPI rank ") +
                          std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
            return {DataHubStatus::kDataHubFailed, data_array_list_};
        }
        if (amr_1d_data.data_len < 0) {
            error_str_ = std::string("Particle count = ") + std::to_string(amr_1d_data.data_len) +
                         std::string(" < 0 for (particle type, gid) = (") + ptype + std::string(", ") +
                         std::to_string(gid) + std::string(") on MPI rank ") +
                         std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
            return {DataHubStatus::kDataHubFailed, data_array_list_};
        } else if (amr_1d_data.data_len == 0) {
            amr_1d_data.data_ptr = nullptr;
            is_new_allocation_list_.emplace_back(false);
            data_array_list_.emplace_back(amr_1d_data);
            continue;
        }

        // Get particle data, it first tries to read in libyt.particle_data, if not, it generates data in get_par_attr
        yt_data par_array;
        ds_status = ds_amr.GetPythonBoundLocalParticleData(gid, ptype.c_str(), pattr.c_str(), &par_array);
        if (ds_status.status == DataStructureStatus::kDataStructureSuccess) {
            // Read from libyt.particle_data
            amr_1d_data.data_ptr = par_array.data_ptr;
            is_new_allocation_list_.emplace_back(false);
            data_array_list_.emplace_back(amr_1d_data);
        } else {
            // Get particle function get_par_attr
            void (*get_par_attr)(const int, const long*, const char*, const char*, yt_array*) =
                particle_list[ptype_index].get_par_attr;
            if (get_par_attr == nullptr) {
                error_str_ = std::string("Get particle function get_par_attr not set in particle type [ ") + ptype +
                             std::string(" ] on MPI rank ") + std::to_string(DataStructureAmr::mpi_rank_) +
                             std::string(".\n");
                return {DataHubStatus::kDataHubFailed, data_array_list_};
            }

            // Generate buffer
            amr_1d_data.data_ptr = dtype_utilities::AllocateMemory(amr_1d_data.data_dtype, amr_1d_data.data_len);
            if (amr_1d_data.data_ptr == nullptr) {
                error_str_ =
                    std::string("Failed to allocate memory for (particle type, attribute, gid, data_len) = (") + ptype +
                    std::string(", ") + pattr + std::string(", ") + std::to_string(gid) + std::string(", ") +
                    std::to_string(amr_1d_data.data_len) + std::string(") on MPI rank ") +
                    std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
                return {DataHubStatus::kDataHubFailed, data_array_list_};
            }
            int list_len = 1;
            long list_gid[1] = {gid};
            yt_array data_array[1];
            data_array[0].gid = gid;
            data_array[0].data_length = amr_1d_data.data_len;
            data_array[0].data_ptr = amr_1d_data.data_ptr;
            (*get_par_attr)(list_len, list_gid, ptype.c_str(), pattr.c_str(), data_array);
            is_new_allocation_list_.emplace_back(true);
            data_array_list_.emplace_back(amr_1d_data);
        }
    }

    return {DataHubStatus::kDataHubSuccess, data_array_list_};
}
