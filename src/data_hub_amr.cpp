#include "data_hub_amr.h"

#include "yt_prototype.h"

//-------------------------------------------------------------------------------------------------------
// Class         :  DataHubAmr
// Public Method :  GetLocalFieldData
//
// Notes       :  1. Get local field data and cache it.
//                2. The method is specific for AMR data defined in libyt.
//                3. The method is designed to be called many times by user and fails fast.
//                4. Can retrieve data type derived_func/cell-centered/face-centered.
//                5. If the data retrival requires new allocation of data buffer (ex: derived function),
//                   then it will be marked in is_new_allocation_list_. It will later be freed by ClearCache.
//                6. TODO: The function is not tested yet
//-------------------------------------------------------------------------------------------------------
DataHubReturn<AmrDataArray3D> DataHubAmr::GetLocalFieldData(const DataStructureAmr& ds_amr,
                                                            const std::string& field_name,
                                                            const std::vector<long>& grid_id_list) {
    // Free cache before doing new query
    ClearCache();

    yt_field* field_list = ds_amr.field_list_;
    int field_id = ds_amr.GetFieldIndex(field_name.c_str());
    if (field_id == -1) {
        error_str_ = std::string("Cannot find field_name [ ") + field_name +
                     std::string(" ] in field_list on MPI rank ") + std::to_string(DataStructureAmr::mpi_rank_) +
                     std::string(".\n");
        return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
    }

    if (strcmp(field_list[field_id].field_type, "derived_func") == 0) {
        for (const long& gid : grid_id_list) {
            AmrDataArray3D amr_data{};

            // Get amr grid info
            int grid_dim[3];
            DataStructureOutput status = ds_amr.GetPythonBoundFullHierarchyGridDimensions(gid, &grid_dim[0]);
            if (status.status != DataStructureStatus::kDataStructureSuccess) {
                error_str_ = status.error;
                error_str_ += std::string("Failed to get grid dim for (field_name, gid) = (") + field_name +
                              std::string(", ") + std::to_string(gid) + std::string(") on MPI rank ") +
                              std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
                return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
            }
            if (field_list[field_id].contiguous_in_x) {
                amr_data.data_dim[0] = grid_dim[2];
                amr_data.data_dim[1] = grid_dim[1];
                amr_data.data_dim[2] = grid_dim[0];
            } else {
                amr_data.data_dim[0] = grid_dim[0];
                amr_data.data_dim[1] = grid_dim[1];
                amr_data.data_dim[2] = grid_dim[2];
            }
            amr_data.id = gid;
            amr_data.contiguous_in_x = field_list[field_id].contiguous_in_x;
            amr_data.data_dtype = field_list[field_id].field_dtype;

            // Get derived function pointer
            void (*derived_func)(const int, const long*, const char*, yt_array*) = field_list[field_id].derived_func;
            if (derived_func == nullptr) {
                error_str_ = std::string("Derived function derived_func not set in field [ ") + field_name +
                             std::string(" ] on MPI rank ") + std::to_string(DataStructureAmr::mpi_rank_) +
                             std::string(".\n");
                return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
            }

            // Allocate memory for data_ptr and generate data
            long data_len = amr_data.data_dim[0] * amr_data.data_dim[1] * amr_data.data_dim[2];
            if (get_dtype_allocation(amr_data.data_dtype, data_len, &amr_data.data_ptr) != YT_SUCCESS) {
                error_str_ = std::string("Failed to allocate memory for (field_name, gid) = (") + field_name +
                             std::string(", ") + std::to_string(gid) + std::string(") on MPI rank ") +
                             std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
                return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
            }
            // TODO: test using OpenMP in derived_func and pass in a group of ids.
            //       What would happen if we didn't compile libyt with OpenMP? (Time Profile This)
            yt_array data_array[1];
            data_array[0].gid = amr_data.id;
            data_array[0].data_length = data_len;
            data_array[0].data_ptr = amr_data.data_ptr;
            int list_len = 1;
            long list_gid[1] = {amr_data.id};
            (*derived_func)(list_len, list_gid, field_name.c_str(), data_array);

            is_new_allocation_list_.emplace_back(true);
            amr_data_array_3d_list_.emplace_back(amr_data);
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
                return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
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
            amr_data_array_3d_list_.emplace_back(amr_data);
        }
    } else {
        error_str_ = std::string("Unknown field type [ ") + std::string(field_list[field_id].field_type) +
                     std::string(" ] in field [ ") + field_name + std::string(" ] on MPI rank ") +
                     std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
        return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
    }

    return {DataHubStatus::kDataHubSuccess, amr_data_array_3d_list_};
}

//-------------------------------------------------------------------------------------------------------
// Class         :  DataHubAmr
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
DataHubReturn<AmrDataArray1D> DataHubAmr::GetLocalParticleData(const DataStructureAmr& ds_amr, const std::string& ptype,
                                                               const std::string& pattr,
                                                               const std::vector<long>& grid_id_list) {
    // Free cache before doing new query
    ClearCache();

    yt_particle* particle_list = ds_amr.particle_list_;
    int ptype_index = ds_amr.GetParticleIndex(ptype.c_str());
    int pattr_index = ds_amr.GetParticleAttributeIndex(ptype_index, pattr.c_str());
    if (ptype_index == -1 || pattr_index == -1) {
        error_str_ = std::string("Cannot find (particle type, attribute) = (") + ptype + std::string(", ") + pattr +
                     std::string(") in particle_list on MPI rank ") + std::to_string(DataStructureAmr::mpi_rank_) +
                     std::string(".\n");
        return {DataHubStatus::kDataHubFailed, amr_data_array_1d_list_};
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
            return {DataHubStatus::kDataHubFailed, amr_data_array_1d_list_};
        }
        if (amr_1d_data.data_len < 0) {
            error_str_ = std::string("Particle count = ") + std::to_string(amr_1d_data.data_len) +
                         std::string(" < 0 for (particle type, gid) = (") + ptype + std::string(", ") +
                         std::to_string(gid) + std::string(") on MPI rank ") +
                         std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
            return {DataHubStatus::kDataHubFailed, amr_data_array_1d_list_};
        } else if (amr_1d_data.data_len == 0) {
            amr_1d_data.data_ptr = nullptr;
            is_new_allocation_list_.emplace_back(false);
            amr_data_array_1d_list_.emplace_back(amr_1d_data);
            continue;
        }

        // Get particle data, it first tries to read in libyt.particle_data, if not, it generates data in get_par_attr
        yt_data par_array;
        ds_status = ds_amr.GetPythonBoundLocalParticleData(gid, ptype.c_str(), pattr.c_str(), &par_array);
        if (ds_status.status == DataStructureStatus::kDataStructureSuccess) {
            // Read from libyt.particle_data
            amr_1d_data.data_ptr = par_array.data_ptr;
            is_new_allocation_list_.emplace_back(false);
            amr_data_array_1d_list_.emplace_back(amr_1d_data);
        } else {
            // Get particle function get_par_attr
            void (*get_par_attr)(const int, const long*, const char*, const char*, yt_array*) =
                particle_list[ptype_index].get_par_attr;
            if (get_par_attr == nullptr) {
                error_str_ = std::string("Get particle function get_par_attr not set in particle type [ ") + ptype +
                             std::string(" ] on MPI rank ") + std::to_string(DataStructureAmr::mpi_rank_) +
                             std::string(".\n");
                return {DataHubStatus::kDataHubFailed, amr_data_array_1d_list_};
            }

            // Generate buffer
            if (get_dtype_allocation(amr_1d_data.data_dtype, amr_1d_data.data_len, &amr_1d_data.data_ptr) !=
                YT_SUCCESS) {
                error_str_ =
                    std::string("Failed to allocate memory for (particle type, attribute, gid, data_len) = (") + ptype +
                    std::string(", ") + pattr + std::string(", ") + std::to_string(gid) + std::string(", ") +
                    std::to_string(amr_1d_data.data_len) + std::string(") on MPI rank ") +
                    std::to_string(DataStructureAmr::mpi_rank_) + std::string(".\n");
                return {DataHubStatus::kDataHubFailed, amr_data_array_1d_list_};
            }
            int list_len = 1;
            long list_gid[1] = {gid};
            yt_array data_array[1];
            data_array[0].gid = gid;
            data_array[0].data_length = amr_1d_data.data_len;
            data_array[0].data_ptr = amr_1d_data.data_ptr;
            (*get_par_attr)(list_len, list_gid, ptype.c_str(), pattr.c_str(), data_array);
            is_new_allocation_list_.emplace_back(true);
            amr_data_array_1d_list_.emplace_back(amr_1d_data);
        }
    }

    return {DataHubStatus::kDataHubSuccess, amr_data_array_1d_list_};
}

//-------------------------------------------------------------------------------------------------------
// Class         :  DataHubAmr
// Public Method :  ClearCache
//
// Notes       :  1. Clear the cache, and decide if new allocation needs to be freed.
//                2. TODO: since the class holds both 1D and 3D data cache, and only one of them will be
//                         used at a time, we need to check which one is is_new_allocation_list_ referring to.
//-------------------------------------------------------------------------------------------------------
void DataHubAmr::ClearCache() {
    if (!take_ownership_) {
        if (is_new_allocation_list_.size() == amr_data_array_3d_list_.size()) {
            for (size_t i = 0; i < amr_data_array_3d_list_.size(); i++) {
                if (is_new_allocation_list_[i]) {
                    free(amr_data_array_3d_list_[i].data_ptr);
                }
            }
        } else if (is_new_allocation_list_.size() == amr_data_array_1d_list_.size()) {
            for (size_t i = 0; i < amr_data_array_1d_list_.size(); i++) {
                if (is_new_allocation_list_[i]) {
                    free(amr_data_array_1d_list_[i].data_ptr);
                }
            }
        }
    }
    is_new_allocation_list_.clear();
    amr_data_array_3d_list_.clear();
    amr_data_array_1d_list_.clear();
    error_str_ = std::string("");
}
