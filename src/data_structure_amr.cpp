#include "data_structure_amr.h"

#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_prototype.h"

DataHubReturn<AmrDataArray3D> DataHubAmr::GetLocalFieldData(const std::string& field_name,
                                                            const std::vector<long>& grid_id_list) {
    // Free cache before doing new query
    ClearCache();

    // Since everything is under LibytProcessControl, we need to include it.
    // TODO: Move data structure in LibytProcessControl to this class later
    yt_field* field_list = LibytProcessControl::Get().field_list;
    int field_id = -1;
    for (int v = 0; v < LibytProcessControl::Get().param_yt_.num_fields; v++) {
        if (field_name == field_list[v].field_name) {
            field_id = v;
            break;
        }
    }
    if (field_id == -1) {
        std::string error_msg = std::string("Cannot find field_name [ ") + field_name +
                                std::string(" ] in field_list on MPI rank ") +
                                std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
        return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
    }

    if (strcmp(field_list[field_id].field_type, "derived_func") == 0) {
        for (const long& gid : grid_id_list) {
            AmrDataArray3D amr_data{};

            // Get amr grid info
            // TODO: Get data using libyt publich API, (this should be fixed later)
            int grid_dim[3];
            if (yt_getGridInfo_Dimensions(gid, &grid_dim) != YT_SUCCESS) {
                std::string error_msg = std::string("Failed to get grid dim for (field_name, gid) = (") + field_name +
                                        std::string(", ") + std::to_string(gid) + std::string(") on MPI rank ") +
                                        std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
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
                std::string error_msg = std::string("Derived function not set in field [ ") + field_name +
                                        std::string(" ] on MPI rank ") +
                                        std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
                return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
            }

            // Allocate memory for data_ptr and generate data
            long data_len = amr_data.data_dim[0] * amr_data.data_dim[1] * amr_data.data_dim[2];
            if (get_dtype_allocation(amr_data.data_dtype, data_len, &amr_data.data_ptr) != YT_SUCCESS) {
                std::string error_msg = std::string("Failed to allocate memory for (field_name, gid) = (") +
                                        field_name + std::string(", ") + std::to_string(gid) +
                                        std::string(") on MPI rank ") +
                                        std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
                return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
            }
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
            // TODO: Get data using libyt publich API, (this should be fixed later)
            yt_data field_data;
            if (yt_getGridInfo_FieldData(gid, field_name.c_str(), &field_data) != YT_SUCCESS) {
                std::string error_msg = std::string("Failed to get data (field_name, gid) = (") + field_name +
                                        std::string(", ") + std::to_string(gid) + std::string(") on MPI rank ") +
                                        std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
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
        std::string error_msg = std::string("Unknown field type [ ") + std::string(field_list[field_id].field_type) +
                                std::string(" ] in field [ ") + field_name + std::string(" ] on MPI rank ") +
                                std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
        return {DataHubStatus::kDataHubFailed, amr_data_array_3d_list_};
    }

    return {DataHubStatus::kDataHubSuccess, amr_data_array_3d_list_};
}

void DataHubAmr::ClearCache() {
    if (!take_ownership_) {
        // TODO: this is just weird (make particle RMA work first)
        //       since the class holds both 1D and 3D data cache, and only one of them will be used,
        //       we need to check which one is is_new_allocation_list_ referring to.
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
        for (size_t i = 0; i < amr_data_array_1d_list_.size(); i++) {
            free(amr_data_array_1d_list_[i].data_ptr);
        }
    }
    is_new_allocation_list_.clear();
    amr_data_array_3d_list_.clear();
    amr_data_array_1d_list_.clear();
    error_str_ = std::string("");
}
