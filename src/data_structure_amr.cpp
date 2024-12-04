#include "data_structure_amr.h"

#include "libyt.h"
#include "libyt_process_control.h"

const std::vector<AmrDataArray3D>& DataHubAmr::GetFieldData(const std::string& field_name,
                                                            const std::vector<long>& grid_id_list) {
    // Free cache before doing new query
    Free();

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
        // TODO: deal with error later, make rma work first
        std::string error_msg = std::string("Cannot find field_name [ ") + field_name +
                                std::string(" ] in field_list on MPI rank ") +
                                std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
        return amr_data_array_3d_list_;
    }

    if (strcmp(field_list[field_id].field_type, "derived_func") == 0) {
        is_new_allocation_ = true;
        for (const long& gid : grid_id_list) {
            // TODO: derived_function is not implemented yet
        }
    } else if (strcmp(field_list[field_id].field_type, "cell-centered") == 0 ||
               strcmp(field_list[field_id].field_type, "face-centered") == 0) {
        is_new_allocation_ = false;
        for (const long& gid : grid_id_list) {
            // TODO: Get data using libyt publich API, (this should be fixed later)
            yt_data field_data;
            if (yt_getGridInfo_FieldData(gid, field_name.c_str(), &field_data) != YT_SUCCESS) {
                // TODO: deal with error later, make rma work first
                std::string error_msg = std::string("Failed to get (field_name, gid) = (") + field_name +
                                        std::string(", ") + std::to_string(gid) + std::string(") on MPI rank ") +
                                        std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
                return amr_data_array_3d_list_;
            }
            AmrDataArray3D amr_data{};
            amr_data.id = gid;
            amr_data.contiguous_in_x = field_list[field_id].contiguous_in_x;
            amr_data.data_dtype = field_data.data_dtype;
            for (int d = 0; d < 3; d++) {
                amr_data.data_dim[d] = field_data.data_dimensions[d];
            }
            amr_data.data_ptr = field_data.data_ptr;

            amr_data_array_3d_list_.emplace_back(amr_data);
        }
    } else {
        // TODO: deal with error later, make rma work first
        std::string error_msg = std::string("Unknown field type [ ") + std::string(field_list[field_id].field_type) +
                                std::string(" ] in field [ ") + field_name + std::string(" ] on MPI rank ") +
                                std::to_string(LibytProcessControl::Get().mpi_rank_) + std::string(".\n");
        return amr_data_array_3d_list_;
    }

    return amr_data_array_3d_list_;
}

void DataHubAmr::Free() {
    if (is_new_allocation_) {
        for (size_t i = 0; i < amr_data_array_3d_list_.size(); i++) {
            free(amr_data_array_3d_list_[i].data_ptr);
        }
    }
    amr_data_array_3d_list_.clear();
}
