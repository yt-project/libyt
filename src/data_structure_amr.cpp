#include "data_structure_amr.h"

#include "libyt_process_control.h"

const std::vector<AmrDataArray3D>& DataStructureAmr::GetFieldData(const std::string& field_name,
                                                                  const std::vector<long>& grid_id_list) {
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

    return amr_data_array_3d_list_;
}
