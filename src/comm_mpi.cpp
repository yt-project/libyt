#ifndef SERIAL_MODE
#include "comm_mpi.h"

#include "comm_mpi_rma.h"
#include "data_hub_amr.h"
#include "timer.h"

int CommMpi::mpi_rank_ = 0;
int CommMpi::mpi_size_ = 1;
int CommMpi::mpi_root_ = 0;

std::map<std::string, MPI_Datatype*> CommMpi::mpi_custom_type_map_;
MPI_Datatype CommMpi::yt_long_mpi_type_;
MPI_Datatype CommMpi::yt_hierarchy_mpi_type_;
MPI_Datatype CommMpi::mpi_rma_address_mpi_type_;
MPI_Datatype CommMpi::amr_data_array_3d_mpi_type_;
MPI_Datatype CommMpi::amr_data_array_1d_mpi_type_;
MPI_Datatype CommMpi::yt_rma_grid_info_mpi_type_;
MPI_Datatype CommMpi::yt_rma_particle_info_mpi_type_;

void CommMpi::InitializeInfo(int mpi_root) {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
    mpi_root_ = mpi_root;
}

void CommMpi::InitializeYtLongMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int length[1] = {1};
    const MPI_Aint displacements[1] = {0};
    MPI_Datatype types[1] = {MPI_LONG};
    MPI_Type_create_struct(1, length, displacements, types, &yt_long_mpi_type_);
    MPI_Type_commit(&yt_long_mpi_type_);
}

void CommMpi::InitializeYtHierarchyMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[7] = {3, 3, 1, 1, 3, 1, 1};
    MPI_Aint displacements[7];
    displacements[0] = offsetof(yt_hierarchy, left_edge);
    displacements[1] = offsetof(yt_hierarchy, right_edge);
    displacements[2] = offsetof(yt_hierarchy, id);
    displacements[3] = offsetof(yt_hierarchy, parent_id);
    displacements[4] = offsetof(yt_hierarchy, dimensions);
    displacements[5] = offsetof(yt_hierarchy, level);
    displacements[6] = offsetof(yt_hierarchy, proc_num);
    MPI_Datatype types[7] = {MPI_DOUBLE, MPI_DOUBLE, MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(7, lengths, displacements, types, &yt_hierarchy_mpi_type_);
    MPI_Type_commit(&yt_hierarchy_mpi_type_);
}

void CommMpi::InitializeMpiRmaAddressMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[2] = {1, 1};
    MPI_Aint displacements[2];
    displacements[0] = offsetof(MpiRmaAddress, mpi_address);
    displacements[1] = offsetof(MpiRmaAddress, mpi_rank);
    MPI_Datatype types[2] = {MPI_AINT, MPI_INT};
    MPI_Type_create_struct(2, lengths, displacements, types, &mpi_rma_address_mpi_type_);
    MPI_Type_commit(&mpi_rma_address_mpi_type_);
}

void CommMpi::InitializeAmrDataArray3DMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Though we won't use the void* pointer in other rank, we still pass it for consistency.
    // Since it is passed as struct, we need to set the memory layout that covers the void* pointer size
    static_assert(sizeof(void*) == sizeof(long), "sizeof(void*) and sizeof(long) have difference size");

    int lengths[5] = {1, 1, 3, 1, 1};
    MPI_Aint displacements[5];
    displacements[0] = offsetof(AmrDataArray3D, id);
    displacements[1] = offsetof(AmrDataArray3D, data_dtype);
    displacements[2] = offsetof(AmrDataArray3D, data_dim);
    displacements[3] = offsetof(AmrDataArray3D, data_ptr);
    displacements[4] = offsetof(AmrDataArray3D, contiguous_in_x);
    MPI_Datatype types[5] = {MPI_LONG, MPI_INT, MPI_INT, MPI_LONG, MPI_CXX_BOOL};
    MPI_Type_create_struct(5, lengths, displacements, types, &amr_data_array_3d_mpi_type_);
    MPI_Type_commit(&amr_data_array_3d_mpi_type_);

    mpi_custom_type_map_["amr_grid"] = &amr_data_array_3d_mpi_type_;
}

void CommMpi::InitializeAmrDataArray1DMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    int lengths[4] = {1, 1, 1, 1};
    MPI_Aint displacements[4];
    displacements[0] = offsetof(AmrDataArray1D, id);
    displacements[1] = offsetof(AmrDataArray1D, data_dtype);
    displacements[2] = offsetof(AmrDataArray1D, data_ptr);
    displacements[3] = offsetof(AmrDataArray1D, data_len);
    MPI_Datatype types[4] = {MPI_LONG, MPI_INT, MPI_AINT, MPI_LONG};
    MPI_Type_create_struct(4, lengths, displacements, types, &amr_data_array_1d_mpi_type_);
    MPI_Type_commit(&amr_data_array_1d_mpi_type_);

    mpi_custom_type_map_["amr_particle"] = &amr_data_array_1d_mpi_type_;
}

void CommMpi::InitializeYtRmaGridInfoMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // TODO: this part will be removed after refactoring RMA
    int lengths[5] = {1, 1, 1, 1, 3};
    const MPI_Aint displacements[5] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 1 * sizeof(int),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 2 * sizeof(int)};
    MPI_Datatype types[5] = {MPI_LONG, MPI_AINT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(5, lengths, displacements, types, &yt_rma_grid_info_mpi_type_);
    MPI_Type_commit(&yt_rma_grid_info_mpi_type_);
}

void CommMpi::InitializeYtRmaParticleInfoMpiDataType() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // TODO: this part will be removed after refactoring RMA
    int lengths[4] = {1, 1, 1, 1};
    const MPI_Aint displacements[4] = {0, 1 * sizeof(long), 1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       2 * sizeof(long) + 1 * sizeof(MPI_Aint)};
    MPI_Datatype types[4] = {MPI_LONG, MPI_AINT, MPI_LONG, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types, &yt_rma_particle_info_mpi_type_);
    MPI_Type_commit(&yt_rma_particle_info_mpi_type_);
}

void CommMpi::SetAllNumGridsLocal(int* all_num_grids_local, int num_grids_local) {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Allgather(&num_grids_local, 1, MPI_INT, all_num_grids_local, 1, MPI_INT, MPI_COMM_WORLD);
}

//-------------------------------------------------------------------------------------------------------
// Class                :  CommMpi
// Public Static Method :  CheckAllStates
//
// Notes       :  1. Get all states from all ranks and check if every state matches the desired state,
//                   if yes, return success value; otherwise return failure value.
//                2. Both success value and failure value are passed in as arguments.
//                3. It supports only integer state, which are states declared in a enum class.
//-------------------------------------------------------------------------------------------------------
int CommMpi::CheckAllStates(int local_state, int desired_state, int success_value, int failure_value) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int* all_results = new int[mpi_size_];
    MPI_Allgather(&local_state, 1, MPI_INT, all_results, 1, MPI_INT, MPI_COMM_WORLD);

    bool match_desired_state = true;
    for (int r = 0; r < mpi_size_; r++) {
        if (all_results[r] != desired_state) {
            match_desired_state = false;
            break;
        }
    }

    delete[] all_results;

    return match_desired_state ? success_value : failure_value;
}

//-------------------------------------------------------------------------------------------------------
// Class                :  CommMpi
// Public Static Method :  SetStringUsingValueOnRank
//
// Notes       :  1. Sync the string on all ranks using the value on the specified rank.
//                2. The string is passed in by reference, so the string will be sync to the src rank.
//                3. Though reference to a string is passed in, only ranks other than src rank will have
//                   the string over-written.
//-------------------------------------------------------------------------------------------------------
void CommMpi::SetStringUsingValueOnRank(std::string& sync_string, int src_mpi_rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Get the length of the string
    unsigned long sync_string_len = 0;
    if (mpi_rank_ == src_mpi_rank) {
        sync_string_len = sync_string.length();
    }
    MPI_Bcast(&sync_string_len, 1, MPI_UNSIGNED_LONG, src_mpi_rank, MPI_COMM_WORLD);

    // Allocate the memory only on other ranks, and broadcast the string
    // Create new string on other ranks
    if (mpi_rank_ == src_mpi_rank) {
        MPI_Bcast((void*)sync_string.c_str(), (int)sync_string_len, MPI_CHAR, src_mpi_rank, MPI_COMM_WORLD);
    } else {
        char* dest_string = nullptr;
        dest_string = new char[sync_string_len + 1];
        MPI_Bcast((void*)dest_string, (int)sync_string_len, MPI_CHAR, src_mpi_rank, MPI_COMM_WORLD);
        dest_string[sync_string_len] = '\0';
        sync_string = std::string(dest_string);
        delete[] dest_string;
    }
}

//-------------------------------------------------------------------------------------------------------
// Class                :  CommMpi
// Public Static Method :  GatherAllStringsToRank
//
// Notes       :  1. It's a collective operation.
//                2. Should I also make it ignore the string on the destination rank?
//-------------------------------------------------------------------------------------------------------
void CommMpi::GatherAllStringsToRank(std::vector<std::string>& all_strings, const std::string& src_string,
                                     int dest_mpi_rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int src_string_len = static_cast<int>(src_string.length());
    if (mpi_rank_ == dest_mpi_rank) {
        // Get the length of all strings
        int* all_src_string_len = new int[mpi_size_];
        MPI_Gather(&src_string_len, 1, MPI_INT, all_src_string_len, 1, MPI_INT, dest_mpi_rank, MPI_COMM_WORLD);

        // Allocate buffer, and gather all the strings to the destination rank
        unsigned long sum_all_string_len = 0;
        int* displacements = new int[mpi_size_];
        for (int r = 0; r < mpi_size_; r++) {
            sum_all_string_len += all_src_string_len[r];
            displacements[r] = 0;
            for (int r1 = 0; r1 < r; r1++) {
                displacements[r] += all_src_string_len[r1];
            }
        }
        char* buffer = new char[sum_all_string_len];
        MPI_Gatherv(src_string.c_str(), src_string_len, MPI_CHAR, buffer, all_src_string_len, displacements, MPI_CHAR,
                    dest_mpi_rank, MPI_COMM_WORLD);

        // Copies the char* to std::string
        all_strings.clear();
        for (int r = 0; r < mpi_size_; r++) {
            all_strings.emplace_back(std::string(&buffer[displacements[r]], all_src_string_len[r]));
        }

        delete[] all_src_string_len;
        delete[] displacements;
        delete[] buffer;
    } else {
        MPI_Gather(&src_string_len, 1, MPI_INT, nullptr, 0, MPI_INT, dest_mpi_rank, MPI_COMM_WORLD);
        MPI_Gatherv(src_string.c_str(), src_string_len, MPI_CHAR, nullptr, nullptr, nullptr, MPI_CHAR, dest_mpi_rank,
                    MPI_COMM_WORLD);
    }
}

//-------------------------------------------------------------------------------------------------------
// Class                :  CommMpi
// Public Static Method :  GatherAllStringsToRank
//
// Notes       :  1. It's a collective operation.
//                2. I'm not sure if using int* instead of std::vector<int> will be faster.
//-------------------------------------------------------------------------------------------------------
void CommMpi::GatherAllIntsToRank(std::vector<int>& all_ints, const int src_int, int dest_mpi_rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

    all_ints.clear();
    if (CommMpi::mpi_rank_ == dest_mpi_rank) {
        all_ints.assign(mpi_size_, 0);
    }
    MPI_Gather(&src_int, 1, MPI_INT, all_ints.data(), 1, MPI_INT, dest_mpi_rank, MPI_COMM_WORLD);
}

#endif
