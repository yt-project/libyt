#ifndef SERIAL_MODE
#include "comm_mpi_rma.h"

#include <iostream>

#include "big_mpi.h"
#include "comm_mpi.h"
#include "timer.h"
#include "yt_prototype.h"

template<typename DataInfoClass, typename DataClass>
CommMPIRma<DataInfoClass, DataClass>::CommMPIRma(const std::string& data_group_name, const std::string& data_format)
    : data_group_name_(data_group_name), data_format_(data_format) {
    SET_TIMER(__PRETTY_FUNCTION__);
}

template<typename DataInfoClass, typename DataClass>
std::pair<CommMPIRmaStatus, const std::vector<DataClass>&> CommMPIRma<DataInfoClass, DataClass>::GetRemoteData(
    const std::vector<DataClass>& prepared_data_list, const std::vector<FetchedFromInfo>& fetch_id_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Reset states to be able to reuse, or even do data chunking in the future
    error_str_ = std::string();
    mpi_prepared_data_info_list_.clear();
    mpi_prepared_data_info_list_.reserve(prepared_data_list.size());
    mpi_fetched_data_.clear();
    mpi_fetched_data_.reserve(fetch_id_list.size());

    // One-sided MPI
    if (InitializeMPIWindow() != CommMPIRmaStatus::kMPISuccess) {
        return std::make_pair<CommMPIRmaStatus, const std::vector<DataClass>&>(CommMPIRmaStatus::kMPIFailed,
                                                                               mpi_fetched_data_);
    }

    if (PrepareData(prepared_data_list) != CommMPIRmaStatus::kMPISuccess) {
        return std::make_pair<CommMPIRmaStatus, const std::vector<DataClass>&>(CommMPIRmaStatus::kMPIFailed,
                                                                               mpi_fetched_data_);
    }

    if (GatherAllPreparedData() != CommMPIRmaStatus::kMPISuccess) {
        return std::make_pair<CommMPIRmaStatus, const std::vector<DataClass>&>(CommMPIRmaStatus::kMPIFailed,
                                                                               mpi_fetched_data_);
    }

    // TODO: make sure it is always called before returning.
    if (CleanUp() != CommMPIRmaStatus::kMPISuccess) {
        return std::make_pair<CommMPIRmaStatus, const std::vector<DataClass>&>(CommMPIRmaStatus::kMPIFailed,
                                                                               mpi_fetched_data_);
    }

    return std::make_pair<CommMPIRmaStatus, const std::vector<DataClass>&>(CommMPIRmaStatus::kMPISuccess,
                                                                           mpi_fetched_data_);
}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::InitializeMPIWindow() {
    MPI_Info mpi_window_info;
    MPI_Info_create(&mpi_window_info);
    MPI_Info_set(mpi_window_info, "no_locks", "true");
    int mpi_return_code = MPI_Win_create_dynamic(mpi_window_info, MPI_COMM_WORLD, &mpi_window_);
    MPI_Info_free(&mpi_window_info);

    if (mpi_return_code != MPI_SUCCESS) {
        error_str_ = std::string("Create one-sided MPI (RMA) window failed!\n"
                                 "Try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".");
        return CommMPIRmaStatus::kMPIFailed;
    }

    return CommMPIRmaStatus::kMPISuccess;
}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::PrepareData(const std::vector<DataClass>& prepared_data_list) {
    for (const DataClass& pdata : prepared_data_list) {
        // Check if data pointer is nullptr
        if (pdata.data_ptr == nullptr) {
            error_str_ = std::string("Data pointer is nullptr in (data_group, gid) = (") + data_group_name_ +
                         std::string(", ") + std::to_string(pdata.id) + std::string(")!");
            return CommMPIRmaStatus::kMPIFailed;
        }

        // Attach buffer to window (TODO: consider particle data too)
        int dtype_size;
        get_dtype_size(pdata.data_type, &dtype_size);
        MPI_Aint data_size = pdata.data_dim[0] * pdata.data_dim[1] * pdata.data_dim[2] * dtype_size;
        int mpi_return_code = MPI_Win_attach(mpi_window_, pdata.data_ptr, data_size);
        if (mpi_return_code != MPI_SUCCESS) {
            error_str_ = std::string("Attach buffer (data_group, gid) = (") + data_group_name_ + std::string(", ") +
                         std::to_string(pdata.id) + std::string(") to one-sided MPI (RMA) window failed on MPI rank ") +
                         std::to_string(CommMPI::mpi_rank_) +
                         std::string("!\n"
                                     "Try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".");
            return CommMPIRmaStatus::kMPIFailed;
        }

        // Get the address of the attached buffer
        MPI_Aint mpi_address;
        if (MPI_Get_address(pdata.data_ptr, &mpi_address) != MPI_SUCCESS) {
            error_str_ = std::string("Get the address of the attached buffer (data_group, id) = (") + data_group_name_ +
                         std::string(", ") + std::to_string(pdata.id) + std::string(") failed on MPI rank ") +
                         std::to_string(CommMPI::mpi_rank_) + std::string("!");
            return CommMPIRmaStatus::kMPIFailed;
        }

        // Add to prepared list (TODO: consider particle data too)
        mpi_prepared_data_info_list_.emplace_back(DataInfoClass{
            pdata.id, pdata.data_type, {pdata.data_dim[0], pdata.data_dim[1], pdata.data_dim[2]}, pdata.swap_axes});
        mpi_prepared_data_address_list_.emplace_back(MPIRmaAddress{CommMPI::mpi_rank_, mpi_address});

        // TODO: After single out loggging, change to debug (debug purpose only)
        printf("Attach buffer (data_group, id) = (%s, %ld) to one-sided MPI (RMA) window on MPI rank %d\n",
               data_group_name_.c_str(), pdata.id, CommMPI::mpi_rank_);
    }

    return CommMPIRmaStatus::kMPISuccess;
}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::GatherAllPreparedData() {
    // Get send count in each rank
    int send_count = mpi_prepared_data_info_list_.size();
    int* all_send_counts = new int[CommMPI::mpi_size_];
    MPI_Allgather(&send_count, 1, MPI_INT, all_send_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate total send count and search range
    long total_send_counts = 0;
    search_range_.clear();
    search_range_.assign(CommMPI::mpi_size_ + 1, 0);
    for (int r1 = 0; r1 < CommMPI::mpi_size_ + 1; r1++) {
        for (int r2 = 0; r2 < r1; r2++) {
            search_range_[r1] += all_send_counts[r2];
        }
    }
    total_send_counts = search_range_[CommMPI::mpi_size_];

    // Get all prepared data
    all_prepared_data_info_list_ = new DataInfoClass[total_send_counts];
    all_prepared_data_address_list_ = new MPIRmaAddress[total_send_counts];
    big_MPI_Gatherv<DataInfoClass>(CommMPI::mpi_root_, all_send_counts, mpi_prepared_data_info_list_.data(),
                                   CommMPI::mpi_custom_type_map_[data_format_], all_prepared_data_info_list_);
    big_MPI_Gatherv<MPIRmaAddress>(CommMPI::mpi_root_, all_send_counts, mpi_prepared_data_address_list_.data(),
                                   &CommMPI::mpi_rma_address_mpi_type_, all_prepared_data_address_list_);
    big_MPI_Bcast<DataInfoClass>(CommMPI::mpi_root_, total_send_counts, all_prepared_data_info_list_,
                                 CommMPI::mpi_custom_type_map_[data_format_]);
    big_MPI_Bcast<MPIRmaAddress>(CommMPI::mpi_root_, total_send_counts, all_prepared_data_address_list_,
                                 &CommMPI::mpi_rma_address_mpi_type_);

    // Clean up
    delete[] all_send_counts;
    mpi_prepared_data_info_list_.clear();
    mpi_prepared_data_address_list_.clear();

    return CommMPIRmaStatus::kMPISuccess;
}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::FetchRemoteData() {}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::CleanUp() {
    MPI_Win_free(&mpi_window_);
    return CommMPIRmaStatus::kMPISuccess;
}

template class CommMPIRma<AMRDataArray3DInfo, AMRDataArray3D>;

#endif