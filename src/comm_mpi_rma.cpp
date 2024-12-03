#ifndef SERIAL_MODE
#include "comm_mpi_rma.h"

#include <iostream>

#include "comm_mpi.h"
#include "timer.h"
#include "yt_prototype.h"

template<typename DataInfoClass, typename DataClass>
CommMPIRma<DataInfoClass, DataClass>::CommMPIRma(const std::string& data_group_name)
    : data_group_name_(data_group_name) {
    SET_TIMER(__PRETTY_FUNCTION__);
}

template<typename DataInfoClass, typename DataClass>
std::pair<CommMPIRmaStatus, const std::vector<DataClass>&> CommMPIRma<DataInfoClass, DataClass>::GetRemoteData(
    const std::vector<DataClass>& prepared_data_list, const std::vector<FetchedFromInfo>& fetch_id_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Reset states to be able to reuse, or even do data chunking in the future
    error_str_ = std::string();
    mpi_prepared_data_.clear();
    mpi_prepared_data_.reserve(prepared_data_list.size());
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
                         std::string(", ") + std::to_string(pdata.data_info.id) + std::string(")!");
            return CommMPIRmaStatus::kMPIFailed;
        }

        // Attach buffer to window
        int dtype_size;
        get_dtype_size(pdata.data_info.data_type, &dtype_size);
        MPI_Aint data_size =
            pdata.data_info.data_dim[0] * pdata.data_info.data_dim[1] * pdata.data_info.data_dim[2] * dtype_size;
        int mpi_return_code = MPI_Win_attach(mpi_window_, pdata.data_ptr, data_size);
        if (mpi_return_code != MPI_SUCCESS) {
            error_str_ = std::string("Attach buffer (data_group, gid) = (") + data_group_name_ + std::string(", ") +
                         std::to_string(pdata.data_info.id) +
                         std::string(") to one-sided MPI (RMA) window failed on MPI rank ") +
                         std::to_string(CommMPI::mpi_rank_) +
                         std::string("!\n"
                                     "Try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".");
            return CommMPIRmaStatus::kMPIFailed;
        }

        // Get the address of the attached buffer
        MPI_Aint mpi_address;
        if (MPI_Get_address(pdata.data_ptr, &mpi_address) != MPI_SUCCESS) {
            error_str_ = std::string("Get the address of the attached buffer (data_group, id) = (") + data_group_name_ +
                         std::string(", ") + std::to_string(pdata.data_info.id) + std::string(") failed on MPI rank ") +
                         std::to_string(CommMPI::mpi_rank_) + std::string("!");
            return CommMPIRmaStatus::kMPIFailed;
        }

        // Add to prepared list
        mpi_prepared_data_.emplace_back(MPIRmaData<DataInfoClass>{
            mpi_address,
            CommMPI::mpi_rank_,
            {pdata.data_info.id,
             pdata.data_info.data_type,
             {pdata.data_info.data_dim[0], pdata.data_info.data_dim[1], pdata.data_info.data_dim[2]},
             pdata.data_info.swap_axes_}});

        printf("Attach buffer (gid, data_group) = (%s, %ld) to one-sided MPI (RMA) window on MPI rank %d\n",
               data_group_name_.c_str(), pdata.data_info.id, CommMPI::mpi_rank_);
    }

    return CommMPIRmaStatus::kMPISuccess;
}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::GatherAllPreparedData() {}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::FetchRemoteData() {}

template<typename DataInfoClass, typename DataClass>
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::CleanUp() {
    MPI_Win_free(&mpi_window_);
    return CommMPIRmaStatus::kMPISuccess;
}

template class CommMPIRma<AMRFieldDataArray3DInfo, AMRFieldDataArray3D>;

#endif