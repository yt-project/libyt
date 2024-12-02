#ifndef SERIAL_MODE
#include "comm_mpi_rma.h"

#include <iostream>

#include "timer.h"

template<typename DataInfoClass, typename DataClass>
CommMPIRma<DataInfoClass, DataClass>::CommMPIRma(const std::string& data_group_name)
    : data_group_name_(data_group_name) {
    SET_TIMER(__PRETTY_FUNCTION__);
}

template<typename DataInfoClass, typename DataClass>
std::pair<CommMPIRmaStatus, const std::vector<DataClass>&> CommMPIRma<DataInfoClass, DataClass>::GetRemoteData(
    const std::vector<DataClass>& prepared_data, const std::vector<FetchedFromInfo>& fetch_id_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Reset states to be able to reuse, or even do data chunking in the future
    error_str_ = std::string();
    mpi_prepared_data_.clear();
    mpi_fetched_data_.clear();

    // One-sided MPI
    //    if (InitializeMPIWindow() != MPIStatus::CommMPIRmaStatus::kMPISuccess) {
    //        return std::make_pair<int, const std::vector<DataClass>&>(kMPIFailed, mpi_fetched_data_);
    //    }

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
CommMPIRmaStatus CommMPIRma<DataInfoClass, DataClass>::PrepareData() {}

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