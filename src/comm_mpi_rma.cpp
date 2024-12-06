#ifndef SERIAL_MODE
#include "comm_mpi_rma.h"

#include <iostream>

#include "big_mpi.h"
#include "comm_mpi.h"
#include "data_structure_amr.h"
#include "timer.h"
#include "yt_prototype.h"

template<typename DataClass>
CommMpiRma<DataClass>::CommMpiRma(const std::string& data_group_name, const std::string& data_format)
    : data_group_name_(data_group_name), data_format_(data_format) {
    SET_TIMER(__PRETTY_FUNCTION__);
}

template<typename DataClass>
CommMpiRmaReturn<DataClass> CommMpiRma<DataClass>::GetRemoteData(
    const std::vector<DataClass>& prepared_data_list, const std::vector<CommMpiRmaQueryInfo>& fetch_id_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Reset states to be able to reuse, or even do data chunking in the future
    error_str_ = std::string();
    mpi_fetched_data_.clear();
    all_prepared_data_list_ = nullptr;
    all_prepared_data_address_list_ = nullptr;

    // One-sided MPI
    bool success = false;
    while (1) {
        if (InitializeMpiWindow() != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }
        if (PrepareData(prepared_data_list) != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }
        if (GatherAllPreparedData(prepared_data_list) != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }
        if (FetchRemoteData(fetch_id_list) != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }

        success = true;
        break;
    }

    CleanUp(prepared_data_list);

    if (!success) {
        return {CommMpiRmaStatus::kMpiFailed, mpi_fetched_data_};
    } else {
        return {CommMpiRmaStatus::kMpiSuccess, mpi_fetched_data_};
    }
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::InitializeMpiWindow() {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Info mpi_window_info;
    MPI_Info_create(&mpi_window_info);
    MPI_Info_set(mpi_window_info, "no_locks", "true");
    int mpi_return_code = MPI_Win_create_dynamic(mpi_window_info, MPI_COMM_WORLD, &mpi_window_);
    MPI_Info_free(&mpi_window_info);

    if (mpi_return_code != MPI_SUCCESS) {
        error_str_ = std::string("Create one-sided MPI (RMA) window failed!\n"
                                 "Try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".");
        return CommMpiRmaStatus::kMpiFailed;
    }

    return CommMpiRmaStatus::kMpiSuccess;
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::PrepareData(const std::vector<DataClass>& prepared_data_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    mpi_prepared_data_address_list_.clear();
    mpi_prepared_data_address_list_.reserve(prepared_data_list.size());

    for (const DataClass& pdata : prepared_data_list) {
        // Check if data pointer is nullptr
        if (pdata.data_ptr == nullptr) {
            error_str_ = std::string("Data pointer is nullptr in (data_group, gid) = (") + data_group_name_ +
                         std::string(", ") + std::to_string(pdata.id) + std::string(")!");
            return CommMpiRmaStatus::kMpiFailed;
        }

        // Attach buffer to window (TODO: consider particle data too)
        int mpi_return_code = MPI_Win_attach(mpi_window_, pdata.data_ptr, (MPI_Aint)GetDataSize(pdata));
        if (mpi_return_code != MPI_SUCCESS) {
            error_str_ = std::string("Attach buffer (data_group, gid) = (") + data_group_name_ + std::string(", ") +
                         std::to_string(pdata.id) + std::string(") to one-sided MPI (RMA) window failed on MPI rank ") +
                         std::to_string(CommMpi::mpi_rank_) +
                         std::string("!\n"
                                     "Try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".");
            return CommMpiRmaStatus::kMpiFailed;
        }

        // Get the address of the attached buffer
        MPI_Aint mpi_address;
        if (MPI_Get_address(pdata.data_ptr, &mpi_address) != MPI_SUCCESS) {
            error_str_ = std::string("Get the address of the attached buffer (data_group, id) = (") + data_group_name_ +
                         std::string(", ") + std::to_string(pdata.id) + std::string(") failed on MPI rank ") +
                         std::to_string(CommMpi::mpi_rank_) + std::string("!");
            return CommMpiRmaStatus::kMpiFailed;
        }

        mpi_prepared_data_address_list_.emplace_back(MpiRmaAddress{mpi_address, CommMpi::mpi_rank_});

        // TODO: After single out loggging, change to debug (debug purpose only)
        printf("Attach buffer (data_group, id) = (%s, %ld) to one-sided MPI (RMA) window on MPI rank %d\n",
               data_group_name_.c_str(), pdata.id, CommMpi::mpi_rank_);
    }

    return CommMpiRmaStatus::kMpiSuccess;
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::GatherAllPreparedData(const std::vector<DataClass>& prepared_data_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Get send count in each rank
    int send_count = prepared_data_list.size();
    int* all_send_counts = new int[CommMpi::mpi_size_];
    MPI_Allgather(&send_count, 1, MPI_INT, all_send_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Calculate total send count and search range
    long total_send_counts = 0;
    search_range_.clear();
    search_range_.assign(CommMpi::mpi_size_ + 1, 0);
    for (int r1 = 0; r1 < CommMpi::mpi_size_ + 1; r1++) {
        for (int r2 = 0; r2 < r1; r2++) {
            search_range_[r1] += all_send_counts[r2];
        }
    }
    total_send_counts = search_range_[CommMpi::mpi_size_];

    // Get all prepared data
    all_prepared_data_list_ = new DataClass[total_send_counts];
    all_prepared_data_address_list_ = new MpiRmaAddress[total_send_counts];
    big_MPI_Gatherv<DataClass>(CommMpi::mpi_root_, all_send_counts, prepared_data_list.data(),
                               CommMpi::mpi_custom_type_map_[data_format_], all_prepared_data_list_);
    big_MPI_Gatherv<MpiRmaAddress>(CommMpi::mpi_root_, all_send_counts, mpi_prepared_data_address_list_.data(),
                                   &CommMpi::mpi_rma_address_mpi_type_, all_prepared_data_address_list_);
    big_MPI_Bcast<DataClass>(CommMpi::mpi_root_, total_send_counts, all_prepared_data_list_,
                             CommMpi::mpi_custom_type_map_[data_format_]);
    big_MPI_Bcast<MpiRmaAddress>(CommMpi::mpi_root_, total_send_counts, all_prepared_data_address_list_,
                                 &CommMpi::mpi_rma_address_mpi_type_);

    // Clean up
    delete[] all_send_counts;
    mpi_prepared_data_address_list_.clear();

    return CommMpiRmaStatus::kMpiSuccess;
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::FetchRemoteData(const std::vector<CommMpiRmaQueryInfo>& fetch_id_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Open the window epoch
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, mpi_window_);

    // Fetch data
    mpi_fetched_data_.reserve(fetch_id_list.size());
    for (const CommMpiRmaQueryInfo& fid : fetch_id_list) {
        bool data_found = false;

        for (long s = search_range_[fid.mpi_rank]; s < search_range_[fid.mpi_rank + 1]; s++) {
            if (all_prepared_data_list_[s].id == fid.id) {
                DataClass fetched_data = all_prepared_data_list_[s];

                // Copy data from remote buffer to local, and set the pointer in fetched_data
                MPI_Datatype mpi_dtype;
                get_mpi_dtype(fetched_data.data_dtype, &mpi_dtype);
                void* fetched_data_buffer = malloc(GetDataSize(fetched_data));
                fetched_data.data_ptr = fetched_data_buffer;
                if (big_MPI_Get_dtype(fetched_data_buffer, GetDataLen(fetched_data), &fetched_data.data_dtype,
                                      &mpi_dtype, all_prepared_data_address_list_[s].mpi_rank,
                                      all_prepared_data_address_list_[s].mpi_address, &mpi_window_) != YT_SUCCESS) {
                    error_str_ = std::string("Fetch remote data buffer (data_group, id, mpi_rank) = (") +
                                 data_group_name_ + std::string(", ") + std::to_string(fid.id) + std::string(", ") +
                                 std::to_string(all_prepared_data_address_list_[s].mpi_rank) +
                                 std::string(") failed on MPI rank ") + std::to_string(CommMpi::mpi_rank_) +
                                 std::string("!");
                    free(fetched_data_buffer);
                    return CommMpiRmaStatus::kMpiFailed;
                }

                // Push to fetched data list
                mpi_fetched_data_.emplace_back(fetched_data);
                data_found = true;
                break;
            }
        }

        if (!data_found) {
            error_str_ = std::string("Cannot find remote data buffer (data_group, id, mpi_rank) = (") +
                         data_group_name_ + std::string(", ") + std::to_string(fid.id) + std::string(", ") +
                         std::to_string(fid.mpi_rank) + std::string(") on MPI rank ") +
                         std::to_string(CommMpi::mpi_rank_) + std::string("!");
            return CommMpiRmaStatus::kMpiFailed;
        }
    }

    // Close the window epoch, every data should be fetched by now
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, mpi_window_);

    return CommMpiRmaStatus::kMpiSuccess;
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::CleanUp(const std::vector<DataClass>& prepared_data_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (const DataClass& pdata : prepared_data_list) {
        MPI_Win_detach(mpi_window_, pdata.data_ptr);
    }
    MPI_Win_free(&mpi_window_);
    mpi_prepared_data_address_list_.clear();
    search_range_.clear();
    delete[] all_prepared_data_list_;
    delete[] all_prepared_data_address_list_;

    return CommMpiRmaStatus::kMpiSuccess;
}

template class CommMpiRma<AmrDataArray3D>;
template class CommMpiRma<AmrDataArray1D>;

std::size_t CommMpiRmaAmrDataArray3D::GetDataSize(const AmrDataArray3D& data) {
    for (int i = 0; i < 3; i++) {
        if (data.data_dim[i] <= 0) {
            return 0;
        }
    }
    int dtype_size;
    get_dtype_size(data.data_dtype, &dtype_size);
    return data.data_dim[0] * data.data_dim[1] * data.data_dim[2] * dtype_size;
}

std::size_t CommMpiRmaAmrDataArray3D::GetDataLen(const AmrDataArray3D& data) {
    for (int i = 0; i < 3; i++) {
        if (data.data_dim[i] <= 0) {
            return 0;
        }
    }
    return data.data_dim[0] * data.data_dim[1] * data.data_dim[2];
}

std::size_t CommMpiRmaAmrDataArray1D::GetDataSize(const AmrDataArray1D& data) {
    if (data.data_len <= 0) {
        return 0;
    }
    int dtype_size;
    get_dtype_size(data.data_dtype, &dtype_size);
    return data.data_len * dtype_size;
}

std::size_t CommMpiRmaAmrDataArray1D::GetDataLen(const AmrDataArray1D& data) {
    if (data.data_len <= 0) {
        return 0;
    }
    return data.data_len;
}

#endif
