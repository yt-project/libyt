#ifndef SERIAL_MODE
#include "comm_mpi_rma.h"

#include "big_mpi.h"
#include "comm_mpi.h"
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
    // Make sure every process can go through each step correctly, otherwise fail fast.
    CommMpiRmaStatus status, all_status;
    int step = 0;
    while (1) {
        status = InitializeMpiWindow();
        step = 1;
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::GetAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));
        if (all_status != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }

        status = PrepareData(prepared_data_list);
        step = 2;
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::GetAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));
        if (all_status != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }

        status = GatherAllPreparedData(prepared_data_list);
        step = 3;
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::GetAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));
        if (all_status != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }

        status = FetchRemoteData(fetch_id_list);
        step = 4;
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::GetAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));

        break;
    }

    if (status == CommMpiRmaStatus::kMpiSuccess && step >= 2) {
        DetachBuffer(prepared_data_list);
    }
    if (status == CommMpiRmaStatus::kMpiSuccess) {
        FreeMpiWindow();
    }
    CleanUp(prepared_data_list);

    return {.status = status, .all_status = static_cast<CommMpiRmaStatus>(all_status), .data_list = mpi_fetched_data_};
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
        // If data pointer is nullptr, we don't need to wrap it.
        if (pdata.data_ptr == nullptr) {
            mpi_prepared_data_address_list_.emplace_back(
                MpiRmaAddress{reinterpret_cast<MPI_Aint>(nullptr), CommMpi::mpi_rank_});
            continue;
        }

        // Attach buffer to window for all the prepared data passed in
        long data_size = GetDataSize(pdata);
        if (data_size < 0) {
            error_str_ = std::string("Prepare data size is invalid in (data_group, id) = (") + data_group_name_ +
                         std::string(", ") + std::to_string(pdata.id) + std::string(") on MPI rank ") +
                         std::to_string(CommMpi::mpi_rank_) + std::string("!");
            return CommMpiRmaStatus::kMpiFailed;
        }
        int mpi_return_code = MPI_Win_attach(mpi_window_, pdata.data_ptr, (MPI_Aint)data_size);
        if (mpi_return_code != MPI_SUCCESS) {
            error_str_ = std::string("Attach buffer (data_group, id) = (") + data_group_name_ + std::string(", ") +
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
    bool data_found = true;
    for (const CommMpiRmaQueryInfo& fid : fetch_id_list) {
        data_found = false;

        for (long s = search_range_[fid.mpi_rank]; s < search_range_[fid.mpi_rank + 1]; s++) {
            if (all_prepared_data_list_[s].id == fid.id) {
                DataClass fetched_data = all_prepared_data_list_[s];

                // If data pointer to fetch is nullptr, we don't need to fetch it.
                if (reinterpret_cast<void*>(all_prepared_data_address_list_[s].mpi_address) == nullptr) {
                    fetched_data.data_ptr = nullptr;
                    mpi_fetched_data_.emplace_back(fetched_data);
                    data_found = true;
                    break;
                }

                // Check the size, length, and data pointer
                MPI_Datatype mpi_dtype;
                get_mpi_dtype(fetched_data.data_dtype, &mpi_dtype);
                long data_size = GetDataSize(fetched_data);
                long data_len = GetDataLen(fetched_data);
                if (data_size < 0) {
                    error_str_ = std::string("Fetch remote data size is invalid in (data_group, id, mpi_rank) = (") +
                                 data_group_name_ + std::string(", ") + std::to_string(fetched_data.id) +
                                 std::string(", ") + std::to_string(all_prepared_data_address_list_[s].mpi_rank) +
                                 std::string(") on MPI rank ") + std::to_string(CommMpi::mpi_rank_) + std::string("!");
                    break;
                }
                if (data_len < 0) {
                    error_str_ = std::string("Fetch remote data length is invalid in (data_group, id, mpi_rank) = (") +
                                 data_group_name_ + std::string(", ") + std::to_string(fetched_data.id) +
                                 std::string(", ") + std::to_string(all_prepared_data_address_list_[s].mpi_rank) +
                                 std::string(") on MPI rank ") + std::to_string(CommMpi::mpi_rank_) + std::string("!");
                    break;
                }

                // Copy data from remote buffer to local, and set the pointer in fetched_data
                void* fetched_data_buffer = malloc(data_size);
                fetched_data.data_ptr = fetched_data_buffer;
                if (big_MPI_Get_dtype(fetched_data_buffer, data_len, &fetched_data.data_dtype, &mpi_dtype,
                                      all_prepared_data_address_list_[s].mpi_rank,
                                      all_prepared_data_address_list_[s].mpi_address, &mpi_window_) != YT_SUCCESS) {
                    error_str_ = std::string("Fetch remote data buffer (data_group, id, mpi_rank) = (") +
                                 data_group_name_ + std::string(", ") + std::to_string(fid.id) + std::string(", ") +
                                 std::to_string(all_prepared_data_address_list_[s].mpi_rank) +
                                 std::string(") failed on MPI rank ") + std::to_string(CommMpi::mpi_rank_) +
                                 std::string("!");
                    free(fetched_data_buffer);
                    break;
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
            break;
        }
    }

    // Close the window epoch, even if the fetch failed
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, mpi_window_);

    if (data_found) {
        return CommMpiRmaStatus::kMpiSuccess;
    } else {
        return CommMpiRmaStatus::kMpiFailed;
    }
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::FreeMpiWindow() {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Win_free(&mpi_window_);

    return CommMpiRmaStatus::kMpiSuccess;
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::DetachBuffer(const std::vector<DataClass>& prepared_data_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    for (const DataClass& pdata : prepared_data_list) {
        MPI_Win_detach(mpi_window_, pdata.data_ptr);
    }

    return CommMpiRmaStatus::kMpiSuccess;
}

template<typename DataClass>
CommMpiRmaStatus CommMpiRma<DataClass>::CleanUp(const std::vector<DataClass>& prepared_data_list) {
    SET_TIMER(__PRETTY_FUNCTION__);

    search_range_.clear();
    delete[] all_prepared_data_list_;
    delete[] all_prepared_data_address_list_;

    return CommMpiRmaStatus::kMpiSuccess;
}

template class CommMpiRma<AmrDataArray3D>;
template class CommMpiRma<AmrDataArray1D>;

long CommMpiRmaAmrDataArray3D::GetDataSize(const AmrDataArray3D& data) {
    for (int i = 0; i < 3; i++) {
        if (data.data_dim[i] < 0) {
            return -1;
        }
    }
    if (data.data_dtype == YT_DTYPE_UNKNOWN) {
        return -1;
    }

    int dtype_size;
    get_dtype_size(data.data_dtype, &dtype_size);
    return data.data_dim[0] * data.data_dim[1] * data.data_dim[2] * dtype_size;
}

long CommMpiRmaAmrDataArray3D::GetDataLen(const AmrDataArray3D& data) {
    for (int i = 0; i < 3; i++) {
        if (data.data_dim[i] < 0) {
            return -1;
        }
    }
    return data.data_dim[0] * data.data_dim[1] * data.data_dim[2];
}

long CommMpiRmaAmrDataArray1D::GetDataSize(const AmrDataArray1D& data) {
    if (data.data_len < 0 || data.data_dtype == YT_DTYPE_UNKNOWN) {
        return -1;
    }

    int dtype_size;
    get_dtype_size(data.data_dtype, &dtype_size);
    return data.data_len * dtype_size;
}

long CommMpiRmaAmrDataArray1D::GetDataLen(const AmrDataArray1D& data) {
    if (data.data_len < 0) {
        return -1;
    }
    return data.data_len;
}

#endif
