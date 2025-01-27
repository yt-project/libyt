#ifndef SERIAL_MODE
#include "comm_mpi_rma.h"

#include "big_mpi.h"
#include "comm_mpi.h"
#include "timer.h"
#include "yt_prototype.h"

template<typename DataClass>
MPI_Datatype CommMpiRma<DataClass>::mpi_rma_data_type_ = nullptr;

//-------------------------------------------------------------------------------------------------------
// Class         :  CommMpiRma<DataClass>
// Public Method :  Constructor
//
// Notes       :  1. Set up data group name and data format.
//                2. Data format maps to custom MPI datatype defined in CommMpi::mpi_custom_type_map_.
//-------------------------------------------------------------------------------------------------------
template<typename DataClass>
CommMpiRma<DataClass>::CommMpiRma(const std::string& data_group_name, const std::string& data_format)
    : data_group_name_(data_group_name), data_format_(data_format) {
    SET_TIMER(__PRETTY_FUNCTION__);
    InitializeMpiAddressDataType();
}

template<typename DataClass>
void CommMpiRma<DataClass>::InitializeMpiAddressDataType() {
    if (mpi_rma_data_type_ != nullptr) {
        return;
    }
    int lengths[2] = {1, 1};
    MPI_Aint displacements[2];
    displacements[0] = offsetof(MpiRmaAddress, mpi_address);
    displacements[1] = offsetof(MpiRmaAddress, mpi_rank);
    MPI_Datatype types[2] = {MPI_AINT, MPI_INT};
    MPI_Type_create_struct(2, lengths, displacements, types, &mpi_rma_data_type_);
    MPI_Type_commit(&mpi_rma_data_type_);
}

//-------------------------------------------------------------------------------------------------------
// Class         :  CommMpiRma<DataClass>
// Public Method :  GetRemoteData
//
// Notes       :  1. Correctly call each sub-step of MPI RMA operation, and make sure every process has
//                   reach the same step before entering the next step.
//                2. If a step failed in any process, the other processes will know and fail fast.
//                   This is because some steps use collective MPI operations, which needs all processes
//                   to participate.
//                3. The returned struct contains status of current MPI process, status of all processes,
//                   and the fetched data if it has.
//                4. Fetched data are cached in class, so it only returns a const reference to the data.
//                5. This function is designed to be called many times by user.
//                6. The class uses template design pattern for other class to inherit and implement the
//                   GetDataLen/GetDataSize function for some specific data struct to pass around.
//                   It is agnostic to what the data is.
//                7. TODO: chunking data?
//-------------------------------------------------------------------------------------------------------
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
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::CheckAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));
        if (all_status != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }
        step = 1;

        status = PrepareData(prepared_data_list);
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::CheckAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));
        if (all_status != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }
        step = 2;

        status = GatherAllPreparedData(prepared_data_list);
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::CheckAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));
        if (all_status != CommMpiRmaStatus::kMpiSuccess) {
            break;
        }
        step = 3;

        status = FetchRemoteData(fetch_id_list);
        all_status = static_cast<CommMpiRmaStatus>(CommMpi::CheckAllStates(
            static_cast<int>(status), static_cast<int>(CommMpiRmaStatus::kMpiSuccess),
            static_cast<int>(CommMpiRmaStatus::kMpiSuccess), static_cast<int>(CommMpiRmaStatus::kMpiFailed)));
        step = 4;

        break;
    }

    if (step >= 1) {
        DetachBuffer(prepared_data_list);
        FreeMpiWindow();
    }
    CleanUp(prepared_data_list);

    return {.status = status, .all_status = static_cast<CommMpiRmaStatus>(all_status), .data_list = mpi_fetched_data_};
}

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRma<DataClass>
// Private Method :  InitializeMpiWindow
//
// Notes       :  1. Initialize one-sided MPI (RMA) window.
//                2. MPI_Win_create_dynamic is a collective operation;
//                   it must be called by all MPI processes in the intra-communicator.
//                   (ref: https://rookiehpc.org/mpi/docs/mpi_win_create_dynamic/index.html)
//-------------------------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRma<DataClass>
// Private Method :  PrepareData
//
// Notes       :  1. Faithfully attaching buffer to window for all the prepared data passed in, even if
//                   nullptr is passed in.
//                2. Does not check the validity of the data. Return error if it is unable to attach the
//                   buffer to the window.
//                3. Call GetDataSize to get the size of the data. The method is implemented by the
//                   derived class.
//-------------------------------------------------------------------------------------------------------
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
        // printf("Attach buffer (data_group, id) = (%s, %ld) to one-sided MPI (RMA) window on MPI rank %d\n",
        //        data_group_name_.c_str(), pdata.id, CommMpi::mpi_rank_);
    }

    return CommMpiRmaStatus::kMpiSuccess;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRma<DataClass>
// Private Method :  GatherAllPreparedData
//
// Notes       :  1. Gather all prepared data from all MPI ranks.
//                2. Use collective MPI operation, which requires all processes to participate.
//-------------------------------------------------------------------------------------------------------
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

    // Get all prepared data (TODO: why didn't I use AllGatherv?)
    all_prepared_data_list_ = new DataClass[total_send_counts];
    all_prepared_data_address_list_ = new MpiRmaAddress[total_send_counts];
    BigMpiGatherv<DataClass>(CommMpi::mpi_root_, all_send_counts, prepared_data_list.data(), &GetMpiDataType(),
                             all_prepared_data_list_);
    BigMpiGatherv<MpiRmaAddress>(CommMpi::mpi_root_, all_send_counts, mpi_prepared_data_address_list_.data(),
                                 &CommMpiRma::mpi_rma_data_type_, all_prepared_data_address_list_);
    big_MPI_Bcast<DataClass>(CommMpi::mpi_root_, total_send_counts, all_prepared_data_list_, &GetMpiDataType());
    big_MPI_Bcast<MpiRmaAddress>(CommMpi::mpi_root_, total_send_counts, all_prepared_data_address_list_,
                                 &CommMpiRma::mpi_rma_data_type_);

    // Clean up
    delete[] all_send_counts;
    mpi_prepared_data_address_list_.clear();

    return CommMpiRmaStatus::kMpiSuccess;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRma<DataClass>
// Private Method :  FetchRemoteData
//
// Notes       :  1. MPI_Win_fence is a collective operation, which requires all processes to participate.
//                2. Close the window epoch even if the fetch failed, since it is a collective operation.
//                3. Allocate new buffer and fetch/copy data from remote buffer to local buffer.
//                4. If fetch id contains nullptr, we don't need to fetch it; just get the data info and
//                   set the pointer to nullptr.
//                5. If unable to fetch data, or the data size/length is invalid, return error.
//                   If there is error, it would ignore the rest of the fetch ids.
//                6. Call GetDataLen/GetDataSize to get the length and size of the data. The method is
//                   implemented by the derived class.
//-------------------------------------------------------------------------------------------------------
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

    // Detach what is attached in PrepareData
    for (std::size_t i = 0; i < mpi_prepared_data_address_list_.size(); i++) {
        MPI_Win_detach(mpi_window_, prepared_data_list[i].data_ptr);
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
MPI_Datatype CommMpiRmaAmrDataArray3D::mpi_data_type_ = nullptr;
MPI_Datatype CommMpiRmaAmrDataArray1D::mpi_data_type_ = nullptr;

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRmaAmrDataArray3D
// Private Method :  GetDataSize
//
// Notes          :  1. The method is used in PrepareData and FetchRemoteData to get the size of the data.
//                   2. For invalid data, return value < 0.
//-------------------------------------------------------------------------------------------------------
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

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRmaAmrDataArray3D
// Private Method :  GetDataLen
//
// Notes          :  1. The method is used in FetchRemoteData to get the length of the data.
//                   2. For invalid data, return value < 0.
//-------------------------------------------------------------------------------------------------------
long CommMpiRmaAmrDataArray3D::GetDataLen(const AmrDataArray3D& data) {
    for (int i = 0; i < 3; i++) {
        if (data.data_dim[i] < 0) {
            return -1;
        }
    }
    return data.data_dim[0] * data.data_dim[1] * data.data_dim[2];
}

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRmaAmrDataArray3D
// Public Method  :  Constructor
//
// Notes          :  1. Also initialize custom mpi data type.
//-------------------------------------------------------------------------------------------------------
CommMpiRmaAmrDataArray3D::CommMpiRmaAmrDataArray3D(const std::string& data_group_name, const std::string& data_format)
    : CommMpiRma<AmrDataArray3D>(data_group_name, data_format) {
    InitializeMpiDataType();
}

//-------------------------------------------------------------------------------------------------------
// Class                 :  CommMpiRmaAmrDataArray3D
// Private Static Method :  InitializeMpiDataType
//
// Notes          :  1. Initialize custom mpi data type for AmrDataArray3D.
//-------------------------------------------------------------------------------------------------------
void CommMpiRmaAmrDataArray3D::InitializeMpiDataType() {
    if (mpi_data_type_ != nullptr) {
        return;
    }

    int lengths[5] = {1, 1, 3, 1, 1};
    MPI_Aint displacements[5];
    displacements[0] = offsetof(AmrDataArray3D, id);
    displacements[1] = offsetof(AmrDataArray3D, data_dtype);
    displacements[2] = offsetof(AmrDataArray3D, data_dim);
    displacements[3] = offsetof(AmrDataArray3D, data_ptr);
    displacements[4] = offsetof(AmrDataArray3D, contiguous_in_x);
    MPI_Datatype types[5] = {MPI_LONG, MPI_INT, MPI_INT, MPI_AINT, MPI_CXX_BOOL};
    MPI_Type_create_struct(5, lengths, displacements, types, &mpi_data_type_);
    MPI_Type_commit(&mpi_data_type_);
}

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRmaAmrDataArray1D
// Private Method :  GetDataSize
//
// Notes          :  1. The method is used in PrepareData and FetchRemoteData to get the size of the data.
//                   2. For invalid data, return value < 0.
//-------------------------------------------------------------------------------------------------------
long CommMpiRmaAmrDataArray1D::GetDataSize(const AmrDataArray1D& data) {
    if (data.data_len < 0 || data.data_dtype == YT_DTYPE_UNKNOWN) {
        return -1;
    }

    int dtype_size;
    get_dtype_size(data.data_dtype, &dtype_size);
    return data.data_len * dtype_size;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRmaAmrDataArray1D
// Private Method :  GetDataLen
//
// Notes          :  1. The method is used in FetchRemoteData to get the length of the data.
//                   2. For invalid data, return value < 0.
//-------------------------------------------------------------------------------------------------------
long CommMpiRmaAmrDataArray1D::GetDataLen(const AmrDataArray1D& data) {
    if (data.data_len < 0) {
        return -1;
    }
    return data.data_len;
}

//-------------------------------------------------------------------------------------------------------
// Class          :  CommMpiRmaAmrDataArray1D
// Public Method  :  Constructor
//
// Notes          :  1. Also initialize custom mpi data type.
//-------------------------------------------------------------------------------------------------------
CommMpiRmaAmrDataArray1D::CommMpiRmaAmrDataArray1D(const std::string& data_group_name, const std::string& data_format)
    : CommMpiRma<AmrDataArray1D>(data_group_name, data_format) {
    InitializeMpiDataType();
}

//-------------------------------------------------------------------------------------------------------
// Class                 :  CommMpiRmaAmrDataArray1D
// Private Static Method :  InitializeMpiDataType
//
// Notes          :  1. Initialize custom mpi data type for AmrDataArray1D.
//-------------------------------------------------------------------------------------------------------
void CommMpiRmaAmrDataArray1D::InitializeMpiDataType() {
    if (mpi_data_type_ != nullptr) {
        return;
    }

    int lengths[4] = {1, 1, 1, 1};
    MPI_Aint displacements[4];
    displacements[0] = offsetof(AmrDataArray1D, id);
    displacements[1] = offsetof(AmrDataArray1D, data_dtype);
    displacements[2] = offsetof(AmrDataArray1D, data_ptr);
    displacements[3] = offsetof(AmrDataArray1D, data_len);
    MPI_Datatype types[4] = {MPI_LONG, MPI_INT, MPI_AINT, MPI_LONG};
    MPI_Type_create_struct(4, lengths, displacements, types, &mpi_data_type_);
    MPI_Type_commit(&mpi_data_type_);
}

#endif
