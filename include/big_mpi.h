#ifndef LIBYT_PROJECT_INCLUDE_BIG_MPI_H_
#define LIBYT_PROJECT_INCLUDE_BIG_MPI_H_

#ifndef SERIAL_MODE

#include <limits.h>
#include <mpi.h>

#include "timer.h"
#include "yt_macro.h"

enum class BigMpiStatus : int { kBigMpiFailed = 0, kBigMpiExceedCounts = 1, kBigMpiSuccess = 2 };

//-------------------------------------------------------------------------------------------------------
// Template    :  BigMpiAllgatherv
// Description :  This is a workaround method for passing big send count of MPI_Allgatherv.
//-------------------------------------------------------------------------------------------------------
template<typename T>
BigMpiStatus BigMpiAllgatherv(int* send_counts, const void* send_buffer, MPI_Datatype mpi_datatype, void* recv_buffer) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int mpi_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Count recv_counts, offsets, and split the buffer, if too large.
    int* recv_counts = new int[mpi_size];
    int* offsets = new int[mpi_size];
    int mpi_start = 0;
    long index_start = 0;
    long accumulate = 0;

    // Workaround method for passing big send count.
    for (int i = 0; i < mpi_size; i++) {
        offsets[i] = 0;
        accumulate = 0;
        for (int j = mpi_start; j < i; j++) {
            offsets[i] += send_counts[j];
            accumulate += send_counts[j];
        }
        // exceeding INT_MAX, start MPI_Allgatherv
        if (accumulate > INT_MAX) {
            // Set recv_counts and offsets.
            for (int k = 0; k < mpi_size; k++) {
                if (mpi_start <= k && k < i) {
                    recv_counts[k] = send_counts[k];
                } else {
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Allgatherv
            if (mpi_start <= mpi_rank && mpi_rank < i) {
                MPI_Allgatherv(send_buffer, send_counts[mpi_rank], mpi_datatype, &(((T*)recv_buffer)[index_start]),
                               recv_counts, offsets, mpi_datatype, MPI_COMM_WORLD);
            } else {
                MPI_Allgatherv(send_buffer, 0, mpi_datatype, &(((T*)recv_buffer)[index_start]), recv_counts, offsets,
                               mpi_datatype, MPI_COMM_WORLD);
            }

            // New start point.
            mpi_start = i;
            offsets[mpi_start] = 0;
            index_start = 0;
            for (int k = 0; k < i; k++) {
                index_start += send_counts[k];
            }
        }
        // Reach last mpi rank, MPI_Gatherv
        // We can ignore the case when there is only one rank left and its offsets exceeds INT_MAX simultaneously.
        // Because one is type int, the other is type long.
        else if (i == mpi_size - 1) {
            // Set recv_counts and offsets.
            for (int k = 0; k < mpi_size; k++) {
                if (mpi_start <= k && k <= i) {
                    recv_counts[k] = send_counts[k];
                } else {
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Allgatherv
            if (mpi_start <= mpi_rank && mpi_rank <= i) {
                MPI_Allgatherv(send_buffer, send_counts[mpi_rank], mpi_datatype, &(((T*)recv_buffer)[index_start]),
                               recv_counts, offsets, mpi_datatype, MPI_COMM_WORLD);
            } else {
                MPI_Allgatherv(send_buffer, 0, mpi_datatype, &(((T*)recv_buffer)[index_start]), recv_counts, offsets,
                               mpi_datatype, MPI_COMM_WORLD);
            }
        }
    }

    // Free resource
    delete[] recv_counts;
    delete[] offsets;

    return BigMpiStatus::kBigMpiSuccess;
}

//-------------------------------------------------------------------------------------------------------
// Template    :  BigMpiGatherv
// Description :  This is a workaround method for passing big send count of MPI_Gatherv.
//-------------------------------------------------------------------------------------------------------
template<typename T>
BigMpiStatus BigMpiGatherv(int root_rank, int* send_counts, const void* send_buffer, MPI_Datatype* mpi_datatype,
                           void* recv_buffer) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int mpi_size, mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Count recv_counts, offsets, and split the buffer, if too large.
    int* recv_counts = new int[mpi_size];
    int* offsets = new int[mpi_size];
    int mpi_start = 0;
    long index_start = 0;
    long accumulate = 0;

    // Workaround method for passing big send count.
    for (int i = 0; i < mpi_size; i++) {
        offsets[i] = 0;
        accumulate = 0;
        for (int j = mpi_start; j < i; j++) {
            offsets[i] += send_counts[j];
            accumulate += send_counts[j];
        }
        // exceeding INT_MAX, start MPI_Gatherv
        if (accumulate > INT_MAX) {
            // Set recv_counts and offsets.
            for (int k = 0; k < mpi_size; k++) {
                if (mpi_start <= k && k < i) {
                    recv_counts[k] = send_counts[k];
                } else {
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Gatherv
            if (mpi_start <= mpi_rank && mpi_rank < i) {
                MPI_Gatherv(send_buffer, send_counts[mpi_rank], *mpi_datatype, &(((T*)recv_buffer)[index_start]),
                            recv_counts, offsets, *mpi_datatype, root_rank, MPI_COMM_WORLD);
            } else {
                MPI_Gatherv(send_buffer, 0, *mpi_datatype, &(((T*)recv_buffer)[index_start]), recv_counts, offsets,
                            *mpi_datatype, root_rank, MPI_COMM_WORLD);
            }

            // New start point.
            mpi_start = i;
            offsets[mpi_start] = 0;
            index_start = 0;
            for (int k = 0; k < i; k++) {
                index_start += send_counts[k];
            }
        }
        // Reach last mpi rank, MPI_Gatherv
        // We can ignore the case when there is only one rank left and its offsets exceeds INT_MAX simultaneously.
        // Because one is type int, the other is type long.
        else if (i == mpi_size - 1) {
            // Set recv_counts and offsets.
            for (int k = 0; k < mpi_size; k++) {
                if (mpi_start <= k && k <= i) {
                    recv_counts[k] = send_counts[k];
                } else {
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Gatherv
            if (mpi_start <= mpi_rank && mpi_rank <= i) {
                MPI_Gatherv(send_buffer, send_counts[mpi_rank], *mpi_datatype, &(((T*)recv_buffer)[index_start]),
                            recv_counts, offsets, *mpi_datatype, root_rank, MPI_COMM_WORLD);
            } else {
                MPI_Gatherv(send_buffer, 0, *mpi_datatype, &(((T*)recv_buffer)[index_start]), recv_counts, offsets,
                            *mpi_datatype, root_rank, MPI_COMM_WORLD);
            }
        }
    }

    // Free resource
    delete[] recv_counts;
    delete[] offsets;

    return BigMpiStatus::kBigMpiSuccess;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  BigMpiBcast
// Description :  This is a workaround method for passing big send count of MPI_Bcast.
//-------------------------------------------------------------------------------------------------------
template<typename T>
BigMpiStatus BigMpiBcast(int root_rank, long send_count, void* buffer, MPI_Datatype* mpi_datatype) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // The maximum MPI_Bcast send count is INT_MAX.
    // If num_grids > INT_MAX chop it to chunks, then broadcast.
    long stride = INT_MAX;
    int part = (int)(send_count / stride) + 1;
    int remain = (int)(send_count % stride);
    long index;
    for (int i = 0; i < part; i++) {
        index = i * stride;
        if (i == part - 1) {
            MPI_Bcast(&(((T*)buffer)[index]), remain, *mpi_datatype, root_rank, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(&(((T*)buffer)[index]), (int)stride, *mpi_datatype, root_rank, MPI_COMM_WORLD);
        }
    }

    return BigMpiStatus::kBigMpiSuccess;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  big_MPI_Get
// Description :  This is a workaround method for passing big send count of MPI_Get.
//
// Note        :  1. big_MPI_Get_dtype delegates calls to here.
//
// Parameter   :  void         *recv_buff   : Store received buffer.
//                long          data_len    : Total length of the data.
//                MPI_Datatype *mpi_dtype   : MPI_Datatype.
//                int           get_rank    : Rank to get.
//                MPI_Aint      base_address: Address of the first element of the target buffer.
//                MPI_Win      *window      : Window.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template<typename T>
int big_MPI_Get(void* recv_buff, long data_len, MPI_Datatype* mpi_dtype, int get_rank, MPI_Aint base_address,
                MPI_Win* window) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // The maximum sendcount of MPI_Get is INT_MAX.
    long stride = INT_MAX;
    int part = (int)(data_len / stride) + 1;
    int remain = (int)(data_len % stride);
    long index;

    // Get size of the data, and get the displacement address.
    int size = sizeof(T);
    MPI_Aint address = base_address;

    // Split to many time if data_len > INT_MAX
    for (int i = 0; i < part; i++) {
        index = i * stride;
        if (i == part - 1) {
            MPI_Get(&(((T*)recv_buff)[index]), remain, *mpi_dtype, get_rank, address, remain, *mpi_dtype, *window);
        } else {
            MPI_Get(&(((T*)recv_buff)[index]), stride, *mpi_dtype, get_rank, address, stride, *mpi_dtype, *window);
        }
        address += stride * size;
    }

    return YT_SUCCESS;
}

#endif  // #ifndef SERIAL_MODE

#endif  // LIBYT_PROJECT_INCLUDE_BIG_MPI_H_
