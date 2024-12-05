#ifndef __BIG_MPI_H__
#define __BIG_MPI_H__

#ifndef SERIAL_MODE

#include <mpi.h>

#include "timer.h"
#include "yt_macro.h"

//-------------------------------------------------------------------------------------------------------
// Template    :  big_MPI_Gatherv
// Description :  This is a workaround method for passing big send count of MPI_Gatherv.
//
// Note        :  1. Use inside yt_commit(), yt_rma_field, yt_rma_particle.
//
// Parameter   :  int            RootRank     : Root rank.
//                int           *sendcounts   : Send counts in each rank.
//                void          *sendbuffer   : Buffer to send.
//                MPI_Datatype  *mpi_datatype : MPI datatype, can be user defined or MPI defined one.
//                void          *recvbuffer   : Store the received buffer.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template<typename T>
int big_MPI_Gatherv(int RootRank, int* sendcounts, const void* sendbuffer, MPI_Datatype* mpi_datatype,
                    void* recvbuffer) {
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

    // Workaround method for passing big sendcount.
    for (int i = 0; i < mpi_size; i++) {
        offsets[i] = 0;
        accumulate = 0;
        for (int j = mpi_start; j < i; j++) {
            offsets[i] += sendcounts[j];
            accumulate += sendcounts[j];
        }
        // exceeding INT_MAX, start MPI_Gatherv
        if (accumulate > INT_MAX) {
            // Set recv_counts and offsets.
            for (int k = 0; k < mpi_size; k++) {
                if (mpi_start <= k && k < i) {
                    recv_counts[k] = sendcounts[k];
                } else {
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Gatherv
            if (mpi_start <= mpi_rank && mpi_rank < i) {
                MPI_Gatherv(sendbuffer, sendcounts[mpi_rank], *mpi_datatype, &(((T*)recvbuffer)[index_start]),
                            recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            } else {
                MPI_Gatherv(sendbuffer, 0, *mpi_datatype, &(((T*)recvbuffer)[index_start]), recv_counts, offsets,
                            *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }

            // New start point.
            mpi_start = i;
            offsets[mpi_start] = 0;
            index_start = 0;
            for (int k = 0; k < i; k++) {
                index_start += sendcounts[k];
            }
        }
        // Reach last mpi rank, MPI_Gatherv
        // We can ignore the case when there is only one rank left and its offsets exceeds INT_MAX simultaneously.
        // Because one is type int, the other is type long.
        else if (i == mpi_size - 1) {
            // Set recv_counts and offsets.
            for (int k = 0; k < mpi_size; k++) {
                if (mpi_start <= k && k <= i) {
                    recv_counts[k] = sendcounts[k];
                } else {
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Gatherv
            if (mpi_start <= mpi_rank && mpi_rank <= i) {
                MPI_Gatherv(sendbuffer, sendcounts[mpi_rank], *mpi_datatype, &(((T*)recvbuffer)[index_start]),
                            recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            } else {
                MPI_Gatherv(sendbuffer, 0, *mpi_datatype, &(((T*)recvbuffer)[index_start]), recv_counts, offsets,
                            *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
        }
    }

    // Free resource
    delete[] recv_counts;
    delete[] offsets;

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  big_MPI_Bcast
// Description :  This is a workaround method for passing big send count of MPI_Bcast.
//
// Note        :  1. Use inside yt_commit(), yt_rma_field, yt_rma_particle.
//
// Parameter   :  int            RootRank     : Root rank.
//                long           sendcount    : Send count.
//                void          *buffer       : Buffer to broadcast.
//                MPI_Datatype  *mpi_datatype : MPI datatype, can be user defined or MPI defined one.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
template<typename T>
int big_MPI_Bcast(int RootRank, long sendcount, void* buffer, MPI_Datatype* mpi_datatype) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // The maximum MPI_Bcast sendcount is INT_MAX.
    // If num_grids > INT_MAX chop it to chunks, then broadcast.
    long stride = INT_MAX;
    int part = (int)(sendcount / stride) + 1;
    int remain = (int)(sendcount % stride);
    long index;
    for (int i = 0; i < part; i++) {
        index = i * stride;
        if (i == part - 1) {
            MPI_Bcast(&(((T*)buffer)[index]), remain, *mpi_datatype, RootRank, MPI_COMM_WORLD);
        } else {
            MPI_Bcast(&(((T*)buffer)[index]), (int)stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
        }
    }

    return YT_SUCCESS;
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

#endif  // __BIG_MPI_H__
