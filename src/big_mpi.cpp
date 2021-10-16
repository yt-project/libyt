#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  big_MPI_Gatherv
// Description :  This is a workaround method for passing big send count of MPI_Gatherv.
//
// Note        :  1. Use inside yt_commit_grids().
//                2. Because void* pointer has no arithmetic, cast_type indicates what C type or struct
//                   to cast to.
//                   cast_type             type
//                   ===================================
//                       0          yt_hierarchy
//                       1          yt_rma_grid_info
//                       2          yt_rma_particle_info
//
// Parameter   :  int            RootRank     : Root rank.
//                int           *sendcounts   : Send counts in each rank.
//                void          *sendbuffer   : Buffer to send.
//                MPI_Datatype  *mpi_datatype : MPI datatype, can be user defined or MPI defined one.
//                void          *recvbuffer   : Store the received buffer.
//                int            cast_type    : What data type to cast the void* ptr to.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int big_MPI_Gatherv(int RootRank, int *sendcounts, void *sendbuffer, MPI_Datatype *mpi_datatype, void *recvbuffer, int cast_type)
{
    // Get NRank, MyRank
    int MyRank;
    int NRank;
    MPI_Comm_size(MPI_COMM_WORLD, &NRank);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

    // Count recv_counts, offsets, and split the buffer, if too large.
    int *recv_counts = new int [NRank];
    int *offsets = new int [NRank];
    int mpi_start = 0;
    long index_start = 0;
    long accumulate = 0;

    // Workaround method for passing big sendcount.
    for (int i = 0; i < NRank; i++){
        offsets[i] = 0;
        accumulate = 0;
        for (int j = mpi_start; j < i; j++){
            offsets[i] += sendcounts[j];
            accumulate += sendcounts[j];
        }
        // exceeding INT_MAX, start MPI_Gatherv
        if ( accumulate > INT_MAX ){
            // Set recv_counts and offsets.
            for (int k = 0; k < NRank; k++){
                if ( mpi_start <= k && k < i ){
                    recv_counts[k] = sendcounts[k];
                }
                else{
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Gatherv
            if(cast_type == 0){
                if ( mpi_start <= MyRank && MyRank < i ){
                    MPI_Gatherv(sendbuffer, sendcounts[MyRank], *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 1){
                if ( mpi_start <= MyRank && MyRank < i ){
                    MPI_Gatherv(sendbuffer, sendcounts[MyRank], *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 2){
                if ( mpi_start <= MyRank && MyRank < i ){
                    MPI_Gatherv(sendbuffer, sendcounts[MyRank], *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }

            // New start point.
            mpi_start = i;
            offsets[mpi_start] = 0;
            index_start = 0;
            for (int k = 0; k < i; k++){
                index_start += sendcounts[k];
            }
        }
            // Reach last mpi rank, MPI_Gatherv
            // We can ignore the case when there is only one rank left and its offsets exceeds INT_MAX simultaneously.
            // Because one is type int, the other is type long.
        else if ( i == NRank - 1 ){
            // Set recv_counts and offsets.
            for (int k = 0; k < NRank; k++){
                if ( mpi_start <= k && k <= i ){
                    recv_counts[k] = sendcounts[k];
                }
                else{
                    offsets[k] = 0;
                    recv_counts[k] = 0;
                }
            }
            // MPI_Gatherv
            if(cast_type == 0){
                if ( mpi_start <= MyRank && MyRank <= i ){
                    MPI_Gatherv(sendbuffer, sendcounts[MyRank], *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 1){
                if ( mpi_start <= MyRank && MyRank <= i ){
                    MPI_Gatherv(sendbuffer, sendcounts[MyRank], *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 2){
                if ( mpi_start <= MyRank && MyRank <= i ){
                    MPI_Gatherv(sendbuffer, sendcounts[MyRank], *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }

        }
    }

    // Free resource
    delete [] recv_counts;
    delete [] offsets;

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  big_MPI_Bcast
// Description :  This is a workaround method for passing big send count of MPI_Bcast.
//
// Note        :  1. Use inside yt_commit_grids().
//                2. Because void* pointer has no arithmetic, cast_type indicates what C type or struct
//                   to cast to.
//                   cast_type             type
//                   ===================================
//                       0          yt_hierarchy
//                       1          yt_rma_grid_info
//                       2          yt_rma_particle_info
//
// Parameter   :  int            RootRank     : Root rank.
//                long           sendcount    : Send count.
//                void          *buffer       : Buffer to broadcast.
//                MPI_Datatype  *mpi_datatype : MPI datatype, can be user defined or MPI defined one.
//                int            cast_type    : What data type to cast the void* ptr to.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int big_MPI_Bcast(int RootRank, long sendcount, void *buffer, MPI_Datatype *mpi_datatype, int cast_type)
{
    // The maximum MPI_Bcast sendcount is INT_MAX.
    // If num_grids > INT_MAX chop it to chunks, then broadcast.
    long stride   = INT_MAX;
    int  part     = (int) (sendcount / stride) + 1;
    int  remain   = (int) (sendcount % stride);
    long index;
    if( cast_type == 0 ){
        for (int i=0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Bcast(&(((yt_hierarchy*)buffer)[index]), remain, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
            else {
                MPI_Bcast(&(((yt_hierarchy*)buffer)[index]), stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
        }
    }
    else if( cast_type == 1 ){
        for (int i=0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Bcast(&(((yt_rma_grid_info*)buffer)[index]), remain, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
            else {
                MPI_Bcast(&(((yt_rma_grid_info*)buffer)[index]), stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
        }
    }
    else if( cast_type == 2 ){
        for (int i=0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Bcast(&(((yt_rma_particle_info*)buffer)[index]), remain, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
            else {
                MPI_Bcast(&(((yt_rma_particle_info*)buffer)[index]), stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
        }
    }

    return YT_SUCCESS;
}