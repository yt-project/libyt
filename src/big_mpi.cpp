#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  big_MPI_Gatherv
// Description :  This is a workaround method for passing big send count of MPI_Gatherv.
//
// Note        :  1. Use inside yt_commit(), yt_rma_field, yt_rma_particle.
//                2. Because void* pointer has no arithmetic, cast_type indicates what C type or struct
//                   to cast to.
//                   cast_type             type
//                   ===================================
//                       0          yt_hierarchy
//                       1          yt_rma_grid_info
//                       2          yt_rma_particle_info
//                       3          long
//                TODO: Using function templates
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
    // Count recv_counts, offsets, and split the buffer, if too large.
    int *recv_counts = new int [g_mysize];
    int *offsets = new int [g_mysize];
    int mpi_start = 0;
    long index_start = 0;
    long accumulate = 0;

    // Workaround method for passing big sendcount.
    for (int i = 0; i < g_mysize; i++){
        offsets[i] = 0;
        accumulate = 0;
        for (int j = mpi_start; j < i; j++){
            offsets[i] += sendcounts[j];
            accumulate += sendcounts[j];
        }
        // exceeding INT_MAX, start MPI_Gatherv
        if ( accumulate > INT_MAX ){
            // Set recv_counts and offsets.
            for (int k = 0; k < g_mysize; k++){
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
                if ( mpi_start <= g_myrank && g_myrank < i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 1){
                if ( mpi_start <= g_myrank && g_myrank < i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 2){
                if ( mpi_start <= g_myrank && g_myrank < i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 3){
                if ( mpi_start <= g_myrank && g_myrank < i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((long*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((long*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
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
        else if ( i == g_mysize - 1 ){
            // Set recv_counts and offsets.
            for (int k = 0; k < g_mysize; k++){
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
                if ( mpi_start <= g_myrank && g_myrank <= i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_hierarchy*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 1){
                if ( mpi_start <= g_myrank && g_myrank <= i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_grid_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 2){
                if ( mpi_start <= g_myrank && g_myrank <= i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((yt_rma_particle_info*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
            }
            else if(cast_type == 3){
                if ( mpi_start <= g_myrank && g_myrank <= i ){
                    MPI_Gatherv(sendbuffer, sendcounts[g_myrank], *mpi_datatype,
                                &(((long*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
                }
                else{
                    MPI_Gatherv(sendbuffer,                  0, *mpi_datatype,
                                &(((long*)recvbuffer)[index_start]), recv_counts, offsets, *mpi_datatype, RootRank, MPI_COMM_WORLD);
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
// Note        :  1. Use inside yt_commit(), yt_rma_field, yt_rma_particle.
//                2. Because void* pointer has no arithmetic, cast_type indicates what C type or struct
//                   to cast to.
//                   cast_type             type
//                   ===================================
//                       0          yt_hierarchy
//                       1          yt_rma_grid_info
//                       2          yt_rma_particle_info
//                       3          long
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
                MPI_Bcast(&(((yt_hierarchy*)buffer)[index]), (int)stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
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
                MPI_Bcast(&(((yt_rma_grid_info*)buffer)[index]), (int)stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
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
                MPI_Bcast(&(((yt_rma_particle_info*)buffer)[index]), (int)stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
        }
    }
    else if( cast_type == 3 ){
        for (int i=0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Bcast(&(((long*)buffer)[index]), remain, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
            else {
                MPI_Bcast(&(((long*)buffer)[index]), (int)stride, *mpi_datatype, RootRank, MPI_COMM_WORLD);
            }
        }
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  big_MPI_Get
// Description :  This is a workaround method for passing big send count of MPI_Get.
//
// Note        :  1. Use inside .
//                2. Because void* pointer has no arithmetic, cast_type indicates what C type or struct
//                   to cast to.
//                     yt_dtype       Cast C Type
//                   ===================================
//                    YT_FLOAT          float
//                    YT_DOUBLE         double
//                    YT_LONGDOUBLE     long double
//                    YT_INT            int
//                    YT_LONG           long
//
// Parameter   :  void         *recv_buff   : Store received buffer.
//                long          data_len    : Total length of the data.
//                yt_dtype     *data_dtype  : Data type.
//                MPI_Datatype *mpi_dtype   : MPI_Datatype.
//                int           get_rank    : Rank to get.
//                MPI_Aint      base_address: Address of the first element of the target buffer.
//                MPI_Win      *window      : Window.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int big_MPI_Get(void *recv_buff, long data_len, yt_dtype *data_dtype, MPI_Datatype *mpi_dtype, int get_rank,
                MPI_Aint base_address, MPI_Win *window)
{
    // The maximum sendcount of MPI_Get is INT_MAX.
    long stride   = INT_MAX;
    int  part     = (int) (data_len / stride) + 1;
    int  remain   = (int) (data_len % stride);
    long index;

    // Get size of the data, and get the displacement address.
    int size;
    get_dtype_size(*data_dtype, &size);
    MPI_Aint address = base_address;

    if( *data_dtype == YT_FLOAT ){
        // Split to many time if data_len > INT_MAX
        for (int i = 0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Get(&(((float*)recv_buff)[index]), remain, *mpi_dtype, get_rank, address, remain, *mpi_dtype, *window);
            }
            else {
                MPI_Get(&(((float*)recv_buff)[index]), stride, *mpi_dtype, get_rank, address, stride, *mpi_dtype, *window);
            }
            address += stride * size;
        }
    }
    else if( *data_dtype == YT_DOUBLE ){
        // Split to many time if data_len > INT_MAX
        for (int i = 0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Get(&(((double*)recv_buff)[index]), remain, *mpi_dtype, get_rank, address, remain, *mpi_dtype, *window);
            }
            else {
                MPI_Get(&(((double*)recv_buff)[index]), stride, *mpi_dtype, get_rank, address, stride, *mpi_dtype, *window);
            }
            address += stride * size;
        }
    }
    else if( *data_dtype == YT_LONGDOUBLE ){
        // Split to many time if data_len > INT_MAX
        for (int i = 0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Get(&(((long double*)recv_buff)[index]), remain, *mpi_dtype, get_rank, address, remain, *mpi_dtype, *window);
            }
            else {
                MPI_Get(&(((long double*)recv_buff)[index]), stride, *mpi_dtype, get_rank, address, stride, *mpi_dtype, *window);
            }
            address += stride * size;
        }
    }
    else if( *data_dtype == YT_INT ){
        // Split to many time if data_len > INT_MAX
        for (int i = 0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Get(&(((int*)recv_buff)[index]), remain, *mpi_dtype, get_rank, address, remain, *mpi_dtype, *window);
            }
            else {
                MPI_Get(&(((int*)recv_buff)[index]), stride, *mpi_dtype, get_rank, address, stride, *mpi_dtype, *window);
            }
            address += stride * size;
        }
    }
    else if( *data_dtype == YT_LONG ){
        // Split to many time if data_len > INT_MAX
        for (int i = 0; i < part; i++){
            index = i * stride;
            if ( i == part - 1 ){
                MPI_Get(&(((long*)recv_buff)[index]), remain, *mpi_dtype, get_rank, address, remain, *mpi_dtype, *window);
            }
            else {
                MPI_Get(&(((long*)recv_buff)[index]), stride, *mpi_dtype, get_rank, address, stride, *mpi_dtype, *window);
            }
            address += stride * size;
        }
    }
    else{
        // Safety check that data_dtype is one of yt_dtype,
        // so that if we cannot match a C type, then it must be user forgot to implement here.
        bool valid = false;
        for ( int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++ ){
            yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
            if ( *data_dtype == dtype ){
                valid = true;
                break;
            }
        }
        if ( valid == true ){
            log_error("You should also match your new yt_dtype to C type in big_MPI_Get function.\n");
        }
        else{
            log_error("If you want to implement your new yt_dtype, you should modify both yt_dtype Enum and big_MPI_Get function.\n");
        }

        return YT_FAIL;
    }

    return YT_SUCCESS;
}