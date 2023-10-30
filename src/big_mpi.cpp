#ifndef SERIAL_MODE

#include "yt_combo.h"
#include "libyt.h"


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
    SET_TIMER(__PRETTY_FUNCTION__);

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

#endif // #ifndef SERIAL_MODE
