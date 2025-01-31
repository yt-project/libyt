#include <cstring>

#include "big_mpi.h"
#include "libyt.h"
#include "numpy_controller.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#include "pybind11/numpy.h"
#endif

#ifndef SERIAL_MODE
//-------------------------------------------------------------------------------------------------------
// Function    :  big_MPI_Get_dtype
// Description :  wrapper of big_MPI_Get
//
// Note        :  1. This function delegates calling big_MPI_Get.
//                2. This function looks totally irrelevant to yt_dtype, but in order to put all yt_dtype
//                   relevant stuff in a file only, I choose to put it here.
//                3.   yt_dtype            C Type
//                  ========================================
//                     YT_FLOAT            float
//                     YT_DOUBLE           double
//                     YT_LONGDOUBLE       long double
//                     YT_INT              int
//                     YT_LONG             long
//                4. TODO: maybe I can move this to a local function in comm_mpi_rma.cpp
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                length    : length of the 1d array
//                data_ptr  : data pointer to allocated buffer
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int big_MPI_Get_dtype(void* recv_buff, long data_len, yt_dtype* data_dtype, MPI_Datatype* mpi_dtype, int get_rank,
                      MPI_Aint base_address, MPI_Win* window) {
    switch (*data_dtype) {
        case YT_FLOAT:
            return big_MPI_Get<float>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_DOUBLE:
            return big_MPI_Get<double>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_LONGDOUBLE:
            return big_MPI_Get<long double>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_CHAR:
            return big_MPI_Get<char>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_UCHAR:
            return big_MPI_Get<unsigned char>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_SHORT:
            return big_MPI_Get<short>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_USHORT:
            return big_MPI_Get<unsigned short>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_INT:
            return big_MPI_Get<int>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_UINT:
            return big_MPI_Get<unsigned int>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_LONG:
            return big_MPI_Get<long>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_ULONG:
            return big_MPI_Get<unsigned long>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_LONGLONG:
            return big_MPI_Get<long long>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_ULONGLONG:
            return big_MPI_Get<unsigned long long>(recv_buff, data_len, mpi_dtype, get_rank, base_address, window);
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
            bool valid = false;
            for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
                yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
                if (*data_dtype == dtype) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                log_error("Forget to delegate new yt_dtype to big_MPI_Get in big_MPI_Get_dtype function.\n");
            } else {
                log_error("No such yt_dtype.\n");
            }
            return YT_FAIL;
    }
}
#endif

#ifdef USE_PYBIND11

#endif