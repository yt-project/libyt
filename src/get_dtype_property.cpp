#include "yt_combo.h"
#include "libyt.h"
#include <cstring>

//-------------------------------------------------------------------------------------------------------
// Function    :  get_npy_dtype
// Description :  Match from yt_dtype YT_* to NumPy enumerate type.
//
// Note        :  1. This function matches yt_dtype to NumPy enumerate type, and will write result in 
//                   npy_dtype.
//                2.   yt_dtype      NumPy Enumerate Type
//                  ========================================
//                     YT_FLOAT            NPY_FLOAT
//                     YT_DOUBLE           NPY_DOUBLE
//                     YT_LONGDOUBLE       NPY_LONGDOUBLE
//                     YT_INT              NPY_INT
//                     YT_LONG             NPY_LONG
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                npy_dtype : NumPy enumerate data type.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_npy_dtype( yt_dtype data_type, int *npy_dtype ){
    SET_TIMER(__PRETTY_FUNCTION__);

    switch (data_type) {
        case YT_FLOAT:
            *npy_dtype = NPY_FLOAT;
            return YT_SUCCESS;
        case YT_DOUBLE:
            *npy_dtype = NPY_DOUBLE;
            return YT_SUCCESS;
        case YT_LONGDOUBLE:
            *npy_dtype = NPY_LONGDOUBLE;
            return YT_SUCCESS;
        case YT_INT:
            *npy_dtype = NPY_INT;
            return YT_SUCCESS;
        case YT_LONG:
            *npy_dtype = NPY_LONG;
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
            // Safety check that data_type is one of yt_dtype,
            // so that if we cannot match a NumPy Enum Type, then it must be user forgot to implement here.
            bool valid = false;
            for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
                yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
                if (data_type == dtype) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                log_error("Forget to match new yt_dtype to NumPy enumerate type in get_npy_dtype function.\n");
            }
            else {
                log_error("No such yt_dtype.\n");
            }

            *npy_dtype = -1;
            return YT_FAIL;
    }
}


//-------------------------------------------------------------------------------------------------------
// Function    :  get_yt_dtype_from_npy
// Description :  Match from NumPy enumerate type to yt_dtype
//
// Note        :  1. This function matches NumPy enumerate type to yt_dtype, and will write result in
//                   data_dtype.
//
// Parameter   :  npy_dtype : NumPy enumerate data type.
//                data_type : yt_dtype, YT_*.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int  get_yt_dtype_from_npy(int npy_dtype, yt_dtype *data_dtype ) {
    SET_TIMER(__PRETTY_FUNCTION__ );

    switch (npy_dtype) {
        case NPY_FLOAT:
            *data_dtype = YT_FLOAT;
            return YT_SUCCESS;
        case NPY_DOUBLE:
            *data_dtype = YT_DOUBLE;
            return YT_SUCCESS;
        case NPY_LONGDOUBLE:
            *data_dtype = YT_LONGDOUBLE;
            return YT_SUCCESS;
        case NPY_INT:
            *data_dtype = YT_INT;
            return YT_SUCCESS;
        case NPY_LONG:
            *data_dtype = YT_LONG;
            return YT_SUCCESS;
        default:
            log_error("No matching yt_dtype for NumPy data type num [%d] in get_yt_dtype_from_npy function.\n",
                      npy_dtype);
            return YT_FAIL;
    }
}


#ifndef SERIAL_MODE
//-------------------------------------------------------------------------------------------------------
// Function    :  get_mpi_dtype
// Description :  Match from yt_dtype YT_* to MPI_Datatype.
//
// Note        :  1. This function matches yt_dtype to MPI_Datatype, and will write result in
//                   mpi_dtype.
//                2.   yt_dtype      NumPy Enumerate Type
//                  ========================================
//                     YT_FLOAT            MPI_FLOAT
//                     YT_DOUBLE           MPI_DOUBLE
//                     YT_LONGDOUBLE       MPI_LONG_DOUBLE
//                     YT_INT              MPI_INT
//                     YT_LONG             MPI_LONG
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                mpi_dtype : MPI_Datatype.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_mpi_dtype( yt_dtype data_type, MPI_Datatype *mpi_dtype ){
    SET_TIMER(__PRETTY_FUNCTION__);

    switch (data_type) {
        case YT_FLOAT:
            *mpi_dtype = MPI_FLOAT;
            return YT_SUCCESS;
        case YT_DOUBLE:
            *mpi_dtype = MPI_DOUBLE;
            return YT_SUCCESS;
        case YT_LONGDOUBLE:
            *mpi_dtype = MPI_LONG_DOUBLE;
            return YT_SUCCESS;
        case YT_INT:
            *mpi_dtype = MPI_INT;
            return YT_SUCCESS;
        case YT_LONG:
            *mpi_dtype = MPI_LONG;
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
            // Safety check that data_type is one of yt_dtype,
            // so that if we cannot match a NumPy Enum Type, then it must be user forgot to implement here.
            bool valid = false;
            for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
                yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
                if (data_type == dtype) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                log_error("Forget to match new yt_dtype to MPI_Datatype in get_mpi_dtype function.\n");
            }
            else {
                log_error("No such yt_dtype.\n");
            }

            *mpi_dtype = 0;
            return YT_FAIL;
    }
}
#endif

//-------------------------------------------------------------------------------------------------------
// Function    :  get_dtype_size
// Description :  Match from yt_dtype YT_* to sizeof(C type).
//
// Note        :  1. This function matches yt_dtype to C type size, and will write result in dtype_size.
//                2.   yt_dtype            C Type
//                  ========================================
//                     YT_FLOAT            float
//                     YT_DOUBLE           double
//                     YT_LONGDOUBLE       long double
//                     YT_INT              int
//                     YT_LONG             long
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                dtype_size: sizeof(C type).
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_dtype_size( yt_dtype data_type, int *dtype_size ){
    SET_TIMER(__PRETTY_FUNCTION__);

    switch (data_type) {
        case YT_FLOAT:
            *dtype_size = sizeof(float);
            return YT_SUCCESS;
        case YT_DOUBLE:
            *dtype_size = sizeof(double);
            return YT_SUCCESS;
        case YT_LONGDOUBLE:
            *dtype_size = sizeof(long double);
            return YT_SUCCESS;
        case YT_INT:
            *dtype_size = sizeof(int);
            return YT_SUCCESS;
        case YT_LONG:
            *dtype_size = sizeof(long);
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
            // Safety check that data_type is one of yt_dtype,
            // so that if we cannot match a NumPy Enum Type, then it must be user forgot to implement here.
            bool valid = false;
            for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
                yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
                if (data_type == dtype) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                log_error("Forget to match new yt_dtype to size in get_dtype_size function.\n");
            }
            else {
                log_error("No such yt_dtype.\n");
            }

            *dtype_size = -1;
            return YT_FAIL;
    }
}


//-------------------------------------------------------------------------------------------------------
// Function    :  get_dtype_allocation
// Description :  Allocate memory with base unit yt_dtype (YT_*) in C
//
// Note        :  1. This function allocates yt_dtype with base C data type unit, and will store
//                   initialized memory at data_ptr.
//                2. It only initializes 1D array and set array to zeros.
//                3.   yt_dtype            C Type
//                  ========================================
//                     YT_FLOAT            float
//                     YT_DOUBLE           double
//                     YT_LONGDOUBLE       long double
//                     YT_INT              int
//                     YT_LONG             long
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                length    : length of the 1d array
//                data_ptr  : data pointer to allocated buffer
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_dtype_allocation(yt_dtype data_type, unsigned long length, void ** data_ptr ) {
    SET_TIMER(__PRETTY_FUNCTION__);

    switch (data_type) {
        case YT_FLOAT:
            *data_ptr = malloc(length * sizeof(float));
            memset(*data_ptr, 0, length * sizeof(float));
            return YT_SUCCESS;
        case YT_DOUBLE:
            *data_ptr = malloc(length * sizeof(double));
            memset(*data_ptr, 0, length * sizeof(double));
            return YT_SUCCESS;
        case YT_LONGDOUBLE:
            *data_ptr = malloc(length * sizeof(long double));
            memset(*data_ptr, 0, length * sizeof(long double));
            return YT_SUCCESS;
        case YT_INT:
            *data_ptr = malloc(length * sizeof(int));
            memset(*data_ptr, 0, length * sizeof(int));
            return YT_SUCCESS;
        case YT_LONG:
            *data_ptr = malloc(length * sizeof(long));
            memset(*data_ptr, 0, length * sizeof(long));
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
            // Safety check that data_type is one of yt_dtype,
            // so that if we cannot match a NumPy Enum Type, then it must be user forgot to implement here.
            bool valid = false;
            for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
                yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
                if (data_type == dtype) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                log_error("Forget to match new yt_dtype to allocation in get_dtype_allocation function.\n");
            }
            else {
                log_error("No such yt_dtype.\n");
            }

            *data_ptr = nullptr;
            return YT_FAIL;
    }
}