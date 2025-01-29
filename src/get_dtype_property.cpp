#include <cstring>
#include <typeinfo>

#include "big_mpi.h"
#include "libyt.h"
#include "numpy_controller.h"
#include "yt_combo.h"

#ifdef USE_PYBIND11
#include "pybind11/embed.h"
#include "pybind11/numpy.h"
#endif

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
int get_npy_dtype(yt_dtype data_type, int* npy_dtype) {
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
        case YT_CHAR:
            *npy_dtype = NPY_BYTE;
            return YT_SUCCESS;
        case YT_UCHAR:
            *npy_dtype = NPY_UBYTE;
            return YT_SUCCESS;
        case YT_SHORT:
            *npy_dtype = NPY_SHORT;
            return YT_SUCCESS;
        case YT_USHORT:
            *npy_dtype = NPY_USHORT;
            return YT_SUCCESS;
        case YT_INT:
            *npy_dtype = NPY_INT;
            return YT_SUCCESS;
        case YT_UINT:
            *npy_dtype = NPY_UINT;
            return YT_SUCCESS;
        case YT_LONG:
            *npy_dtype = NPY_LONG;
            return YT_SUCCESS;
        case YT_ULONG:
            *npy_dtype = NPY_ULONG;
            return YT_SUCCESS;
        case YT_LONGLONG:
            *npy_dtype = NPY_LONGLONG;
            return YT_SUCCESS;
        case YT_ULONGLONG:
            *npy_dtype = NPY_ULONGLONG;
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
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
            } else {
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
int get_yt_dtype_from_npy(int npy_dtype, yt_dtype* data_dtype) {
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
        case NPY_BYTE:
            *data_dtype = YT_CHAR;
            return YT_SUCCESS;
        case NPY_UBYTE:
            *data_dtype = YT_UCHAR;
            return YT_SUCCESS;
        case NPY_SHORT:
            *data_dtype = YT_SHORT;
            return YT_SUCCESS;
        case NPY_USHORT:
            *data_dtype = YT_USHORT;
            return YT_SUCCESS;
        case NPY_INT:
            *data_dtype = YT_INT;
            return YT_SUCCESS;
        case NPY_UINT:
            *data_dtype = YT_UINT;
            return YT_SUCCESS;
        case NPY_LONG:
            *data_dtype = YT_LONG;
            return YT_SUCCESS;
        case NPY_ULONG:
            *data_dtype = YT_ULONG;
            return YT_SUCCESS;
        case NPY_LONGLONG:
            *data_dtype = YT_LONGLONG;
            return YT_SUCCESS;
        case NPY_ULONGLONG:
            *data_dtype = YT_ULONGLONG;
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
int get_mpi_dtype(yt_dtype data_type, MPI_Datatype* mpi_dtype) {
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
        case YT_CHAR:
            *mpi_dtype = MPI_SIGNED_CHAR;
            return YT_SUCCESS;
        case YT_UCHAR:
            *mpi_dtype = MPI_UNSIGNED_CHAR;
            return YT_SUCCESS;
        case YT_SHORT:
            *mpi_dtype = MPI_SHORT;
            return YT_SUCCESS;
        case YT_USHORT:
            *mpi_dtype = MPI_UNSIGNED_SHORT;
            return YT_SUCCESS;
        case YT_INT:
            *mpi_dtype = MPI_INT;
            return YT_SUCCESS;
        case YT_UINT:
            *mpi_dtype = MPI_UNSIGNED;
            return YT_SUCCESS;
        case YT_LONG:
            *mpi_dtype = MPI_LONG;
            return YT_SUCCESS;
        case YT_ULONG:
            *mpi_dtype = MPI_UNSIGNED_LONG;
            return YT_SUCCESS;
        case YT_LONGLONG:
            *mpi_dtype = MPI_LONG_LONG;
            return YT_SUCCESS;
        case YT_ULONGLONG:
            *mpi_dtype = MPI_UNSIGNED_LONG_LONG;
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
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
            } else {
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
int get_dtype_size(yt_dtype data_type, int* dtype_size) {
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
        case YT_CHAR:
            *dtype_size = sizeof(char);
            return YT_SUCCESS;
        case YT_UCHAR:
            *dtype_size = sizeof(unsigned char);
            return YT_SUCCESS;
        case YT_SHORT:
            *dtype_size = sizeof(short);
            return YT_SUCCESS;
        case YT_USHORT:
            *dtype_size = sizeof(unsigned short);
            return YT_SUCCESS;
        case YT_INT:
            *dtype_size = sizeof(int);
            return YT_SUCCESS;
        case YT_UINT:
            *dtype_size = sizeof(unsigned int);
            return YT_SUCCESS;
        case YT_LONG:
            *dtype_size = sizeof(long);
            return YT_SUCCESS;
        case YT_ULONG:
            *dtype_size = sizeof(unsigned long);
            return YT_SUCCESS;
        case YT_LONGLONG:
            *dtype_size = sizeof(long long);
            return YT_SUCCESS;
        case YT_ULONGLONG:
            *dtype_size = sizeof(unsigned long long);
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
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
            } else {
                log_error("No such yt_dtype.\n");
            }

            *dtype_size = -1;
            return YT_FAIL;
    }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  get_dtype_typeid
// Description :  Match from yt_dtype (YT_*) to type id (C type).
//
// Note        :  1. This function matches yt_dtype to C type id, and will write result as a pointer
//                   in dtype_id.
//                2.   yt_dtype            C Type
//                  ========================================
//                     YT_FLOAT            float
//                     YT_DOUBLE           double
//                     YT_LONGDOUBLE       long double
//                     YT_INT              int
//                     YT_LONG             long
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                dtype_id  : address of a pointer to type info.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_dtype_typeid(yt_dtype data_type, const std::type_info** dtype_id) {
    switch (data_type) {
        case YT_FLOAT:
            *dtype_id = &typeid(float);
            return YT_SUCCESS;
        case YT_DOUBLE:
            *dtype_id = &typeid(double);
            return YT_SUCCESS;
        case YT_LONGDOUBLE:
            *dtype_id = &typeid(long double);
            return YT_SUCCESS;
        case YT_CHAR:
            *dtype_id = &typeid(char);
            return YT_SUCCESS;
        case YT_UCHAR:
            *dtype_id = &typeid(unsigned char);
            return YT_SUCCESS;
        case YT_SHORT:
            *dtype_id = &typeid(short);
            return YT_SUCCESS;
        case YT_USHORT:
            *dtype_id = &typeid(unsigned short);
            return YT_SUCCESS;
        case YT_INT:
            *dtype_id = &typeid(int);
            return YT_SUCCESS;
        case YT_UINT:
            *dtype_id = &typeid(unsigned int);
            return YT_SUCCESS;
        case YT_LONG:
            *dtype_id = &typeid(long);
            return YT_SUCCESS;
        case YT_ULONG:
            *dtype_id = &typeid(unsigned long);
            return YT_SUCCESS;
        case YT_LONGLONG:
            *dtype_id = &typeid(long long);
            return YT_SUCCESS;
        case YT_ULONGLONG:
            *dtype_id = &typeid(unsigned long long);
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
            bool valid = false;
            for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
                yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
                if (data_type == dtype) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                log_error("Forget to match new yt_dtype to C type in get_dtype_typeid function.\n");
            } else {
                log_error("No such yt_dtype.\n");
            }

            *dtype_id = nullptr;
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
int get_dtype_allocation(yt_dtype data_type, unsigned long length, void** data_ptr) {
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
        case YT_CHAR:
            *data_ptr = malloc(length * sizeof(char));
            memset(*data_ptr, 0, length * sizeof(char));
            return YT_SUCCESS;
        case YT_UCHAR:
            *data_ptr = malloc(length * sizeof(unsigned char));
            memset(*data_ptr, 0, length * sizeof(unsigned char));
            return YT_SUCCESS;
        case YT_SHORT:
            *data_ptr = malloc(length * sizeof(short));
            memset(*data_ptr, 0, length * sizeof(short));
            return YT_SUCCESS;
        case YT_USHORT:
            *data_ptr = malloc(length * sizeof(unsigned short));
            memset(*data_ptr, 0, length * sizeof(unsigned short));
            return YT_SUCCESS;
        case YT_INT:
            *data_ptr = malloc(length * sizeof(int));
            memset(*data_ptr, 0, length * sizeof(int));
            return YT_SUCCESS;
        case YT_UINT:
            *data_ptr = malloc(length * sizeof(unsigned int));
            memset(*data_ptr, 0, length * sizeof(unsigned int));
            return YT_SUCCESS;
        case YT_LONG:
            *data_ptr = malloc(length * sizeof(long));
            memset(*data_ptr, 0, length * sizeof(long));
            return YT_SUCCESS;
        case YT_ULONG:
            *data_ptr = malloc(length * sizeof(unsigned long));
            memset(*data_ptr, 0, length * sizeof(unsigned long));
            return YT_SUCCESS;
        case YT_LONGLONG:
            *data_ptr = malloc(length * sizeof(long long));
            memset(*data_ptr, 0, length * sizeof(long long));
            return YT_SUCCESS;
        case YT_ULONGLONG:
            *data_ptr = malloc(length * sizeof(unsigned long long));
            memset(*data_ptr, 0, length * sizeof(unsigned long long));
            return YT_SUCCESS;
        case YT_DTYPE_UNKNOWN:
            log_warning("Forget to set yt_dtype, yt_dtype is YT_DTYPE_UNKNOWN.\n");
            return YT_FAIL;
        default:
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
            } else {
                log_error("No such yt_dtype.\n");
            }

            *data_ptr = nullptr;
            return YT_FAIL;
    }
}

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
pybind11::array get_pybind11_allocate_array_dtype(yt_dtype data_type, const std::vector<long>& shape,
                                                  const std::vector<long>& stride) {
    switch (data_type) {
        case YT_FLOAT:
            return pybind11::array_t<float>(shape, stride);
        case YT_DOUBLE:
            return pybind11::array_t<double>(shape, stride);
        case YT_LONGDOUBLE:
            return pybind11::array_t<long double>(shape, stride);
        case YT_CHAR:
            return pybind11::array_t<char>(shape, stride);
        case YT_UCHAR:
            return pybind11::array_t<unsigned char>(shape, stride);
        case YT_SHORT:
            return pybind11::array_t<short>(shape, stride);
        case YT_USHORT:
            return pybind11::array_t<unsigned short>(shape, stride);
        case YT_INT:
            return pybind11::array_t<int>(shape, stride);
        case YT_UINT:
            return pybind11::array_t<unsigned int>(shape, stride);
        case YT_LONG:
            return pybind11::array_t<long>(shape, stride);
        case YT_ULONG:
            return pybind11::array_t<unsigned long>(shape, stride);
        case YT_LONGLONG:
            return pybind11::array_t<long long>(shape, stride);
        case YT_ULONGLONG:
            return pybind11::array_t<unsigned long long>(shape, stride);
        case YT_DTYPE_UNKNOWN:
            log_warning("yt_dtype is YT_DTYPE_UNKNOWN. Unable to create pybind11::array\n");
            return pybind11::array();
        default:
            bool valid = false;
            for (int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++) {
                yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
                if (data_type == dtype) {
                    valid = true;
                    break;
                }
            }
            if (valid) {
                log_error("Forget to delegate new yt_dtype to pybind11::array_t.\n");
            } else {
                log_error("No such yt_dtype.\n");
            }
            return pybind11::array();
    }
}
#endif