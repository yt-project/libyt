#include "dtype_utilities.h"

#include "numpy_controller.h"

namespace dtype_utilities {

#ifndef SERIAL_MODE
/*****************************************************************************************
 * \brief Get MPI data type based on yt_dtype.
 *
 * \details
 * 1. Get the MPI data type based on yt_dtype.
 * 2. If the data type is unknown, it will return 0.
 *
 * @param data_type[in] yt_dtype data type
 * @return MPI data type map from yt_dtype
 ****************************************************************************************/
MPI_Datatype YtDtype2MpiDtype(yt_dtype data_type) {
  switch (data_type) {
    case YT_FLOAT:
      return MPI_FLOAT;
    case YT_DOUBLE:
      return MPI_DOUBLE;
    case YT_LONGDOUBLE:
      return MPI_LONG_DOUBLE;
    case YT_CHAR:
      return MPI_SIGNED_CHAR;
    case YT_UCHAR:
      return MPI_UNSIGNED_CHAR;
    case YT_SHORT:
      return MPI_SHORT;
    case YT_USHORT:
      return MPI_UNSIGNED_SHORT;
    case YT_INT:
      return MPI_INT;
    case YT_UINT:
      return MPI_UNSIGNED;
    case YT_LONG:
      return MPI_LONG;
    case YT_ULONG:
      return MPI_UNSIGNED_LONG;
    case YT_LONGLONG:
      return MPI_LONG_LONG;
    case YT_ULONGLONG:
      return MPI_UNSIGNED_LONG_LONG;
    case YT_DTYPE_UNKNOWN:
      return 0;
    default:
      return 0;
  }
}
#endif  // #ifndef SERIAL_MODE

/*****************************************************************************************
 * \brief Get yt_dtype based on NumPy data type.

 * \details
 * 1. Get the yt_dtype based on NumPy data type.
 * 2. If the data type is unknown, it will return YT_DTYPE_UNKNOWN.
 *
 * @param npy_dtype[in] NumPy data type
 * @return yt_dtype enum map from NumPy data type
 ****************************************************************************************/
yt_dtype NumPyDtype2YtDtype(int npy_dtype) {
  switch (npy_dtype) {
    case NPY_FLOAT:
      return YT_FLOAT;
    case NPY_DOUBLE:
      return YT_DOUBLE;
    case NPY_LONGDOUBLE:
      return YT_LONGDOUBLE;
    case NPY_BYTE:
      return YT_CHAR;
    case NPY_UBYTE:
      return YT_UCHAR;
    case NPY_SHORT:
      return YT_SHORT;
    case NPY_USHORT:
      return YT_USHORT;
    case NPY_INT:
      return YT_INT;
    case NPY_UINT:
      return YT_UINT;
    case NPY_LONG:
      return YT_LONG;
    case NPY_ULONG:
      return YT_ULONG;
    case NPY_LONGLONG:
      return YT_LONGLONG;
    case NPY_ULONGLONG:
      return YT_ULONGLONG;
    default:
      return YT_DTYPE_UNKNOWN;
  }
}

/*****************************************************************************************
 * \brief Get NumPy data type based on yt_dtype.
 *
 * \details
 * 1. Map yt_dtype to NumPy data type. Assuming NumPy data type has enum > 0.
 * 2. If the data type is unknown, it will return -1.
 *
 * @param data_type[in] yt data type
 * @return Size of the data type in bytes
 ****************************************************************************************/
int YtDtype2NumPyDtype(yt_dtype data_type) {
  switch (data_type) {
    case YT_FLOAT:
      return NPY_FLOAT;
    case YT_DOUBLE:
      return NPY_DOUBLE;
    case YT_LONGDOUBLE:
      return NPY_LONGDOUBLE;
    case YT_CHAR:
      return NPY_BYTE;
    case YT_UCHAR:
      return NPY_UBYTE;
    case YT_SHORT:
      return NPY_SHORT;
    case YT_USHORT:
      return NPY_USHORT;
    case YT_INT:
      return NPY_INT;
    case YT_UINT:
      return NPY_UINT;
    case YT_LONG:
      return NPY_LONG;
    case YT_ULONG:
      return NPY_ULONG;
    case YT_LONGLONG:
      return NPY_LONGLONG;
    case YT_ULONGLONG:
      return NPY_ULONGLONG;
    case YT_DTYPE_UNKNOWN:
      return -1;
    default:
      return -1;
  }
}

/*****************************************************************************************
 * \brief Get yt_dtype size in bytes.
 *
 * \details
 * 1. Get the size of the data type in bytes.
 * 2. If the data type is unknown, it will return -1.
 *
 * @param data_type[in] yt data type
 * @return Size of the data type in bytes
 ****************************************************************************************/
int GetYtDtypeSize(yt_dtype data_type) {
  switch (data_type) {
    case YT_FLOAT:
      return sizeof(float);
    case YT_DOUBLE:
      return sizeof(double);
    case YT_LONGDOUBLE:
      return sizeof(long double);
    case YT_CHAR:
      return sizeof(char);
    case YT_UCHAR:
      return sizeof(unsigned char);
    case YT_SHORT:
      return sizeof(short);
    case YT_USHORT:
      return sizeof(unsigned short);
    case YT_INT:
      return sizeof(int);
    case YT_UINT:
      return sizeof(unsigned int);
    case YT_LONG:
      return sizeof(long);
    case YT_ULONG:
      return sizeof(unsigned long);
    case YT_LONGLONG:
      return sizeof(long long);
    case YT_ULONGLONG:
      return sizeof(unsigned long long);
    case YT_DTYPE_UNKNOWN:
      return -1;
    default:
      return -1;
  }
}

/*****************************************************************************************
 * \brief Allocate memory based on yt_dtype and length.
 *
 * \details
 * 1. The memory is allocated based on yt_dtype and length and initialized to zero.
 * 2. If the data type is unknown, it will return nullptr.
 *
 * @param data_type[in] yt data type
 * @param length[in] length of the array
 * @return The pointer to the allocated memory, or nullptr if failed.
 ****************************************************************************************/
void* AllocateMemory(yt_dtype data_type, unsigned long length) {
  switch (data_type) {
    case YT_FLOAT: {
      void* data_ptr = malloc(length * sizeof(float));
      memset(data_ptr, 0, length * sizeof(float));
      return data_ptr;
    }
    case YT_DOUBLE: {
      void* data_ptr = malloc(length * sizeof(double));
      memset(data_ptr, 0, length * sizeof(double));
      return data_ptr;
    }
    case YT_LONGDOUBLE: {
      void* data_ptr = malloc(length * sizeof(long double));
      memset(data_ptr, 0, length * sizeof(long double));
      return data_ptr;
    }
    case YT_CHAR: {
      void* data_ptr = malloc(length * sizeof(char));
      memset(data_ptr, 0, length * sizeof(char));
      return data_ptr;
    }
    case YT_UCHAR: {
      void* data_ptr = malloc(length * sizeof(unsigned char));
      memset(data_ptr, 0, length * sizeof(unsigned char));
      return data_ptr;
    }
    case YT_SHORT: {
      void* data_ptr = malloc(length * sizeof(short));
      memset(data_ptr, 0, length * sizeof(short));
      return data_ptr;
    }
    case YT_USHORT: {
      void* data_ptr = malloc(length * sizeof(unsigned short));
      memset(data_ptr, 0, length * sizeof(unsigned short));
      return data_ptr;
    }
    case YT_INT: {
      void* data_ptr = malloc(length * sizeof(int));
      memset(data_ptr, 0, length * sizeof(int));
      return data_ptr;
    }
    case YT_UINT: {
      void* data_ptr = malloc(length * sizeof(unsigned int));
      memset(data_ptr, 0, length * sizeof(unsigned int));
      return data_ptr;
    }
    case YT_LONG: {
      void* data_ptr = malloc(length * sizeof(long));
      memset(data_ptr, 0, length * sizeof(long));
      return data_ptr;
    }
    case YT_ULONG: {
      void* data_ptr = malloc(length * sizeof(unsigned long));
      memset(data_ptr, 0, length * sizeof(unsigned long));
      return data_ptr;
    }
    case YT_LONGLONG: {
      void* data_ptr = malloc(length * sizeof(long long));
      memset(data_ptr, 0, length * sizeof(long long));
      return data_ptr;
    }
    case YT_ULONGLONG: {
      void* data_ptr = malloc(length * sizeof(unsigned long long));
      memset(data_ptr, 0, length * sizeof(unsigned long long));
      return data_ptr;
    }
    case YT_DTYPE_UNKNOWN:
      return nullptr;
    default:
      return nullptr;
  }
}

}  // namespace dtype_utilities
