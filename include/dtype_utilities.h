#ifndef LIBYT_PROJECT_INCLUDE_DTYPE_UTILITIES_H_
#define LIBYT_PROJECT_INCLUDE_DTYPE_UTILITIES_H_

#ifndef SERIAL_MODE
#include <mpi.h>
#endif

#include "yt_type.h"

namespace dtype_utilities {

#ifndef SERIAL_MODE
MPI_Datatype YtDtype2MpiDtype(yt_dtype data_type);
#endif
yt_dtype NumPyDtype2YtDtype(int npy_dtype);
int YtDtype2NumPyDtype(yt_dtype data_type);
int GetYtDtypeSize(yt_dtype data_type);
void* AllocateMemory(yt_dtype data_type, unsigned long length);
}  // namespace dtype_utilities

#endif  // LIBYT_PROJECT_INCLUDE_DTYPE_UTILITIES_H_
