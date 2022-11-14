#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  get_npy_dtype
// Description :  Match from yt_dtype YT_* to NumPy enumerate type.
//
// Note        :  1. This function matches yt_dtype to NumPy enumerate type, and will write result in 
//                   npy_dtype.
//                2. If you want to add new type to wrap your data, add it here. So that libyt knows how
//                   to match it to NumPy enumerate type.
//                3.   yt_dtype      NumPy Enumerate Type
//                  ========================================
//                     YT_FLOAT            NPY_FLOAT
//                     YT_DOUBLE           NPY_DOUBLE
//                     YT_INT              NPY_INT
//                     YT_LONG             NPY_LONG
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                npy_dtype : NumPy enumerate data type.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_npy_dtype( yt_dtype data_type, int *npy_dtype ){
	
	if ( data_type == YT_FLOAT ){
		*npy_dtype = NPY_FLOAT;
	}
	else if ( data_type == YT_DOUBLE ){
		*npy_dtype = NPY_DOUBLE;
	}
	else if ( data_type == YT_INT ){
		*npy_dtype = NPY_INT;
	}
	else if ( data_type == YT_LONG ){
		*npy_dtype = NPY_LONG;
	}
	else{
		// Safety check that data_type is one of yt_dtype, 
		// so that if we cannot match a NumPy Enum Type, then it must be user forgot to implement here.
		bool valid = false;
		for ( int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++ ){
			yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
			if ( data_type == dtype ){
				valid = true;
				break;
			}
		}
		if ( valid == true ){
			log_error("You should also match your new yt_dtype to NumPy enumerate type in get_npy_dtype function.\n");
		}
		
		*npy_dtype = -1;
		return YT_FAIL;
	}

	return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  get_mpi_dtype
// Description :  Match from yt_dtype YT_* to MPI_Datatype.
//
// Note        :  1. This function matches yt_dtype to MPI_Datatype, and will write result in
//                   mpi_dtype.
//                2. If you want to add new type to wrap your data, add it here. So that libyt knows how
//                   to match it to mpi_dtype.
//                3.   yt_dtype      NumPy Enumerate Type
//                  ========================================
//                     YT_FLOAT            MPI_FLOAT
//                     YT_DOUBLE           MPI_DOUBLE
//                     YT_INT              MPI_INT
//                     YT_LONG             MPI_LONG
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                mpi_dtype : MPI_Datatype.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_mpi_dtype( yt_dtype data_type, MPI_Datatype *mpi_dtype ){

    if ( data_type == YT_FLOAT ){
        *mpi_dtype = MPI_FLOAT;
    }
    else if ( data_type == YT_DOUBLE ){
        *mpi_dtype = MPI_DOUBLE;
    }
    else if ( data_type == YT_INT ){
        *mpi_dtype = MPI_INT;
    }
    else if ( data_type == YT_LONG ){
        *mpi_dtype = MPI_LONG;
    }
    else{
        // Safety check that data_type is one of yt_dtype,
        // so that if we cannot match a MPI_Datatype, then it must be user forgot to implement here.
        bool valid = false;
        for ( int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++ ){
            yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
            if ( data_type == dtype ){
                valid = true;
                break;
            }
        }
        if ( valid == true ){
            log_error("You should also match your new yt_dtype to MPI_Datatype in get_mpi_dtype function.\n");
        }

        *mpi_dtype = 0;
        return YT_FAIL;
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  get_dtype_size
// Description :  Match from yt_dtype YT_* to sizeof(C type).
//
// Note        :  1. This function matches yt_dtype to C type size, and will write result in dtype_size.
//                2. If you want to add new type to wrap your data, add it here too. So that libyt knows how
//                   to match it to sizeof(C type).
//                3.   yt_dtype            C Type
//                  ========================================
//                     YT_FLOAT            float
//                     YT_DOUBLE           double
//                     YT_INT              int
//                     YT_LONG             long
//
// Parameter   :  data_type : yt_dtype, YT_*.
//                dtype_size: sizeof(C type).
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int get_dtype_size( yt_dtype data_type, int *dtype_size ){

    if ( data_type == YT_FLOAT ){
        *dtype_size = sizeof(float);
    }
    else if ( data_type == YT_DOUBLE ){
        *dtype_size = sizeof(double);
    }
    else if ( data_type == YT_INT ){
        *dtype_size = sizeof(int);
    }
    else if ( data_type == YT_LONG ){
        *dtype_size = sizeof(long);
    }
    else{
        // Safety check that data_type is one of yt_dtype,
        // so that if we cannot match a C type, then it must be user forgot to implement here.
        bool valid = false;
        for ( int yt_dtypeInt = YT_FLOAT; yt_dtypeInt < YT_DTYPE_UNKNOWN; yt_dtypeInt++ ){
            yt_dtype dtype = static_cast<yt_dtype>(yt_dtypeInt);
            if ( data_type == dtype ){
                valid = true;
                break;
            }
        }
        if ( valid == true ){
            log_error("You should also match your new yt_dtype to C type in get_dtype_size function.\n");
        }

        *dtype_size = -1;
        return YT_FAIL;
    }

    return YT_SUCCESS;
}