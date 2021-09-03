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
		*npy_dtype = -1;
		return YT_FAIL;
	}

	return YT_SUCCESS;
}