#include "yt_combo.h"
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_check_grid.cpp
// Description :  Check that all the grids are loaded.
//
// Note        :  1. Check that all the grids (the hierarchy) are set, every MPI rank need to do this.
// 				  2. Check that the hierarchy is correct.
// 				  3. Use inside yt_inline(), before perform yt operation "def yt_inline():"
// 				  
//
// Parameter   :  param_libyt : pointer to the yt_param_libyt
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------

int yt_check_grid(const yt_param_libyt *param_libyt) {
	
	// Check that all the grids hierarchy are set.
	
	// Check that data set are well collected.
	
	// Check that the hierarchy relationship are correct. 
	

	return YT_SUCCESS;
}