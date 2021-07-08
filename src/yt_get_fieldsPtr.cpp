#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_get_fieldsPtr
// Description :  Get pointer of the array of struct yt_field with length num_fields.
//
// Note        :  1. User should call this function after yt_set_parameter(), since we need num_fields.
//
// Parameter   :  yt_field **field_list  : Initialize and store the field list array under this pointer 
//                                         points to.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
//
int yt_get_fieldsPtr( yt_field **field_list )
{
	// check if libyt has been initialized
   	if ( !g_param_libyt.libyt_initialized ){
    	YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );
   	}

	// check if yt_set_parameter() have been called
   	if ( !g_param_libyt.param_yt_set ) {
    	YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );
    }

   	log_info( "Getting pointer to field list information ...\n" );

	// Initialize the field_list array.
	*field_list = new yt_field [g_param_yt.num_fields];

	// Store the field_list to g_param_yt
	g_param_yt.field_list = *field_list;

	// Above all works like charm
	g_param_libyt.get_fieldsPtr = true;
	log_info( "Getting pointer to field list information  ... done.\n" );
	
	return YT_SUCCESS;
}