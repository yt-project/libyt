#include "yt_combo.h"
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_set_parameter
// Description :  Set YT-specific parameters
//
// Note        :  1. Store the input "param_yt" to libyt.param_yt
//
// Parameter   :  param_yt : Structure storing all YT-specific parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_set_parameter( const yt_param_yt *param_yt )
{

// check if libyt has been initialized
   if ( g_initialized )
      log_info( "Setting YT parameters ...\n" );
   else
      YT_ABORT( "Please invoke yt_init before calling %s!\n", __FUNCTION__ );


// check if all parameters have been set properly
   if ( param_yt->validate() )
      log_debug( "Validating YT parameters ... done\n" );
   else
      YT_ABORT(  "Validating YT parameters ... failed\n" );


// print out all parameters
   log_debug( "Listing all YT parameters ...\n" );
   param_yt->show();


// store data into the Python module libyt.param_yt


   return YT_SUCCESS;

} // FUNCTION : yt_set_parameter
