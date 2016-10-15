#include <stdio.h>
#include "yt.h"
#include "yt_macro.h"
#include "yt_type.h"
#include "yt_prototype.h"
#include "yt_global.h"


// all libyt global variables are defined here (with a prefix g_)
// --> they must also be declared in "yt_global.h" with the keyword extern
bool     g_initialized = false;  // record whether libyt has been initialized
yt_param g_param;                // libyt runtime parameters
                                 // --> do not defined as a pointer so that it is always initialized
                                 //     --> log functions can be called safely everywhere




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_init
// Description :  Initialize libyt
//
// Note        :  1. User-provided parameters "param" will be backed up to a libyt global variable
//
// Parameter   :  argc  : Argument count
//                argv  : Argument vector
//                param : libyt runtime parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_init( int argc, char *argv[], const yt_param *param )
{

// nothing to do if libyt has been initialized
   if ( g_initialized )
   {
      log_warning( "libyt has already been initialized!\n" );

      return YT_SUCCESS;
   }


// store user-provided parameters to a libyt global variable
// --> better do it **before** calling any log function since they will query g_param.verbose
   g_param = *param;

   log_info( "Initializing libyt ...\n" );
   log_debug( "   verbose = %d\n", g_param.verbose );
   log_debug( "   script  = %s\n", g_param.script );


// initialize Python interpreter
   if ( init_python(argc,argv) == YT_FAIL )   return YT_FAIL;


   g_initialized = true;
   return YT_SUCCESS;

} // FUNCTION : yt_init
