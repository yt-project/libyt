// define DEFINE_GLOBAL since this file **defines** all global variables
#define DEFINE_GLOBAL
#include "yt_combo.h"
#undef DEFINE_GLOBAL
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_init
// Description :  Initialize libyt
//
// Note        :  1. Input "param_libyt" will be backed up to a libyt global variable
//                2. This function should not be called more than once (even if yt_finalize has been called)
//                   since some extensions (e.g., NumPy) may not work properly
//
// Parameter   :  argc        : Argument count
//                argv        : Argument vector
//                param_libyt : libyt runtime parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_init( int argc, char *argv[], const yt_param_libyt *param_libyt )
{

// yt_init should only be called once
   static int init_count = 0;
   init_count ++;

// still need to check "init_count" since yt_finalize() will set "g_param_libyt.libyt_initialized = false"
   if ( g_param_libyt.libyt_initialized  ||  init_count >= 2 )
      YT_ABORT( "yt_init() should not be called more than once!\n" );


// store user-provided parameters to a libyt internal variable
// --> better do it **before** calling any log function since they will query g_param_libyt.verbose
   g_param_libyt.verbose    = param_libyt->verbose;
   g_param_libyt.script     = param_libyt->script;
   g_param_libyt.counter    = param_libyt->counter;   // useful during restart, where the initial counter can be non-zero
   g_param_libyt.check_data = param_libyt->check_data;

   log_info( "Initializing libyt ...\n" );
   log_info( "   verbose = %d\n", g_param_libyt.verbose );
   log_info( "    script = %s\n", g_param_libyt.script  );
   log_info( "   counter = %ld\n", g_param_libyt.counter);
   log_info( "check_data = %s\n", (g_param_libyt.check_data ? "true" : "false"));

// create libyt module, should be before init_python
   if ( create_libyt_module() == YT_FAIL ) {
      return YT_FAIL;
   }

// initialize Python interpreter
   if ( init_python(argc,argv) == YT_FAIL ) {
      return YT_FAIL;
   }  

// initialize libyt python module such as parameters.
   if ( init_libyt_module() == YT_FAIL ) {
      return YT_FAIL;
   }
   
   g_param_libyt.libyt_initialized = true;
   return YT_SUCCESS;

} // FUNCTION : yt_init
