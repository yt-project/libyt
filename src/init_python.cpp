#include <Python.h>
#include "numpy/arrayobject.h"
#include <stdio.h>
#include "yt.h"
#include "yt_macro.h"
#include "yt_type.h"
#include "yt_prototype.h"

static int check_numpy();




//-------------------------------------------------------------------------------------------------------
// Function    :  init_python
// Description :  Initialize Python interpreter
//
// Note        :  1. Called by yt_init
//
// Parameter   :  argc  : Argument count
//                argv  : Argument vector
//                param : libyt runtime parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int init_python( int argc, char *argv[], const yt_param *param )
{

// initialize Python interpreter
   Py_SetProgramName( "yt_inline" );

// 0: kips initialization registration of signal handlers
   Py_InitializeEx( 0 );

   if ( !Py_IsInitialized() ) {
      YT_ABORT( "Couldn't initialize Python!\n" ); }
   else if ( param->verbose == YT_VERBOSE_DEBUG )
      log_info( "Initialize Python interpreter successfully\n" );

// set sys.argv
   PySys_SetArgv( argc, argv );


// check numpy
   if ( !check_numpy() )
   {
//    call _import_array and PyErr_PrintEx(0) to print out traceback error messages to stderr
      _import_array();
      PyErr_PrintEx( 0 );
      YT_ABORT( "Couldn't import Numpy!" );
   }
   else if ( param->verbose == YT_VERBOSE_DEBUG )  log_info( "Import Numpy successfully\n" );


// add the current location to the search path for modules (sys._parallel = True --> run yt in parallel )
   PyRun_SimpleString( "import sys; sys.path.insert(0,'.'); sys._parallel = True" );


// import the garbage collector interface
   PyRun_SimpleString( "import gc\n" );


   return YT_SUCCESS;

} // FUNCTION : init_python



//-------------------------------------------------------------------------------------------------------
// Function    :  check_numpy
// Description :  Check if Numpy can be imported properly
//
// Note        :  1. Static function called by init_python
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int check_numpy()
{

// import_array1() is a macro which calls _import_array() and returns the given value (YT_FAIL here) on error
   import_array1( YT_FAIL );

   return YT_SUCCESS;

} // FUNCTION : check_numpy
