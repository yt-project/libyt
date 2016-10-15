// to get rid of the warning messages about using deprecated NumPy API
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

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
// Parameter   :  argc : Argument count
//                argv : Argument vector
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int init_python( int argc, char *argv[] )
{

// initialize Python interpreter
   Py_SetProgramName( "yt_inline" );

// 0: kips initialization registration of signal handlers
   Py_InitializeEx( 0 );

   if ( Py_IsInitialized() )
      log_debug( "Initialize Python interpreter successfully\n" );
   else {
      YT_ABORT( "Couldn't initialize Python!\n" ); }

// set sys.argv
   PySys_SetArgv( argc, argv );


// check numpy
   if ( check_numpy() )
      log_debug( "Import NumPy successfully\n" );
   else
   {
//    call _import_array and PyErr_PrintEx(0) to print out traceback error messages to stderr
      _import_array();
      PyErr_PrintEx( 0 );
      YT_ABORT( "Couldn't import NumPy!\n" );
   }


// add the current location to the module search path (sys._parallel = True --> run yt in parallel )
   if ( PyRun_SimpleString( "import sys; sys.path.insert(0,'.'); sys._parallel = True" ) != 0 )
      YT_ABORT( "Couldn't import sys module properly!\n" );


// import the garbage collector interface
   if ( PyRun_SimpleString( "import gc" ) != 0 )
      YT_ABORT( "Couldn't import Python garbage collector!\n" );


   return YT_SUCCESS;

} // FUNCTION : init_python



//-------------------------------------------------------------------------------------------------------
// Function    :  check_numpy
// Description :  Check if NumPy can be imported properly
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
