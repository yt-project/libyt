// define CALL_IMPORT_ARRAY since this file will invoke import_array to import NumPy
#define CALL_IMPORT_ARRAY
#include "yt_combo.h"
#undef CALL_IMPORT_ARRAY

static int import_numpy();




//-------------------------------------------------------------------------------------------------------
// Function    :  init_python
// Description :  Initialize Python interpreter
//
// Note        :  1. Called by yt_init()
//
// Parameter   :  argc : Argument count
//                argv : Argument vector
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int init_python( int argc, char *argv[] )
{

// TODO: Where do we need this?
// initialize Python interpreter
   Py_SetProgramName( Py_DecodeLocale("yt_inline", NULL) );

// 0: skips the initialization registration of signal handlers
   Py_InitializeEx( 0 );

   if ( Py_IsInitialized() )
      log_debug( "Initializing Python interpreter ... done\n" );
   else {
      YT_ABORT(  "Initializing Python interpreter ... failed!\n" ); }

// TODO: What are argc, argv use for?
//       Length is hardcoded, each argv string size cannot longer than 1000.
// set sys.argv
   wchar_t **wchar_t_argv = (wchar_t **) malloc(argc * sizeof(wchar_t *));
   wchar_t wchar_temp[1000];
   for (int i = 0; i < argc; i = i+1) {
	  printf("argv[%d] = %s\n", i, argv[i]);
      mbtowc(wchar_temp, argv[i], 1000);
      wchar_t_argv[i] = wchar_temp;
   }
// TODO: Comment out, since sometimes the typecasting cannot work
//       and leads to error in OpenMPI.
//   PySys_SetArgv( argc, wchar_t_argv );


// import numpy
   if ( import_numpy() )
      log_debug( "Importing NumPy ... done\n" );
   else
   {
//    call _import_array and PyErr_PrintEx(0) to print out traceback error messages to stderr
      _import_array();
      PyErr_PrintEx( 0 );
      YT_ABORT(  "Importing NumPy ... failed!\n" );
   }


// add the current location to the module search path (sys._parallel = True --> run yt in parallel )
   if ( PyRun_SimpleString( "import sys; sys.path.insert(0,'.'); sys._parallel = True" ) == 0 )
      log_debug( "Adding search path for modules ... done\n" );
   else
      YT_ABORT(  "Adding search path for modules ... failed!\n" );


// import the garbage collector interface
   if ( PyRun_SimpleString( "import gc" ) == 0 )
      log_debug( "Importing Python garbage collector ... done\n" );
   else
      YT_ABORT(  "Importing Python garbage collector ... failed!\n" );


   return YT_SUCCESS;

} // FUNCTION : init_python



//-------------------------------------------------------------------------------------------------------
// Function    :  import_numpy
// Description :  Import NumPy
//
// Note        :  1. Static function called by init_python
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int import_numpy()
{

// TODO: Cannot find import_array1, but compile success
// import_array1() is a macro which calls _import_array() and returns the given value (YT_FAIL here) on error
   import_array1( YT_FAIL );

   return YT_SUCCESS;

} // FUNCTION : import_numpy
