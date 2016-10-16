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


// export data to libyt.param_yt
// convenient macros for converting data to Python objects
// ==> Py_XDECREF is the same as Py_DECREF() except that it supports NULL input
#  define TMP_PYINT( A )   Py_XDECREF( tmp_int ); tmp_int = PyLong_FromLong   ( (long  )A );
#  define TMP_PYFLT( A )   Py_XDECREF( tmp_flt ); tmp_flt = PyFloat_FromDouble( (double)A );

// scalars
   PyObject *tmp_int=NULL, *tmp_flt=NULL;

   TMP_PYFLT( param_yt->current_time );
   PyDict_SetItemString( g_param_yt, "current_time", tmp_flt );



// vectors (stored as Python tuples)
// PyObject *tgd_tuple, *tgd0, *tgd1, *tgd2;


// release resource
   Py_XDECREF( tmp_int );
   Py_XDECREF( tmp_flt );
#  undef TMP_PYINT
#  undef TMP_PYFLT

   return YT_SUCCESS;

} // FUNCTION : yt_set_parameter
