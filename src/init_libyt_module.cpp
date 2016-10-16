#include "yt_combo.h"
#include "string.h"


// list all libyt module methods here
static PyMethodDef libyt_method_list[] =
{
// { "method_name", c_function_name, METH_VARARGS, "Description"},
   { NULL, NULL, 0, NULL } // sentinel
};



//-------------------------------------------------------------------------------------------------------
// Function    :  init_libyt_module
// Description :  Initialize the libyt module
//
// Note        :  1. libyt module is used for sharing data between simulation code and YT
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int init_libyt_module()
{

// create module and obtain its __dict__ attribute
   PyObject *libyt_module=NULL, *libyt_module_dict=NULL;

   libyt_module = Py_InitModule( "libyt", libyt_method_list );
   if ( libyt_module != NULL)
      log_debug( "Creating libyt module ... done\n" );
   else
      YT_ABORT(  "Creating libyt module ... failed!\n" );

   libyt_module_dict = PyModule_GetDict( libyt_module );
   if ( libyt_module_dict != NULL )
      log_debug( "Obtaining the __dict__ attribute of libyt ... done\n" );
   else
      YT_ABORT(  "Obtaining the __dict__ attribute of libyt ... failed!\n" );


// attach empty dictionaries
   g_grid_data  = PyDict_New();
   g_hierarchy  = PyDict_New();
   g_param_yt   = PyDict_New();
   g_param_user = PyDict_New();

   PyDict_SetItemString( libyt_module_dict, "grid_data",   g_grid_data  );
   PyDict_SetItemString( libyt_module_dict, "hierarchy",   g_hierarchy  );
   PyDict_SetItemString( libyt_module_dict, "paramm_yt",   g_param_yt   );
   PyDict_SetItemString( libyt_module_dict, "paramm_user", g_param_user );

   log_debug( "Attaching empty dictionaries to libyt module ... done\n" );


// import YT inline analysis script
   const int CallYT_CommandWidth = 8 + strlen( g_param_libyt.script );   // 8 = "import " + '\0'
   char *CallYT = (char*) malloc( CallYT_CommandWidth*sizeof(char) );
   sprintf( CallYT, "import %s", g_param_libyt.script );

   if ( PyRun_SimpleString( CallYT ) == 0 )
      log_debug( "Importing YT inline analysis script \"%s\" ... done\n", g_param_libyt.script );
   else
      YT_ABORT(  "Importing YT inline analysis script \"%s\" ... failed (please do not include the \".py\" extension)!\n",
                g_param_libyt.script );

   free( CallYT );


   return YT_SUCCESS;

} // FUNCTION : init_libyt_module



/*
//-------------------------------------------------------------------------------------------------------
// Function    :  Template
// Description :  ???
//
// Note        :  1.
//
// Parameter   :  None
//
// Return      :  ?
//-------------------------------------------------------------------------------------------------------
static PyObject * c_function_name( PyObject *self, PyObject *args )
{

    const char *command;
    int sts;

    if ( !PyArg_ParseTuple( args, "s", &command ) )   return NULL;
    sts = system( command );
    return Py_BuildValue( "i", sts );

} // METHOD : ?
*/



