#include "yt_combo.h"
#include "string.h"

//-------------------------------------------------------------------------------------------------------
// Description :  List of libyt C extension python methods
//
// Note        :  1. List of python C extension methods functions.
// 
// Lists       :  libyt_method
//-------------------------------------------------------------------------------------------------------
static PyObject* libyt_method(PyObject *self, PyObject *args){
  printf("Inside libyt_method!!!\n");
  Py_INCREF(Py_None);
  return Py_None; // return nothing in pthon function
}


// Define functions in module, list all libyt module methods here
static PyMethodDef libyt_method_list[] =
{
// { "method_name", c_function_name, METH_VARARGS, "Description"},
   {"method1", libyt_method, METH_VARARGS, "test method"},
   { NULL, NULL, 0, NULL } // sentinel
};

// Declare the definition of libyt_module
static struct PyModuleDef libyt_module_definition = 
{
    PyModuleDef_HEAD_INIT,
    "libyt",
    "libyt documentation",
    -1,
    libyt_method_list
};

// Create libyt python module
static PyObject* PyInit_libyt(void)
{
  // Create libyt module
  PyObject *libyt_module = PyModule_Create( &libyt_module_definition );
  if ( libyt_module != NULL ){
    log_debug( "Creating libyt module ... done\n" );
  }
  else {
    YT_ABORT(  "Creating libyt module ... failed!\n");
  }

  // Add objects dictionary
  g_py_grid_data  = PyDict_New();
  g_py_hierarchy  = PyDict_New();
  g_py_param_yt   = PyDict_New();
  g_py_param_user = PyDict_New();

  PyModule_AddObject(libyt_module, "grid_data",  g_py_grid_data );
  PyModule_AddObject(libyt_module, "hierarchy",  g_py_hierarchy );
  PyModule_AddObject(libyt_module, "param_yt",   g_py_param_yt  );
  PyModule_AddObject(libyt_module, "param_user", g_py_param_user);

  log_debug( "Attaching empty dictionaries to libyt module ... done\n" );

  return libyt_module;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  create_libyt_module
// Description :  Create the libyt module
//
// Note        :  1. Create libyt module, should be called before Py_Initialize().
//                2. It is used for sharing data between simulation code and YT.
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int create_libyt_module()
{
  PyImport_AppendInittab("libyt", &PyInit_libyt);

  return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  init_libyt_module
// Description :  Initialize the libyt module
//
// Note        :  1. Import newly created libyt module.
//                2. Load user script to python.
//                
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int init_libyt_module()
{

// import newly created libyt module
   if ( PyRun_SimpleString("import libyt\n") == 0 )
      log_debug( "Import libyt module ... done\n" );
   else
      YT_ABORT(  "Import libyt module ... failed!\n" );


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



