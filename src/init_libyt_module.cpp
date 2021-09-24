#include "yt_combo.h"
#include "string.h"

//-------------------------------------------------------------------------------------------------------
// Description :  List of libyt C extension python methods
//
// Note        :  1. List of python C extension methods functions.
//                2. These function will be called in python, so the parameters indicate python 
//                   input type.
// 
// Lists       :       Python Method         C Extension Function         
//              .............................................................
//                     derived_func          libyt_field_derived_func
//                     get_attr              libyt_particle_get_attr
//                     get_field_remote      libyt_field_get_field_remote
//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_field_derived_func
// Description :  Use the derived_func defined inside yt_field struct to derived the field according to 
//                this function.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. We assume that parallelism in yt will make each rank only has to deal with the local
//                   grids. So we can always find one grid with id = gid inside grids_local.
//                   (Maybe we can add feature get grids data from other rank in the future!)
//                3. The returned numpy array data type is numpy.double.
//                4. grid_dimensions[3] is in [x][y][z] coordinate.
//                
// Parameter   :  int : GID of the grid
//                str : field name
//
// Return      :  numpy.3darray
//-------------------------------------------------------------------------------------------------------
static PyObject* libyt_field_derived_func(PyObject *self, PyObject *args){

    // Parse the input arguments input by python.
    // If not in the format libyt.derived_func( int , str ), raise an error
    long  gid;
    char *field_name;
    int   field_id;

    if ( !PyArg_ParseTuple(args, "ls", &gid, &field_name) ){
        PyErr_SetString(PyExc_TypeError, "Wrong input type, expect to be libyt.derived_func(int, str).");
        return NULL;
    }

    // Get the derived_func define in field_list according to field_name.
    // If cannot find field_name inside field_list, raise an error.
    // If we successfully find the field_name, but the derived_func is not assigned (is NULL), raise an error.
    void (*derived_func) (long, double*);
    bool have_FieldName = false;

    for (int v = 0; v < g_param_yt.num_fields; v++){
        if ( strcmp(g_param_yt.field_list[v].field_name, field_name) == 0 ){
            have_FieldName = true;
            field_id = v;
            if ( g_param_yt.field_list[v].derived_func != NULL ){
                derived_func = g_param_yt.field_list[v].derived_func;
            }
            else {
                PyErr_Format(PyExc_NotImplementedError, "In field_list, field_name [ %s ], derived_func does not set properly.\n", 
                             g_param_yt.field_list[v].field_name);
                return NULL;
            }
            break;
        }
    }

    if ( !have_FieldName ) {
        PyErr_Format(PyExc_ValueError, "Cannot find field_name [ %s ] in field_list.\n", field_name);
        return NULL;
    }

    // Get the grid's dimension[3] according to the gid.
    // We assume that parallelism in yt will make each rank only has to deal with the local grids.
    // We can always find grid with id = gid inside grids_local.
    int  grid_dimensions[3];
    bool have_Grid = false;

    for (int lid = 0; lid < g_param_yt.num_grids_local; lid++){
        if ( g_param_yt.grids_local[lid].id == gid ){
            have_Grid = true;
            grid_dimensions[0] = g_param_yt.grids_local[lid].grid_dimensions[0];
            grid_dimensions[1] = g_param_yt.grids_local[lid].grid_dimensions[1];
            grid_dimensions[2] = g_param_yt.grids_local[lid].grid_dimensions[2];
            break;
        }
    }

    if ( !have_Grid ){
        int MyRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
        PyErr_Format(PyExc_ValueError, "Cannot find grid with GID [ %ld ] on MPI rank [%d].\n", gid, MyRank);
        return NULL;
    }

    // Allocate 1D array with size of grid dimension, initialized with 0.
    // derived_func will make changes to this array.
    // This array will be wrapped by Numpy API and will be return. 
    // The called object will then OWN this numpy array, so that we don't have to free it.
    long gridTotalSize = grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2];
    double *output = (double *) malloc( gridTotalSize * sizeof(double) );
    for (long i = 0; i < gridTotalSize; i++) {
        output[i] = (double) 0;
    }

    // Call the derived_func, result will be made inside output 1D array.
    (*derived_func) (gid, output);

    // Wrapping the C allocated 1D array into 3D numpy array.
    // grid_dimensions[3] is in [x][y][z] coordinate, 
    // thus we have to check if the field has swap_axes == true or false.
    int      nd = 3;
    int      typenum = NPY_DOUBLE;
    npy_intp dims[3];
    if ( g_param_yt.field_list[field_id].swap_axes == true ){
        dims[0] = grid_dimensions[2];
        dims[1] = grid_dimensions[1];
        dims[2] = grid_dimensions[0];
    }
    else{
        dims[0] = grid_dimensions[0];
        dims[1] = grid_dimensions[1];
        dims[2] = grid_dimensions[2];
    }

    PyObject *derived_NpArray = PyArray_SimpleNewFromData(nd, dims, typenum, output);
    PyArray_ENABLEFLAGS( (PyArrayObject*) derived_NpArray, NPY_ARRAY_OWNDATA);

    return derived_NpArray;
}


//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_particle_get_attr
// Description :  Use the get_attr defined inside yt_particle struct to get the particle attributes.
//
// Note        :  1. Support only grid dimension = 3 for now, which is "coor_x", "coor_y", "coor_z" in
//                   yt_particle must be set.
//                2. We assume that parallelism in yt will make each rank only has to deal with the local
//                   grids. So we can always find one grid with id = gid inside grids_local. We will only
//                   get particle that belongs to this id.
//                   (Maybe we can add feature get grids data from other rank in the future!)
//                3. The returned numpy array data type well be set by attr_dtype.
//                4. We will always return 1D numpy array, with length equal particle count of the species 
//                   in that grid.
//                5. Return Py_None if number of ptype particle == 0.
//                
// Parameter   :  int : GID of the grid
//                str : ptype, particle species, ex:"io"
//                str : attribute, or in terms in yt, which is particle.
//
// Return      :  numpy.1darray
//-------------------------------------------------------------------------------------------------------
static PyObject* libyt_particle_get_attr(PyObject *self, PyObject *args){
    // Parse the input arguments input by python.
    // If not in the format libyt.get_attr( int , str , str ), raise an error
    long  gid;
    char *ptype;
    char *attr_name;

    if ( !PyArg_ParseTuple(args, "lss", &gid, &ptype, &attr_name) ){
        PyErr_SetString(PyExc_TypeError, "Wrong input type, expect to be libyt.get_attr(int, str, str).");
        return NULL;
    }

    
    // Get get_attr function pointer defined in particle_list according to ptype and attr_name.
    // Get attr_dtype of the attr_name.
    // If cannot find ptype or attr_name, raise an error.
    // If find them successfully, but get_attr not set, which is == NULL, raise an error.
    void    (*get_attr) (long, char*, void*);
    yt_dtype attr_dtype;
    bool     have_ptype = false;
    bool     have_attr_name = false;
    int      species_index;

    for ( int s = 0; s < g_param_yt.num_species; s++ ){
        if ( strcmp(g_param_yt.particle_list[s].species_name, ptype) == 0 ){
            have_ptype = true;
            species_index = s;

            // Get get_attr
            if ( g_param_yt.particle_list[s].get_attr != NULL ){
                get_attr = g_param_yt.particle_list[s].get_attr;
            }
            else {
                PyErr_Format(PyExc_NotImplementedError, "In particle_list, species_name [ %s ], get_attr does not set properly.\n",
                             g_param_yt.particle_list[s].species_name);
                return NULL;
            }

            // Get attr_dtype
            for ( int p = 0; p < g_param_yt.particle_list[s].num_attr; p++ ){
                if ( strcmp(g_param_yt.particle_list[s].attr_list[p].attr_name, attr_name) == 0 ){
                    have_attr_name = true;
                    // Since in yt_attribute validate(), we have already make sure that attr_dtype is set.
                    // So we don't need additional check
                    attr_dtype = g_param_yt.particle_list[s].attr_list[p].attr_dtype;
                    break;
                }
            }

            break;
        }
    }

    if ( !have_ptype ){
        PyErr_Format(PyExc_ValueError, "Cannot find species_name [ %s ] in particle_list.\n", ptype);
        return NULL;
    }
    if ( !have_attr_name ){
        PyErr_Format(PyExc_ValueError, "species_name [ %s ], attr_name [ %s ] not in particle_list.\n",
                     ptype, attr_name);
        return NULL;
    }


    // Get lenght of the returned 1D numpy array, which is equal to particle_count_list in the grid.
    long  array_length;
    bool  have_Grid = false;

    for (int lid = 0; lid < g_param_yt.num_grids_local; lid++){
        if ( g_param_yt.grids_local[lid].id == gid ){
            have_Grid = true;
            array_length = g_param_yt.grids_local[lid].particle_count_list[species_index];
            
            if ( array_length == 0 ){
                Py_INCREF(Py_None);
                return Py_None;
            }
            
            break;
        }
    }

    if ( !have_Grid ){
        int MyRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);
        PyErr_Format(PyExc_ValueError, "Cannot find grid with GID [ %ld ] on MPI rank [%d].\n", gid, MyRank);
        return NULL;
    }


    // Allocate the output array with size = array_length, type = attr_dtype, and initialize as 0
    // Then pass in to get_attr(long, char*, void*) function
    // Finally, return numpy 1D array, by wrapping the output.
    // We do not need to free output, since we make python owns this data after returning.
    int      nd = 1;
    int      typenum;
    npy_intp dims[1] = { array_length };
    void     *output;

    if ( get_npy_dtype(attr_dtype, &typenum) != YT_SUCCESS ){
        PyErr_Format(PyExc_ValueError, "Unknown yt_dtype, cannot get the NumPy enumerate type properly.\n");
        return NULL;
    }

    // Initialize output array
    if ( attr_dtype == YT_INT ){
        output = malloc( array_length * sizeof(int) );
        for ( long i = 0; i < array_length; i++ ){ 
            ((int *)output)[i] = 0;
        }
    }
    else if ( attr_dtype == YT_FLOAT ){
        output = malloc( array_length * sizeof(float) );
        for ( long i = 0; i < array_length; i++ ){ 
            ((float *)output)[i] = 0;
        }
    }
    else if ( attr_dtype == YT_DOUBLE ){
        output = malloc( array_length * sizeof(double) );
        for ( long i = 0; i < array_length; i++ ){ 
            ((double *)output)[i] = 0;
        }
    }
    else if ( attr_dtype == YT_LONG ){
        output = malloc( array_length * sizeof(long) );
        for ( long i = 0; i < array_length; i++ ){
            ((long *)output)[i] = 0;
        }
    }
    
    // Call get_attr function pointer
    get_attr(gid, attr_name, output);

    // Wrap the output and return back to python
    PyObject *outputNumpyArray = PyArray_SimpleNewFromData(nd, dims, typenum, output);
    PyArray_ENABLEFLAGS( (PyArrayObject*) outputNumpyArray, NPY_ARRAY_OWNDATA);

    return outputNumpyArray;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_field_get_field_remote
// Description :  Get non-local field data from remote ranks. 
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. We return in dictionary objects.
//                
// Parameter   :  list obj : fname_list   : list of field name to get.
//                list obj : to_prepare   : list of grid ids you need to prepare.
//                list obj : nonlocal_id  : nonlocal grid id that you want to get.
//                list obj : nonlocal_rank: where to get those nonlocal grid.
//
// Return      :  dict obj data[grid id][field_name][:,:,:]
//-------------------------------------------------------------------------------------------------------
static PyObject* libyt_field_get_field_remote(PyObject *self, PyObject *args){
    // Parse the input list arguments by python
    PyObject *arg1; // fname_list
    PyObject *arg2; // prepare_grid_id_list
    PyObject *arg3; // get_grid_id_list
    PyObject *arg4; // get_grid_rank_list
    if ( !PyArg_ParseTuple(args, "OOOO", &arg1, &arg2, &arg3, &arg4) ){
        PyErr_SetString(PyExc_TypeError, "Wrong input type, expect to be libyt.get_field_remote(list, list, list, list).\n");
        return NULL;
    }

    // Make these input lists iterators.
    PyObject *fname_list           = PyObject_GetIter( arg1 );
    PyObject *prepare_grid_id_list = PyObject_GetIter( arg2 );
    PyObject *get_grid_id_list     = PyObject_GetIter( arg3 );
    PyObject *get_grid_rank_list   = PyObject_GetIter( arg4 );
    if( fname_list == NULL || prepare_grid_id_list == NULL || get_grid_id_list == NULL || get_grid_rank_list == NULL ){
        PyErr_SetString(PyExc_TypeError, "One of the input arguments are not iterable!\n");
        return NULL;
    }

    // fname -> grid id
    PyObject *py_fname;
    while( py_fname = PyIter_Next( fname_list )){
        // Get fname, grid id to prepare, grid id to get, rank to get from.
        char *fname = PyBytes_AsString( py_fname );
        
        

        // Done with this py_fname, dereference it.
        Py_DECREF( py_fname );
    }

    // Dereference Python objects
    Py_DECREF( fname_list );
    Py_DECREF( prepare_grid_id_list );
    Py_DECREF( get_grid_id_list );
    Py_DECREF( get_grid_rank_list );

    // Return to Python 
    
}

//-------------------------------------------------------------------------------------------------------
// Description :  Preparation for creating libyt python module
//
// Note        :  1. Contains data blocks for creating libyt python module.
//                2. Only initialize libyt python module, not import to system yet.
// 
// Lists:      :  libyt_method_list       : Declare libyt C extension python methods.
//                libyt_module_definition : Definition to libyt python module.
//                PyInit_libyt            : Create libyt python module, and append python objects, 
//                                          ex: dictionary.
//-------------------------------------------------------------------------------------------------------

// Define functions in module, list all libyt module methods here
static PyMethodDef libyt_method_list[] =
{
// { "method_name", c_function_name, METH_VARARGS, "Description"},
   {"derived_func",     libyt_field_derived_func, METH_VARARGS, 
    "Input GID and field name, and get the field data derived by derived_func."},
   {"get_attr",         libyt_particle_get_attr,  METH_VARARGS,
    "Input GID, ptype, particle (which is attribute), and get the particle attribute by get_attr."},
   {"get_field_remote", libyt_field_get_field_remote, METH_VARARGS,
    "Input fields, prepare grids, id of grids to get, and which rank to get from. It returns a dict object contains all the data."},
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
