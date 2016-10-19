#include "yt_combo.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  allocate_hierarchy
// Description :  Fill the libyt.hierarchy dictionary with NumPy arrays allocated but uninitialized
//
// Note        :  1. Called by yt_set_parameter()
//                2. These NumPy array will be set when calling yt_add_grid()
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int allocate_hierarchy()
{

// remove all key-value pairs if one wants to overwrite the existing dictionary
// ==> it should happen only if one calls yt_set_parameter() more than once
   if ( PyDict_Size( g_py_hierarchy ) > 0 )
   {
      PyDict_Clear( g_py_hierarchy );
      log_warning( "Removing existing key-value pairs in libyt.hierarchy ... done\n" );
   }


// allocate NumPy arrays and attach them to libyt.hierarchy
   npy_intp np_dim[2];
   PyObject *py_obj;

   np_dim[0] = (npy_intp)g_param_yt.num_grids;

// convenient macro
#  define ADD_DICT( DIM1, KEY, TYPE )                                               \
   {                                                                                \
      np_dim[1] = (npy_intp)DIM1;                                                   \
      py_obj    = PyArray_SimpleNew( 2, np_dim, TYPE );                             \
                                                                                    \
      if ( PyDict_SetItemString( g_py_hierarchy, KEY, py_obj ) != 0 )               \
         YT_ABORT( "Inserting key \"%s\" to libyt.hierarchy ... failed!\n", KEY );  \
                                                                                    \
      Py_DECREF( py_obj );                                                          \
   }

   ADD_DICT( 3, "grid_left_edge",      NPY_DOUBLE );
   ADD_DICT( 3, "grid_right_edge",     NPY_DOUBLE );
   ADD_DICT( 3, "grid_dimensions",     NPY_LONG );
   ADD_DICT( 1, "grid_particle_count", NPY_LONG );
   ADD_DICT( 1, "grid_id",             NPY_LONG );
   ADD_DICT( 1, "grid_parent_id",      NPY_LONG );
   ADD_DICT( 1, "grid_level",          NPY_LONG );


   return YT_SUCCESS;

} // FUNCTION : allocate_hierarchy
