#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  append_grid
// Description :  Add a single full grid to the libyt Python module
//
// Note        :  1. Store the input "grid" to libyt.hierarchy and libyt.grid_data to python
//                2. Called and use by yt_add_grids().
//
// Parameter   :  yt_grid *grid
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int append_grid( yt_grid *grid ){

// export grid info to libyt.hierarchy
   PyArrayObject *py_array_obj;

// convenient macro
// note that PyDict_GetItemString() returns a **borrowed** reference ==> no need to call Py_DECREF
#  define FILL_ARRAY( KEY, ARRAY, DIM, TYPE )                                                            \
   {                                                                                                     \
      for (int t=0; t<DIM; t++)                                                                          \
      {                                                                                                  \
         if (  ( py_array_obj = (PyArrayObject*)PyDict_GetItemString( g_py_hierarchy, KEY ) ) == NULL )  \
            YT_ABORT( "Accessing the key \"%s\" from libyt.hierarchy ... failed!\n", KEY );              \
                                                                                                         \
         *(TYPE*)PyArray_GETPTR2( py_array_obj, grid->id, t ) = (TYPE)(ARRAY)[t];                        \
      }                                                                                                  \
   }

   FILL_ARRAY( "grid_left_edge",       grid->left_edge,      3, npy_double );
   FILL_ARRAY( "grid_right_edge",      grid->right_edge,     3, npy_double );
   FILL_ARRAY( "grid_dimensions",      grid->dimensions,     3, npy_long   );
   FILL_ARRAY( "grid_particle_count", &grid->particle_count, 1, npy_long   );
   FILL_ARRAY( "grid_parent_id",      &grid->parent_id,      1, npy_long   );
   FILL_ARRAY( "grid_levels",         &grid->level,          1, npy_long   );
   FILL_ARRAY( "proc_num",            &grid->proc_num,       1, npy_int    );
   log_debug( "Inserting grid [%15ld] info to libyt.hierarchy ... done\n", grid->id );

// export grid data to libyt.grid_data as "libyt.grid_data[grid_id][field_label][field_data]"
   int      grid_ftype   = (grid->field_ftype == YT_FLOAT ) ? NPY_FLOAT : NPY_DOUBLE;
   npy_intp grid_dims[3] = { grid->dimensions[0], grid->dimensions[1], grid->dimensions[2] };
   PyObject *py_grid_id, *py_field_labels, *py_field_data;

// allocate [grid_id][field_label]
   py_grid_id      = PyLong_FromLong( grid->id );
   py_field_labels = PyDict_New();

   PyDict_SetItem( g_py_grid_data, py_grid_id, py_field_labels );

// fill [grid_id][field_label][field_data]
   for (int v=0; v<g_param_yt.num_fields; v++)
   {
//    PyArray_SimpleNewFromData simply creates an array wrapper and does note allocate and own the array
      py_field_data = PyArray_SimpleNewFromData( 3, grid_dims, grid_ftype, grid->field_data[v] );

//    add the field data to "libyt.grid_data[grid_id][field_label]"
      PyDict_SetItemString( py_field_labels, g_param_yt.field_labels[v], py_field_data );

//    call decref since PyDict_SetItemString() returns a new reference
      Py_DECREF( py_field_data );

//    we assume that field data of specific range are not disperse, they contain in one MPI rank only
      if ( grid->field_data[v] != NULL ) {
         log_debug( "Inserting grid [%15ld] field data [%s] to libyt.grid_data ... done\n", 
                     grid->id, g_param_yt.field_labels[v] );
      }
   }

// call decref since both PyLong_FromLong() and PyDict_New() return a new reference
   Py_DECREF( py_grid_id );
   Py_DECREF( py_field_labels );

   return YT_SUCCESS;
}