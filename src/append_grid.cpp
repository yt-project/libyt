#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  append_grid
// Description :  Add a single full grid to the libyt Python module
//
// Note        :  1. Store the input "grid" to libyt.hierarchy and libyt.grid_data.
//                2. Called and use by yt_commit_grids().
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

    FILL_ARRAY( "grid_left_edge",       grid->left_edge,           3, npy_double )
    FILL_ARRAY( "grid_right_edge",      grid->right_edge,          3, npy_double )
    FILL_ARRAY( "grid_dimensions",      grid->grid_dimensions,     3, npy_int    )
    FILL_ARRAY( "grid_parent_id",      &grid->parent_id,           1, npy_long   )
    FILL_ARRAY( "grid_levels",         &grid->level,               1, npy_int    )
    FILL_ARRAY( "proc_num",            &grid->proc_num,            1, npy_int    )
    if ( g_param_yt.num_par_types > 0 ) {
        FILL_ARRAY("particle_count_list", grid->particle_count_list, g_param_yt.num_par_types, npy_long)
    }

    log_debug( "Inserting grid [%ld] info to libyt.hierarchy ... done\n", grid->id );

    // return if no field data, no need to do anything.
    if ( grid->field_data == NULL ) return YT_SUCCESS;

    // export grid data to libyt.grid_data as "libyt.grid_data[grid_id][field_list.field_name][field_data.data_ptr]"
    // if there is data to append.
    PyObject *py_grid_id, *py_field_labels, *py_field_data;

    // append data to dict only if data is not NULL.
    py_grid_id = PyLong_FromLong( grid->id );
    py_field_labels = PyDict_New();
    for (int v=0; v<g_param_yt.num_fields; v++) {
        if ( (grid->field_data)[v].data_ptr == NULL ) continue;

        // check if dictionary exists, if no add new dict under key gid
        if ( PyDict_Contains( g_py_grid_data, py_grid_id ) != 1 ) {
            PyDict_SetItem( g_py_grid_data, py_grid_id, py_field_labels );
        }

        // insert data under py_field_labels dict
        // (1) Grab NumPy Enumerate Type in order: (1)data_dtype (2)field_dtype
        int grid_dtype;
        if ( get_npy_dtype((grid->field_data)[v].data_dtype, &grid_dtype) == YT_SUCCESS ){
            log_debug("Grid ID [ %ld ], field data [ %s ], grab NumPy enumerate type from data_dtype.\n",
                      grid->id, g_param_yt.field_list[v].field_name);
        }
        else if ( get_npy_dtype(g_param_yt.field_list[v].field_dtype, &grid_dtype) == YT_SUCCESS ){
            (grid->field_data)[v].data_dtype = g_param_yt.field_list[v].field_dtype;
            log_debug("Grid ID [ %ld ], field data [ %s ], grab NumPy enumerate type from field_dtype.\n",
                      grid->id, g_param_yt.field_list[v].field_name);
        }
        else{
            YT_ABORT("Grid ID [ %ld ], field data [ %s ], cannot get the NumPy enumerate type properly.\n",
                     grid->id, g_param_yt.field_list[v].field_name);
        }

        // (2) Get the dimension of the input array
        // Only "cell-centered" will be set to grid_dimensions + ghost cell, else should be set in data_dimensions.
        if ( strcmp(g_param_yt.field_list[v].field_type, "cell-centered") == 0 ){
            // Get grid_dimensions and consider contiguous_in_x or not, since grid_dimensions is defined as [x][y][z].
            if ( g_param_yt.field_list[v].contiguous_in_x ){
                for ( int d=0; d<3; d++ ) { (grid->field_data)[v].data_dimensions[d] = (grid->grid_dimensions)[2-d]; }
            }
            else{
                for ( int d=0; d<3; d++ ) { (grid->field_data)[v].data_dimensions[d] = (grid->grid_dimensions)[d]; }
            }
            // Plus the ghost cell to get the actual array dimensions.
            for(int d = 0; d < 6; d++) {
                (grid->field_data)[v].data_dimensions[ d / 2 ] += g_param_yt.field_list[v].field_ghost_cell[d];
            }
        }
        // See if all data_dimensions > 0, abort if not.
        for (int d = 0; d < 3; d++){
            if ( (grid->field_data)[v].data_dimensions[d] <= 0 ){
                YT_ABORT("Grid ID [ %ld ], field data [ %s ], data_dimensions[%d] = %d <= 0.\n",
                         grid->id, g_param_yt.field_list[v].field_name, d, (grid->field_data)[v].data_dimensions[d]);
            }
        }

        npy_intp grid_dims[3] = { (grid->field_data)[v].data_dimensions[0],
                                  (grid->field_data)[v].data_dimensions[1],
                                  (grid->field_data)[v].data_dimensions[2]};

        // (3) Insert data to dict
        // PyArray_SimpleNewFromData simply creates an array wrapper and does not allocate and own the array
        py_field_data = PyArray_SimpleNewFromData( 3, grid_dims, grid_dtype, (grid->field_data)[v].data_ptr );

        // Mark this memory (NumPy array) read-only
        PyArray_CLEARFLAGS( (PyArrayObject*) py_field_data, NPY_ARRAY_WRITEABLE);

        // add the field data to dict "libyt.grid_data[grid_id][field_list.field_name]"
        PyDict_SetItemString( py_field_labels, g_param_yt.field_list[v].field_name, py_field_data );

        // call decref since PyDict_SetItemString() returns a new reference
        Py_DECREF( py_field_data );

        log_debug( "Inserting grid [%ld] field data [%s] to libyt.grid_data ... done\n",
                   grid->id, g_param_yt.field_list[v].field_name );

    }

    // call decref since both PyLong_FromLong() and PyDict_New() return a new reference
    Py_DECREF( py_grid_id );
    Py_DECREF( py_field_labels );

    return YT_SUCCESS;
}