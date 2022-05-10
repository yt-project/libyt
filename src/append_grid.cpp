#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  append_grid
// Description :  Add a single full grid to the libyt Python module
//
// Note        :  1. Store the input "grid" to libyt.hierarchy and libyt.grid_data to python
//                2. Called and use by yt_commit_grids().
//                3. If field_data == NULL, we append Py_None to the dictionary.
//                4. We assign data_dimensions and data_dtype in yt_data if user only sets its macros in grid_dimensions
//                   and field_dtype.
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
    FILL_ARRAY( "grid_dimensions",      grid->grid_dimensions,     3, npy_long   )
    FILL_ARRAY( "grid_particle_count", &grid->grid_particle_count, 1, npy_long   )
    FILL_ARRAY( "grid_parent_id",      &grid->parent_id,           1, npy_long   )
    FILL_ARRAY( "grid_levels",         &grid->level,               1, npy_long   )
    FILL_ARRAY( "proc_num",            &grid->proc_num,            1, npy_int    )
    log_debug( "Inserting grid [%ld] info to libyt.hierarchy ... done\n", grid->id );

// export grid data to libyt.grid_data as "libyt.grid_data[grid_id][field_list.field_name][field_data.data_ptr]"
    PyObject *py_grid_id, *py_field_labels, *py_field_data;

// allocate [grid_id][field_list.field_name]
    py_grid_id      = PyLong_FromLong( grid->id );
    py_field_labels = PyDict_New();

    PyDict_SetItem( g_py_grid_data, py_grid_id, py_field_labels );

// fill [grid_id][field_list.field_name][field_data.data_ptr]
    for (int v=0; v<g_param_yt.num_fields; v++)
    {
//    If grid->field_data == NULL, append Py_None to dictionary
        if ( grid->field_data == NULL ) {
            // add Py_None to dict "libyt.grid_data[grid_id][field_list.field_name]"
            PyDict_SetItemString( py_field_labels, g_param_yt.field_list[v].field_name, Py_None );
        }

//    Else grid->field_data != NULL, append the data.
//    Or else append Py_None.
        else {
            if ( (grid->field_data)[v].data_ptr == NULL ){
                // add Py_None to dict "libyt.grid_data[grid_id][field_list.field_name]"
                PyDict_SetItemString( py_field_labels, g_param_yt.field_list[v].field_name, Py_None );

                log_debug( "Inserting [ None ] to grid [%ld] field data [%s] to libyt.grid_data, since data_ptr == NULL\n",
                           grid->id, g_param_yt.field_list[v].field_name );
            }
            else {
                // Grab NumPy Enumerate Type in order: (1)data_dtype (2)field_dtype
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

                // Get the dimension of the input array
                // Only "cell-centered" will be set to grid_dimensions + ghost cell, else should be set in data_dimensions.
                if ( strcmp(g_param_yt.field_list[v].field_define_type, "cell-centered") == 0 ){
                    // Get grid_dimensions and consider swap_axes or not, since grid_dimensions is defined as [x][y][z].
                    if ( g_param_yt.field_list[v].swap_axes == true ){
                        (grid->field_data)[v].data_dimensions[0] = (grid->grid_dimensions)[2];
                        (grid->field_data)[v].data_dimensions[1] = (grid->grid_dimensions)[1];
                        (grid->field_data)[v].data_dimensions[2] = (grid->grid_dimensions)[0];
                    }
                    else{
                        (grid->field_data)[v].data_dimensions[0] = (grid->grid_dimensions)[0];
                        (grid->field_data)[v].data_dimensions[1] = (grid->grid_dimensions)[1];
                        (grid->field_data)[v].data_dimensions[2] = (grid->grid_dimensions)[2];
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

                // PyArray_SimpleNewFromData simply creates an array wrapper and does not allocate and own the array
                py_field_data = PyArray_SimpleNewFromData( 3, grid_dims, grid_dtype, (grid->field_data)[v].data_ptr );

                // add the field data to dict "libyt.grid_data[grid_id][field_list.field_name]"
                PyDict_SetItemString( py_field_labels, g_param_yt.field_list[v].field_name, py_field_data );

                // call decref since PyDict_SetItemString() returns a new reference
                Py_DECREF( py_field_data );

                log_debug( "Inserting grid [%ld] field data [%s] to libyt.grid_data ... done\n",
                           grid->id, g_param_yt.field_list[v].field_name );
            }

        }

    }

// call decref since both PyLong_FromLong() and PyDict_New() return a new reference
    Py_DECREF( py_grid_id );
    Py_DECREF( py_field_labels );

    return YT_SUCCESS;
}