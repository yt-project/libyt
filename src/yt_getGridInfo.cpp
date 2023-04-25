#include "yt_combo.h"
#include "libyt.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_*
// Description :  Get dimension of the grid with grid id = gid.
//
// Note        :  1. It searches libyt.hierarchy for grid's dimensions, and does not check whether the grid
//                   is on this rank or not.
//                2. grid_dimensions is defined in [x][y][z] <-> [0][1][2] coordinate.
//                3.    Function                                              Search NumPy Array
//                   --------------------------------------------------------------------------------
//                    yt_getGridInfo_Dimensions(const long, int (*)[3])       libyt.hierarchy["grid_dimensions"]
//                    yt_getGridInfo_LefgEdge(const long, double (*)[3])      libyt.hierarchy["grid_left_edge"]
//                    yt_getGridInfo_RightEdge(const long, double (*)[3])     libyt.hierarchy["grid_right_edge"]
//                    yt_getGridInfo_ParentId(const long, long *)             libyt.hierarchy["grid_parent_id"]
//                    yt_getGridInfo_Level(const long, int *)                 libyt.hierarchy["grid_levels"]
//                    yt_getGridInfo_ProcNum(const long, int *)               libyt.hierarchy["proc_num"]
//
// Example     :  long gid = 0;
//                int dim[3];
//                yt_getGridInfo_Dimensions( gid, &dim );
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
#define GET_ARRAY(KEY, ARRAY, DIM, TYPE, GID)                                                            \
    {                                                                                                    \
        PyArrayObject *py_array_obj = (PyArrayObject*) PyDict_GetItemString( g_py_hierarchy, KEY );      \
        for (int t=0; t<DIM; t++) {                                                                      \
            (ARRAY)[t] = *(TYPE*)PyArray_GETPTR2( py_array_obj, GID, t );                                \
        }                                                                                                \
    }

// function factory
#define GET_GRIDINFO_DIM3(NAME, KEY, TYPE)                                                                            \
    int yt_getGridInfo_##NAME(const long gid, TYPE (*NAME)[3])                                                        \
    {                                                                                                                 \
        if (!g_param_libyt.commit_grids) {                                                                            \
            YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",  \
                     __FUNCTION__);                                                                                   \
        }                                                                                                             \
        GET_ARRAY(KEY, *NAME, 3, TYPE, gid)                                                                           \
        return YT_SUCCESS;                                                                                            \
    }                                                                                                                 \

// function factory
#define GET_GRIDINFO_DIM1(NAME, KEY, TYPE)                                                                            \
    int yt_getGridInfo_##NAME(const long gid, TYPE *NAME)                                                             \
    {                                                                                                                 \
        if (!g_param_libyt.commit_grids) {                                                                            \
            YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",  \
                     __FUNCTION__);                                                                                   \
        }                                                                                                             \
        TYPE temp[1];                                                                                                 \
        GET_ARRAY(KEY, temp, 1, TYPE, gid)                                                                            \
        *NAME = temp[0];                                                                                              \
        return YT_SUCCESS;                                                                                            \
    }                                                                                                                 \

// int yt_getGridInfo_Dimensions( const long gid, int (*dimensions)[3] )
GET_GRIDINFO_DIM3(Dimensions, "grid_dimensions", int)

// int yt_getGridInfo_LeftEdge(const long, double (*)[3])
GET_GRIDINFO_DIM3(LeftEdge, "grid_left_edge", double)

// int yt_getGridInfo_RightEdge(const long, double (*)[3])
GET_GRIDINFO_DIM3(RightEdge, "grid_right_edge", double)

//int yt_getGridInfo_ParentId(const long, long *)
GET_GRIDINFO_DIM1(ParentId, "grid_parent_id", long)

// int yt_getGridInfo_Level(const long, int *)
GET_GRIDINFO_DIM1(Level, "grid_levels", int)

// int yt_getGridInfo_ProcNum(const long, int *)
GET_GRIDINFO_DIM1(ProcNum, "proc_num", int)


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_ParticleCount
// Description :  Get particle count of particle type ptype in grid gid.
//
// Note        :  1. It searches libyt.hierarchy["par_count_list"][gid][ptype], and does not check whether the grid
//                   is on this rank or not.
//
// Parameter   :  const long   gid           : Target grid id.
//                const char  *ptype         : Target particle type.
//                long        *par_count     : Store particle count here.
//
// Example     :  long count;
//                yt_getGridInfo_ParticleCount( gid, "par_type", &count );
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_ParticleCount(const long gid, const char *ptype, long *par_count) {

    if (!g_param_libyt.commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    // find index of ptype
    int label = -1;
    for (int s=0; s<g_param_yt.num_par_types; s++) {
        if (strcmp(g_param_yt.particle_list[s].par_type, ptype) == 0) {
            label = s;
            break;
        }
    }
    if ( label == -1 ) YT_ABORT("Cannot find species name [%s] in particle_list.\n", ptype);

    // get particle count NumPy array in libyt.hierarchy["par_count_list"]
    PyArrayObject *py_array_obj = (PyArrayObject*)PyDict_GetItemString(g_py_hierarchy, "par_count_list");
    if ( py_array_obj == NULL ) YT_ABORT("Cannot find key \"par_count_list\" in libyt.hierarchy dict.\n");

    // read libyt.hierarchy["par_count_list"][gid][ptype]
    *par_count = *(long*)PyArray_GETPTR2(py_array_obj, gid, label);

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_FieldData
// Description :  Get field_data of field_name in the grid with grid id = gid .
//
// Note        :  1. It searches libyt.grid_data[gid][fname], and return YT_FAIL if it cannot find
//                   corresponding data.
//                2. User should cast to their own datatype after receiving the pointer.
//                3. Returns an existing data pointer and data dimensions user passed in,
//                   and does not make a copy of it!!
//                4. Works only for 3-dim data.
//
// Parameter   :  const long   gid              : Target grid id.
//                const char  *field_name       : Target field name.
//                yt_data     *field_data       : Store the yt_data struct pointer that points to data here.
//                
// Example     :  yt_data Data;
//                yt_getGridInfo_FieldData( gid, "field_name", &Data );
//                double *FieldData = (double *) Data.data_ptr;
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_FieldData(const long gid, const char *field_name, yt_data *field_data) {

    if (!g_param_libyt.commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    // get dictionary libyt.grid_data[gid]
    PyObject *py_grid_id = PyLong_FromLong(gid);
    PyObject *py_field_data_list = PyDict_GetItem(g_py_grid_data, py_grid_id);
    if ( py_field_data_list == NULL ) {
        YT_ABORT("Cannot find grid data [%ld] on MPI rank [%d].\n", gid, g_myrank);
    }

    // read NumPy array libyt.grid_data[gid][field_name]
    PyArrayObject *py_array_obj = (PyArrayObject*) PyDict_GetItemString(py_field_data_list, field_name);
    if ( py_array_obj == NULL ) {
        YT_ABORT("Cannot find grid [%ld] data [%s] on MPI rank [%d].\n", gid, field_name, g_myrank);
    }

    // get NumPy array dimensions.
    npy_intp *py_array_dims = PyArray_DIMS(py_array_obj);
    for ( int d=0; d<3; d++ ){
        (*field_data).data_dimensions[d] = (int) py_array_dims[d];
    }

    // get NumPy data pointer.
    (*field_data).data_ptr = PyArray_DATA(py_array_obj);

    // get NumPy data dtype, and convert to YT_DTYPE.
    PyArray_Descr *py_array_info = PyArray_DESCR(py_array_obj);
    if ((py_array_info->type_num) == NPY_FLOAT)           (*field_data).data_dtype = YT_FLOAT;
    else if ((py_array_info->type_num) == NPY_DOUBLE)     (*field_data).data_dtype = YT_DOUBLE;
    else if ((py_array_info->type_num) == NPY_LONGDOUBLE) (*field_data).data_dtype = YT_LONGDOUBLE;
    else if ((py_array_info->type_num) == NPY_INT)        (*field_data).data_dtype = YT_INT;
    else if ((py_array_info->type_num) == NPY_LONG)       (*field_data).data_dtype = YT_LONG;
    else {
        YT_ABORT("No matching yt_dtype for NumPy data type num [%d].\n", py_array_info->type_num);
    }

    // dereference
    Py_DECREF(py_grid_id);

    return YT_SUCCESS;
}