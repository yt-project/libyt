#include "libyt.h"
#include "libyt_process_control.h"
#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_*
// Description :  Get dimension of the grid with grid id = gid.
//
// Note        :  1. It searches libyt.hierarchy for grid's dimensions, and does not check whether the grid
//                   is on this rank or not.
//                2. grid_dimensions is defined in [x][y][z] <-> [0][1][2] coordinate.
//                3. Since gid doesn't need to be 0-indexed, we need to minus
//                LibytProcessControl::Get().param_yt_.index_offset
//                   when look up.
//                4.    Function                                              Search NumPy Array
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
#ifndef USE_PYBIND11
#define GET_ARRAY(KEY, ARRAY, DIM, TYPE, GID)                                                                          \
    {                                                                                                                  \
        PyArrayObject* py_array_obj =                                                                                  \
            (PyArrayObject*)PyDict_GetItemString(LibytProcessControl::Get().py_hierarchy_, KEY);                       \
        for (int t = 0; t < DIM; t++) {                                                                                \
            (ARRAY)[t] =                                                                                               \
                *(TYPE*)PyArray_GETPTR2(py_array_obj, GID - LibytProcessControl::Get().param_yt_.index_offset, t);     \
        }                                                                                                              \
    }

// function factory
#define GET_GRIDINFO_DIM3(NAME, KEY, TYPE)                                                                             \
    int yt_getGridInfo_##NAME(const long gid, TYPE(*NAME)[3]) {                                                        \
        SET_TIMER(__PRETTY_FUNCTION__);                                                                                \
        if (!LibytProcessControl::Get().commit_grids) {                                                                \
            YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",         \
                     __FUNCTION__);                                                                                    \
        }                                                                                                              \
        GET_ARRAY(KEY, *NAME, 3, TYPE, gid)                                                                            \
        return YT_SUCCESS;                                                                                             \
    }

// function factory
#define GET_GRIDINFO_DIM1(NAME, KEY, TYPE)                                                                             \
    int yt_getGridInfo_##NAME(const long gid, TYPE* NAME) {                                                            \
        SET_TIMER(__PRETTY_FUNCTION__);                                                                                \
        if (!LibytProcessControl::Get().commit_grids) {                                                                \
            YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",         \
                     __FUNCTION__);                                                                                    \
        }                                                                                                              \
        TYPE temp[1];                                                                                                  \
        GET_ARRAY(KEY, temp, 1, TYPE, gid)                                                                             \
        *NAME = temp[0];                                                                                               \
        return YT_SUCCESS;                                                                                             \
    }

// int yt_getGridInfo_Dimensions( const long gid, int (*dimensions)[3] )
GET_GRIDINFO_DIM3(Dimensions, "grid_dimensions", int)

// int yt_getGridInfo_LeftEdge(const long, double (*)[3])
GET_GRIDINFO_DIM3(LeftEdge, "grid_left_edge", double)

// int yt_getGridInfo_RightEdge(const long, double (*)[3])
GET_GRIDINFO_DIM3(RightEdge, "grid_right_edge", double)

// int yt_getGridInfo_ParentId(const long, long *)
GET_GRIDINFO_DIM1(ParentId, "grid_parent_id", long)

// int yt_getGridInfo_Level(const long, int *)
GET_GRIDINFO_DIM1(Level, "grid_levels", int)

// int yt_getGridInfo_ProcNum(const long, int *)
GET_GRIDINFO_DIM1(ProcNum, "proc_num", int)

#undef GET_GRIDINFO_DIM1
#undef GET_GRIDINFO_DIM3
#undef GET_ARRAY
#else
int yt_getGridInfo_Dimensions(const long gid, int (*dimensions)[3]) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    for (int d = 0; d < 3; d++) {
        (*dimensions)[d] = LibytProcessControl::Get()
                               .grid_dimensions[(gid - LibytProcessControl::Get().param_yt_.index_offset) * 3 + d];
    }

    return YT_SUCCESS;
}

int yt_getGridInfo_LeftEdge(const long gid, double (*left_edge)[3]) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    for (int d = 0; d < 3; d++) {
        (*left_edge)[d] = LibytProcessControl::Get()
                              .grid_left_edge[(gid - LibytProcessControl::Get().param_yt_.index_offset) * 3 + d];
    }

    return YT_SUCCESS;
}

int yt_getGridInfo_RightEdge(const long gid, double (*right_edge)[3]) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    for (int d = 0; d < 3; d++) {
        (*right_edge)[d] = LibytProcessControl::Get()
                               .grid_right_edge[(gid - LibytProcessControl::Get().param_yt_.index_offset) * 3 + d];
    }

    return YT_SUCCESS;
}

int yt_getGridInfo_ParentId(const long gid, long* parent_id) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    *parent_id = LibytProcessControl::Get().grid_parent_id[gid - LibytProcessControl::Get().param_yt_.index_offset];

    return YT_SUCCESS;
}

int yt_getGridInfo_Level(const long gid, int* level) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    *level = LibytProcessControl::Get().grid_levels[gid - LibytProcessControl::Get().param_yt_.index_offset];

    return YT_SUCCESS;
}

int yt_getGridInfo_ProcNum(const long gid, int* proc_num) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    *proc_num = LibytProcessControl::Get().proc_num[gid - LibytProcessControl::Get().param_yt_.index_offset];

    return YT_SUCCESS;
}
#endif  // #ifndef USE_PYBIND11

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_ParticleCount
// Description :  Get particle count of particle type ptype in grid gid.
//
// Note        :  1. It searches libyt.hierarchy["par_count_list"][index][ptype],
//                   and does not check whether the grid is on this rank or not.
//                2. Since gid doesn't need to be 0-indexed, we need to minus
//                LibytProcessControl::Get().param_yt_.index_offset
//                   when look up in hierarchy.
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
int yt_getGridInfo_ParticleCount(const long gid, const char* ptype, long* par_count) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    // find index of ptype
    yt_particle* particle_list = LibytProcessControl::Get().particle_list;
    if (particle_list == nullptr) YT_ABORT("No particle.\n");

    int label = -1;
    for (int s = 0; s < LibytProcessControl::Get().param_yt_.num_par_types; s++) {
        if (strcmp(particle_list[s].par_type, ptype) == 0) {
            label = s;
            break;
        }
    }
    if (label == -1) YT_ABORT("Cannot find species name [%s] in particle_list.\n", ptype);

#ifndef USE_PYBIND11
    // get particle count NumPy array in libyt.hierarchy["par_count_list"]
    PyArrayObject* py_array_obj =
        (PyArrayObject*)PyDict_GetItemString(LibytProcessControl::Get().py_hierarchy_, "par_count_list");
    if (py_array_obj == NULL) YT_ABORT("Cannot find key \"par_count_list\" in libyt.hierarchy dict.\n");

    // read libyt.hierarchy["par_count_list"][index][ptype]
    *par_count = *(long*)PyArray_GETPTR2(py_array_obj, gid - LibytProcessControl::Get().param_yt_.index_offset, label);
#else
    long* par_count_list = LibytProcessControl::Get().par_count_list;
    *par_count = par_count_list[(gid - LibytProcessControl::Get().param_yt_.index_offset) *
                                    LibytProcessControl::Get().param_yt_.num_par_types +
                                label];
#endif

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_FieldData
// Description :  Get libyt.grid_data of field_name in the grid with grid id = gid .
//
// Note        :  1. It searches libyt.grid_data[gid][fname], and return YT_FAIL if it cannot find
//                   corresponding data.
//                2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//                3. User should cast to their own datatype after receiving the pointer.
//                4. Returns an existing data pointer and data dimensions user passed in,
//                   and does not make a copy of it!!
//                5. Works only for 3-dim data.
//                6. libyt also use this function to look up data.
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
int yt_getGridInfo_FieldData(const long gid, const char* field_name, yt_data* field_data) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    // get dictionary libyt.grid_data[gid][field_name]
    PyObject* py_grid_id = PyLong_FromLong(gid);
    PyObject* py_field = PyUnicode_FromString(field_name);

    if (PyDict_Contains(LibytProcessControl::Get().py_grid_data_, py_grid_id) != 1 ||
        PyDict_Contains(PyDict_GetItem(LibytProcessControl::Get().py_grid_data_, py_grid_id), py_field) != 1) {
        log_error("Cannot find grid [%ld] data [%s] on MPI rank [%d].\n", gid, field_name,
                  LibytProcessControl::Get().mpi_rank_);
        Py_DECREF(py_grid_id);
        Py_DECREF(py_field);
        return YT_FAIL;
    }
    PyArrayObject* py_array_obj =
        (PyArrayObject*)PyDict_GetItem(PyDict_GetItem(LibytProcessControl::Get().py_grid_data_, py_grid_id), py_field);

    Py_DECREF(py_grid_id);
    Py_DECREF(py_field);

    // get NumPy array dimensions.
    npy_intp* py_array_dims = PyArray_DIMS(py_array_obj);
    for (int d = 0; d < 3; d++) {
        (*field_data).data_dimensions[d] = (int)py_array_dims[d];
    }

    // get NumPy data pointer.
    (*field_data).data_ptr = PyArray_DATA(py_array_obj);

    // get NumPy data dtype, and convert to YT_DTYPE.
    PyArray_Descr* py_array_info = PyArray_DESCR(py_array_obj);
    if (get_yt_dtype_from_npy(py_array_info->type_num, &(*field_data).data_dtype) != YT_SUCCESS) {
        YT_ABORT("No matching yt_dtype for NumPy data type num [%d].\n", py_array_info->type_num);
    }

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_ParticleData
// Description :  Get libyt.particle_data of ptype attr attributes in the grid with grid id = gid.
//
// Note        :  1. It searches libyt.particle_data[gid][ptype][attr], and return YT_FAIL if it cannot
//                   find corresponding data.
//                2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//                3. User should cast to their own datatype after receiving the pointer.
//                4. Returns an existing data pointer and data dimensions user passed in,
//                   and does not make a copy of it!!
//                5. For 1-dim data (like particles), the higher dimensions is set to 0.
//
// Parameter   :  const long   gid              : Target grid id.
//                const char  *ptype            : Target particle type.
//                const char  *attr             : Target attribute name.
//                yt_data     *par_data         : Store the yt_data struct pointer that points to data.
//
// Example     :
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_ParticleData(const long gid, const char* ptype, const char* attr, yt_data* par_data) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    // get dictionary libyt.particle_data[gid][ptype]
    PyObject* py_grid_id = PyLong_FromLong(gid);
    PyObject* py_ptype = PyUnicode_FromString(ptype);
    PyObject* py_attr = PyUnicode_FromString(attr);

    if (PyDict_Contains(LibytProcessControl::Get().py_particle_data_, py_grid_id) != 1 ||
        PyDict_Contains(PyDict_GetItem(LibytProcessControl::Get().py_particle_data_, py_grid_id), py_ptype) != 1 ||
        PyDict_Contains(
            PyDict_GetItem(PyDict_GetItem(LibytProcessControl::Get().py_particle_data_, py_grid_id), py_ptype),
            py_attr) != 1) {
        log_error("Cannot find particle type [%s] attribute [%s] data in grid [%ld] on MPI rank [%d].\n", ptype, attr,
                  gid, LibytProcessControl::Get().mpi_rank_);
        Py_DECREF(py_grid_id);
        Py_DECREF(py_ptype);
        Py_DECREF(py_attr);
        return YT_FAIL;
    }
    PyArrayObject* py_data = (PyArrayObject*)PyDict_GetItem(
        PyDict_GetItem(PyDict_GetItem(LibytProcessControl::Get().py_particle_data_, py_grid_id), py_ptype), py_attr);

    Py_DECREF(py_grid_id);
    Py_DECREF(py_ptype);
    Py_DECREF(py_attr);

    // extracting py_data to par_data
    npy_intp* py_data_dims = PyArray_DIMS(py_data);
    (*par_data).data_dimensions[0] = (int)py_data_dims[0];
    (*par_data).data_dimensions[1] = 0;
    (*par_data).data_dimensions[2] = 0;

    (*par_data).data_ptr = PyArray_DATA(py_data);

    PyArray_Descr* py_data_info = PyArray_DESCR(py_data);
    if (get_yt_dtype_from_npy(py_data_info->type_num, &(*par_data).data_dtype) != YT_SUCCESS) {
        YT_ABORT("No matching yt_dtype for NumPy data type num [%d].\n", py_data_info->type_num);
    }

    return YT_SUCCESS;
}
