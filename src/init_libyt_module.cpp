#include "yt_combo.h"
#include <string.h>
#include "yt_rma_field.h"
#include "yt_rma_particle.h"
#include "yt_type_array.h"
#include "libyt.h"

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
//                     get_attr_remote       libyt_particle_get_attr_remote
//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_field_derived_func
// Description :  Use the derived function inside yt_field struct to generate the field, then pass back
//                to Python.
//
// Note        :  1. Support only grid dimension = 3 for now.
//                2. This function only needs to deal with the local grids.
//                3. The returned numpy array data type is according to field's field_dtype defined at
//                   yt_field.
//                4. grid_dimensions[3] is in [x][y][z] coordinate.
//                5. Now, input from Python only contains gid and field name. In the future, when we
//                   support hybrid OpenMP/MPI, it can accept list and a string.
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
    yt_dtype field_dtype;

    // TODO: Hybrid OpenMP/MPI, accept a list of gid and a string.
    if ( !PyArg_ParseTuple(args, "ls", &gid, &field_name) ){
        PyErr_SetString(PyExc_TypeError, "Wrong input type, expect to be libyt.derived_func(int, str).");
        return NULL;
    }

    // Get the derived_func define in field_list according to field_name.
    //  (1) If cannot find field_name inside field_list, raise an error.
    //  (2) If we successfully find the field_name, but the derived_func or derived_func_with_name
    //      is not assigned (is NULL), raise an error.
    void (*derived_func) (int, long*, yt_array*);
    void (*derived_func_with_name) (int, long*, char*, yt_array*);
    bool have_FieldName = false;
    short derived_func_option = 0;

    derived_func = NULL;
    derived_func_with_name = NULL;

    for (int v = 0; v < g_param_yt.num_fields; v++){
        if ( strcmp(g_param_yt.field_list[v].field_name, field_name) == 0 ){
            have_FieldName = true;
            field_id = v;
            field_dtype = g_param_yt.field_list[v].field_dtype;
            // The order of derived function being used: (1) derived_func (2) derived_func_with_name
            if ( g_param_yt.field_list[v].derived_func != NULL ){
                derived_func = g_param_yt.field_list[v].derived_func;
                derived_func_option = 1;
            }
            else if ( g_param_yt.field_list[v].derived_func_with_name != NULL ){
                derived_func_with_name = g_param_yt.field_list[v].derived_func_with_name;
                derived_func_option = 2;
            }
            else {
                PyErr_Format(PyExc_NotImplementedError, "In field_list, field_name [ %s ], derived_func or derived_func_with_name does not set properly.\n",
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

    // Get the grid's dimension[3], proc_num according to the gid.
    int  grid_dimensions[3], proc_num;
    if ( yt_getGridInfo_ProcNum(gid, &proc_num) != YT_SUCCESS ||
         yt_getGridInfo_Dimensions(gid, &grid_dimensions) != YT_SUCCESS ){
        PyErr_Format(PyExc_ValueError, "Cannot get grid [%ld] dimensions or MPI rank.\n", gid);
        return NULL;
    }
    if ( proc_num != g_myrank ){
        PyErr_Format(PyExc_ValueError, "Trying to prepare nonlocal grid. Grid [%ld] is on MPI rank [%d].\n", gid, proc_num);
        return NULL;
    }
    for (int d=0; d<3; d++){
        if (grid_dimensions[d] < 0){
            PyErr_Format(PyExc_ValueError, "Trying to prepare grid [%ld] that has grid_dimensions[%d] = %d < 0.\n", gid, d, grid_dimensions[d]);
            return NULL;
        }
    }

    // Generate data using derived_func or derived_func_with_name
    //  (1) Allocate 1D array with size of grid dimension, initialized with 0.
    //  (2) Call derived function.
    //  (3) This array will be wrapped by Numpy API and will be return.
    //      The called object will then OWN this numpy array, so that we don't have to free it.
    // TODO: Hybrid OpenMP/MPI, need to allocate for a list of gid.
    long gridTotalSize = grid_dimensions[0] * grid_dimensions[1] * grid_dimensions[2];
    void *output;
    if ( field_dtype == YT_FLOAT ){
        output = malloc( gridTotalSize * sizeof(float) );
        for (long i = 0; i < gridTotalSize; i++) { ((float *) output)[i] = 0.0; }
    }
    else if ( field_dtype == YT_DOUBLE ){
        output = malloc( gridTotalSize * sizeof(double) );
        for (long i = 0; i < gridTotalSize; i++) { ((double *) output)[i] = 0.0; }
    }
    else if ( field_dtype == YT_INT ){
        output = malloc( gridTotalSize * sizeof(int) );
        for (long i = 0; i < gridTotalSize; i++) { ((int *) output)[i] = 0; }
    }
    else if ( field_dtype == YT_LONG ){
        output = malloc( gridTotalSize * sizeof(long) );
        for (long i = 0; i < gridTotalSize; i++) { ((long *) output)[i] = 0; }
    }
    else{
        PyErr_Format(PyExc_ValueError, "Unknown field_dtype in field [%s]\n", field_name);
        return NULL;
    }

    // Call (1)derived_func or (2)derived_func_with_name, result will be made inside output 1D array.
    // TODO: Hybrid OpenMP/OpenMPI, dynamically ask a list of grid data from derived function.
    //       I assume we get one grid at a time here. Will change later...
    int  list_length = 1;
    long list_gid[1] = {gid};
    yt_array data_array[1];
    data_array[0].gid = gid; data_array[0].data_length = gridTotalSize; data_array[0].data_ptr = output;

    if ( derived_func_option == 1 ){
        (*derived_func) (list_length, list_gid, data_array);
    }
    else if ( derived_func_option == 2 ){
        (*derived_func_with_name) (list_length, list_gid, field_name, data_array);
    }

    // Wrapping the C allocated 1D array into 3D numpy array.
    // grid_dimensions[3] is in [x][y][z] coordinate, 
    // thus we have to check if the field has swap_axes == true or false.
    // TODO: Hybrid OpenMP/MPI, we will need to further pack up a list of gid's field data into Python dictionary.
    int      nd = 3;
    int      typenum;
    npy_intp dims[3];

    if ( get_npy_dtype(field_dtype, &typenum) != YT_SUCCESS ){
        PyErr_Format(PyExc_ValueError, "Unknown yt_dtype, cannot get the NumPy enumerate type properly.\n");
        return NULL;
    }

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
//                2. Deal with local particles only.
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
    void    (*get_attr) (int, long*, char*, yt_array*);
    yt_dtype attr_dtype = YT_DTYPE_UNKNOWN;
    int      species_index = -1;

    for ( int s = 0; s < g_param_yt.num_species; s++ ){
        if ( strcmp(g_param_yt.particle_list[s].species_name, ptype) == 0 ){
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
                    attr_dtype = g_param_yt.particle_list[s].attr_list[p].attr_dtype;
                    break;
                }
            }

            break;
        }
    }

    if ( species_index == -1 ){
        PyErr_Format(PyExc_ValueError, "Cannot find species_name [ %s ] in particle_list.\n", ptype);
        return NULL;
    }
    if ( attr_dtype == YT_DTYPE_UNKNOWN ){
        PyErr_Format(PyExc_ValueError, "species_name [ %s ], attr_name [ %s ] not in particle_list.\n",
                     ptype, attr_name);
        return NULL;
    }

    // Get length of the returned 1D numpy array, which is equal to particle_count_list in the grid.
    long  array_length;
    int proc_num;
    if ( yt_getGridInfo_ProcNum(gid, &proc_num) != YT_SUCCESS ||
         yt_getGridInfo_ParticleCount(gid, ptype, &array_length) != YT_SUCCESS ){
        PyErr_Format(PyExc_ValueError, "Cannot get particle number in grid [%ld] or MPI rank.\n", gid);
        return NULL;
    }
    if ( proc_num != g_myrank ){
        PyErr_Format(PyExc_ValueError, "Trying to prepare nonlocal particles. Grid [%ld] is on MPI rank [%d].\n", gid, proc_num);
        return NULL;
    }
    if ( array_length == 0 ){
        Py_INCREF(Py_None);
        return Py_None;
    }
    if ( array_length < 0 ) {
        PyErr_Format(PyExc_ValueError, "Grid [%ld] particle species [%s] has particle number = %ld < 0.\n",
                     gid, ptype, array_length);
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
        for ( long i = 0; i < array_length; i++ ){ ((int *)output)[i] = 0; }
    }
    else if ( attr_dtype == YT_FLOAT ){
        output = malloc( array_length * sizeof(float) );
        for ( long i = 0; i < array_length; i++ ){ ((float *)output)[i] = 0.0; }
    }
    else if ( attr_dtype == YT_DOUBLE ){
        output = malloc( array_length * sizeof(double) );
        for ( long i = 0; i < array_length; i++ ){ ((double *)output)[i] = 0.0; }
    }
    else if ( attr_dtype == YT_LONG ){
        output = malloc( array_length * sizeof(long) );
        for ( long i = 0; i < array_length; i++ ){ ((long *)output)[i] = 0; }
    }
    else{
        PyErr_Format(PyExc_ValueError, "In species [ %s ] attribute [ %s ], unknown yt_dtype.\n", ptype, attr_name);
        return NULL;
    }
    
    // Call get_attr function pointer
    int  list_length = 1;
    long list_gid[1] = { gid };
    yt_array data_array[1];
    data_array[0].gid = gid; data_array[0].data_length = array_length; data_array[0].data_ptr = output;
    get_attr(list_length, list_gid, attr_name, data_array);

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
//                3. We assume that the fname_list passed in has the same fname order in each rank.
//                4. This function will get all the desired fields and grids.
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
    PyObject *arg1; // fname_list, we will make it an iterable object.
    PyObject *py_prepare_grid_id_list;
    PyObject *py_get_grid_id_list;
    PyObject *py_get_grid_rank_list;

    int  len_fname_list;   // Max number of field is INT_MAX
    int  len_prepare;      // Since maximum number of local grid is INT_MAX
    long len_get_grid;     // Max of total grid number is LNG_MAX

    if ( !PyArg_ParseTuple(args, "OiOiOOl", &arg1, &len_fname_list, &py_prepare_grid_id_list, &len_prepare,
                                            &py_get_grid_id_list, &py_get_grid_rank_list, &len_get_grid) ){
        PyErr_SetString(PyExc_TypeError, "Wrong input type, "
                                         "expect to be libyt.get_field_remote(list, int, list, int, list, list, long).\n");
        return NULL;
    }

    // Make these input lists iterators.
    PyObject *fname_list = PyObject_GetIter( arg1 );
    if( fname_list == NULL ){
        PyErr_SetString(PyExc_TypeError, "fname_list is not an iterable object!\n");
        return NULL;
    }

    // Create Python dictionary for storing remote data.
    // py_output for returning back to python, the others are used temporary inside this method.
    PyObject *py_output = PyDict_New();
    PyObject *py_grid_id, *py_field_label, *py_field_data;

    // Get all remote grid id in field name fname, get one field at a time.
    PyObject *py_fname;
    PyObject *py_prepare_grid_id;
    PyObject *py_get_grid_id;
    PyObject *py_get_grid_rank;
    int root = 0;
    while( py_fname = PyIter_Next( fname_list ) ){
        // Get fname, and create yt_rma_field class.
        char *fname = PyBytes_AsString( py_fname );
        yt_rma_field RMAOperation = yt_rma_field( fname, len_prepare, len_get_grid );

        // Prepare grid with field fname and id = gid.
        // TODO: Hybrid OpenMP/OpenMPI, we might want to prepare a list of gid at one call
        //       if it is a derived field.
        for(int i = 0; i < len_prepare; i++){
            py_prepare_grid_id = PyList_GetItem(py_prepare_grid_id_list, i);
            long gid = PyLong_AsLong( py_prepare_grid_id );
            if( RMAOperation.prepare_data( gid ) != YT_SUCCESS ){
                PyErr_SetString(PyExc_RuntimeError, "Something went wrong in yt_rma_field when preparing data.\n");
                return NULL;
            }
        }
        RMAOperation.gather_all_prepare_data( root );

        // Fetch remote data.
        for(long i = 0; i < len_get_grid; i++){
            py_get_grid_id = PyList_GetItem(py_get_grid_id_list, i);
            py_get_grid_rank = PyList_GetItem(py_get_grid_rank_list, i);
            long get_gid  = PyLong_AsLong( py_get_grid_id );
            int  get_rank = (int) PyLong_AsLong( py_get_grid_rank );
            if( RMAOperation.fetch_remote_data( get_gid, get_rank ) != YT_SUCCESS ){
                PyErr_SetString(PyExc_RuntimeError, "Something went wrong in yt_rma_field when fetching remote data.\n");
                return NULL;
            }
        }

        // Clean up prepared data.
        RMAOperation.clean_up();

        // Get those fetched data and wrap it to NumPy array
        long      get_gid;
        char     *get_fname;
        yt_dtype  get_data_dtype;
        int       get_data_dim[3];
        void     *get_data_ptr;
        long      num_to_get = len_get_grid;
        while( num_to_get > 0 ){
            // Step1: Fetched data.
            if( RMAOperation.get_fetched_data(&get_gid, &get_fname, &get_data_dtype, &get_data_dim, &get_data_ptr) != YT_SUCCESS ){
                // It means we have reached the end of the fetched data container.
                // This if clause is just a safety check.
                break;
            }
            num_to_get -= 1;

            // Step2: Get Python dictionary to append.
            // Check if grid id key exist in py_output, if not create one.
            py_grid_id = PyLong_FromLong( get_gid );
            if( PyDict_Contains(py_output, py_grid_id) == 0 ){
                py_field_label = PyDict_New();
                PyDict_SetItem( py_output, py_grid_id, py_field_label );
                Py_DECREF(py_field_label);
            }
            // Get the Python dictionary under key: grid id, and stored in py_field_label.
            // PyDict_GetItem returns a borrowed reference.
            py_field_label = PyDict_GetItem( py_output, py_grid_id );

            // Step3: Wrap the data to NumPy array and append to dictionary.
            npy_intp npy_dim[3] = { get_data_dim[0], get_data_dim[1], get_data_dim[2] };
            int      npy_dtype;
            get_npy_dtype(get_data_dtype, &npy_dtype);
            py_field_data = PyArray_SimpleNewFromData(3, npy_dim, npy_dtype, get_data_ptr);
            PyArray_ENABLEFLAGS((PyArrayObject*) py_field_data, NPY_ARRAY_OWNDATA);
            PyDict_SetItemString(py_field_label, get_fname, py_field_data);

            // Dereference
            Py_DECREF(py_grid_id);
            Py_DECREF(py_field_data);
        }

        // Done with this py_fname, dereference it.
        Py_DECREF( py_fname );
    }

    // Dereference Python objects
    Py_DECREF( fname_list );

    // Return to Python
    return py_output;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  libyt_particle_get_attr_remote
// Description :  Get non-local particle data from remote ranks.
//
// Note        :  1. We return in dictionary objects.
//                2. We assume that the list of to-get attribute has the same ptype and attr order in each
//                   rank.
//                3. If there are no particles in one grid, then we write Py_None to it.
//
// Parameter   :  dict obj : ptf          : {<ptype>: [<attr1>, <attr2>, ...]} particle type and attributes
//                                          to read.
//                list obj : to_prepare   : list of grid ids you need to prepare.
//                list obj : nonlocal_id  : nonlocal grid id that you want to get.
//                list obj : nonlocal_rank: where to get those nonlocal grid.
//
// Return      :  dict obj data[grid id][ptype][attribute]
//-------------------------------------------------------------------------------------------------------
static PyObject* libyt_particle_get_attr_remote(PyObject *self, PyObject *args){
    // Parse the input list arguments by Python
    PyObject *py_ptf_dict;
    PyObject *arg2, *py_ptf_keys;
    PyObject *py_prepare_list;
    PyObject *py_to_get_list;
    PyObject *py_get_rank_list;

    int  len_prepare;
    long len_to_get;

    if( !PyArg_ParseTuple(args, "OOOiOOl", &py_ptf_dict, &arg2, &py_prepare_list, &len_prepare,
                                            &py_to_get_list, &py_get_rank_list, &len_to_get) ){
        PyErr_SetString(PyExc_TypeError, "Wrong input type, "
                                         "expect to be libyt.get_attr_remote(dict, iter, list, int, list, list, long).\n");
        return NULL;
    }

    py_ptf_keys = PyObject_GetIter( arg2 );
    if( py_ptf_keys == NULL ){
        PyErr_SetString(PyExc_TypeError, "py_ptf_keys is not an iterable object!\n");
        return NULL;
    }

    // Variables for creating output.
    PyObject *py_output = PyDict_New();
    PyObject *py_grid_id, *py_ptype_dict, *py_ptype_key, *py_attribute_dict, *py_par_data;

    // Run through all the py_ptf_dict and its value.
    PyObject *py_ptype;
    PyObject *py_value;
    PyObject *py_attribute, *py_attr_iter;
    PyObject *py_prepare_id, *py_get_id, *py_get_rank;
    int root = 0;
    while( py_ptype = PyIter_Next( py_ptf_keys ) ){

        char *ptype = PyBytes_AsString( py_ptype );

        // Get attribute list inside key ptype in py_ptf_dict.
        // PyDict_GetItemWithError returns a borrowed reference.
        py_value = PyDict_GetItemWithError( py_ptf_dict, py_ptype );
        if( py_value == NULL ){
            PyErr_Format(PyExc_KeyError, "py_ptf_dict has no key [ %s ].\n", ptype);
            return NULL;
        }
        py_attr_iter = PyObject_GetIter( py_value );

        // Iterate through attribute list, and perform RMA operation.
        while( py_attribute = PyIter_Next( py_attr_iter ) ){
            // Initialize RMA operation
            char *attr = PyBytes_AsString( py_attribute );
            yt_rma_particle RMAOperation = yt_rma_particle( ptype, attr, len_prepare, len_to_get );

            // Prepare particle data in grid gid.
            for(int i = 0; i < len_prepare; i++){
                py_prepare_id = PyList_GetItem( py_prepare_list, i );
                long gid = PyLong_AsLong( py_prepare_id );
                if( RMAOperation.prepare_data( gid ) != YT_SUCCESS ){
                    PyErr_SetString(PyExc_RuntimeError, "Something went wrong in yt_rma_particle when preparing data.\n");
                    return NULL;
                }
            }
            RMAOperation.gather_all_prepare_data( root );

            // Fetch remote data.
            for(long i = 0; i < len_to_get; i++){
                py_get_id = PyList_GetItem( py_to_get_list, i );
                py_get_rank = PyList_GetItem( py_get_rank_list, i );
                long get_gid = PyLong_AsLong( py_get_id );
                int  get_rank = (int) PyLong_AsLong( py_get_rank );
                if( RMAOperation.fetch_remote_data( get_gid, get_rank ) != YT_SUCCESS ){
                    PyErr_SetString(PyExc_RuntimeError, "Something went wrong in yt_rma_particle when fetching remote data.\n");
                    return NULL;
                }
            }

            // Clean up.
            RMAOperation.clean_up();

            // Get fetched data, and wrap up to NumPy Array, then store inside py_output.
            long      get_gid;
            char     *get_ptype;
            char     *get_attr;
            yt_dtype  get_data_dtype;
            long      get_data_len;
            void     *get_data_ptr;
            long num_to_get = len_to_get;
            while( num_to_get > 0 ){
                // Step1: Fetch data.
                if ( RMAOperation.get_fetched_data(&get_gid, &get_ptype, &get_attr, &get_data_dtype, &get_data_len, &get_data_ptr) != YT_SUCCESS ){
                    break;
                }
                num_to_get -= 1;

                // Step2: Get python dictionary to append data to.
                // Check if the grid id key exist in py_output, if not create one.
                py_grid_id = PyLong_FromLong( get_gid );
                if( PyDict_Contains(py_output, py_grid_id) == 0 ){
                    py_ptype_dict = PyDict_New();
                    PyDict_SetItem( py_output, py_grid_id, py_ptype_dict );
                    Py_DECREF( py_ptype_dict );
                }
                // Get python dictionary under key: py_grid_id. Stored in py_ptype_dict.
                py_ptype_dict = PyDict_GetItem( py_output, py_grid_id );
                Py_DECREF(py_grid_id);

                // Check if py_ptype_key exist in py_ptype_dict, if not create one.
                py_ptype_key = PyUnicode_FromString( get_ptype );
                if( PyDict_Contains(py_ptype_dict, py_ptype_key) == 0 ){
                    py_attribute_dict = PyDict_New();
                    PyDict_SetItem( py_ptype_dict, py_ptype_key, py_attribute_dict );
                    Py_DECREF( py_attribute_dict );
                }
                // Get python dictionary under key: py_ptype_key. Stored in py_attribute_dict.
                py_attribute_dict = PyDict_GetItem( py_ptype_dict, py_ptype_key );
                Py_DECREF(py_ptype_key);

                // Step3: Wrap the data to NumPy array if ptr is not NULL and append to dictionary.
                //        Or else append None to dictionary.
                if( get_data_len == 0 ){
                    PyDict_SetItemString(py_attribute_dict, get_attr, Py_None);
                }
                else if ( get_data_len > 0 && get_data_ptr != NULL ) {
                    int nd = 1;
                    int npy_type;
                    npy_intp dims[1] = { get_data_len };
                    get_npy_dtype( get_data_dtype, &npy_type );
                    py_par_data = PyArray_SimpleNewFromData(nd, dims, npy_type, get_data_ptr);
                    PyArray_ENABLEFLAGS( (PyArrayObject*) py_par_data, NPY_ARRAY_OWNDATA );
                    PyDict_SetItemString(py_attribute_dict, get_attr, py_par_data);
                    Py_DECREF(py_par_data);
                }
                else {
                    PyErr_SetString(PyExc_RuntimeError, "Something went wrong in yt_rma_particle when fetching remote data.\n");
                    return NULL;
                }
            }

            // Free unused resource
            Py_DECREF( py_attribute );
        }

        // Free unused resource.
        Py_DECREF( py_attr_iter );
        Py_DECREF( py_ptype );
    }

    // Free unneeded resource.
    Py_DECREF( py_ptf_keys );

    // Return.
    return py_output;
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
    "Get local derived field data."},
   {"get_attr",         libyt_particle_get_attr,  METH_VARARGS,
    "Get local particle attribute data."},
   {"get_field_remote", libyt_field_get_field_remote, METH_VARARGS,
    "Get remote field data."},
   {"get_attr_remote",  libyt_particle_get_attr_remote, METH_VARARGS,
    "Get remote particle attribute data."},
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

#ifdef INTERACTIVE_MODE
  g_py_interactive_mode = PyDict_New();
  PyModule_AddObject(libyt_module, "interactive_mode", g_py_interactive_mode);
#endif // #ifdef INTERACTIVE_MODE

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
#ifdef SUPPORT_TIMER
  g_timer->record_time("create_libyt_module", 0);
#endif

  PyImport_AppendInittab("libyt", &PyInit_libyt);

#ifdef SUPPORT_TIMER
  g_timer->record_time("create_libyt_module", 1);
#endif

  return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Function    :  init_libyt_module
// Description :  Initialize the libyt module
//
// Note        :  1. Import newly created libyt module.
//                2. Load user script to python.
//                3. Add imported script's __dict__ under in libyt.interactive_mode["script_namespace"]
//                
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int init_libyt_module()
{
#ifdef SUPPORT_TIMER
   g_timer->record_time("init_libyt_module", 0);
#endif

#ifdef SUPPORT_TIMER
    g_timer->record_time("import-libyt", 0);
#endif

// import newly created libyt module
   if ( PyRun_SimpleString("import libyt\n") == 0 )
      log_debug( "Import libyt module ... done\n" );
   else
      YT_ABORT(  "Import libyt module ... failed!\n" );

#ifdef SUPPORT_TIMER
    g_timer->record_time("import-libyt", 1);
#endif

#ifdef SUPPORT_TIMER
    g_timer->record_time("import-userscript", 0);
#endif
// import YT inline analysis script
   int command_width = 8 + strlen( g_param_libyt.script );   // 8 = "import " + '\0'
   char *command = (char*) malloc( command_width * sizeof(char) );
   sprintf( command, "import %s", g_param_libyt.script );

   if ( PyRun_SimpleString( command ) == 0 )
      log_debug( "Importing YT inline analysis script \"%s\" ... done\n", g_param_libyt.script );
   else
      YT_ABORT(  "Importing YT inline analysis script \"%s\" ... failed (please do not include the \".py\" extension)!\n",
                g_param_libyt.script );

   free( command );
#ifdef SUPPORT_TIMER
    g_timer->record_time("import-userscript", 1);
#endif

#ifdef INTERACTIVE_MODE
    // add imported script's namespace under in libyt.interactive_mode["script_globals"]
    // 67 -> libyt.interactive_mode["script_globals"] = sys.modules[""].__dict__
    //  1 -> '\0'
    command_width = 68 + strlen(g_param_libyt.script);
    command = (char*) malloc( command_width * sizeof(char) );
    sprintf( command, "libyt.interactive_mode[\"script_globals\"] = sys.modules[\"%s\"].__dict__", g_param_libyt.script);

    if ( PyRun_SimpleString( command ) == 0 ){
        log_debug("Loading imported script's global variables ... done\n");
    }
    else {
        YT_ABORT("Loading imported script's global variables ... failed\n");
    }

    free(command);
#endif // #ifdef INTERACTIVE_MODE

#ifdef SUPPORT_TIMER
   g_timer->record_time("init_libyt_module", 1);
#endif

   return YT_SUCCESS;

} // FUNCTION : init_libyt_module
