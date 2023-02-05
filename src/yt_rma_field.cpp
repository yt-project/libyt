#include "yt_rma_field.h"
#include "yt_combo.h"
#include "libyt.h"
#include <string.h>

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  Constructor
//
// Notes       :  1. Initialize m_Window, which used inside OpenMPI RMA operation. And set m_Window info
//                   to "no_locks".
//                2. Copy the input fname to m_FieldName, in case it is freed.
//                3. Find the corresponding field_type and contiguous_in_x in field_list, and assign to
//                   m_FieldDefineType and m_FieldSwapAxes.
//                4. Find field index inside field_list and assign to m_FieldIndex.
//                5. Set the std::vector capacity.
//
// Arguments   :  const char *fname       : Field name.
//                int         len_prepare : Number of grid to prepare.
//                long        len_get_grid: Number of grid to get.
//-------------------------------------------------------------------------------------------------------
yt_rma_field::yt_rma_field(const char* fname, int len_prepare, long len_get_grid)
: m_LenAllPrepare(0), m_FieldIndex(-1)
{
    // Initialize m_Window and set info to "no_locks".
    MPI_Info windowInfo;
    MPI_Info_create( &windowInfo );
    MPI_Info_set( windowInfo, "no_locks", "true" );
    int mpi_return_code = MPI_Win_create_dynamic( windowInfo, MPI_COMM_WORLD, &m_Window );
    MPI_Info_free( &windowInfo );

    if (mpi_return_code != MPI_SUCCESS) {
        log_error("yt_rma_field: create one-sided MPI (RMA) window failed!\n");
        log_error("yt_rma_field: try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".\n");
    }

    for(int v=0; v < g_param_yt.num_fields; v++){
        if( strcmp(fname, g_param_yt.field_list[v].field_name) == 0){
            m_FieldName       = g_param_yt.field_list[v].field_name;
            m_FieldDefineType = g_param_yt.field_list[v].field_type;
            m_FieldIndex      = v;
            m_FieldSwapAxes   = g_param_yt.field_list[v].contiguous_in_x;
            break;
        }
    }

    // Set std::vector capacity
    m_Prepare.reserve(len_prepare);
    m_PrepareData.reserve(len_prepare);
    m_Fetched.reserve(len_get_grid);
    m_FetchedData.reserve(len_get_grid);

    log_debug("yt_rma_field: Field Name = %s\n", m_FieldName);
    log_debug("yt_rma_field: Field Define Type = %s\n", m_FieldDefineType);
    log_debug("yt_rma_field: Field Swap Axes = %s\n", m_FieldSwapAxes ? "true" : "false");
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  Destructor
//
// Notes       :  1. Free m_Window.
//                2. Clear m_Fetched and m_FetchedData.
//
// Arguments   :  None
//-------------------------------------------------------------------------------------------------------
yt_rma_field::~yt_rma_field()
{
    MPI_Win_free(&m_Window);
    m_Fetched.clear();
    m_FetchedData.clear();
    log_debug("yt_rma_field: Destructor called.\n");
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  prepare_data
// Description :  Prepare data grid = gid and field = m_FieldName, then attach grid data to window and
//                get rma address.
//
// Notes       :  1. Prepare the grid data with id = gid, and attach grid data to window.
//                2. Insert data pointer into m_PrepareData.
//                3. Insert data information into m_Prepare.
//                4. In "cell-centered" and "face-centered", we pass full data_ptr, including ghost cell.
//                5. "derived_func" data_dimensions must be the same as grid dim after considering
//                   contiguous_in_x, since derived_func generates only data without ghost cell.
//                6. We assume that all input gid can be found on this rank.
//                7. Check that we indeed get data pointer and its dimensions and data type.
//
// Arguments   :  long gid : Grid id to prepare.
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_field::prepare_data(long& gid)
{
    // Make sure that the field exist.
    if( m_FieldIndex == -1 ){
        YT_ABORT("yt_rma_field: Cannot find field name [ %s ] in field_list on MPI rank [ %d ].\n", m_FieldName, g_myrank);
    }

    // Get grid info
    yt_rma_grid_info grid_info;
    grid_info.id = gid;
    yt_getGridInfo_ProcNum(gid, &(grid_info.rank));
    if ( grid_info.rank != g_myrank ) {
        YT_ABORT("yt_rma_field: Trying to prepare nonlocal grid [%ld] locate on MPI rank [%d].\n", gid, grid_info.rank);
    }

    // Get data dimensions, data type, and data pointer
    void *data_ptr = NULL;
    if ( strcmp(m_FieldDefineType, "derived_func") == 0 ) {
        // get data dimensions
        int grid_dimensions[3];
        if ( yt_getGridInfo_Dimensions( gid, &grid_dimensions ) != YT_SUCCESS ) {
            YT_ABORT("yt_rma_field: Failed to get grid [%ld] dimensions.\n", gid);
        }
        if ( m_FieldSwapAxes ) {
            for (int d=0; d<3; d++) {grid_info.data_dim[d] = grid_dimensions[2 - d]; }
        }
        else{
            for (int d=0; d<3; d++){ grid_info.data_dim[d] = grid_dimensions[d]; }
        }

        // get data type
        grid_info.data_dtype = g_param_yt.field_list[m_FieldIndex].field_dtype;

        // allocate data_ptr
        long gridLength = grid_info.data_dim[0] * grid_info.data_dim[1] * grid_info.data_dim[2];
        if ( grid_info.data_dtype == YT_FLOAT ){
            data_ptr = malloc( gridLength * sizeof(float) );
            for(long i = 0; i < gridLength; i++){ ((float *) data_ptr)[i] = 0.0; }
        }
        else if ( grid_info.data_dtype == YT_DOUBLE ){
            data_ptr = malloc( gridLength * sizeof(double) );
            for(long i = 0; i < gridLength; i++){ ((double *) data_ptr)[i] = 0.0; }
        }
        else if ( grid_info.data_dtype == YT_INT ){
            data_ptr = malloc( gridLength * sizeof(int) );
            for(long i = 0; i < gridLength; i++){ ((int *) data_ptr)[i] = 0; }
        }
        else if ( grid_info.data_dtype == YT_LONG ){
            data_ptr = malloc( gridLength * sizeof(long) );
            for(long i = 0; i < gridLength; i++){ ((long *) data_ptr)[i] = 0; }
        }
        else{
            YT_ABORT("yt_rma_field: Unknown field_dtype.\n");
        }

        // get derived function and generate data
        int  list_length = 1;
        long list_gid[1] = {gid};
        yt_array data_array[1];
        data_array[0].gid = gid; data_array[0].data_length = gridLength; data_array[0].data_ptr = data_ptr;

        if ( g_param_yt.field_list[m_FieldIndex].derived_func != NULL ){
            void (*derived_func) (const int, const long*, const char*, yt_array*);
            derived_func = g_param_yt.field_list[m_FieldIndex].derived_func;
            (*derived_func) (list_length, list_gid, m_FieldName, data_array);
        }
        else{
            YT_ABORT("yt_rma_field: In field [%s], field_type == %s, but derived_func not set!\n",
                     m_FieldName, m_FieldDefineType);
        }
    }
    else if ( strcmp(m_FieldDefineType, "cell-centered") == 0 || strcmp(m_FieldDefineType, "face-centered") == 0 ) {
        // get data dimensions, data type, and data pointer
        yt_data field_data;
        if ( yt_getGridInfo_FieldData(gid, m_FieldName, &field_data) != YT_SUCCESS ){
            YT_ABORT("yt_rma_field: Failed to get field data %s in grid [%ld].\n", m_FieldName, gid);
        }
        for (int d=0; d<3; d++){ grid_info.data_dim[d] = field_data.data_dimensions[d]; }
        grid_info.data_dtype = field_data.data_dtype;
        data_ptr = field_data.data_ptr;
    }
    else{
        YT_ABORT("yt_rma_field: Unknown field define type %s.\n", m_FieldDefineType);
    }

    // Attach buffer to window.
    int size;
    if( get_dtype_size( grid_info.data_dtype, &size ) != YT_SUCCESS ){
        YT_ABORT("yt_rma_field: Unknown data type.\n");
    }
    if( data_ptr == NULL ){
        YT_ABORT("yt_rma_field: Unable to get GID [%ld] field [%s] data.\n", gid, m_FieldName);
    }

    int mpi_return_code = MPI_Win_attach(m_Window, data_ptr, grid_info.data_dim[0] * grid_info.data_dim[1] * grid_info.data_dim[2] * size);
    if( mpi_return_code != MPI_SUCCESS ){
        log_error("yt_rma_field: attach data buffer to one-sided MPI (RMA) window failed!\n");
        log_error("yt_rma_field: try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".\n");
        YT_ABORT("yt_rma_field: Attach buffer to window failed.\n");
    }

    // Get the address of the attached buffer.
    if( MPI_Get_address(data_ptr, &(grid_info.address)) != MPI_SUCCESS ){
        YT_ABORT("yt_rma_field: Get attached buffer address failed.\n");
    }

    // Push back data_ptr to m_PrepareData, and grid_info to m_Prepare.
    m_PrepareData.push_back( data_ptr );
    m_Prepare.push_back( grid_info );

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  gather_all_prepare_data
// Description :  Gather all prepared data in each rank.
//
// Notes       :  1. This should be called after preparing all the needed grid data.
//                2. Perform big_MPI_Gatherv and big_MPI_Bcast at root rank.
//                3. Set up m_SearchRange and m_LenAllPrepare, will later be used in fetch_remote_data to
//                   search grid id in m_AllPrepare.
//                4. Open the window epoch.
//
// Parameter   :  int root : root rank.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_field::gather_all_prepare_data(int root)
{
    int NRank;
    MPI_Comm_size(MPI_COMM_WORLD, &NRank);

    // Get m_Prepare data and its length
    int PreparedInfoListLength = m_Prepare.size();
    yt_rma_grid_info *PreparedInfoList = m_Prepare.data();

    // MPI_Gather send count in each rank, then MPI_Bcast.
    int *SendCount = new int [NRank];
    MPI_Gather(&PreparedInfoListLength, 1, MPI_INT, SendCount, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(SendCount, NRank, MPI_INT, root, MPI_COMM_WORLD);
    
    // Calculate m_SearchRange, and m_LenAllPrepare.
    m_SearchRange = new long [ NRank + 1 ];
    for(int i = 0; i < NRank + 1; i++) {
        m_SearchRange[i] = 0;
        for (int rank = 0; rank < i; rank++) {
            m_SearchRange[i] += SendCount[rank];
        }
    }
    m_LenAllPrepare = m_SearchRange[ NRank ];

    // Gather PreparedInfoList, which is m_Prepare in each rank, perform big_MPI_Gatherv and big_MPI_Bcast
    m_AllPrepare = new yt_rma_grid_info [m_LenAllPrepare];
    big_MPI_Gatherv(root, SendCount, (void*)PreparedInfoList, &yt_rma_grid_info_mpi_type, (void*)m_AllPrepare, 1);
    big_MPI_Bcast(root, m_LenAllPrepare, (void*)m_AllPrepare, &yt_rma_grid_info_mpi_type, 1);

    // Open window epoch.
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, m_Window);

    // Free unused resource
    delete [] SendCount;

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  fetch_remote_data
// Description :  Allocate spaces and fetch remote data, and store in std::vector m_Fetched, m_FetchData.
//
// Notes       :  1. Look for grid info in m_AllPrepare, and allocate memory.
//                2. We allocate buffer to store fetched data, but we do not free them. It is Python's
//                   responsibility.
//
// Parameters  : long gid  : Grid id to fetch.
//               int  rank : Fetch grid from rank.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_field::fetch_remote_data(long& gid, int& rank)
{
    // Look for gid in m_AllPrepare, and allocate memory.
    bool get_remote_grid = false;
    long  gridLength;
    int  dtype_size;
    MPI_Datatype mpi_dtype;
    yt_rma_grid_info fetched;
    for(long aid = m_SearchRange[rank]; aid < m_SearchRange[rank+1]; aid++){
        if( m_AllPrepare[aid].id == gid ){
            // Copy fetched grid info to fetch.
            fetched = m_AllPrepare[aid];
            for(int d = 0; d < 3; d++){
                fetched.data_dim[d] = m_AllPrepare[aid].data_dim[d];
            }
            // Grab data length, size, mpi type.
            gridLength = fetched.data_dim[0] * fetched.data_dim[1] * fetched.data_dim[2];
            get_dtype_size(fetched.data_dtype, &dtype_size);
            get_mpi_dtype(fetched.data_dtype, &mpi_dtype);

            get_remote_grid = true;
            break;
        }
    }
    if( get_remote_grid != true ){
        YT_ABORT("yt_rma_field: Cannot get remote grid id [ %ld ] located in rank [ %d ] on MPI rank [ %d ].\n",
                 gid, rank, g_myrank);
    }
    void *fetchedData = malloc( gridLength * dtype_size );

    // Fetch data and info
    if( big_MPI_Get(fetchedData, gridLength, &(fetched.data_dtype), &mpi_dtype, rank, fetched.address, &m_Window) != YT_SUCCESS ){
        YT_ABORT("yt_rma_field: big_MPI_Get fetch remote grid [ %ld ] located on rank [ %d ] failed!\n", gid, rank);
    }

    // Push back to m_Fetched and m_FetchedData
    m_Fetched.push_back(fetched);
    m_FetchedData.push_back(fetchedData);

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  clean_up
// Description :  Clean up prepared data.
//
// Notes       :  1. Close the window epoch, and detach the prepared data buffer.
//                2. Free m_AllPrepare, m_SearchRange.
//                3. Free local prepared data, drop m_Prepare vector and free pointers in m_PrepareData
//                   vector if m_FieldDefineType is "derived_func".
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_field::clean_up()
{
    // Close the window epoch
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, m_Window);

    // Detach m_PrepareData from m_Window.
    for(int i = 0; i < (int)m_PrepareData.size(); i++){
        MPI_Win_detach(m_Window, m_PrepareData[i]);
    }

    // Free local prepared data m_Prepare, m_PrepareData if field_type == "derived_func".
    if( strcmp(m_FieldDefineType, "derived_func") == 0 ) {
        for(int i = 0; i < (int)m_PrepareData.size(); i++) {
            free( m_PrepareData[i] );
        }
    }

    // Free m_AllPrepare, m_SearchRange, m_PrepareData, m_Prepare
    delete [] m_AllPrepare;
    delete [] m_SearchRange;
    m_PrepareData.clear();
    m_Prepare.clear();

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  get_fetched_data
// Description :  Get fetched data.
//
// Notes       :  1. Get fetched data one by one. If vector is empty, then it will return YT_FAIL.
//                2. Whenever one gets the data from m_Fetched and m_FetchedData, it will also remove it
//                   from std::vector.
//                3. Write fetched data and info to passed in parameters.
//
// Parameters  :  long *gid     :  Grid id fetched.
//                char **fname  :  Field name fetched.
//                yt_dtype *data_dtype : Fetched data type.
//                int (*data_dim)[3]   : Fetched data dimensions, in the point of view of the data_ptr.
//                void **data_ptr      : Fetched data pointer.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_field::get_fetched_data(long *gid, const char **fname, yt_dtype *data_dtype, int (*data_dim)[3], void **data_ptr){
    // Check if there are left fetched data to get.
    if( m_Fetched.size() == 0 ){
        return YT_FAIL;
    }

    // Access m_Fetched and m_FetchedData from the back.
    yt_rma_grid_info fetched = m_Fetched.back();
    *gid = fetched.id;
    *fname = m_FieldName;
    *data_dtype = fetched.data_dtype;
    for(int d = 0; d < 3; d++){
        (*data_dim)[d] = fetched.data_dim[d];
    }
    *data_ptr = m_FetchedData.back();

    // Pop back fetched data.
    m_Fetched.pop_back();
    m_FetchedData.pop_back();

    return YT_SUCCESS;
}