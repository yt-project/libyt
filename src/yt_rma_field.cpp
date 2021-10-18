#include "yt_rma_field.h"
#include <string.h>

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  Constructor
//
// Notes       :  1. Initialize m_Window, which used inside OpenMPI RMA operation. And set m_Window info
//                   to "no_locks".
//                2. Copy the input fname to m_FieldName, in case it is freed.
//                3. Find the corresponding field_define_type and swap_axes in field_list, and assign to
//                   m_FieldDefineType and m_FieldSwapAxes.
//                   (We assume that the lifetime of its element exist.)
//                4. Find field index inside field_list and assign to m_FieldIndex.
//                5. Set the std::vector capacity.
//
// Arguments   :  char*       fname: Field name.
//                int   len_prepare: Number of grid to prepare.
//                long len_get_grid: Number of grid to get.
//-------------------------------------------------------------------------------------------------------
yt_rma_field::yt_rma_field(char* fname, int len_prepare, long len_get_grid)
: m_LenAllPrepare(0), m_FieldIndex(-1)
{
    // Initialize m_Window and set info to "no_locks".
    MPI_Info windowInfo;
    MPI_Info_create( &windowInfo );
    MPI_Info_set( windowInfo, "no_locks", "true" );
    MPI_Win_create_dynamic( windowInfo, MPI_COMM_WORLD, &m_Window );
    MPI_Info_free( &windowInfo );

    // Copy input fname, and find its field_define_type
    int len = strlen(fname);
    m_FieldName = new char [len+1];
    strcpy(m_FieldName, fname);

    for(int v=0; v < g_param_yt.num_fields; v++){
        if( strcmp(m_FieldName, g_param_yt.field_list[v].field_name) == 0){
            m_FieldDefineType = g_param_yt.field_list[v].field_define_type;
            m_FieldIndex      = v;
            m_FieldSwapAxes   = g_param_yt.field_list[v].swap_axes;
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
// Notes       :  1. Free m_Window, m_FieldName.
//                2. Clear m_Fetched and m_FetchedData.
//
// Arguments   :  None
//-------------------------------------------------------------------------------------------------------
yt_rma_field::~yt_rma_field()
{
    MPI_Win_free(&m_Window);
    delete [] m_FieldName;
    m_Fetched.clear();
    m_FetchedData.clear();
    log_debug("yt_rma_field: Destructor called.\n");
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_field
// Method      :  prepare_data
// Description :  Prepare data grid = gid and field = m_FieldName, create data if not exist,
//                then attach grid data to window and get address.
//
// Notes       :  1. Prepare the grid data with id = gid, and attach grid data to window.
//                2. Insert data pointer into m_PrepareData.
//                3. Insert data information into m_Prepare.
//                4. The data_dim is in the point of view of the data array, only "face-centered" data
//                   can have data dim different from grid dim.
//                5. "derived_func" data type must be YT_DOUBLE.
//                6. We assume that all gid can be found on this rank.
//
// Arguments   :  long gid : Grid id to prepare.
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_field::prepare_data(long& gid)
{
    // Make sure that the field exist.
    if( m_FieldIndex == -1 ){
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        YT_ABORT("yt_rma_field: Cannot find field name [ %s ] in field_list on MPI rank [ %d ].\n", m_FieldName, myrank);
    }

    // Get grid info
    int local_index = -1;
    yt_rma_grid_info grid_info;
    grid_info.id = gid;

    for(int lid = 0; lid < g_param_yt.num_grids_local; lid++){
        if( g_param_yt.grids_local[lid].id == gid ){
            local_index = lid;
            grid_info.rank = g_param_yt.grids_local[lid].proc_num;

            // Get data dimensions. "face-centered" can differ from grid's.
            if( strcmp(m_FieldDefineType, "face-centered") == 0 ){
                for(int d = 0; d < 3; d++){
                    grid_info.data_dim[d] = g_param_yt.grids_local[lid].field_data[m_FieldIndex].data_dim[d];
                }
            }
            else{
                if( m_FieldSwapAxes ){
                    grid_info.data_dim[0] = g_param_yt.grids_local[lid].grid_dimensions[2];
                    grid_info.data_dim[1] = g_param_yt.grids_local[lid].grid_dimensions[1];
                    grid_info.data_dim[2] = g_param_yt.grids_local[lid].grid_dimensions[0];
                }
                else{
                    grid_info.data_dim[0] = g_param_yt.grids_local[lid].grid_dimensions[0];
                    grid_info.data_dim[1] = g_param_yt.grids_local[lid].grid_dimensions[1];
                    grid_info.data_dim[2] = g_param_yt.grids_local[lid].grid_dimensions[2];
                }
            }

            // Get data type. "derived_func" can only be YT_DOUBLE type.
            if( strcmp(m_FieldDefineType, "derived_func") == 0 ){
                grid_info.data_dtype = YT_DOUBLE;
            }
            else{
                grid_info.data_dtype = g_param_yt.grids_local[lid].field_data[m_FieldIndex].data_dtype;
            }
            break;
        }
    }

    if(local_index == -1){
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        YT_ABORT("yt_rma_field: Cannot find grid id [ %ld ] on MPI rank [ %d ].\n", gid, myrank);
    }

    // Get data pointer
    void *data_ptr;
    if( strcmp(m_FieldDefineType, "cell-centered") == 0 ){
        data_ptr = g_param_yt.grids_local[local_index].field_data[m_FieldIndex].data_ptr;
    }
    else if( strcmp(m_FieldDefineType, "face-centered") == 0 ){
        data_ptr = g_param_yt.grids_local[local_index].field_data[m_FieldIndex].data_ptr;
    }
    else if( strcmp(m_FieldDefineType, "derived_func") == 0 ){
        // Allocate memory.
        int gridLength = grid_info.data_dim[0] * grid_info.data_dim[1] * grid_info.data_dim[2];
        data_ptr = malloc( gridLength * sizeof(double) );
        double *temp = (double *) data_ptr;
        for(int i = 0; i < gridLength; i++){
            temp[i] = 0.0;
        }
        // Generate data.
        void (*derived_func) (long, double*);
        derived_func = g_param_yt.field_list[m_FieldIndex].derived_func;
        if( derived_func == NULL ){
            YT_ABORT("yt_rma_field: In field [%s], field_define_type == %s, but derived_func not set!\n",
                      m_FieldName, m_FieldDefineType);
        }
        (*derived_func) (gid, (double*) data_ptr);
    }

    // Attach buffer to window.
    int size;
    get_dtype_size( grid_info.data_dtype, &size );
    if( MPI_Win_attach(m_Window, data_ptr, grid_info.data_dim[0] * grid_info.data_dim[1] * grid_info.data_dim[2] * size) != MPI_SUCCESS ){
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
//                3. Open the window epoch.
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

    // Calculate m_LenAllPrepare.
    for(int rank = 0; rank < NRank; rank++){
        m_LenAllPrepare += SendCount[rank];
    }

    // Gather PreparedInfoList, which is m_Prepare in each rank
    // (1) Create MPI_Datatype for yt_rma_grid_info
    MPI_Datatype yt_rma_grid_info_mpi_type;
    int lengths[5] = {1, 1, 1, 1, 3};
    const MPI_Aint displacements[5] = {0,
                                       1 * sizeof(long),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 1 * sizeof(int),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 2 * sizeof(int)};
    MPI_Datatype types[5] = {MPI_LONG, MPI_AINT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(5, lengths, displacements, types, &yt_rma_grid_info_mpi_type);
    MPI_Type_commit(&yt_rma_grid_info_mpi_type);

    // (2) Perform big_MPI_Gatherv and big_MPI_Bcast
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
    int  gridLength;
    int  dtype_size;
    MPI_Datatype mpi_dtype;
    yt_rma_grid_info fetched;
    for(long aid = 0; aid < m_LenAllPrepare; aid++){
        if( m_AllPrepare[aid].id == gid ){
            // Copy fetched grid info to fetched.
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
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        YT_ABORT("yt_rma_field: Cannot get remote grid id [ %ld ] located in rank [ %d ] on MPI rank [ %d ].\n",
                 gid, rank, myrank);
    }
    void *fetchedData = malloc( gridLength * dtype_size );

    // Fetch data and info
    if( MPI_Get(fetchedData, gridLength, mpi_dtype, rank, fetched.address, gridLength, mpi_dtype, m_Window) != MPI_SUCCESS ){
        YT_ABORT("yt_rma_field: MPI_Get fetch remote data failed!\n");
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
//                2. Free m_AllPrepare.
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
    for(int i = 0; i < m_PrepareData.size(); i++){
        MPI_Win_detach(m_Window, m_PrepareData[i]);
    }

    // Free local prepared data m_Prepare, m_PrepareData if field_define_type == "derived_func".
    if( strcmp(m_FieldDefineType, "derived_func") == 0 ) {
        for(int i = 0; i < m_PrepareData.size(); i++) {
            delete [] m_PrepareData[i];
        }
    }

    // Free m_AllPrepare, m_PrepareData, m_Prepare
    delete [] m_AllPrepare;
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
int yt_rma_field::get_fetched_data(long *gid, char **fname, yt_dtype *data_dtype, int (*data_dim)[3], void **data_ptr){
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