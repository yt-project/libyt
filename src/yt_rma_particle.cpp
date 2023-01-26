#include "yt_rma_particle.h"
#include "yt_combo.h"
#include "libyt.h"
#include <string.h>

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  Constructor
//
// Notes       :  1. Initialize m_Window, which used inside OpenMPI RMA operation. And set m_Window info
//                   to "no_locks".
//                2. Copy the input ptype to m_ParticleType and attribute to m_AttributeName, in case
//                   it is freed.
//                3. Find particle and its attribute index inside particle_list and assign to
//                   m_ParticleIndex and m_AttributeIndex. We assume that we can always find them.
//                4. Grab the data type of the attribute, and store in m_AttributeDataType.
//                5. Set the std::vector capacity.
//
// Arguments   :  char*       ptype: Particle type.
//                char*   attribute: Attribute name.
//                int   len_prepare: Number of grid to prepare.
//                long      len_get: Number of grid to get.
//-------------------------------------------------------------------------------------------------------
yt_rma_particle::yt_rma_particle(char *ptype, char *attribute, int len_prepare, long len_get)
 : m_LenAllPrepare(0), m_ParticleIndex(-1), m_AttributeIndex(-1)
{
    // Initialize m_Window and set info to "no_locks".
    MPI_Info windowInfo;
    MPI_Info_create( &windowInfo );
    MPI_Info_set( windowInfo, "no_locks", "true" );
    int mpi_return_code = MPI_Win_create_dynamic( windowInfo, MPI_COMM_WORLD, &m_Window );
    MPI_Info_free( &windowInfo );

    if (mpi_return_code != MPI_SUCCESS) {
        log_error("yt_rma_particle: create one-sided MPI (RMA) window failed!\n");
        log_error("yt_rma_particle: try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".\n");
    }

    // Copy input ptype and attribute.
    int len = strlen(ptype);
    m_ParticleType = new char [len+1];
    strcpy(m_ParticleType, ptype);

    len = strlen(attribute);
    m_AttributeName = new char [len+1];
    strcpy(m_AttributeName, attribute);

    for(int v = 0; v < g_param_yt.num_species; v++){
        if( strcmp(m_ParticleType, g_param_yt.particle_list[v].species_name) == 0 ){
            m_ParticleIndex = v;
            for(int a = 0; a < g_param_yt.particle_list[v].num_attr; a++) {
                if( strcmp(m_AttributeName, g_param_yt.particle_list[v].attr_list[a].attr_name) == 0 ){
                    m_AttributeIndex = a;
                    m_AttributeDataType = g_param_yt.particle_list[v].attr_list[a].attr_dtype;
                    break;
                }
            }
            break;
        }
    }

    // Set std::vector capacity
    m_Prepare.reserve(len_prepare);
    m_PrepareData.reserve(len_prepare);
    m_Fetched.reserve(len_get);
    m_FetchedData.reserve(len_get);

    log_debug("yt_rma_particle: Particle Type  = %s\n", m_ParticleType);
    log_debug("yt_rma_particle: Attribute Name = %s\n", m_AttributeName);
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  Destructor
//
// Notes       :  1. Free m_ParticleType, m_AttributeName, m_Window.
//                2. Clear m_Fetched and m_FetchedData, even though it is already empty.
//
// Arguments   :  None
//-------------------------------------------------------------------------------------------------------
yt_rma_particle::~yt_rma_particle()
{
    MPI_Win_free(&m_Window);
    delete [] m_ParticleType;
    delete [] m_AttributeName;
    m_Fetched.clear();
    m_FetchedData.clear();
    log_debug("yt_rma_particle: Destructor called.\n");
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  prepare_data
// Description :  Prepare particle data in grid = gid and species_name = m_ParticleType, then attach
//                particle data to m_Window and get the address.
//
// Notes       :  1. Prepare the particle data in grid = gid, and attach particle data to m_Window.
//                2. Insert data pointer into m_PrepareData.
//                3. Insert data information into m_Prepare.
//                4. data_len is the length of the particle data array.
//                5. We assume that all gid can be found on this rank.
//
// Arguments   :  long gid : Particle data to prepare in grid id = gid.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_particle::prepare_data(long& gid)
{
    // Make sure particle type and its attribute name exist.
    if( m_ParticleIndex == -1 ){
        YT_ABORT("yt_rma_particle: Cannot find species name [ %s ] in particle_list on MPI rank [ %d ].\n", m_ParticleType, g_myrank);
    }
    if( m_AttributeIndex == -1 ){
        YT_ABORT("yt_rma_particle: Cannot find attribute name [ %s ] in species name [ %s ] on MPI rank [ %d ].\n",
                 m_AttributeName, m_ParticleType, g_myrank);
    }

    // Get particle info
    yt_rma_particle_info par_info;
    par_info.id = gid;
    yt_getGridInfo_ProcNum(gid, &(par_info.rank));
    if ( par_info.rank != g_myrank ) {
        YT_ABORT("yt_rma_particle: Trying to prepare nonlocal particle data in grid [%ld] that is on MPI rank [%d].\n",
                 gid, par_info.rank);
    }
    if ( yt_getGridInfo_ParticleCount(gid, m_ParticleType, &(par_info.data_len)) != YT_SUCCESS ) {
        YT_ABORT("yt_rma_particle: Failed to get number of particle %s in grid [%ld].\n", m_ParticleType, gid);
    }
    if ( par_info.data_len < 0 ) {
        YT_ABORT("yt_rma_particle: Particle %s count = %ld < 0 in grid [%ld].\n", m_ParticleType, par_info.data_len, gid);
    }

    // Generate particle data
    void (*get_attr) (int, long*, char*, yt_array*);
    get_attr = g_param_yt.particle_list[m_ParticleIndex].get_attr;
    if( get_attr == NULL ){
        YT_ABORT("yt_rma_particle: In species [%s], get_attr not set!\n", m_ParticleType);
    }

    int dtype_size;
    if ( get_dtype_size(m_AttributeDataType, &dtype_size) != YT_SUCCESS ){
        YT_ABORT("yt_rma_particle: In species [%s] attribute [%s], unknown yt_dtype.\n", m_ParticleType, m_AttributeName);
    }

    void *data_ptr;
    if( par_info.data_len > 0 ){
        // Generate buffer.
        data_ptr = malloc( par_info.data_len * dtype_size );
        int list_len = 1;
        long list_gid[1] = { gid };
        yt_array data_array[1];
        data_array[0].gid = gid; data_array[0].data_length = par_info.data_len; data_array[0].data_ptr = data_ptr;

        (*get_attr) (list_len, list_gid, m_AttributeName, data_array);

        // Attach buffer to window.
        int mpi_return_code = MPI_Win_attach(m_Window, data_ptr, par_info.data_len * dtype_size );
        if (mpi_return_code != MPI_SUCCESS) {
            log_error("yt_rma_particle: attach data buffer to one-sided MPI (RMA) window failed!\n");
            log_error("yt_rma_particle: try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\"\n");
            YT_ABORT("yt_rma_particle: Attach particle [%s] attribute [%s] to window failed!\n",
                     m_ParticleType, m_AttributeName);
        }

        // Get the address of the attached buffer.
        if( MPI_Get_address(data_ptr, &(par_info.address)) != MPI_SUCCESS ){
            YT_ABORT("yt_rma_particle: Get attached particle [%s] attribute [%s] buffer address failed!\n",
                     m_ParticleType, m_AttributeName);
        }
    }
    else{
        data_ptr = NULL;
        par_info.address = NULL;
    }

    // Push back to m_Prepare, m_PrepareData.
    m_PrepareData.push_back( data_ptr );
    m_Prepare.push_back( par_info );

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  gather_all_prepare_data
// Description :  Gather all prepared data in each rank.
//
// Notes       :  1. This should be called after preparing all the needed particle data.
//                2. Perform big_MPI_Gatherv and big_MPI_Bcast at root rank.
//                3. Set up m_SearchRange and m_LenAllPrepare, will later be used in fetch_remote_data to
//                   search grid id in m_AllPrepare.
//                4. Open the window epoch.
//
// Parameter   :  int root : root rank.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_particle::gather_all_prepare_data(int root)
{
    int NRank;
    MPI_Comm_size(MPI_COMM_WORLD, &NRank);

    // Get m_Prepare data and its length
    int PreparedInfoListLength = m_Prepare.size();
    yt_rma_particle_info *PreparedInfoList = m_Prepare.data();

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
    m_AllPrepare = new yt_rma_particle_info [m_LenAllPrepare];
    big_MPI_Gatherv(root, SendCount, (void*)PreparedInfoList, &yt_rma_particle_info_mpi_type, (void*)m_AllPrepare, 2);
    big_MPI_Bcast(root, m_LenAllPrepare, (void*)m_AllPrepare, &yt_rma_particle_info_mpi_type, 2);

    // Open window epoch.
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, m_Window);

    // Free unused resource
    delete [] SendCount;

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  fetch_remote_data
// Description :  Allocate spaces and fetch remote data, and store in std::vector m_Fetched, m_FetchData.
//
// Notes       :  1. Look for particle info in m_AllPrepare, and allocate memory.
//                2. We allocate buffer to store fetched data, but we do not free them. It is Python's
//                   responsibility.
//                3. Allocate and fetch data if data length > 0.
//
// Parameters  : long gid  : From which grid id to fetch the particle data.
//               int  rank : Fetch data from rank.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_particle::fetch_remote_data(long& gid, int& rank)
{
    // Look for gid in m_AllPrepare.
    bool get_remote_gid = false;
    yt_rma_particle_info fetched;
    for(long aid = m_SearchRange[rank]; aid < m_SearchRange[rank+1]; aid++){
        if( m_AllPrepare[aid].id == gid ){
            fetched = m_AllPrepare[aid];

            get_remote_gid = true;
            break;
        }
    }
    if( get_remote_gid != true ){
        YT_ABORT("yt_rma_particle: Cannot get remote grid id [ %ld ] located in rank [ %d ] on MPI rank [ %d ].\n",
                 gid, rank, g_myrank);
    }
    void *fetchedData;

    if( fetched.data_len > 0 ){
        int  dtype_size;
        MPI_Datatype mpi_dtype;
        get_dtype_size( m_AttributeDataType, &dtype_size );
        get_mpi_dtype( m_AttributeDataType, &mpi_dtype );
        fetchedData = malloc( fetched.data_len * dtype_size );

        if( big_MPI_Get(fetchedData, fetched.data_len, &m_AttributeDataType, &mpi_dtype, rank, fetched.address, &m_Window) != YT_SUCCESS ){
            YT_ABORT("yt_rma_particle: big_MPI_Get fetch particle [%s] attribute [%s] in grid [%ld] failed!\n",
                     m_ParticleType, m_AttributeName, gid);
        }
    }
    else{
        fetchedData = NULL;
    }

    // Push back to m_Fetched, m_FetchedData.
    m_Fetched.push_back(fetched);
    m_FetchedData.push_back(fetchedData);

    return YT_SUCCESS;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  clean_up
// Description :  Clean up prepared data.
//
// Notes       :  1. Close the window epoch, and detach the prepared data buffer.
//                2. Free m_AllPrepare, m_SearchRange.
//                3. Free local prepared data, drop m_Prepare and m_PrepareData vector.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_particle::clean_up()
{
    // Close the window epoch
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, m_Window);

    // Detach m_PrepareData from m_Window and free it, if data length > 0.
    for(int i = 0; i < (int)m_Prepare.size(); i++){
        if( m_Prepare[i].data_len > 0 ){
            MPI_Win_detach(m_Window, m_PrepareData[i]);
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
// Class       :  yt_rma_particle
// Method      :  get_fetched_data
// Description :  Get fetched data.
//
// Notes       :  1. Get fetched data one by one. If vector is empty, then it will return YT_FAIL.
//                2. Whenever one gets the data from m_Fetched and m_FetchedData, it will also remove it
//                   from std::vector.
//                3. Write fetched data and info to passed in parameters.
//                4. If fetched data has data length 0, then we write in NULL to data_ptr
//                   (see also fetch_remote_data).
//
// Parameters  :  long     *gid        : Grid id fetched.
//                char    **ptype      : Particle type fetched.
//                char    **attribute  : Attribute fetched.
//                yt_dtype *data_dtype : Fetched data type.
//                long     *data_len   : Fetched data length.
//                void    **data_ptr   : Fetched data pointer.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_particle::get_fetched_data(long *gid, char **ptype, char **attribute, yt_dtype *data_dtype, long *data_len, void **data_ptr)
{
    // Check if there are left fetched data to get.
    if( m_Fetched.size() == 0 ){
        return YT_FAIL;
    }

    // Access m_Fetched and m_FetchedData from the back.
    yt_rma_particle_info fetched = m_Fetched.back();
    *gid = fetched.id;
    *ptype = m_ParticleType;
    *attribute = m_AttributeName;
    *data_dtype = m_AttributeDataType;
    *data_len = fetched.data_len;
    *data_ptr = m_FetchedData.back();

    // Pop back fetched data.
    m_Fetched.pop_back();
    m_FetchedData.pop_back();

    return YT_SUCCESS;
}
