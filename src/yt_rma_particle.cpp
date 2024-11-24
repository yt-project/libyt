#ifndef SERIAL_MODE

#include "yt_rma_particle.h"

#include <string.h>

#include "LibytProcessControl.h"
#include "big_mpi.h"
#include "libyt.h"

static int get_particle_data(const long gid, const char* ptype, const char* attr, yt_data* par_data);

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  Constructor
//
// Notes       :  1. Initialize m_Window, which used inside OpenMPI RMA operation. And set m_Window info
//                   to "no_locks".
//                2. Assume the lifetime of par_type and attr_name cover the in situ analysis process,
//                   libyt only borrows their names and does not make a copy.
//                3. Find particle and its attribute index inside particle_list and assign to
//                   m_ParticleIndex and m_AttributeIndex. We assume that we can always find them.
//                4. Grab the data type of the attribute, and store in m_AttributeDataType.
//                5. Set the std::vector capacity.
//
// Arguments   :  const char*     ptype: Particle type.
//                const char* attribute: Attribute name.
//                int       len_prepare: Number of grid to prepare.
//                long          len_get: Number of grid to get.
//-------------------------------------------------------------------------------------------------------
yt_rma_particle::yt_rma_particle(const char* ptype, const char* attribute, int len_prepare, long len_get)
    : m_ParticleIndex(-1), m_AttributeIndex(-1), m_LenAllPrepare(0) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Initialize m_Window and set info to "no_locks".
    MPI_Info windowInfo;
    MPI_Info_create(&windowInfo);
    MPI_Info_set(windowInfo, "no_locks", "true");
    int mpi_return_code = MPI_Win_create_dynamic(windowInfo, MPI_COMM_WORLD, &m_Window);
    MPI_Info_free(&windowInfo);

    if (mpi_return_code != MPI_SUCCESS) {
        log_error("yt_rma_particle: create one-sided MPI (RMA) window failed!\n");
        log_error("yt_rma_particle: try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\".\n");
    }

    yt_particle* particle_list = LibytProcessControl::Get().particle_list;
    for (int v = 0; v < LibytProcessControl::Get().param_yt_.num_par_types; v++) {
        if (strcmp(ptype, particle_list[v].par_type) == 0) {
            m_ParticleType = particle_list[v].par_type;
            m_ParticleIndex = v;
            for (int a = 0; a < particle_list[v].num_attr; a++) {
                if (strcmp(attribute, particle_list[v].attr_list[a].attr_name) == 0) {
                    m_AttributeName = particle_list[v].attr_list[a].attr_name;
                    m_AttributeIndex = a;
                    m_AttributeDataType = particle_list[v].attr_list[a].attr_dtype;
                    break;
                }
            }
            break;
        }
    }

    // Set std::vector capacity
    m_Prepare.reserve(len_prepare);
    m_PrepareData.reserve(len_prepare);
    m_FreePrepareData.reserve(len_prepare);
    m_Fetched.reserve(len_get);
    m_FetchedData.reserve(len_get);

    log_debug("yt_rma_particle: Particle Type  = %s\n", m_ParticleType);
    log_debug("yt_rma_particle: Attribute Name = %s\n", m_AttributeName);
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  Destructor
//
// Notes       :  1. Free m_Window.
//                2. Clear m_Fetched and m_FetchedData, even though it is already empty.
//
// Arguments   :  None
//-------------------------------------------------------------------------------------------------------
yt_rma_particle::~yt_rma_particle() {
    SET_TIMER(__PRETTY_FUNCTION__);

    MPI_Win_free(&m_Window);
    m_Fetched.clear();
    m_FetchedData.clear();
    log_debug("yt_rma_particle: Destructor called.\n");
}

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Method      :  prepare_data
// Description :  Prepare particle data in grid = gid and par_type = m_ParticleType, then attach
//                particle data to m_Window and get the address.
//
// Notes       :  1. Prepare the particle data in grid = gid, and attach particle data to m_Window.
//                2. Insert data pointer into m_PrepareData, and if data gets from get_par_attr,
//                   we push m_FreePrepareData to true.
//                3. Insert data information into m_Prepare.
//                4. data_len is the length of the particle data array.
//                5. Only prepare data length > 0. data_ptr with length = 0 will be set as NULL.
//                5. We assume that all gid can be found on this rank.
//
// Arguments   :  long gid : Particle data to prepare in grid id = gid.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_particle::prepare_data(long& gid) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Make sure particle type and its attribute name exist.
    if (m_ParticleIndex == -1) {
        YT_ABORT("yt_rma_particle: Cannot find particle type [ %s ] in particle_list on MPI rank [ %d ].\n",
                 m_ParticleType, LibytProcessControl::Get().mpi_rank_);
    }
    if (m_AttributeIndex == -1) {
        YT_ABORT("yt_rma_particle: Cannot find attribute name [ %s ] in particle type [ %s ] on MPI rank [ %d ].\n",
                 m_AttributeName, m_ParticleType, LibytProcessControl::Get().mpi_rank_);
    }

    // Get particle info
    yt_rma_particle_info par_info;
    par_info.id = gid;
    yt_getGridInfo_ProcNum(gid, &(par_info.rank));
    if (par_info.rank != LibytProcessControl::Get().mpi_rank_) {
        YT_ABORT("yt_rma_particle: Trying to prepare nonlocal particle data in grid [%ld] that is on MPI rank [%d].\n",
                 gid, par_info.rank);
    }
    if (yt_getGridInfo_ParticleCount(gid, m_ParticleType, &(par_info.data_len)) != YT_SUCCESS) {
        YT_ABORT("yt_rma_particle: Failed to get number of particle %s in grid [%ld].\n", m_ParticleType, gid);
    }
    if (par_info.data_len < 0) {
        YT_ABORT("yt_rma_particle: Particle %s count = %ld < 0 in grid [%ld].\n", m_ParticleType, par_info.data_len,
                 gid);
    }

    // Get particle data
    void* data_ptr = nullptr;
    par_info.address = NULL;
    int dtype_size;
    bool to_free = false;
    if (par_info.data_len > 0) {
        // get data type
        if (get_dtype_size(m_AttributeDataType, &dtype_size) != YT_SUCCESS) {
            YT_ABORT("yt_rma_particle: Particle type [%s] attribute [%s], unknown yt_dtype.\n", m_ParticleType,
                     m_AttributeName);
        }

        // get data_ptr: check if the data can be get in libyt.particle_data[g.id][ptype][attr] first,
        // if not, generate data in get_par_attr
        yt_data par_array;
        if (get_particle_data(gid, m_ParticleType, m_AttributeName, &par_array) == YT_SUCCESS) {
            data_ptr = par_array.data_ptr;
            to_free = false;
        } else {
            log_info("Trying to get particle data through user-defined function.\n");

            // Generate particle data through get_par_attr function pointer, if we cannot find it in libyt.particle_data
            void (*get_par_attr)(const int, const long*, const char*, const char*, yt_array*);
            get_par_attr = LibytProcessControl::Get().particle_list[m_ParticleIndex].get_par_attr;
            if (get_par_attr == nullptr) {
                YT_ABORT("yt_rma_particle: Particle type [%s], get_par_attr not set!\n", m_ParticleType);
            }

            // Generate buffer
            if (get_dtype_allocation(m_AttributeDataType, par_info.data_len, &data_ptr) != YT_SUCCESS) {
                YT_ABORT("yt_rma_particle: Cannot allocate memory, unknown attr_dtype.\n");
            }
            to_free = true;
            int list_len = 1;
            long list_gid[1] = {gid};
            yt_array data_array[1];
            data_array[0].gid = gid;
            data_array[0].data_length = par_info.data_len;
            data_array[0].data_ptr = data_ptr;
            (*get_par_attr)(list_len, list_gid, m_ParticleType, m_AttributeName, data_array);
        }

        // Attach buffer to window.
        int mpi_return_code = MPI_Win_attach(m_Window, data_ptr, par_info.data_len * dtype_size);
        if (mpi_return_code != MPI_SUCCESS) {
            log_error("yt_rma_particle: attach data buffer to one-sided MPI (RMA) window failed!\n");
            log_error("yt_rma_particle: try setting \"OMPI_MCA_osc=sm,pt2pt\" when using \"mpirun\"\n");
            if (to_free) free(data_ptr);
            YT_ABORT("yt_rma_particle: Attach particle [%s] attribute [%s] to window failed!\n", m_ParticleType,
                     m_AttributeName);
        }

        // Get the address of the attached buffer.
        if (MPI_Get_address(data_ptr, &(par_info.address)) != MPI_SUCCESS) {
            if (to_free) free(data_ptr);
            YT_ABORT("yt_rma_particle: Get attached particle [%s] attribute [%s] buffer address failed!\n",
                     m_ParticleType, m_AttributeName);
        }
    }

    // Push back to m_Prepare, m_PrepareData.
    m_PrepareData.push_back(data_ptr);
    m_FreePrepareData.push_back(to_free);
    m_Prepare.push_back(par_info);

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
int yt_rma_particle::gather_all_prepare_data(int root) {
    SET_TIMER(__PRETTY_FUNCTION__);

    int NRank;
    MPI_Comm_size(MPI_COMM_WORLD, &NRank);

    // Get m_Prepare data and its length
    int PreparedInfoListLength = m_Prepare.size();
    yt_rma_particle_info* PreparedInfoList = m_Prepare.data();

    // MPI_Gather send count in each rank, then MPI_Bcast.
    int* SendCount = new int[NRank];
    MPI_Gather(&PreparedInfoListLength, 1, MPI_INT, SendCount, 1, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(SendCount, NRank, MPI_INT, root, MPI_COMM_WORLD);

    // Calculate m_SearchRange, and m_LenAllPrepare.
    m_SearchRange = new long[NRank + 1];
    for (int i = 0; i < NRank + 1; i++) {
        m_SearchRange[i] = 0;
        for (int rank = 0; rank < i; rank++) {
            m_SearchRange[i] += SendCount[rank];
        }
    }
    m_LenAllPrepare = m_SearchRange[NRank];

    // Gather PreparedInfoList, which is m_Prepare in each rank, perform big_MPI_Gatherv and big_MPI_Bcast
    m_AllPrepare = new yt_rma_particle_info[m_LenAllPrepare];
    big_MPI_Gatherv<yt_rma_particle_info>(root, SendCount, (void*)PreparedInfoList,
                                          &LibytProcessControl::Get().yt_rma_particle_info_mpi_type_,
                                          (void*)m_AllPrepare);
    big_MPI_Bcast<yt_rma_particle_info>(root, m_LenAllPrepare, (void*)m_AllPrepare,
                                        &LibytProcessControl::Get().yt_rma_particle_info_mpi_type_);

    // Open window epoch.
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, m_Window);

    // Free unused resource
    delete[] SendCount;

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
int yt_rma_particle::fetch_remote_data(long& gid, int& rank) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Look for gid in m_AllPrepare.
    bool get_remote_gid = false;
    yt_rma_particle_info fetched;
    for (long aid = m_SearchRange[rank]; aid < m_SearchRange[rank + 1]; aid++) {
        if (m_AllPrepare[aid].id == gid) {
            fetched = m_AllPrepare[aid];

            get_remote_gid = true;
            break;
        }
    }
    if (get_remote_gid != true) {
        YT_ABORT("yt_rma_particle: Cannot get remote grid id [ %ld ] located in rank [ %d ] on MPI rank [ %d ].\n", gid,
                 rank, LibytProcessControl::Get().mpi_rank_);
    }
    void* fetchedData;

    if (fetched.data_len > 0) {
        int dtype_size;
        MPI_Datatype mpi_dtype;
        get_dtype_size(m_AttributeDataType, &dtype_size);
        get_mpi_dtype(m_AttributeDataType, &mpi_dtype);
        fetchedData = malloc(fetched.data_len * dtype_size);

        if (big_MPI_Get_dtype(fetchedData, fetched.data_len, &m_AttributeDataType, &mpi_dtype, rank, fetched.address,
                              &m_Window) != YT_SUCCESS) {
            YT_ABORT("yt_rma_particle: big_MPI_Get_dtype fetch particle [%s] attribute [%s] in grid [%ld] failed!\n",
                     m_ParticleType, m_AttributeName, gid);
        }
    } else {
        fetchedData = nullptr;
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
//                3. Free local prepared data if m_FreePrepareData is true.
//                4. Drop m_Prepare, m_PrepareData, m_FreePrepareData vector.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_rma_particle::clean_up() {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Close the window epoch
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, m_Window);

    // Detach m_PrepareData from m_Window and free it, if data length > 0.
    for (int i = 0; i < (int)m_Prepare.size(); i++) {
        if (m_Prepare[i].data_len > 0) {
            MPI_Win_detach(m_Window, m_PrepareData[i]);
            if (m_FreePrepareData[i]) {
                free(m_PrepareData[i]);
            }
        }
    }

    // Free m_AllPrepare, m_SearchRange, m_PrepareData, m_FreePrepareData, m_Prepare
    delete[] m_AllPrepare;
    delete[] m_SearchRange;
    m_PrepareData.clear();
    m_FreePrepareData.clear();
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
int yt_rma_particle::get_fetched_data(long* gid, const char** ptype, const char** attribute, yt_dtype* data_dtype,
                                      long* data_len, void** data_ptr) {
    SET_TIMER(__PRETTY_FUNCTION__);

    // Check if there are left fetched data to get.
    if (m_Fetched.size() == 0) {
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

//-------------------------------------------------------------------------------------------------------
// Private Function :  get_particle_data
// Description      :  Get libyt.particle_data of ptype attr attributes in the grid with grid id = gid.
//
// Note             :  1. It searches libyt.particle_data[gid][ptype][attr], and return YT_FAIL if it cannot
//                        find corresponding data.
//                     2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//                     3. User should cast to their own datatype after receiving the pointer.
//                     4. Returns an existing data pointer and data dimensions user passed in,
//                        and does not make a copy of it!!
//                     5. For 1-dim data (like particles), the higher dimensions is set to 0.
//                     6. TODO: This is a COPY of yt_getGridInfo_ParticleData, but doesn't print error msg.
//                              Will abstracting the layer when refactoring.
//
// Parameter        :  const long   gid              : Target grid id.
//                     const char  *ptype            : Target particle type.
//                     const char  *attr             : Target attribute name.
//                     yt_data     *par_data         : Store the yt_data struct pointer that points to data.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
static int get_particle_data(const long gid, const char* ptype, const char* attr, yt_data* par_data) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    // get dictionary libyt.particle_data[gid][ptype]
    PyObject* py_grid_id = PyLong_FromLong(gid);
    PyObject* py_ptype = PyUnicode_FromString(ptype);
    PyObject* py_attr = PyUnicode_FromString(attr);

    if (PyDict_Contains(g_py_particle_data, py_grid_id) != 1 ||
        PyDict_Contains(PyDict_GetItem(g_py_particle_data, py_grid_id), py_ptype) != 1 ||
        PyDict_Contains(PyDict_GetItem(PyDict_GetItem(g_py_particle_data, py_grid_id), py_ptype), py_attr) != 1) {
        Py_DECREF(py_grid_id);
        Py_DECREF(py_ptype);
        Py_DECREF(py_attr);
        return YT_FAIL;
    }
    PyArrayObject* py_data = (PyArrayObject*)PyDict_GetItem(
        PyDict_GetItem(PyDict_GetItem(g_py_particle_data, py_grid_id), py_ptype), py_attr);

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
        return YT_FAIL;
    }

    return YT_SUCCESS;
}

#endif
