#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

/**
 * \addtogroup api_yt_getGridInfo libyt API: Getting Grid and Data Information
 * \name api_yt_getGridInfo
 * Get grid information or data after loading information into libyt using \ref yt_commit.
 * This is used in data generating functions, like derived field function and get particle
 * data.
 */

/**
 * \brief Get grid dimensions in [x][y][z] coordinates of grid with grid id = gid.
 * \details
 * 1. It searches full hierarchy loaded in Python, and returns \ref YT_FAIL if error
 *    occurs.
 * 2. grid_dimensions is defined in `[x][y][z]` <-> `[0][1][2]` coordinate.
 * 3. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 * 4. If it's a 2D/1D simulation, we still need to pass in `dim[3]`.
 *
 * @param gid[in] grid id
 * @param dimensions[out] grid dimensions
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    long gid = 0;
 *    int dim[3];
 *    yt_getGridInfo_Dimensions( gid, &dim );
 * \endverbatim
 */
int yt_getGridInfo_Dimensions(const long gid, int (*dimensions)[3]) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get()
          .data_structure_amr_.GetPythonBoundFullHierarchyGridDimensions(
              gid, &(*dimensions)[0]);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}

/**
 * \brief Get grid left edges in [x][y][z] coordinates of grid with grid id = gid.
 * \details
 * 1. It searches full hierarchy loaded in Python, and returns \ref YT_FAIL if error
 *    occurs.
 * 2. Returned left edge is defined in `[x][y][z]` <-> `[0][1][2]` coordinate.
 * 3. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 * 4. If it's a 2D/1D simulation, we still need to pass in `left_edge[3]`.
 *
 * @param gid[in] grid id
 * @param left_edge[out] grid left edge
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    long gid = 0;
 *    double left_edge[3];
 *    yt_getGridInfo_LeftEdge( gid, &left_edge );
 * \endverbatim
 */
int yt_getGridInfo_LeftEdge(const long gid, double (*left_edge)[3]) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get()
          .data_structure_amr_.GetPythonBoundFullHierarchyGridLeftEdge(gid,
                                                                       &(*left_edge)[0]);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}

/**
 * \brief Get grid right edges in [x][y][z] coordinates of grid with grid id = gid.
 * \details
 * 1. It searches full hierarchy loaded in Python, and returns \ref YT_FAIL if error
 *    occurs.
 * 2. Returned right edge is defined in `[x][y][z]` <-> `[0][1][2]` coordinate.
 * 3. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 * 4. If it's a 2D/1D simulation, we still need to pass in `right_edge[3]`.
 *
 * @param gid[in] grid id
 * @param right_edge[out] grid right edge
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    long gid = 0;
 *    double right_edge[3];
 *    yt_getGridInfo_RightEdge( gid, &right_edge );
 * \endverbatim
 */
int yt_getGridInfo_RightEdge(const long gid, double (*right_edge)[3]) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get()
          .data_structure_amr_.GetPythonBoundFullHierarchyGridRightEdge(
              gid, &(*right_edge)[0]);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}

/**
 * \brief Get parent id of grid with grid id = gid.
 * \details
 * 1. It searches full hierarchy loaded in Python, and returns \ref YT_FAIL if error
 *    occurs.
 * 2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 *
 * @param gid[in] grid id
 * @param parent_id[out] parent id
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    long parent_id;
 *    yt_getGridInfo_ParentId( gid, &parent_id );
 * \endverbatim
 */
int yt_getGridInfo_ParentId(const long gid, long* parent_id) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get()
          .data_structure_amr_.GetPythonBoundFullHierarchyGridParentId(gid, parent_id);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_SUCCESS;
  }
}

/**
 * \brief Get level of grid with grid id = gid.
 * \details
 * 1. It searches full hierarchy loaded in Python, and returns \ref YT_FAIL if error
 *    occurs.
 * 2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 * 3. Root level starts at 0.
 *
 * @param gid[in] grid id
 * @param level[out] level
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    int level;
 *    yt_getGridInfo_Level( gid, &level );
 * \endverbatim
 */
int yt_getGridInfo_Level(const long gid, int* level) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridLevel(
          gid, level);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}

/**
 * \brief Get processor number (mpi rank) of grid with grid id = gid.
 * \details
 * 1. It searches full hierarchy loaded in Python, and returns \ref YT_FAIL if error
 *    occurs.
 * 2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 * 3. Processor number starts at 0.
 *
 * @param gid[in] grid id
 * @param proc_num[out] processor number
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    int proc_num;
 *    yt_getGridInfo_ProcNum( gid, &proc_num );
 * \endverbatim
 */
int yt_getGridInfo_ProcNum(const long gid, int* proc_num) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get()
          .data_structure_amr_.GetPythonBoundFullHierarchyGridProcNum(gid, proc_num);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}

/**
 * \brief Get particle count of a particle type inside grid with grid id = gid.
 * \details
 * 1. It searches full hierarchy loaded in Python, and returns \ref YT_FAIL if error
 *    occurs.
 * 2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 *
 * @param gid[in] grid id
 * @param ptype[in] particle type
 * @param par_count[out] particle count
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    long par_count;
 *    yt_getGridInfo_ParticleCount( gid, "io", &par_count );
 * \endverbatim
 */
int yt_getGridInfo_ParticleCount(const long gid, const char* ptype, long* par_count) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get()
          .data_structure_amr_.GetPythonBoundFullHierarchyGridParticleCount(
              gid, ptype, par_count);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}

/**
 * \brief Get field data of grid with grid id = gid.
 * \details
 * 1. It searches simulation field data registered under libyt Python module,
 *    and returns \ref YT_FAIL if error occurs.
 * 2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 * 3. User should cast to their own datatype after receiving the pointer.
 * 4. Returns an existing data pointer and data dimensions user passed in,
 *    and does not make a copy of it!!
 * 5. For 2/1-dim data, only [0][1]/[0] is used, and the higher dimensions are set to 1.
 *
 * @param gid[in] grid id
 * @param field_name[in] queried field name
 * @param field_data[out] field data pointer and metadata
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    yt_data data;
 *    yt_getGridInfo_FieldData(gid, "field_name", &data);
 *    double *field_data = (double *) data.data_ptr;
 * \endverbatim
 */
int yt_getGridInfo_FieldData(const long gid, const char* field_name,
                             yt_data* field_data) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get().data_structure_amr_.GetPythonBoundLocalFieldData(
          gid, field_name, field_data);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_ParticleData
// Description :  Get libyt.particle_data of ptype attr attributes in the grid with grid
// id = gid.
//
// Note        :  1. It searches simulation particle data registered under libyt Python
// module,
//                   and return YT_FAIL if error occurs.
//                2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//                3. User should cast to their own datatype after receiving the pointer.
//                4. Returns an existing data pointer and data dimensions user passed in,
//                   and does not make a copy of it!!
//                5. For 1-dim data (like particles), the higher dimensions is set to 0.
//-------------------------------------------------------------------------------------------------------
/**
 * \brief Get particle data of grid with grid id = gid.
 * \details
 * 1. It searches particle data registered under libyt Python module,
 *    and returns \ref YT_FAIL if error occurs.
 * 2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
 * 3. User should cast to their own datatype after receiving the pointer.
 * 4. Returns an existing data pointer and data dimensions user passed in,
 *    and does not make a copy of it!!
 * 5. For 1-dim data (like particles), the higher dimensions is set to 0.
 *
 * @param gid[in] grid id
 * @param ptype[in] queried particle type
 * @param attr[in] queried attribute
 * @param par_data[out] particle data pointer and metadata
 * @return \ref YT_SUCCESS or \ref YT_FAIL
 *
 * \verbatim embed:rst:leading-asterisk
 * .. code-block:: c
 *
 *    yt_data data;
 *    yt_getGridInfo_ParticleData(gid, "io", "PosX", &data);
 *    double *par_data = (double *) data.data_ptr;
 * \endverbatim
 */
int yt_getGridInfo_ParticleData(const long gid, const char* ptype, const char* attr,
                                yt_data* par_data) {
  SET_TIMER(__PRETTY_FUNCTION__);

  if (!LibytProcessControl::Get().commit_grids_) {
    YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before "
             "calling %s()!\n",
             __FUNCTION__);
  }

  DataStructureOutput status =
      LibytProcessControl::Get().data_structure_amr_.GetPythonBoundLocalParticleData(
          gid, ptype, attr, par_data);

  if (status.status == DataStructureStatus::kDataStructureSuccess) {
    return YT_SUCCESS;
  } else {
    logging::LogError(status.error.c_str());
    return YT_FAIL;
  }
}
