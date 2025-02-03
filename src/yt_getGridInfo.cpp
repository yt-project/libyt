#include "libyt.h"
#include "libyt_process_control.h"
#include "logging.h"
#include "timer.h"

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_*
// Description :  Get dimension of the grid with grid id = gid.
//
// Note        :  1. It searches full hierarchy loaded in Python, and returns YT_FAIL if error occurs.
//                2. grid_dimensions/left_edge/right_edge is defined in [x][y][z] <-> [0][1][2] coordinate.
//                3. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//                4.    Function                                              Search NumPy Array
//                   --------------------------------------------------------------------------------
//                    yt_getGridInfo_Dimensions(const long, int (*)[3])       libyt.hierarchy["grid_dimensions"]
//                    yt_getGridInfo_LeftEdge(const long, double (*)[3])      libyt.hierarchy["grid_left_edge"]
//                    yt_getGridInfo_RightEdge(const long, double (*)[3])     libyt.hierarchy["grid_right_edge"]
//                    yt_getGridInfo_ParentId(const long, long *)             libyt.hierarchy["grid_parent_id"]
//                    yt_getGridInfo_Level(const long, int *)                 libyt.hierarchy["grid_levels"]
//                    yt_getGridInfo_ProcNum(const long, int *)               libyt.hierarchy["proc_num"]
//
// Example     :  long gid = 0;
//                int dim[3];
//                yt_getGridInfo_Dimensions( gid, &dim );
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_Dimensions(const long gid, int (*dimensions)[3]) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridDimensions(gid,
                                                                                                 &(*dimensions)[0]);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}

int yt_getGridInfo_LeftEdge(const long gid, double (*left_edge)[3]) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridLeftEdge(gid, &(*left_edge)[0]);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}

int yt_getGridInfo_RightEdge(const long gid, double (*right_edge)[3]) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridRightEdge(gid, &(*right_edge)[0]);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}

int yt_getGridInfo_ParentId(const long gid, long* parent_id) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridParentId(gid, parent_id);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_SUCCESS;
    }
}

int yt_getGridInfo_Level(const long gid, int* level) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridLevel(gid, level);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}

int yt_getGridInfo_ProcNum(const long gid, int* proc_num) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridProcNum(gid, proc_num);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_ParticleCount
// Description :  Get particle count of particle type ptype in grid gid.
//
// Note        :  1. It searches libyt.hierarchy["par_count_list"][index][ptype],
//                   and does not check whether the grid is on this rank or not.
//                2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//
// Example     :  long count;
//                yt_getGridInfo_ParticleCount( gid, "par_type", &count );
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_ParticleCount(const long gid, const char* ptype, long* par_count) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundFullHierarchyGridParticleCount(gid, ptype,
                                                                                                    par_count);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_FieldData
// Description :  Get libyt.grid_data of field_name in the grid with grid id = gid .
//
// Note        :  1. It searches simulation field data registered under libyt Python module,
//                   and return YT_FAIL if error occurs.
//                2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//                3. User should cast to their own datatype after receiving the pointer.
//                4. Returns an existing data pointer and data dimensions user passed in,
//                   and does not make a copy of it!!
//                5. Works only for 3-dim data.
//
// Example     :  yt_data Data;
//                yt_getGridInfo_FieldData( gid, "field_name", &Data );
//                double *FieldData = (double *) Data.data_ptr;
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_FieldData(const long gid, const char* field_name, yt_data* field_data) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundLocalFieldData(gid, field_name, field_data);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  yt_getGridInfo_ParticleData
// Description :  Get libyt.particle_data of ptype attr attributes in the grid with grid id = gid.
//
// Note        :  1. It searches simulation particle data registered under libyt Python module,
//                   and return YT_FAIL if error occurs.
//                2. gid is grid id passed in by user, it doesn't need to be 0-indexed.
//                3. User should cast to their own datatype after receiving the pointer.
//                4. Returns an existing data pointer and data dimensions user passed in,
//                   and does not make a copy of it!!
//                5. For 1-dim data (like particles), the higher dimensions is set to 0.
//-------------------------------------------------------------------------------------------------------
int yt_getGridInfo_ParticleData(const long gid, const char* ptype, const char* attr, yt_data* par_data) {
    SET_TIMER(__PRETTY_FUNCTION__);

    if (!LibytProcessControl::Get().commit_grids_) {
        YT_ABORT("Please follow the libyt procedure, forgot to invoke yt_commit() before calling %s()!\n",
                 __FUNCTION__);
    }

    DataStructureOutput status =
        LibytProcessControl::Get().data_structure_amr_.GetPythonBoundLocalParticleData(gid, ptype, attr, par_data);

    if (status.status == DataStructureStatus::kDataStructureSuccess) {
        return YT_SUCCESS;
    } else {
        logging::LogError(status.error.c_str());
        return YT_FAIL;
    }
}
