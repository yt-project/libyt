#ifndef __YT_RMA_PARTICLE_H__
#define __YT_RMA_PARTICLE_H__

/*******************************************************************************
/
/  yt_rma_particle class.
/
/  ==> included by init_libyt_module.cpp.
/
********************************************************************************/

#ifndef SERIAL_MODE

#include <vector>

#include "yt_combo.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  yt_rma_particle
// Description :  Class to deal with RMA operation, to get non-local particle data.
//
// Notes       :  1. This class deals with OpenMPI RMA operation, should be called by every rank.
//                2. This class isn't aim to deal with generic RMA operation, it is meant to work with
//                   libyt frontend in YT when getting non-local particle data.
//                3. First assume that each rank has the same particle list, and their order are the
//                   same.
//                4. One instance with one field and one window.
//                5. One instance only deals with one particle type and one attribute.
//
//-------------------------------------------------------------------------------------------------------
class yt_rma_particle {
private:
    MPI_Win m_Window;
    const char* m_ParticleType;
    const char* m_AttributeName;
    int m_ParticleIndex;
    int m_AttributeIndex;
    yt_dtype m_AttributeDataType;

    std::vector<yt_rma_particle_info> m_Prepare;
    std::vector<void*> m_PrepareData;
    std::vector<bool> m_FreePrepareData;

    long m_LenAllPrepare;
    long* m_SearchRange;
    yt_rma_particle_info* m_AllPrepare;

    std::vector<yt_rma_particle_info> m_Fetched;
    std::vector<void*> m_FetchedData;

public:
    yt_rma_particle(const char* ptype, const char* attribute, int len_prepare, long len_get);
    ~yt_rma_particle();

    // OpenMPI RMA operation
    int prepare_data(long& gid);
    int gather_all_prepare_data(int root);
    int fetch_remote_data(long& gid, int& rank);
    int clean_up();
    int get_fetched_data(long* gid, const char** ptype, const char** attribute, yt_dtype* data_dtype, long* data_len,
                         void** data_ptr);
};

#endif

#endif  // #ifndef __YT_RMA_PARTICLE_H__
