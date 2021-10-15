#ifndef __YT_RMA_PARTICLE_H__
#define __YT_RMA_PARTICLE_H__

/*******************************************************************************
/
/  yt_rma_field class.
/
/  ==> included by init_libyt_module.cpp.
/
********************************************************************************/

#include <mpi.h>
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
// Data Member :  //TODO
//
// Method      :  //TODO
//-------------------------------------------------------------------------------------------------------
class yt_rma_particle
{
private:

    MPI_Win   m_Window;
    char     *m_ParticleType;
    char     *m_Attribute;
    int       m_ParticleIndex;
    int       m_AttributeIndex;

    std::vector<yt_rma_particle_info> m_Prepare;
    std::vector<void*> m_PrepareData;
};

#endif // #ifndef __YT_RMA_PARTICLE_H__

