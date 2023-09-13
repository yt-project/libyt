#include "LibytProcessControl.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Data member :  Static private LibytProcessControl instance.
// Description :  Singleton instance.
//-------------------------------------------------------------------------------------------------------
LibytProcessControl LibytProcessControl::s_Instance;

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Method      :  Private constructor
// Description :  Initialize the class.
//
// Notes       :  1. The private constructor only get called once at the start of the libyt process.
//                2. I didn't use member initialization list for readability.
//                3. Allocating and freeing memories are manage by other function, this method only
//                   stores it.
//                4. This acts like a global.
//                5. This is helpful when there are other initialization processes added to libyt.
//                   (What I'm doing in this commit is trying to make libyt a C-library with minimal
//                    change.)
//                6. This is not thread-safe.
//
// Data Member : libyt_initialized : true ==> yt_initialize() has been called successfully
//               param_yt_set      : true ==> yt_set_Parameters() has been called successfully
//               get_fieldsPtr     : true ==> yt_get_FieldsPtr() has been called successfully
//               get_particlesPtr  : true ==> yt_get_ParticlesPtr() has been called successfully
//               get_gridsPtr      : true ==> yt_get_GridsPtr() has been called successfully
//               commit_grids      : true ==> yt_commit() has been called successfully
//               free_gridsPtr     : true ==> yt_free() has been called successfully, everything is reset and freed.
//
//               field_list          : field list, including field name, field data type, field definition ...
//               particle_list       : particle list, including particle name, data type ...
//               grids_local         : a data structure for storing local grids hierarchy and data memory
//                                     mapping temporary.
//               num_grids_local_MPI : for gathering different MPI processes hierarchy.
//-------------------------------------------------------------------------------------------------------
LibytProcessControl::LibytProcessControl()
{
    libyt_initialized   = false;
    param_yt_set        = false;
    get_fieldsPtr       = false;
    get_particlesPtr    = false;
    get_gridsPtr        = false;
    commit_grids        = false;
    free_gridsPtr       = true;

    field_list          = nullptr;
    particle_list       = nullptr;
    grids_local         = nullptr;
    num_grids_local_MPI = nullptr;
}

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Method      :  Public static method
// Description :  Get the singleton instance.
//
// Notes       :  1. Return the reference of LibytProcessControl instance.
//-------------------------------------------------------------------------------------------------------
LibytProcessControl& LibytProcessControl::Get()
{
    return s_Instance;
}