#ifndef __YT_TYPE_PARAM_LIBYT_H__
#define __YT_TYPE_PARAM_LIBYT_H__



/*******************************************************************************
/
/  yt_param_libyt structure
/
/  ==> included by yt_type.h
/
********************************************************************************/


//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_param_libyt
// Description :  Data structure of libyt runtime parameters
//
// Data Member :  [public ] ==> Set by users when calling yt_initialize()
//                verbose : Verbose level
//                script  : Name of the YT inline analysis script (without the .py extension)
//                counter : Number of rounds doing inline-analysis
//                check_data: Check the input data (ex: hierarchy, grid information...), if it is true.
//
//                [private] ==> Set and used by libyt internally. Don't touch these parts, since they will
//                              be removed in the later update.
//                libyt_initialized           : true ==> yt_initialize() has been called successfully
//                param_yt_set                : true ==> yt_set_Parameters() has been called successfully
//                get_fieldsPtr               : true ==> yt_get_FieldsPtr() has been called successfully
//                get_particlesPtr            : true ==> yt_get_ParticlesPtr() has been called successfully
//                get_gridsPtr                : true ==> yt_get_GridsPtr() has been called successfully
//                commit_grids                : true ==> yt_commit() has been called successfully
//                free_gridsPtr               : true ==> yt_free() has been called successfully,
//                                                       everything is reset and freed.
//
// Method      :  yt_param_libyt : Constructor
//               ~yt_param_libyt : Destructor
//-------------------------------------------------------------------------------------------------------
struct yt_param_libyt
{

// public data members
// ===================================================================================
   yt_verbose verbose;
   const char *script;
   long       counter;   
   bool       check_data;

// private data members
// ===================================================================================
   bool  libyt_initialized;
   bool  param_yt_set;
   bool  get_fieldsPtr;
   bool  get_particlesPtr;
   bool  get_gridsPtr;
   bool  commit_grids;
   bool  free_gridsPtr;
   


   //===================================================================================
   // Method      :  yt_param_libyt
   // Description :  Constructor of the structure "yt_param_libyt"
   //
   // Note        :  Initialize all data members
   //
   // Parameter   :  None
   //===================================================================================
   yt_param_libyt()
   {

//    Set by user
      verbose    = YT_VERBOSE_WARNING;
      script     = "yt_inline_script";
      counter    = 0;
      check_data = true;

//    Set by libyt
      libyt_initialized  = false;
      param_yt_set       = false;
      get_fieldsPtr      = false;
      get_particlesPtr   = false;
      get_gridsPtr       = false;
      commit_grids       = false;
      free_gridsPtr      = true;
   } // METHOD : yt_param_libyt


   //===================================================================================
   // Method      :  ~yt_param_libyt
   // Description :  Destructor of the structure "yt_param"_libyt
   //
   // Note        :  Free memory
   //===================================================================================
   ~yt_param_libyt()
   {

   } // METHOD : ~yt_param_libyt

}; // struct yt_param_libyt



#endif // #ifndef __YT_TYPE_PARAM_LIBYT_H__
