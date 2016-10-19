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
// Data Member :  [public ] ==> Set by users when calling yt_init()
//                verbose : Verbose level
//                script  : Name of the YT inline analysis script (without the .py extension)
//
//                [private] ==> Set and used by libyt internally
//                libyt_initialized : true ==> yt_init() has been called successfully
//                param_yt_set      : true ==> yt_set_parameter() has been called successfully
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


// private data members
// ===================================================================================
   bool libyt_initialized;
   bool param_yt_set;


   //===================================================================================
   // Constructor :  yt_param_libyt
   // Description :  Constructor of the structure "yt_param_libyt"
   //
   // Note        :  Initialize all data members
   //
   // Parameter   :  None
   //===================================================================================
   yt_param_libyt()
   {

//    set defaults
      verbose = YT_VERBOSE_WARNING;
      script  = "yt_script.py";

      libyt_initialized = false;
      param_yt_set      = false;

   } // METHOD : yt_param_libyt


   //===================================================================================
   // Destructor  :  ~yt_param_libyt
   // Description :  Destructor of the structure "yt_param"_libyt
   //
   // Note        :  Not used currently
   //===================================================================================
   ~yt_param_libyt()
   {

   } // METHOD : ~yt_param_libyt

}; // struct yt_param_libyt



#endif // #ifndef __YT_TYPE_PARAM_LIBYT_H__