#ifndef __YT_TYPE_H__
#define __YT_TYPE_H__



// enumerate types
// ===================================================================================
enum yt_verbose { YT_VERBOSE_NONE=0, YT_VERBOSE_INFO=1, YT_VERBOSE_WARNING=2, YT_VERBOSE_DEBUG=3 };



// structures used by libyt
// ===================================================================================

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_param_libyt
// Description :  Data structure of libyt runtime parameters
//
// Data Member :  verbose : Verbose level
//                script  : Name of the YT inline analysis script (without the .py extension)
//
// Method      :  yt_param_libyt : Constructor
//               ~yt_param_libyt : Destructor
//-------------------------------------------------------------------------------------------------------
struct yt_param_libyt
{

// data members
// ===================================================================================
   yt_verbose verbose;
   const char *script;


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

   } // METHOD : yt_param_libyt



   //===================================================================================
   // Destructor  :  ~yt_param_libyt
   // Description :  Destructor of the structure "yt_param"_libyt
   //
   // Note        :  Free memory
   //===================================================================================
   ~yt_param_libyt()
   {

   } // METHOD : ~yt_param_libyt

}; // struct yt_param_libyt



#endif // #ifndef __YT_TYPE_H__
