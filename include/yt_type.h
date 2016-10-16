#ifndef __YT_TYPE_H__
#define __YT_TYPE_H__



// enumerate types
// ===================================================================================
enum yt_verbose { YT_VERBOSE_NONE=0, YT_VERBOSE_INFO=1, YT_VERBOSE_WARNING=2, YT_VERBOSE_DEBUG=3 };



// structures used by libyt
// ===================================================================================

//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_param
// Description :  Data structure of libyt runtime parameters
//
// Data Member :  verbose : Verbose level
//                script  : Name of the YT inline analysis script (without the .py extension)
//
// Method      :  yt_param : Constructor
//               ~yt_param : Destructor
//-------------------------------------------------------------------------------------------------------
struct yt_param
{

// data members
// ===================================================================================
   yt_verbose verbose;
   const char *script;


   //===================================================================================
   // Constructor :  yt_param
   // Description :  Constructor of the structure "yt_param"
   //
   // Note        :  Initialize all data members
   //
   // Parameter   :  None
   //===================================================================================
   yt_param()
   {

//    set defaults
      verbose = YT_VERBOSE_WARNING;
      script  = "yt_script.py";

   } // METHOD : yt_param



   //===================================================================================
   // Destructor  :  ~yt_param
   // Description :  Destructor of the structure "yt_param"
   //
   // Note        :  Free memory
   //===================================================================================
   ~yt_param()
   {

   } // METHOD : ~yt_param

}; // struct yt_param



#endif // #ifndef __YT_TYPE_H__
