#include <stdio.h>
#include "yt.h"
#include "yt_macro.h"
#include "yt_type.h"
#include "yt_prototype.h"


// global variables
yt_verbose_t verbose = YT_VERBOSE_BASIC;




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_init
// Description :  Initialize libyt
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_init()
{

   static bool initialized = false;

// nothing to do if libyt has been initialized
   if ( initialized )
   {
      if ( verbose >= YT_VERBOSE_DETAIL )    log_warning( "libyt has been initialized already\n" );

      return YT_SUCCESS;
   }

   if ( verbose )    log_info( "Initializing libyt ...\n" );

// init_python();

   initialized = true;

   return YT_SUCCESS;

} // FUNCTION : yt_init
