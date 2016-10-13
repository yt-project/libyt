#include <stdio.h>
#include "yt.h"
#include "yt_macro.h"
#include "yt_type.h"


// global variables
yt_verbose_t verbose = YT_VERBOSE_INFO;




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_init
// Description :  Initialize libyt
//
// Note        :  None
//
// Parameter   :  None
//
// Return      :  None
//-------------------------------------------------------------------------------------------------------
int yt_init()
{

   static bool initialized = false;

// check whether libyt has been initialized already
   if ( initialized )   return YT_SUCCESS;

   if ( verbose )    fprintf( stdout, "%s ...\n", __FUNCTION__ );



   if ( verbose )    fprintf( stdout, "%s ... done\n", __FUNCTION__ );


   initialized = true;

   return YT_SUCCESS;

} // FUNCTION : yt_init
