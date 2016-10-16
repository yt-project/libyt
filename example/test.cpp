#include "libyt.h"


int main( int argc, char *argv[] )
{

   yt_param yt_param;

// yt_param.verbose = YT_VERBOSE_NONE;
// yt_param.verbose = YT_VERBOSE_INFO;
// yt_param.verbose = YT_VERBOSE_WARNING;
   yt_param.verbose = YT_VERBOSE_DEBUG;
   yt_param.script  = "inline_script";

   yt_init( argc, argv, &yt_param );
   yt_finalize();

// yt_init( argc, argv, &yt_param );
// yt_finalize();

}
