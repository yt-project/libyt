#include "libyt.h"


int main( int argc, char *argv[] )
{

   yt_param_libyt param_libyt;

// param_libyt.verbose = YT_VERBOSE_NONE;
// param_libyt.verbose = YT_VERBOSE_INFO;
// param_libyt.verbose = YT_VERBOSE_WARNING;
   param_libyt.verbose = YT_VERBOSE_DEBUG;
   param_libyt.script  = "inline_script";

   yt_init( argc, argv, &param_libyt );
   yt_finalize();

// yt_init( argc, argv, &param_libty );
// yt_finalize();

}
