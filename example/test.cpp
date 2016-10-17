#include "libyt.h"


int main( int argc, char *argv[] )
{

// initialize libyt
   yt_param_libyt param_libyt;

// param_libyt.verbose = YT_VERBOSE_NONE;
// param_libyt.verbose = YT_VERBOSE_INFO;
// param_libyt.verbose = YT_VERBOSE_WARNING;
   param_libyt.verbose = YT_VERBOSE_DEBUG;
   param_libyt.script  = "inline_script";

   yt_init( argc, argv, &param_libyt );


// provide YT-specific parameters
   yt_param_yt param_yt;

// set defaults
   param_yt.frontend                = "yt_frontend";

   for (int d=0; d<3; d++)
   {
      param_yt.domain_left_edge [d] = 1.0 + d;
      param_yt.domain_right_edge[d] = 5.9 + d;
   }
   param_yt.current_time            = 1.0;
   param_yt.current_redshift        = 2.0;
   param_yt.omega_lambda            = 3.0;
   param_yt.omega_matter            = 4.0;
   param_yt.hubble_constant         = 5.0;
   param_yt.length_unit             = 6.0;
   param_yt.mass_unit               = 7.0;
   param_yt.time_unit               = 8.0;

   for (int d=0; d<3; d++)
   {
      param_yt.periodicity      [d] = d%2;
      param_yt.domain_dimensions[d] = 100*(d+1);
   }
// param_yt.cosmological_simulation = 0;
   param_yt.cosmological_simulation = 1;
   param_yt.dimensionality          = 3;

   yt_set_parameter( &param_yt );


// perform inline analysis
   yt_inline();


// exit libyt
   yt_finalize();

}
