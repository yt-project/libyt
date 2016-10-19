#include <stdlib.h>
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
      param_yt.domain_left_edge [d] = 0.0;
      param_yt.domain_right_edge[d] = 100.0 + d;
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
   param_yt.num_grids               = 5L;

   yt_set_parameter( &param_yt );


// overwrite existing parameter
   param_yt.length_unit             = 1.1;
   param_yt.mass_unit               = 1.2;
   param_yt.time_unit               = 1.3;

   yt_set_parameter( &param_yt );


// add code-specific parameters
   const  int   user_int        = 1;
   const long   user_long       = 2;
   const uint   user_uint       = 3;
   const ulong  user_ulong      = 4;
   const float  user_float      = 5.0;
   const double user_double     = 6.0;
   const char  *user_string     = "my_string";

   const int    user_int3   [3] = { 7, 8, 9 };
   const double user_double3[3] = { 10.0, 11.0, 12.0 };

   yt_add_user_parameter_int   ( "user_int",     1, &user_int     );
   yt_add_user_parameter_long  ( "user_long",    1, &user_long    );
   yt_add_user_parameter_uint  ( "user_uint",    1, &user_uint    );
   yt_add_user_parameter_ulong ( "user_ulong",   1, &user_ulong   );
   yt_add_user_parameter_float ( "user_float",   1, &user_float   );
   yt_add_user_parameter_double( "user_double",  1, &user_double  );
   yt_add_user_parameter_string( "user_string",      user_string  );

   yt_add_user_parameter_int   ( "user_int3",    3,  user_int3    );
   yt_add_user_parameter_double( "user_double3", 3,  user_double3 );


// add grids
   const int    seed      = 123;
   const int    grid_size = 8;
   const double dh        = 1.0;

   srand( seed );

   yt_grid *grids = new yt_grid [param_yt.num_grids];

   for (int g=0; g<param_yt.num_grids; g++)
   {
      for (int d=0; d<3; d++)
      {
         grids[g].left_edge [d] = (double)rand()/RAND_MAX;
         grids[g].right_edge[d] = grids[g].left_edge[d] + grid_size*dh;
         grids[g].dimensions[d] = grid_size;
      }

      grids[g].particle_count = rand();
      grids[g].id             = g;
      grids[g].parent_id      = -1;
      grids[g].level          = 0;

      yt_add_grid( &grids[g] );
   }


// perform inline analysis
   yt_inline();

   delete [] grids;


// exit libyt
   yt_finalize();

}
