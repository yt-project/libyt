#include <stdlib.h>
#include <typeinfo>
// ==========================================
// include libyt header
// ==========================================
#include "libyt.h"



int main( int argc, char *argv[] )
{

// ==========================================
// 1. initialize libyt
// ==========================================
// set libyt runtime parameters
   yt_param_libyt param_libyt;

// verbose level
// param_libyt.verbose = YT_VERBOSE_OFF;
// param_libyt.verbose = YT_VERBOSE_INFO;
// param_libyt.verbose = YT_VERBOSE_WARNING;
   param_libyt.verbose = YT_VERBOSE_DEBUG;

// YT analysis script without the ".py" extension (default="yt_inline_script")
   param_libyt.script  = "inline_script";

// *** libyt API ***
   yt_init( argc, argv, &param_libyt );


// ==========================================
// 2. provide YT-specific parameters
// ==========================================
   yt_param_yt param_yt;

   param_yt.frontend                = "name_of_the_target_frontend";
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
   param_yt.num_grids               = 3L;    // total number of grids

// *** libyt API ***
   yt_set_parameter( &param_yt );


// ==========================================
// 3. [optional] add code-specific parameters
// ==========================================
   const  int   user_int        = 1;
   const long   user_long       = 2;
   const uint   user_uint       = 3;
   const ulong  user_ulong      = 4;
   const float  user_float      = 5.0;
   const double user_double     = 6.0;
   const char  *user_string     = "my_string";

   const int    user_int3   [3] = { 7, 8, 9 };
   const double user_double3[3] = { 10.0, 11.0, 12.0 };

// *** libyt API ***
   yt_add_user_parameter_int   ( "user_int",     1, &user_int     );
   yt_add_user_parameter_long  ( "user_long",    1, &user_long    );
   yt_add_user_parameter_uint  ( "user_uint",    1, &user_uint    );
   yt_add_user_parameter_ulong ( "user_ulong",   1, &user_ulong   );
   yt_add_user_parameter_float ( "user_float",   1, &user_float   );
   yt_add_user_parameter_double( "user_double",  1, &user_double  );
   yt_add_user_parameter_string( "user_string",      user_string  );

   yt_add_user_parameter_int   ( "user_int3",    3,  user_int3    );
   yt_add_user_parameter_double( "user_double3", 3,  user_double3 );


// ==========================================
// 4. add grids
// ==========================================
   const int    grid_width = 4;     // number of cells along each direction
   const int    num_fields = 2;     // number of fields
   const double dh         = 1.0;   // cell size in code units

   const int grid_size   = grid_width*grid_width*grid_width;
   const int random_seed = 123;

   const char *field_labels[num_fields] = { "density", "temperature" };   // field names

// data of all fields in all grids
// ==> please set "grids[*].field_ftype" to YT_FLOAT or YT_DOUBLE based on the type of field_data
// typedef float  real;
   typedef double real;
   real (*field_data)[num_fields][grid_size]
      = new real [param_yt.num_grids][num_fields][grid_size];

   srand( random_seed );

   yt_grid *grids = new yt_grid [param_yt.num_grids];

   for (int g=0; g<param_yt.num_grids; g++)
   {
//    arbitrarily set the hierarchy information of this grid
      for (int d=0; d<3; d++)
      {
         grids[g].left_edge [d] = (double)rand()/RAND_MAX;
         grids[g].right_edge[d] = grids[g].left_edge[d] + grid_width*dh;
         grids[g].dimensions[d] = grid_width;   // this example assumes that the grid is a cube
      }

      grids[g].particle_count = rand();
      grids[g].id             = g;        // 0-indexed
      grids[g].parent_id      = -1;       // 0-indexed (-1 for grids on the root level)
      grids[g].level          = 0;        // 0-indexed

//    arbitraryly intialize the field data of this grid
      for (int v=0; v<num_fields; v++)
      for (int t=0; t<grid_size; t++)
         field_data[g][v][t] = (double)rand()/RAND_MAX + v;

//    set pointers pointing to different field data
      grids[g].field_data = new void* [num_fields];
      for (int v=0; v<num_fields; v++)   grids[g].field_data[v] = field_data[g][v];

//    set other field parameters
      grids[g].num_fields   = num_fields;
      grids[g].field_labels = field_labels;
      grids[g].field_ftype  = ( typeid(real) == typeid(float) ) ? YT_FLOAT : YT_DOUBLE;

//    *** libyt API ***
      yt_add_grid( &grids[g] );
   } //for (int g=0; g<param_yt.num_grids; g++)


// ==========================================
// 5. perform inline analysis
// ==========================================
// *** libyt API ***
   yt_inline();

// one needs to repeat steps [2-4] before calling yt_inline() again
   yt_set_parameter( &param_yt );
   yt_add_user_parameter_double( "user_double3", 3,  user_double3 );
   for (int g=0; g<param_yt.num_grids; g++)  yt_add_grid( &grids[g] );

   yt_inline();


// ==========================================
// 6. exit libyt
// ==========================================
// *** libyt API ***
   yt_finalize();


// free resource
   delete [] field_data;
   for (int g=0; g<param_yt.num_grids; g++)  delete [] grids[g].field_data;
   delete [] grids;


   return EXIT_SUCCESS;

} // FUNCTION : main

