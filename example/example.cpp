/*
This example is to test how libyt can work with 
yt python package with OpenMPI.

1. Assign grids to one MPI rank
2. Pass the grid to yt when that grid belongs to the MPI rank
3. Each MPI rank will only pass "its" hierarchy to yt
   ---> Here, we calculate all the grids (libyt_grids) first, 
        then distribute them to grids_local, to simulate the working process.

And also, to illustrates the basic usage of libyt.
In steps 0 - 9.
 */

#include <stdlib.h>
#include <math.h>
#include <typeinfo>
#include <mpi.h>
#include <time.h>

// ==========================================
// 0. include libyt header
// ==========================================
#include "libyt.h"


// single or double precision in the field data
//typedef float real;
typedef double real;

// grid information
#define NGRID_1D  5  // number of root grids along each direction
#define GRID_DIM  8  // grid dimension (this example assumes cubic grids)
#define REFINE_BY 2  // refinement factor between two AMR levels

// convenient macros
#define SQR(a)  ( (a)*(a) )
#define CUBE(a) ( (a)*(a)*(a) )


real set_density( const double x, const double y, const double z, const double t, const double v );
void get_randArray(int *array, int length);


//-------------------------------------------------------------------------------------------------------
// Function    :  main
// Description :  Main function
//-------------------------------------------------------------------------------------------------------
int main( int argc, char *argv[] )
{
   /*
   MPI Initialize, and settings
    */
   int myrank;
   int nrank;
   int RootRank = 0;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   MPI_Comm_size(MPI_COMM_WORLD, &nrank );

// ==========================================
// 1. initialize libyt
// ==========================================
// libyt runtime parameters
   yt_param_libyt param_libyt;

// verbose level
// param_libyt.verbose = YT_VERBOSE_OFF;
// param_libyt.verbose = YT_VERBOSE_INFO;
// param_libyt.verbose = YT_VERBOSE_WARNING;
   param_libyt.verbose = YT_VERBOSE_DEBUG;

// YT analysis script without the ".py" extension (default="yt_inline_script")
   param_libyt.script  = "inline_script";

// *** libyt API ***
   if ( yt_init( argc, argv, &param_libyt ) != YT_SUCCESS )
   {
      fprintf( stderr, "ERROR: yt_init() failed!\n" );
      exit( EXIT_FAILURE );
   }



// **********************************************
// following mimic the simulation evolution loop
// **********************************************
   const int    total_steps = 11;                               // total number of evolution steps
   const double velocity    = 1.0;                              // velocity for setting the density field
   const double dt          = 0.05;                             // evolution time-step
   const double box_size    = 1.0;                              // simulation box size
   const double dh0         = box_size / (NGRID_1D*GRID_DIM);   // cell size at level 0
   const double dh1         = dh0 / REFINE_BY;                  // cell size at level 1
   const int    num_fields  = 1;                                // number of fields
   const int    num_grids   = CUBE(NGRID_1D)+CUBE(REFINE_BY);   // number of grids
                                                                // here we refine one root grid
   
   int *grids_MPI = new int [num_grids];                        // Record MPI rank in each grids

   double time = 0.0;

// this array represents the simulation data stored in memory
   real (*field_data)[num_fields][ CUBE(GRID_DIM) ]
      = new real [num_grids][num_fields][ CUBE(GRID_DIM) ];

   for (int step=0; step<total_steps; step++)
   {
//    ==========================================
//    2. provide YT-specific parameters
//    ==========================================
      yt_param_yt param_yt;

      param_yt.frontend                = "gamer";           // simulation frontend
      param_yt.fig_basename            = "FigName";         // figure base name (default=Fig), will append number of calls to libyt
                                                            // at the end
      param_yt.length_unit             = 3.0857e21;         // units are in cgs
      param_yt.mass_unit               = 1.9885e33;
      param_yt.time_unit               = 3.1557e13;

      param_yt.current_time            = time;
      param_yt.dimensionality          = 3;
      param_yt.refine_by               = REFINE_BY;
      param_yt.num_grids               = num_grids;
      param_yt.num_fields              = num_fields;
      param_yt.field_ftype             = ( typeid(real) == typeid(float) ) ? YT_FLOAT : YT_DOUBLE;

      for (int d=0; d<3; d++)
      {
         param_yt.domain_dimensions[d] = NGRID_1D*GRID_DIM;
         param_yt.domain_left_edge [d] = 0.0;
         param_yt.domain_right_edge[d] = box_size;
         param_yt.periodicity      [d] = 0;
      }

      param_yt.cosmological_simulation = 0;
      param_yt.current_redshift        = 0.5;
      param_yt.omega_lambda            = 0.7;
      param_yt.omega_matter            = 0.3;
      param_yt.hubble_constant         = 0.7;

//    Distribute grids to MPI rank, to simulate simulation code process.
      if (myrank == RootRank){
         get_randArray(grids_MPI, num_grids);
      }
      MPI_Bcast(grids_MPI, num_grids, MPI_INT, RootRank, MPI_COMM_WORLD);

//    Count the number of grids at local
      int num_grids_local = 0;
      for (int i = 0; i < num_grids; i = i+1){
         if (grids_MPI[i] == myrank){
            num_grids_local = num_grids_local + 1;
         }
      }

//    We can either pass in param_yt.grids_MPI or param_yt.num_grids_local
      // param_yt.grids_MPI               = grids_MPI;
      param_yt.num_grids_local         = num_grids_local;

//    *** libyt API ***
      if ( yt_set_parameter( &param_yt ) != YT_SUCCESS )
      {
         fprintf( stderr, "ERROR: yt_set_parameter() failed!\n" );
         exit( EXIT_FAILURE );
      }


//    ==========================================
//    3. [optional] add code-specific parameters
//    ==========================================
      const  int   user_int        = 1;
      const long   user_long       = 2;
      const uint   user_uint       = 3;
      const ulong  user_ulong      = 4;
      const float  user_float      = 5.0;
      const double user_double     = 6.0;
      const char  *user_string     = "my_string";

      const int    user_int3   [3] = { 7, 8, 9 };
      const double user_double3[3] = { 10.0, 11.0, 12.0 };

//    *** libyt API ***
//    to be cautious, one can also check the returns for all these calls
      yt_add_user_parameter_int   ( "user_int",     1, &user_int     );
      yt_add_user_parameter_long  ( "user_long",    1, &user_long    );
      yt_add_user_parameter_uint  ( "user_uint",    1, &user_uint    );
      yt_add_user_parameter_ulong ( "user_ulong",   1, &user_ulong   );
      yt_add_user_parameter_float ( "user_float",   1, &user_float   );
      yt_add_user_parameter_double( "user_double",  1, &user_double  );
      yt_add_user_parameter_string( "user_string",      user_string  );

      yt_add_user_parameter_int   ( "user_int3",    3,  user_int3    );
      yt_add_user_parameter_double( "user_double3", 3,  user_double3 );

//    ============================================================
//    4. Get pointer to field list array, then set up field list
//    ============================================================
      yt_field *field_list;
      yt_get_fieldsPtr( &field_list );

//    We only have one field in this example.
      field_list[0].field_name = "Dens";
      field_list[0].field_define_type = "cell-centered";
      char *field_name_alias[] = {"Name Alias 1", "Name Alias 2", "Name Alias 3"};
      field_list[0].field_name_alias = field_name_alias;
      field_list[0].num_field_name_alias = 3;

//    ============================================================
//    5. Get pointer to local grids array, then set up local grids
//    ============================================================
      yt_grid *grids_local;
      yt_get_gridsPtr( &grids_local );

//    ============================================================
//    Here, we calculate all the grids (libyt_grids) first, 
//    then distribute them to grids_local, to simulate the working process.
//    ============================================================
      yt_grid *libyt_grids = new yt_grid [param_yt.num_grids];

//    set level-0 grids
      int grid_order[3];
      for (grid_order[2]=0; grid_order[2]<NGRID_1D; grid_order[2]++)
      for (grid_order[1]=0; grid_order[1]<NGRID_1D; grid_order[1]++)
      for (grid_order[0]=0; grid_order[0]<NGRID_1D; grid_order[0]++)
      {
         const int gid = (grid_order[2]*NGRID_1D + grid_order[1])*NGRID_1D + grid_order[0];

//       set the hierarchy information of this grid
         for (int d=0; d<3; d++)
         {
            libyt_grids[gid].left_edge [d] = grid_order[d]*GRID_DIM*dh0;
            libyt_grids[gid].right_edge[d] = libyt_grids[gid].left_edge[d] + GRID_DIM*dh0;
            libyt_grids[gid].grid_dimensions[d] = GRID_DIM;   // this example assumes cubic grids
         }

         libyt_grids[gid].particle_count = 0;      // particles are not supported yet
         libyt_grids[gid].id             = gid;    // 0-indexed
         libyt_grids[gid].parent_id      = -1;     // 0-indexed (-1 for grids on the root level)
         libyt_grids[gid].level          = 0;      // 0-indexed

//       in this example we arbitrarily set the field data of this grid
         for (int k=0; k<GRID_DIM; k++)
         for (int j=0; j<GRID_DIM; j++)
         for (int i=0; i<GRID_DIM; i++)
         {
            const double x = libyt_grids[gid].left_edge[0] + (i+0.5)*dh0;
            const double y = libyt_grids[gid].left_edge[1] + (j+0.5)*dh0;
            const double z = libyt_grids[gid].left_edge[2] + (k+0.5)*dh0;

            for (int v=0; v<num_fields; v++) {
               field_data[gid][v][ (k*GRID_DIM+j)*GRID_DIM+i ] = set_density( x, y, z, time, velocity );
            }
         }
      } // for grid_order[0/1/2]


//    in this example we refine the root grid with the peak density into REFINE_BY^3 subgrids
      const double peak[3]       = { 0.5*box_size + velocity*time,
                                     0.5*box_size + velocity*time,
                                     0.5*box_size };
      const double grid_width    = GRID_DIM*dh0;
      const int    center_idx[3] = { int(peak[0]/grid_width),
                                     int(peak[1]/grid_width),
                                     int(peak[2]/grid_width) };
      const int    gid_refine    = ( center_idx[2]*NGRID_1D + center_idx[1] )*NGRID_1D + center_idx[0];
      const int    gid_offset    = CUBE(NGRID_1D);

      for (grid_order[2]=0; grid_order[2]<param_yt.refine_by; grid_order[2]++)
      for (grid_order[1]=0; grid_order[1]<param_yt.refine_by; grid_order[1]++)
      for (grid_order[0]=0; grid_order[0]<param_yt.refine_by; grid_order[0]++)
      {
         const int gid = (grid_order[2]*param_yt.refine_by + grid_order[1])*param_yt.refine_by
                         + grid_order[0] + gid_offset;

//       set the hierarchy information of this grid
         for (int d=0; d<3; d++)
         {
            libyt_grids[gid].left_edge [d] = libyt_grids[gid_refine].left_edge[d] + grid_order[d]*GRID_DIM*dh1;
            libyt_grids[gid].right_edge[d] = libyt_grids[gid].left_edge[d] + GRID_DIM*dh1;
            libyt_grids[gid].grid_dimensions[d] = GRID_DIM;   // this example assumes cubic grids
         }

         libyt_grids[gid].particle_count = 0;            // particles are not supported yet
         libyt_grids[gid].id             = gid;          // 0-indexed
         libyt_grids[gid].parent_id      = gid_refine;   // 0-indexed (-1 for grids on the root level)
         libyt_grids[gid].level          = 1;            // 0-indexed

//       here we arbitrarily set the field data of this grid
         for (int k=0; k<GRID_DIM; k++)
         for (int j=0; j<GRID_DIM; j++)
         for (int i=0; i<GRID_DIM; i++)
         {
            const double x = libyt_grids[gid].left_edge[0] + (i+0.5)*dh1;
            const double y = libyt_grids[gid].left_edge[1] + (j+0.5)*dh1;
            const double z = libyt_grids[gid].left_edge[2] + (k+0.5)*dh1;

            for (int v=0; v<num_fields; v++) {
               field_data[gid][v][ (k*GRID_DIM+j)*GRID_DIM+i ] = set_density( x, y, z, time, velocity );
            }
         }
      } // for grid_order[0/1/2]


//    set general grid attributes and invoke inline analysis
      for (int gid=0; gid<param_yt.num_grids; gid++)
      {
         libyt_grids[gid].field_data = new yt_data [num_fields];

         if (grids_MPI[gid] == myrank){
            for (int v=0; v<num_fields; v++){
               libyt_grids[gid].field_data[v].data_ptr = field_data[gid][v];
            }
         }
         else {
            for (int v=0; v<num_fields; v++){
               // if no data, set it as NULL, so we can make sure each rank contains its own grids only
               libyt_grids[gid].field_data[v].data_ptr = NULL;
            }
         }

      } // for (int gid=0; gid<param_yt.num_grids; gid++) 


//    distribute libyt_grids to grids_local
      int index_local = 0;
      for (int gid = 0; gid < param_yt.num_grids; gid = gid + 1){

         if (grids_MPI[gid] == myrank) {

            for (int d = 0; d < 3; d = d+1) {
               grids_local[index_local].left_edge[d]  = libyt_grids[gid].left_edge[d];
               grids_local[index_local].right_edge[d] = libyt_grids[gid].right_edge[d];
               grids_local[index_local].grid_dimensions[d] = libyt_grids[gid].grid_dimensions[d];
            }
            grids_local[index_local].particle_count = libyt_grids[gid].particle_count;
            grids_local[index_local].id             = libyt_grids[gid].id;
            grids_local[index_local].parent_id      = libyt_grids[gid].parent_id;
            grids_local[index_local].level          = libyt_grids[gid].level;

            for (int v = 0; v < param_yt.num_fields; v = v + 1){
               grids_local[index_local].field_data[v].data_ptr = libyt_grids[gid].field_data[v].data_ptr;
            }

            index_local = index_local + 1;
         }

      }

//    ==============================================
//    6. tell libyt that you have done loading grids
//    ==============================================
//    *** libyt API ***
      if ( yt_commit_grids() != YT_SUCCESS ) {
         fprintf( stderr, "ERROR: yt_commit_grids() failed!\n" );
         exit( EXIT_FAILURE );
      }


//    =============================================================
//    7. perform inline analysis, execute function in python script
//    =============================================================
//    *** libyt API ***
      if ( yt_inline_argument( "yt_inline_ProjectionPlot", 1, "\'density\'" ) != YT_SUCCESS )
      {
         fprintf( stderr, "ERROR: yt_inline() failed!\n" );
         exit( EXIT_FAILURE );
      }

      if ( yt_inline( "yt_inline_ProfilePlot" ) != YT_SUCCESS )
      {
         fprintf( stderr, "ERROR: yt_inline() failed!\n" );
         exit( EXIT_FAILURE );
      }

	  if ( yt_inline("test_user_parameter") != YT_SUCCESS )
	  {
		 fprintf( stderr, "ERROR: yt_inline() failed!\n" );
		 exit( EXIT_FAILURE );
	  }

//    =============================================================================
//    8. end of the inline-analysis at this step, free grid info loaded into python
//    =============================================================================
//    *** libyt API ***
      if ( yt_free_gridsPtr() != YT_SUCCESS )
      {
         fprintf( stderr, "ERROR: yt_free_gridsPtr() failed!\n" );
         exit( EXIT_FAILURE );
      }


//    free resources
      for (int g=0; g<num_grids; g++)  delete [] libyt_grids[g].field_data;
      delete [] libyt_grids;

      time += dt;
   } // for (int step=0; step<total_steps; step++)

// ==========================================
// 9. exit libyt
// ==========================================
// *** libyt API ***
   if ( yt_finalize() != YT_SUCCESS )
   {
      fprintf( stderr, "ERROR: yt_finalize() failed!\n" );
      exit( EXIT_FAILURE );
   }


   delete [] field_data;
   delete [] grids_MPI;
   
   /*
   MPI Finalize
    */
   MPI_Finalize();
   
   return EXIT_SUCCESS;

} // FUNCTION : main



//-------------------------------------------------------------------------------------------------------
// Function    :  set_density
// Description :  Return density at give coordinates and time
//-------------------------------------------------------------------------------------------------------
real set_density( const double x, const double y, const double z, const double t, const double v )
{

   const double center[3]  = { 0.5+v*t, 0.5+v*t, 0.5};   // drift with v along (1,1,0)
   const double sigma      = 0.05;
   const double amplitude  = 1.0e6;
   const double background = 1.0;

   return amplitude*exp(  -0.5*( SQR(x-center[0]) + SQR(y-center[1]) + SQR(z-center[2]) ) / SQR(sigma)  ) + background;

} // FUNCTION : set_density

//-------------------------------------------------------------------------------------------------------
// Function    :  get an array of random number in range 0 ~ NRank-1
// Description :  To random distribute grids to MPI rank
//-------------------------------------------------------------------------------------------------------
void get_randArray(int *array, int length) {
   
   int NRank;
   MPI_Comm_size(MPI_COMM_WORLD, &NRank);
   
   srand((unsigned) time(0));
   
   for (int i = 0; i < length; i = i+1){
      array[i] = rand() % NRank;
   }
} // FUNCTION : get_randArray
