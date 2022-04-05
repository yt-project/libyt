/*
 [Description]
 This example is to show how libyt loads all the data and information into yt,
 then use yt to do analysis on the go. And also, to illustrate the basic
 usage of libyt in steps 0 - 9.

 We have a set of pre-calculated data. We assign each grid to one MPI rank
 randomly to stimulate the actual code of having grid data on different ranks.
 Then we demonstrate how to pass in the grid data and their hierarchies located
 on current MPI rank to libyt. Finally, we use those passed in data to do a
 series of analysis with yt.

 Above description can be concentrated to following steps:
     1. Assign pre-calculated grids to one MPI rank.
     2. Each MPI rank will only pass local data and their hierarchies to libyt
        ---> Here, we calculate all the grids (sim_grids) first,
             then distribute them to grids_local, to simulate the working process.
     3. Do analysis using yt.

 [Compile and Run]
   1. Compile libyt and move libyt.so.* library to ../lib/.
   2. Run this command to set LD_LIBRARY_PATH:
      sh set_ld_path.sh
   3. Update Makefile MPI_PATH.
   4. make clean; make;
   5. mpirun -np 4 --output-filename log ./example
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
#define GHOST_CELL 1  // number of ghost cell in each direction, you can change whatever you like.
                      // We make it like a shift in data array.

// convenient macros
#define SQR(a)  ( (a)*(a) )
#define CUBE(a) ( (a)*(a)*(a) )


real set_density( const double x, const double y, const double z, const double t, const double v );
void get_randArray(int *array, int length);


void level_derived_func(long gid, double *data);                         // derived function for derived field
void level_derived_func_with_name(long gid, char *field, double *data);


void par_io_get_attr(long gid, char *attribute, void *data);   // Must have for yt to get particle's attribute.
                                                               // One particle type need to setup one get_attr.
void getPositionByGID( long gid, real (*Pos)[3] );  // These function is for getting particle position and level info.
void getLevelByGID( long gid, int *Level );         // Used by par_io_get_attr.
yt_grid *gridsPtr;
long     num_total_grids;

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

// Check interface, default is true.
// If it is set false, libyt won't check each input data.
// You can turn off is you have already make sure that everything is input correctly.
   param_libyt.check_data = false;

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
   const int    num_species = 2;                                // number of particle species
   yt_species  *species_list = new yt_species [num_species];    // define species list, so that libyt knows particle species name,
                                                                // and their number of attribute in each of them.
   species_list[0].species_name = "io";                         // particle species "io", with 4 attributes
   species_list[0].num_attr     = 4;
   species_list[1].species_name = "par2";
   species_list[1].num_attr     = 4;
   
   int *grids_MPI = new int [num_grids];                        // Record MPI rank in each grids

   double time = 0.0;
   num_total_grids = (long) num_grids;

// this array represents the simulation data stored in memory
// there can be GHOST_CELL at the grid's boundary.
   real (*field_data)[num_fields][ CUBE(GRID_DIM+GHOST_CELL*2) ]
      = new real [num_grids][num_fields][ CUBE(GRID_DIM+GHOST_CELL*2) ];

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
      param_yt.num_fields              = num_fields + 2;    // The additional fields are for demonstrating derived_func.
      param_yt.num_species             = num_species;
      param_yt.species_list            = species_list;

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

//    Pass in param_yt.num_grids_local
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
      // Since we are now using "gamer" as frontend, we need to set code specific parameter.
      // mhd must be defined in gamer frontend fields.py.
      const int mhd = 0;   
      yt_add_user_parameter_int("mhd", 1, &mhd);
      const int srhd = 0;
      yt_add_user_parameter_int("srhd", 1, &srhd);

      // You can also input your own code specific parameter to match your frontend's fields.py
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
//    4. Get pointer to field list array and particle list array, 
//       and set them up
//    ============================================================
//    (1) Set up field list
      yt_field *field_list;
      yt_get_fieldsPtr( &field_list );
//    (1)-1 Field "Dens" is a cell-centered pre-calculated data.
      field_list[0].field_name = "Dens";
      field_list[0].field_define_type = "cell-centered";
      field_list[0].swap_axes = true; // default is true, which means data is represented in [z][y][x] coordinate.
      field_list[0].field_dtype = ( typeid(real) == typeid(float) ) ? YT_FLOAT : YT_DOUBLE;
      char *field_name_alias[] = {"Name Alias 1", "Name Alias 2", "Name Alias 3"};
      field_list[0].field_name_alias = field_name_alias;
      field_list[0].num_field_name_alias = 3;
      for(int d = 0; d < 6; d++){
          // Number of ghost cell in the beginning and the end of the data array in each dimension in the point of view
          // of the data array.
          // field_ghost_cell[0] -> number of ghost cell to be ignored at the beginning of 0-dim of the data.
          // field_ghost_cell[1] -> number of ghost cell to be ignored at the end of 0-dim of the data.
          field_list[0].field_ghost_cell[d] = GHOST_CELL;
      }
//    (1)-2 Field "level_derived_func" is a derived field, which its data is generated by level_derived_func.
//          For field_define_type = "derived_func", one should set derived_func or derived_func_with_name according to
//          the generating function type.
//          Since there is no pre-existing field to wrap, we don't need to set field_dtype.
      field_list[1].field_name = "level_derived_func";
      field_list[1].field_define_type = "derived_func";
      field_list[1].swap_axes = true;
      field_list[1].derived_func = level_derived_func;
//    (1)-3 Field "level_derived_func_with_name" is a derived field, its data is generated by level_derived_func_with_name.
      field_list[2].field_name = "level_derived_func_with_name";
      field_list[2].field_define_type = "derived_func";
      field_list[2].swap_axes = true;
      field_list[2].derived_func_with_name = level_derived_func_with_name;

//    (2) Set up particle list
      yt_particle *particle_list;
      yt_get_particlesPtr( &particle_list );
//    We have one particle species, with 4 attributes in this example, so the case is simple.
//    Be careful that the order you filled in particle_list, should be the same yt_species *species_list.
      particle_list[0].species_name = "io";     // This two line is redundant, since libyt has already filled in.
      particle_list[0].num_attr     = 4;        // I type it here just to make things clear.

      char     *attr_name[]  = {"ParPosX", "ParPosY", "ParPosZ", "Level"}; // Attribute name
      char     *attr_name_alias[] = {"grid_level"};                        // Alias name for attribute level
      for ( int v=0; v < 4; v++ ){
         
         particle_list[0].attr_list[v].attr_name  = attr_name[v];    // Must fill in attribute name.
         
         if ( v == 3 ){
            particle_list[0].attr_list[v].attr_dtype = YT_INT;       // Must fill in attribute data type.
            particle_list[0].attr_list[v].attr_unit  = "";           // [Optional] if not filled in, libyt will use XXXFieldInfo 
                                                                     // set by param_yt.frontend if it exists.
            particle_list[0].attr_list[v].num_attr_name_alias = 1;   // [Optional] set name alias of this attribute.
            particle_list[0].attr_list[v].attr_name_alias     = attr_name_alias;
            particle_list[0].attr_list[v].attr_display_name   = "Level of the Grid"; // [Optional] if not fill in, libyt will 
                                                                                     // display attr_name. 
         }   
         else{ 
            particle_list[0].attr_list[v].attr_dtype = ( typeid(real) == typeid(float) ) ? YT_FLOAT : YT_DOUBLE; 
         }
      }

      particle_list[0].coor_x = attr_name[0];  // Must fill in this to tell libyt how to find the position attributes
      particle_list[0].coor_y = attr_name[1];  // for this type of particle.
      particle_list[0].coor_z = attr_name[2];

      particle_list[0].get_attr = par_io_get_attr;   // par_io_get_attr is a function ptr that takes arguments (long, char* void*)
                                                     // and returns void.

      for ( int v=0; v<4; v++ ){
         particle_list[1].attr_list[v].attr_name = attr_name[v];     // Fill in for particle species "par2"
      }
      particle_list[1].coor_x = attr_name[0];
      particle_list[1].coor_y = attr_name[1];
      particle_list[1].coor_z = attr_name[2];

//    ============================================================
//    5. Get pointer to local grids array, then set up local grids
//    ============================================================
      yt_grid *grids_local;
      yt_get_gridsPtr( &grids_local );

//    ========================================================================================
//    Here, we calculate all the grids (sim_grids) first, which is grids in simulation code.
//    Then distribute them to (grids_local), to simulate the working process.
//    ========================================================================================
      yt_grid *sim_grids = new yt_grid [param_yt.num_grids];
      gridsPtr = sim_grids;

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
            sim_grids[gid].left_edge [d] = grid_order[d]*GRID_DIM*dh0;
            sim_grids[gid].right_edge[d] = sim_grids[gid].left_edge[d] + GRID_DIM*dh0;
            sim_grids[gid].grid_dimensions[d] = GRID_DIM;
         }

         sim_grids[gid].id                     = gid;    // 0-indexed
         sim_grids[gid].parent_id              = -1;     // 0-indexed (-1 for grids on the root level)
         sim_grids[gid].level                  = 0;      // 0-indexed

         // In this example we arbitrarily set the field data of this grid.
         // But we exclude ghost cell, since ghost cells will be ignored in yt.
         for (int k=GHOST_CELL; k<GRID_DIM+GHOST_CELL; k++)
         for (int j=GHOST_CELL; j<GRID_DIM+GHOST_CELL; j++)
         for (int i=GHOST_CELL; i<GRID_DIM+GHOST_CELL; i++)
         {
            const double x = sim_grids[gid].left_edge[0] + ((i - GHOST_CELL)+0.5)*dh0;
            const double y = sim_grids[gid].left_edge[1] + ((j - GHOST_CELL)+0.5)*dh0;
            const double z = sim_grids[gid].left_edge[2] + ((k - GHOST_CELL)+0.5)*dh0;

            for (int v=0; v<num_fields; v++) {
               field_data[gid][v][ (k*(GRID_DIM+GHOST_CELL*2)+j)*(GRID_DIM+GHOST_CELL*2)+i ] = set_density( x, y, z, time, velocity );
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
            sim_grids[gid].left_edge [d] = sim_grids[gid_refine].left_edge[d] + grid_order[d]*GRID_DIM*dh1;
            sim_grids[gid].right_edge[d] = sim_grids[gid].left_edge[d] + GRID_DIM*dh1;
            sim_grids[gid].grid_dimensions[d] = GRID_DIM;   // this example assumes cubic grids
         }

         sim_grids[gid].id             = gid;          // 0-indexed
         sim_grids[gid].parent_id      = gid_refine;   // 0-indexed (-1 for grids on the root level)
         sim_grids[gid].level          = 1;            // 0-indexed

//       here we arbitrarily set the field data of this grid
         for (int k=GHOST_CELL; k<GRID_DIM+GHOST_CELL; k++)
         for (int j=GHOST_CELL; j<GRID_DIM+GHOST_CELL; j++)
         for (int i=GHOST_CELL; i<GRID_DIM+GHOST_CELL; i++)
         {
            const double x = sim_grids[gid].left_edge[0] + ((i - GHOST_CELL)+0.5)*dh1;
            const double y = sim_grids[gid].left_edge[1] + ((j - GHOST_CELL)+0.5)*dh1;
            const double z = sim_grids[gid].left_edge[2] + ((k - GHOST_CELL)+0.5)*dh1;

            for (int v=0; v<num_fields; v++) {
               field_data[gid][v][ (k*(GRID_DIM+GHOST_CELL*2)+j)*(GRID_DIM+GHOST_CELL*2)+i ] = set_density( x, y, z, time, velocity );
            }
         }
      } // for grid_order[0/1/2]


//    set field_data in sim_grid, which represents grid and data in simulation code.
      for (int gid=0; gid<param_yt.num_grids; gid++)
      {
         sim_grids[gid].field_data = new yt_data [num_fields];

         if (grids_MPI[gid] == myrank){
            for (int v=0; v<num_fields; v++){
               sim_grids[gid].field_data[v].data_ptr = field_data[gid][v];
            }
         }
         else {
            for (int v=0; v<num_fields; v++){
               // if no data, set it as NULL, so we can make sure each rank contains its own grids only
               sim_grids[gid].field_data[v].data_ptr = NULL;
            }
         }

      } // for (int gid=0; gid<param_yt.num_grids; gid++) 


//    Load the actual information to libyt.
      int index_local = 0;
      for (int gid = 0; gid < param_yt.num_grids; gid = gid + 1){

         if (grids_MPI[gid] == myrank) {

            for (int d = 0; d < 3; d = d+1) {
               grids_local[index_local].left_edge[d]  = sim_grids[gid].left_edge[d];
               grids_local[index_local].right_edge[d] = sim_grids[gid].right_edge[d];
               grids_local[index_local].grid_dimensions[d] = sim_grids[gid].grid_dimensions[d];
            }
            grids_local[index_local].particle_count_list[0] = 1; // set the num of particle in species.
            grids_local[index_local].id             = sim_grids[gid].id;
            grids_local[index_local].parent_id      = sim_grids[gid].parent_id;
            grids_local[index_local].level          = sim_grids[gid].level;

            for (int v = 0; v < param_yt.num_fields; v = v + 1){
               grids_local[index_local].field_data[v].data_ptr = sim_grids[gid].field_data[v].data_ptr;
            }

            index_local = index_local + 1;
         }

      }

//    =========================================================================
//    6. tell libyt that you have done loading grids, field_list, particle_list
//    =========================================================================
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
         fprintf( stderr, "ERROR: yt_inline_argument() failed!\n" );
         exit( EXIT_FAILURE );
      }

      if ( yt_inline( "yt_inline_ProfilePlot" ) != YT_SUCCESS )
      {
         fprintf( stderr, "ERROR: yt_inline() failed!\n" );
         exit( EXIT_FAILURE );
      }

      if ( yt_inline( "yt_inline_ParticlePlot" ) != YT_SUCCESS )
      {
         fprintf( stderr, "ERROR: yt_inline() failed!\n" );
         exit( EXIT_FAILURE );
      }

	   if ( yt_inline("test_function") != YT_SUCCESS )
	   {
         fprintf( stderr, "ERROR: test_function() failed!\n" );
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
      for (int g=0; g<num_grids; g++)  delete [] sim_grids[g].field_data;
      delete [] sim_grids;

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
   delete [] species_list;
   
   /*
   MPI Finalize
    */
   MPI_Finalize();
   
   return EXIT_SUCCESS;

} // FUNCTION : main



//-------------------------------------------------------------------------------------------------------
// Function    :  set_density
// Description :  Return density at given coordinates and time
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
// Description :  To random distribute grids to MPI rank, so that we can mimic what happen in real world
//                simulation code.
//-------------------------------------------------------------------------------------------------------
void get_randArray(int *array, int length) {
   
   int NRank;
   MPI_Comm_size(MPI_COMM_WORLD, &NRank);
   
   srand((unsigned) time(0));
   
   for (int i = 0; i < length; i = i+1){
      array[i] = rand() % NRank;
   }
} // FUNCTION : get_randArray


//-------------------------------------------------------------------------------------------------------
// Function    :  level_derived_func
// Description :  Generate data in field name "level_derived_func" by gid.
//
// Notes       :  1. Prototype must be void func(long, double*) or void func(long, char*, double*).
//                2. This function will be concatenated into python C extension, so that yt can get
//                   derived field data when needed.
//                3. In this example, we will write the grid's level to data.
//                4. Write results to double *data in [z][y][x] order, since we set swap_axes = true for
//                   this field.
//
// Parameter   : long     gid      : derived field data in grid gid to be return
//               double  *data     : write the requested data to this array
//-------------------------------------------------------------------------------------------------------
void level_derived_func(long gid, double *data){
    // Get grid level by gid.
    int level;
    getLevelByGID(gid, &level);

    // Get grid dimension by libyt API.
    int dim[3];
    yt_getGridInfo_Dimensions(gid, &dim );

    // Set data array
    int index;
    for(int k=0; k<dim[2]; k++){
        for(int j=0; j<dim[1]; j++){
            for(int i=0; i<dim[0]; i++){
                index = k * dim[1] * dim[0] + j * dim[0] + i;
                data[index] = (double) level;
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------------
// Function    :  level_derived_func_with_name
// Description :  Generate data in field name "level_derived_func_with_name" by gid and a given field name.
//
// Notes       :  1. Prototype must be void func(long, double*) or void func(long, char*, double*).
//                2. This function will be concatenated into python C extension, so that yt can get
//                   derived field data when needed.
//                3. In this example, we will write the grid's level to data.
//                4. Write results to double *data in [z][y][x] order, since we set swap_axes = true for
//                   this field.
//
// Parameter   : long     gid      : derived field data in grid gid to be return
//               char    *field    : target field name
//               double  *data     : write the requested data to this array
//-------------------------------------------------------------------------------------------------------
void level_derived_func_with_name(long gid, char *field, double *data){
    // Get grid level by gid.
    int level;
    getLevelByGID(gid, &level);

    // Get grid dimension by libyt API.
    int dim[3];
    yt_getGridInfo_Dimensions(gid, &dim );

    // Set data array
    int index;
    for(int k=0; k<dim[2]; k++){
        for(int j=0; j<dim[1]; j++){
            for(int i=0; i<dim[0]; i++){
                index = k * dim[1] * dim[0] + j * dim[0] + i;
                data[index] = (double) level;
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  par_io_get_attr
// Description :  For particle type "io" to return their attribute.
// 
// Notes       :  1. Prototype must be void func(long, char*, void*)
//                2. This function will be concatenated into python C extension, so that yt can reach
//                   particle attributes when it is needed.
//                3. In this example, we will create particle with position at the center of the grid it
//                   belongs to with Level equals to the level of the grid.
//                4. Write results to void *data in series.
// 
// Parameter   : long  gid      : particle in grid gid to be return
//               char *attribute: get the attribute of the particle
//               void *data     : write the requested particle data to this array
//-------------------------------------------------------------------------------------------------------
void par_io_get_attr(long gid, char *attribute, void *data){
   // Since we set the particle data to be level, which means length = 1.
   long len_array = 1;
   
   real Pos[3];
   getPositionByGID( gid, &Pos );

   int Level;
   getLevelByGID( gid, &Level );

   // Since this example is very simple, we only have one particle in each grids.
   // So we ignore the for loop.
   if ( strcmp(attribute, "ParPosX") == 0 ){
      ((real *)data)[0] = Pos[0];
   }
   else if ( strcmp(attribute, "ParPosY") == 0 ){
      ((real *)data)[0] = Pos[1];
   }
   else if ( strcmp(attribute, "ParPosZ") == 0 ){
      ((real *)data)[0] = Pos[2];
   }
   else if ( strcmp(attribute, "Level") == 0 ){
      ((int  *)data)[0] = Level;
   }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  getPositionByGID
// Description :  Get the center position of the grid id = gid.
//-------------------------------------------------------------------------------------------------------
void getPositionByGID( long gid, real (*Pos)[3] ){
   for ( long i = 0; i < num_total_grids; i++ ){
      if ( gridsPtr[i].id == gid ){
         for ( int d = 0; d < 3; d++ ){
            (*Pos)[d] = (real) 0.5 * (gridsPtr[i].left_edge[d] + gridsPtr[i].right_edge[d]);
         }
         break;
      }
   }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  getLevelByGID
// Description :  Get the level of the grid id = gid.
//-------------------------------------------------------------------------------------------------------
void getLevelByGID( long gid, int *Level ){
   for ( long i = 0; i < num_total_grids; i++ ){
      if ( gridsPtr[i].id == gid ){
         *Level = gridsPtr[i].level;
         break;
      }
   }
}
