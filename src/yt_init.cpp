// define DEFINE_GLOBAL since this file **defines** all global variables
#define DEFINE_GLOBAL
#include "yt_combo.h"
#undef DEFINE_GLOBAL
#include "libyt.h"

static void init_yt_long_mpi_type();
static void init_yt_hierarchy_mpi_type();
static void init_yt_rma_grid_info_mpi_type();
static void init_yt_rma_particle_info_mpi_type();
static void init_general_info();


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_init
// Description :  Initialize libyt
//
// Note        :  1. Input "param_libyt" will be backed up to a libyt global variable
//                2. This function should not be called more than once (even if yt_finalize has been called)
//                   since some extensions (e.g., NumPy) may not work properly.
//                3. Initialize general info and user-defined MPI data type.
//
// Parameter   :  argc        : Argument count
//                argv        : Argument vector
//                param_libyt : libyt runtime parameters
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_init( int argc, char *argv[], const yt_param_libyt *param_libyt )
{
#ifdef SUPPORT_TIMER
   // initialize timer
   char filename[50];
   sprintf(filename, "RecordTime_%d", g_myrank);
   g_timer = new Timer(filename);
   // start timer.
   g_timer->record_time("yt_init", 0);
#endif

// yt_init should only be called once
   static int init_count = 0;
   init_count ++;

// still need to check "init_count" since yt_finalize() will set "g_param_libyt.libyt_initialized = false"
   if ( g_param_libyt.libyt_initialized  ||  init_count >= 2 )
      YT_ABORT( "yt_init() should not be called more than once!\n" );


// store user-provided parameters to a libyt internal variable
// --> better do it **before** calling any log function since they will query g_param_libyt.verbose
   g_param_libyt.verbose    = param_libyt->verbose;
   g_param_libyt.script     = param_libyt->script;
   g_param_libyt.counter    = param_libyt->counter;   // useful during restart, where the initial counter can be non-zero
   g_param_libyt.check_data = param_libyt->check_data;

   log_info( "Initializing libyt ...\n" );
   log_info( "   verbose = %d\n", g_param_libyt.verbose );
   log_info( "    script = %s\n", g_param_libyt.script  );
   log_info( "   counter = %ld\n", g_param_libyt.counter);
   log_info( "check_data = %s\n", (g_param_libyt.check_data ? "true" : "false"));

// create libyt module, should be before init_python
   if ( create_libyt_module() == YT_FAIL ) {
      return YT_FAIL;
   }

// initialize Python interpreter
   if ( init_python(argc,argv) == YT_FAIL ) {
      return YT_FAIL;
   }  

// import libyt and inline python script.
   if ( init_libyt_module() == YT_FAIL ) {
      return YT_FAIL;
   }

   // Initialize general info
   init_general_info();

   // Initialize user-defined MPI data type
   init_yt_long_mpi_type();
   init_yt_hierarchy_mpi_type();
   init_yt_rma_grid_info_mpi_type();
   init_yt_rma_particle_info_mpi_type();

   g_param_libyt.libyt_initialized = true;

#ifdef SUPPORT_TIMER
   // end timer.
   g_timer->record_time("yt_init", 1);
#endif

   return YT_SUCCESS;

} // FUNCTION : yt_init

static void init_general_info(){
    MPI_Comm_size(MPI_COMM_WORLD, &g_mysize);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_myrank);
}

static void init_yt_long_mpi_type(){
    int length[1] = {1};
    const MPI_Aint displacements[1] = {0};
    MPI_Datatype types[1] = {MPI_LONG};
    MPI_Type_create_struct(1, length, displacements, types, &yt_long_mpi_type);
    MPI_Type_commit(&yt_long_mpi_type);
}

static void init_yt_hierarchy_mpi_type(){
    int lengths[7] = { 3, 3, 1, 1, 3, 1, 1 };
    const MPI_Aint displacements[7] = { 0,
                                        3 * sizeof(double),
                                        6 * sizeof(double),
                                        6 * sizeof(double) + sizeof(long),
                                        6 * sizeof(double) + 2 * sizeof(long),
                                        6 * sizeof(double) + 2 * sizeof(long) + 3 * sizeof(int),
                                        6 * sizeof(double) + 2 * sizeof(long) + 4 * sizeof(int)};
    MPI_Datatype types[7] = { MPI_DOUBLE, MPI_DOUBLE, MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT };
    MPI_Type_create_struct(7, lengths, displacements, types, &yt_hierarchy_mpi_type);
    MPI_Type_commit(&yt_hierarchy_mpi_type);
}

static void init_yt_rma_grid_info_mpi_type(){
    int lengths[5] = {1, 1, 1, 1, 3};
    const MPI_Aint displacements[5] = {0,
                                       1 * sizeof(long),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 1 * sizeof(int),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint) + 2 * sizeof(int)};
    MPI_Datatype types[5] = {MPI_LONG, MPI_AINT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(5, lengths, displacements, types, &yt_rma_grid_info_mpi_type);
    MPI_Type_commit(&yt_rma_grid_info_mpi_type);
}

static void init_yt_rma_particle_info_mpi_type(){
    int lengths[4] = {1, 1, 1, 1};
    const MPI_Aint displacements[4] = {0,
                                       1 * sizeof(long),
                                       1 * sizeof(long) + 1 * sizeof(MPI_Aint),
                                       2 * sizeof(long) + 1 * sizeof(MPI_Aint)};
    MPI_Datatype types[4] = {MPI_LONG, MPI_AINT, MPI_LONG, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types, &yt_rma_particle_info_mpi_type);
    MPI_Type_commit(&yt_rma_particle_info_mpi_type);
}