#include "yt_combo.h"
#include "libyt.h"




//-------------------------------------------------------------------------------------------------------
// Function    :  yt_set_parameter
// Description :  Set YT-specific parameters
//
// Note        :  1. Store yt relavent data in input "param_yt" to libyt.param_yt, not all the data are
//                   passed in.
//                2. Should be called after yt_init().
//                3. Check the validaty of the data in param_yt.
//                4. Organize and generate other information, for later runtime usage.
//
// Parameter   :  param_yt : Structure storing YT-specific parameters that will later pass to YT, and
//                           other relavent data.
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_set_parameter( yt_param_yt *param_yt )
{

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized ){
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );
   }

// check if libyt has free all the resource in previous inline-analysis
   if ( !g_param_libyt.free_gridsPtr ){
      YT_ABORT( "Please invoke yt_free_gridsPtr() before calling %s() for next iteration!\n", __FUNCTION__ );
   }

   log_info( "Setting YT parameters ...\n" );

// check if this function has been called previously
   if ( g_param_libyt.param_yt_set )
   {
      log_warning( "%s() has been called already!\n", __FUNCTION__ );
      log_warning( "==> Are you trying to overwrite existing parameters?\n" );
   }


// reset all cosmological parameters to zero for non-cosmological datasets
   if ( !param_yt->cosmological_simulation ) {
      param_yt->current_redshift = param_yt->omega_lambda = param_yt->omega_matter = param_yt->hubble_constant = 0.0; }


// check if all parameters have been set properly
   if ( param_yt->validate() )
      log_debug( "Validating YT parameters ... done\n" );
   else
      YT_ABORT(  "Validating YT parameters ... failed\n" );


// print out all parameters
   log_debug( "List of YT parameters:\n" );
   param_yt->show();


// store user-provided parameters to a libyt internal variable
// ==> must do this before calling allocate_hierarchy() since it will need "g_param_yt.num_grids"
// ==> must do this before setting the default figure base name since it will overwrite g_param_yt.fig_basename
   g_param_yt = *param_yt;


// set the default figure base name if it's not set by users.
   if ( param_yt->fig_basename == NULL )
   {
      char fig_basename[15];
      sprintf( fig_basename, "Fig%09ld", g_param_libyt.counter );

      g_param_yt.fig_basename = fig_basename;
   }
// append g_param_libyt.counter to prevent over-written
   else {
      char fig_basename[1000];
      sprintf( fig_basename, "%s%09ld", param_yt->fig_basename, g_param_libyt.counter );

      g_param_yt.fig_basename = fig_basename;
   }

// export data to libyt.param_yt
// strings
   add_dict_string(  g_py_param_yt, "frontend",                g_param_yt.frontend                );
   add_dict_string(  g_py_param_yt, "fig_basename",            g_param_yt.fig_basename            );

// scalars
   add_dict_scalar(  g_py_param_yt, "current_time",            g_param_yt.current_time            );
   add_dict_scalar(  g_py_param_yt, "current_redshift",        g_param_yt.current_redshift        );
   add_dict_scalar(  g_py_param_yt, "omega_lambda",            g_param_yt.omega_lambda            );
   add_dict_scalar(  g_py_param_yt, "omega_matter",            g_param_yt.omega_matter            );
   add_dict_scalar(  g_py_param_yt, "hubble_constant",         g_param_yt.hubble_constant         );
   add_dict_scalar(  g_py_param_yt, "length_unit",             g_param_yt.length_unit             );
   add_dict_scalar(  g_py_param_yt, "mass_unit",               g_param_yt.mass_unit               );
   add_dict_scalar(  g_py_param_yt, "time_unit",               g_param_yt.time_unit               );
   add_dict_scalar(  g_py_param_yt, "magnetic_unit",           g_param_yt.magnetic_unit           );
   add_dict_scalar(  g_py_param_yt, "cosmological_simulation", g_param_yt.cosmological_simulation );
   add_dict_scalar(  g_py_param_yt, "dimensionality",          g_param_yt.dimensionality          );
   add_dict_scalar(  g_py_param_yt, "refine_by",               g_param_yt.refine_by               );
   add_dict_scalar(  g_py_param_yt, "num_grids",               g_param_yt.num_grids               );

// vectors (stored as Python tuples)
   add_dict_vector3( g_py_param_yt, "domain_left_edge",        g_param_yt.domain_left_edge        );
   add_dict_vector3( g_py_param_yt, "domain_right_edge",       g_param_yt.domain_right_edge       );
   add_dict_vector3( g_py_param_yt, "periodicity",             g_param_yt.periodicity             );
   add_dict_vector3( g_py_param_yt, "domain_dimensions",       g_param_yt.domain_dimensions       );

   log_debug( "Inserting YT parameters to libyt.param_yt ... done\n" );


// fill libyt.hierarchy with NumPy arrays allocated but uninitialized
   if ( allocate_hierarchy() )
      log_debug( "Allocating libyt.hierarchy ... done\n" );
   else
      YT_ABORT(  "Allocating libyt.hierarchy ... failed!\n" );


// Organize other information, for later runtime usage
   int MyRank;
   MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

   // Make sure g_param_yt.num_grids_local is set, otherwise count from grids_MPI
   if ( g_param_yt.num_grids_local != INT_UNDEFINED ) {
      
      // Prevent input long type, exceed int storage
      if ( g_param_yt.num_grids_local < 0 ){
         YT_ABORT("Number of local grids = %d at MPI rank %d, probably because of exceeding int storage or wrong input!\n",
                   g_param_yt.num_grids_local, MyRank);
      }
   
   }
   else {

      int num_grids_local = 0;
      int flag = 0;
      for ( long i = 0; i < g_param_yt.num_grids; i = i+1 ){
         if ( g_param_yt.grids_MPI[i] == MyRank ) {

            num_grids_local = num_grids_local + 1;
            
            // Prevent exceed int storage
            if ( num_grids_local == INT_MAX ){
               log_warning("Number of local grids at MPI rank %d reach its maximum [%d]!\n", MyRank, INT_MAX);
               flag = 1;
            }
            if (flag == 1){
               YT_ABORT("Number of local grids at MPI rank %d exceed its maximum!\n", MyRank);
            }

         }
      }
      g_param_yt.num_grids_local = num_grids_local;
   }

   // Gather num_grids_local in every rank and store at num_grids_local_MPI, with "MPI_Gather"
   // We need num_grids_local_MPI in MPI_Gatherv in yt_commit_grids()
   int NRank;
   int RootRank = 0;
   MPI_Comm_size(MPI_COMM_WORLD, &NRank);
   int *num_grids_local_MPI = new int [NRank];
   g_param_yt.num_grids_local_MPI = num_grids_local_MPI;

   MPI_Gather(&(g_param_yt.num_grids_local), 1, MPI_INT, num_grids_local_MPI, 1, MPI_INT, RootRank, MPI_COMM_WORLD);
   MPI_Bcast(num_grids_local_MPI, NRank, MPI_INT, RootRank, MPI_COMM_WORLD);

   // Check that sum of num_grids_local_MPI is equal to num_grids (total number of grids)
   long num_grids = 0;
   for (int rid = 0; rid < NRank; rid = rid+1){
      num_grids = num_grids + (long)num_grids_local_MPI[rid];
   }
   if (num_grids != g_param_yt.num_grids){
      YT_ABORT("Sum of grids in each MPI rank [%ld] are not equal to total number of grids [%ld]!\n", 
                num_grids, g_param_yt.num_grids );
   }

// If the above all works like charm.
   g_param_libyt.param_yt_set  = true;
   g_param_libyt.free_gridsPtr = false;
   log_info( "Setting YT parameters ... done.\n" );

   return YT_SUCCESS;

} // FUNCTION : yt_set_parameter
