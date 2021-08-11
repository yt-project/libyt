#include "yt_combo.h"
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_commit_grids
// Description :  Add local grids, append field list and particle list info to the libyt Python module.
//
// Note        :  1. Must call yt_set_parameter() in advance, which will preallocate memory for NumPy arrays.
//                2. Must call yt_get_fieldsPtr (if num_fields>0), yt_get_particlesPtr (if num_species>0), 
//                   yt_get_gridsPtr, which gets data info from user.
//                3. Check the local grids, field list, and particle list. 
//                4. Sum up particle_count_list in each individual grid and store in grid_particle_count.
//                5. Force the "cell-centered" field data_dim read from grid_dimensions.
//                6. Append field_list info and particle_list info to libyt.param_yt['field_list'] and 
//                   libyt.param_yt['particle_list'].
//                7. Gather hierarchy in different rank, and check hierarchy in check_hierarchy().
//                8. Pass the grids and hierarchy to YT in function append_grid().
//                9. We assume that one grid contains all the fields belong to that grid.
//
// Parameter   :
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_commit_grids()
{

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized ){
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );
   }

// check if YT parameters have been set
   if ( !g_param_libyt.param_yt_set ){
      YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );
   }

// check if user sets field_list
   if ( !g_param_libyt.get_fieldsPtr ){
      YT_ABORT( "num_fields == %d, please invoke yt_get_fieldsPtr() before calling %s()!\n",
                 g_param_yt.num_fields, __FUNCTION__ );
   }

// check if user sets particle_list
   if ( !g_param_libyt.get_particlesPtr ){
      YT_ABORT( "num_species == %d, please invoke yt_get_particlesPtr() before calling %s()!\n",
                 g_param_yt.num_species, __FUNCTION__ );
   }

// check if user has call yt_get_gridsPtr()
   if ( !g_param_libyt.get_gridsPtr ){
      YT_ABORT( "Please invoke yt_get_gridsPtr() before calling %s()!\n", __FUNCTION__ );
   }

   log_info("Loading grids to yt ...\n");


// Check yt_field* field_list
   if ( g_param_yt.num_fields > 0 ){
      if ( check_field_list() != YT_SUCCESS ){
         YT_ABORT("Check field_list failed in %s!\n", __FUNCTION__);
      }
   }

// Check yt_particle* particle_list
   if ( g_param_yt.num_species > 0 ){
      if ( check_particle_list() != YT_SUCCESS ){
         YT_ABORT("Check particle_list failed in %s!\n", __FUNCTION__);
      }
   }

// Check yt_grid* grids_local
   if ( g_param_yt.num_grids_local > 0 ){
      if ( check_grid() != YT_SUCCESS ){
         YT_ABORT("Check grids_local failed in %s!\n", __FUNCTION__);
      }
   }


// Add field_list to libyt.param_yt['field_list'] dictionary
   if ( g_param_yt.num_fields > 0 ){
      if ( add_dict_field_list() != YT_SUCCESS ){
         YT_ABORT("Inserting dictionary libyt.param_yt['field_list'] failed!\n");
      }
   }

// Add particle_list to libyt.param_yt['particle_list'] dictionary
   if ( g_param_yt.num_species > 0 ){
      if ( add_dict_particle_list() != YT_SUCCESS ){
         YT_ABORT("Inserting dictionary libyt.param_yt['particle_list'] failed!\n");
      }
   }

// Set up grid_particle_count, field_data's data_dim, field_data's data_dtype in grids_local
   for (int i = 0; i < g_param_yt.num_grids_local; i = i+1) {
      yt_grid grid = g_param_yt.grids_local[i];
      
      // (1) Calculate grid_particle_count, 
      // check particle_count_list element > 0 if this array is set.
      // and sum it up. Be careful the copy of struct, we wish to write changes in g_param_yt.grids_local
      g_param_yt.grids_local[i].grid_particle_count = 0;
      for ( int s = 0; s < g_param_yt.num_species; s++ ){
         if ( grid.particle_count_list[s] >= 0 ){
            g_param_yt.grids_local[i].grid_particle_count += grid.particle_count_list[s];
         }
         else{
            YT_ABORT("Grid ID [%ld], particle count == %ld < 0, in particle species [%s]!\n",
                      grid.id, grid.particle_count_list[s], g_param_yt.particle_list[s].species_name);
         }
      }

      // Deal with field_data in grid
      for (int v = 0; v < g_param_yt.num_fields; v++){

         // (2) set data_dim in field_data if field_define_type == "cell-centered"
         if ( strcmp(g_param_yt.field_list[v].field_define_type, "cell-centered") == 0 ){
            // set the field_data data_dim, base on grid_dimensions and swap_axes
            if ( g_param_yt.field_list[v].swap_axes == true ){
               grid.field_data[v].data_dim[0] = grid.grid_dimensions[2];
               grid.field_data[v].data_dim[1] = grid.grid_dimensions[1];
               grid.field_data[v].data_dim[2] = grid.grid_dimensions[0];
            }
            else {
               grid.field_data[v].data_dim[0] = grid.grid_dimensions[0];
               grid.field_data[v].data_dim[1] = grid.grid_dimensions[1];
               grid.field_data[v].data_dim[2] = grid.grid_dimensions[2];
            }
         }

         // (3) Check field_data data_dtype, if it is not one of enum yt_dtype or YT_DTYPE_UNKNOWN, set to field_dtype.
         if ( grid.field_data[v].data_dtype == YT_DTYPE_UNKNOWN ){
            grid.field_data[v].data_dtype = g_param_yt.field_list[v].field_dtype;
         }
         else if ( grid.field_data[v].data_dtype != YT_FLOAT && grid.field_data[v].data_dtype != YT_DOUBLE &&
                   grid.field_data[v].data_dtype != YT_INT ){
            log_warning("Grid [%ld], field_data [%s], data_dtype is not one of YT_FLOAT, YT_DOUBLE, YT_INT, so set to field_dtype.\n", 
                         grid.id, g_param_yt.field_list[v].field_name);
            grid.field_data[v].data_dtype = g_param_yt.field_list[v].field_dtype;
         }
      }
   }

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Prepare to gather full hierarchy from different rank to root rank.
// Get MPI rank and size
   int MyRank;
   int NRank;
   int RootRank = 0;

   MPI_Comm_size(MPI_COMM_WORLD, &NRank);
   MPI_Comm_rank(MPI_COMM_WORLD, &MyRank);

// Create MPI_Datatype for struct yt_hierarchy
   MPI_Datatype yt_hierarchy_mpi_type;
   int lengths[8] = { 3, 3, 1, 1, 1, 3, 1, 1 };
   const MPI_Aint displacements[8] = { 0, 3 * sizeof(double), 6 * sizeof(double),
                                       6 * sizeof(double) + sizeof(long), 6 * sizeof(double) + 2 * sizeof(long), 
                                       6 * sizeof(double) + 3 * sizeof(long),
                                       6 * sizeof(double) + 3 * sizeof(long) + 3 * sizeof(int), 
                                       6 * sizeof(double) + 3 * sizeof(long) + 4 * sizeof(int)};
   MPI_Datatype types[8] = { MPI_DOUBLE, MPI_DOUBLE, MPI_LONG, MPI_LONG, MPI_LONG, MPI_INT, MPI_INT, MPI_INT };
   MPI_Type_create_struct(8, lengths, displacements, types, &yt_hierarchy_mpi_type);
   MPI_Type_commit(&yt_hierarchy_mpi_type);

// Grep hierarchy data from g_param_yt.grids_local, and allocate receive buffer
   yt_hierarchy *hierarchy_full  = new yt_hierarchy [g_param_yt.num_grids];
   yt_hierarchy *hierarchy_local;
   
// To avoid using new [0]
   if ( g_param_yt.num_grids_local > 0 ){
      hierarchy_local = new yt_hierarchy [g_param_yt.num_grids_local];
   }

   for (int i = 0; i < g_param_yt.num_grids_local; i = i+1) {

      yt_grid grid = g_param_yt.grids_local[i];

      for (int d = 0; d < 3; d = d+1) {
         hierarchy_local[i].left_edge[d]  = grid.left_edge[d];
         hierarchy_local[i].right_edge[d] = grid.right_edge[d];
         hierarchy_local[i].dimensions[d] = grid.grid_dimensions[d];
      }

      hierarchy_local[i].grid_particle_count = grid.grid_particle_count;
      hierarchy_local[i].id                  = grid.id;
      hierarchy_local[i].parent_id           = grid.parent_id;
      hierarchy_local[i].level               = grid.level;
      hierarchy_local[i].proc_num            = grid.proc_num;
   }

// MPI_Gatherv to RootRank
// Reference: https://www.rookiehpc.com/mpi/docs/mpi_gatherv.php
   int *recv_counts = new int [NRank]; 
   int *offsets = new int [NRank];
      
   for (int i = 0; i < NRank; i = i+1){
      recv_counts[i] = g_param_yt.num_grids_local_MPI[i];
      offsets[i] = 0;
      for (int j = 0; j < i; j = j+1){
         offsets[i] = offsets[i] + recv_counts[j];

         // Prevent exceeding int storage.
         if ( offsets[i] < 0 ){
            YT_ABORT("Exceeding int storage, libyt not support number of grids larger than %d yet.\n", INT_MAX);
         }
      }
   }

// Not sure if we need this MPI_Barrier
   MPI_Barrier(MPI_COMM_WORLD);

// Gather all the grids hierarchy
   MPI_Gatherv(hierarchy_local, g_param_yt.num_grids_local, yt_hierarchy_mpi_type, 
               hierarchy_full, recv_counts, offsets, yt_hierarchy_mpi_type, RootRank, MPI_COMM_WORLD);

// Check that the hierarchy are correct, do the test on RootRank only
   if ( MyRank == RootRank ){
      if ( check_hierarchy( hierarchy_full ) == YT_SUCCESS ) {
         log_debug("Validating the parent-children relationship ... done!\n");
      }
      else{
         YT_ABORT("Validating the parent-children relationship ... failed!")
      }
   }

// Not sure if we need this MPI_Barrier
   MPI_Barrier(MPI_COMM_WORLD);

// We pass hierarchy to each rank as well. The maximum MPI_Bcast sendcount is INT_MAX.
// If num_grids > INT_MAX chop it to chunks, then broadcast.
   long stride   = INT_MAX;
   int  part     = (int) (g_param_yt.num_grids / stride) + 1;
   int  remain   = (int) (g_param_yt.num_grids % stride);
   long index;
   for (int i=0; i < part; i++){
      index = i * stride;
      if ( i == part - 1 ){
         MPI_Bcast(&(hierarchy_full[index]), remain, yt_hierarchy_mpi_type, RootRank, MPI_COMM_WORLD);
      }
      else {
         MPI_Bcast(&(hierarchy_full[index]), stride, yt_hierarchy_mpi_type, RootRank, MPI_COMM_WORLD);
      }
   }

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// append grid to YT
// We pass hierarchy to each rank as well.
// Combine full hierarchy and the grid data that one rank has, otherwise fill in NULL in grid data.
   
   yt_grid grid_combine;
   int start_block = offsets[MyRank];
   int end_block = start_block + g_param_yt.num_grids_local;

   for (long i = 0; i < g_param_yt.num_grids; i = i+1) {

      // Load from hierarchy_full
      for (int d = 0; d < 3; d = d+1) {
         grid_combine.left_edge[d]       = hierarchy_full[i].left_edge[d];
         grid_combine.right_edge[d]      = hierarchy_full[i].right_edge[d];
         grid_combine.grid_dimensions[d] = hierarchy_full[i].dimensions[d];
      }
      grid_combine.grid_particle_count = hierarchy_full[i].grid_particle_count;
      grid_combine.id                  = hierarchy_full[i].id;
      grid_combine.parent_id           = hierarchy_full[i].parent_id;
      grid_combine.level               = hierarchy_full[i].level;
      grid_combine.proc_num            = hierarchy_full[i].proc_num;
      
      // load from g_param_yt.grids_local
      if ( start_block <= i && i < end_block ) {
         // Get the pointer to data from grids_local
         grid_combine.field_data   = g_param_yt.grids_local[i - start_block].field_data;
      }
      else {
         // Make it points to NULL
         grid_combine.field_data = NULL;
      }

      // Append grid to YT
      append_grid( &grid_combine );
   }

   log_debug( "Append grids to libyt.grid_data ... done!\n" );
   MPI_Barrier( MPI_COMM_WORLD );

   // Freed resource 
   if ( g_param_yt.num_grids_local > 0 ){
      delete [] hierarchy_local;
   }
   delete [] hierarchy_full;
   delete [] recv_counts;
   delete [] offsets;

   // Above all works like charm
   g_param_libyt.commit_grids = true;
   log_info("Loading grids to yt ... done.\n");

   return YT_SUCCESS;

} // FUNCTION : yt_commit_grids
