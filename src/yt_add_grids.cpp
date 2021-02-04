#include "yt_combo.h"
#include "libyt.h"


//-------------------------------------------------------------------------------------------------------
// Function    :  yt_add_grids
// Description :  Add local grids to the libyt Python module
//
// Note        :  1. Store the input "grid" to libyt.hierarchy and libyt.grid_data to python
//                2. Must call yt_set_parameter() in advance, which will  preallocate memory for NumPy arrays.
//                3. Must call yt_get_gridsPtr() in advance, so that g_param_yt knows the grids_local array
//                   pointer.
//                4. Pass the grids and hierarchy to YT in function append_grid()
//
// Parameter   :
//
// Return      :  YT_SUCCESS or YT_FAIL
//-------------------------------------------------------------------------------------------------------
int yt_add_grids()
{

// check if libyt has been initialized
   if ( !g_param_libyt.libyt_initialized )
      YT_ABORT( "Please invoke yt_init() before calling %s()!\n", __FUNCTION__ );

// check if YT parameters have been set
   if ( !g_param_libyt.param_yt_set )
      YT_ABORT( "Please invoke yt_set_parameter() before calling %s()!\n", __FUNCTION__ );

// check if user has call yt_get_gridsPtr(), so that libyt knows the local grids array ptr.
   if ( !g_param_libyt.get_gridsPtr )
      YT_ABORT( "Please invoke yt_get_gridsPtr() before calling %s()!\n", __FUNCTION__ );

// check each grids individually
   for (int i = 0; i < g_param_yt.num_grids_local; i = i+1) {

      yt_grid grid = g_param_yt.grids_local[i];

      // check if all parameters have been set properly and are in used.
      if ( !(grid.validate()) )
         YT_ABORT(  "Validating input grid ID [%ld] ... failed\n", grid.id );

      // additional checks that depend on input YT parameters, and grid itself only.
      // number of fields, although we merge appending num_fields in yt_get_gridsPtr,
      // the user might alter them unintentionally.
      if (grid.num_fields != g_param_yt.num_fields)
         YT_ABORT(  "Grid ID [%ld] number of fields = %ld, should be %ld!\n", 
                     grid.id, grid.num_fields, g_param_yt.num_fields);

      // grid ID
      if (grid.id >= g_param_yt.num_grids)
         YT_ABORT(  "Grid ID [%ld] >= total number of grids [%ld]!\n", 
                     grid.id, g_param_yt.num_grids );
   
      if (grid.parent_id >= g_param_yt.num_grids)
         YT_ABORT(  "Grid [%ld] parent ID [%ld] >= total number of grids [%ld]!\n",
                    grid.id, grid.parent_id, g_param_yt.num_grids );

      if (grid.level > 0 && grid.parent_id < 0)
         YT_ABORT(  "Grid [%ld] parent ID [%ld] < 0 at level [%d]!\n",
                    grid.id, grid.parent_id, grid.level );

      // edge
      for (int d = 0; d < 3; d = d+1) {
         
         if (grid.left_edge[d] < g_param_yt.domain_left_edge[d])
            YT_ABORT( "Grid [%ld] left edge [%13.7e] < domain left edge [%13.7e] along the dimension [%d]!\n",
                      grid.id, grid.left_edge[d], g_param_yt.domain_left_edge[d], d );

         if (grid.right_edge[d] > g_param_yt.domain_right_edge[d])
            YT_ABORT( "Grid [%ld] right edge [%13.7e] > domain right edge [%13.7e] along the dimension [%d]!\n",
                      grid.id, grid.right_edge[d], g_param_yt.domain_right_edge[d], d );
         if (grid.right_edge[d] < grid.left_edge[d])
            YT_ABORT( "Grid [%ld], right edge [%13.7e] < left edge [%13.7e]!\n", 
                      grid.id, grid.right_edge[d], grid.left_edge[d]);
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
   yt_hierarchy *hierarchy_local = new yt_hierarchy [g_param_yt.num_grids_local];
   yt_hierarchy *hierarchy_full  = new yt_hierarchy [g_param_yt.num_grids];

   for (int i = 0; i < g_param_yt.num_grids_local; i = i+1) {

      yt_grid grid = g_param_yt.grids_local[i];

      for (int d = 0; d < 3; d = d+1) {
         hierarchy_local[i].left_edge[d]  = grid.left_edge[d];
         hierarchy_local[i].right_edge[d] = grid.right_edge[d];
         hierarchy_local[i].dimensions[d] = grid.dimensions[d];
      }

      hierarchy_local[i].particle_count = grid.particle_count;
      hierarchy_local[i].id             = grid.id;
      hierarchy_local[i].parent_id      = grid.parent_id;
      hierarchy_local[i].level          = grid.level;
      hierarchy_local[i].proc_num       = grid.proc_num;
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

// We pass hierarchy to each rank as well
   MPI_Bcast(hierarchy_full, g_param_yt.num_grids, yt_hierarchy_mpi_type, RootRank, MPI_COMM_WORLD);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// append grid to YT
// We pass hierarchy to each rank as well.
// Combine full hierarchy and the grid data that one rank has, otherwise fill in NULL in grid data.
   
   yt_grid grid_combine;
   int start_block = offsets[MyRank];
   int end_block = start_block + g_param_yt.num_grids_local;

   for (int i = 0; i < g_param_yt.num_grids; i = i+1) {

      // From hierarchy_full
      for (int d = 0; d < 3; d = d+1) {
         grid_combine.left_edge[d]  = hierarchy_full[i].left_edge[d];
         grid_combine.right_edge[d] = hierarchy_full[i].right_edge[d];
         grid_combine.dimensions[d] = hierarchy_full[i].dimensions[d];
      }
      grid_combine.particle_count = hierarchy_full[i].particle_count;
      grid_combine.id             = hierarchy_full[i].id;
      grid_combine.parent_id      = hierarchy_full[i].parent_id;
      grid_combine.level          = hierarchy_full[i].level;
      grid_combine.proc_num       = hierarchy_full[i].proc_num;

      // From g_param_yt
      grid_combine.num_fields   = g_param_yt.num_fields;
      grid_combine.field_labels = (const char **) g_param_yt.field_labels;      
      grid_combine.field_ftype  = g_param_yt.field_ftype;
      
      // From g_param_yt.grids_local
      if ( start_block <= i && i < end_block ) {
         // Get the pointer to data from grids_local
         grid_combine.field_data   = g_param_yt.grids_local[i - start_block].field_data;
      }
      else {
         // Append each field, and make them points to NULL
         grid_combine.field_data = new void* [g_param_yt.num_fields];
         for ( int v = 0; v < g_param_yt.num_fields; v = v+1 ) {
            grid_combine.field_data[v]   = NULL;
         }
      }

      // Append grid to YT
      append_grid( &grid_combine );
   }

   MPI_Barrier( MPI_COMM_WORLD );

///////////////////////////////////////////////////////////////////////////////////////////////////////////
// check we have all the grids data, when we have loaded all the grids
   if ( check_grids() == YT_SUCCESS ){
      log_debug( "Append grids to libyt.grid_data ... done!\n" );
   }
   else{
      YT_ABORT(  "Append grids to libyt.grid_data ... failed!\n")
   }

   // Freed resource 
   delete [] hierarchy_local;
   delete [] hierarchy_full;
   delete [] recv_counts;
   delete [] offsets;

   // Above all works like charm
   g_param_libyt.add_grids = true;

   return YT_SUCCESS;

} // FUNCTION : yt_add_grids
