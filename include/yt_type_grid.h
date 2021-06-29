#ifndef __YT_TYPE_GRID_H__
#define __YT_TYPE_GRID_H__



/*******************************************************************************
/
/  yt_grid structure
/
/  ==> included by yt_type.h
/
********************************************************************************/


//-------------------------------------------------------------------------------------------------------
// Structure   :  yt_grid
// Description :  Data structure to store a full single grid with data pointer
// 
// Notes       :  1. We assume that each element in array_name[3] are all in use.
//
// Data Member :  dimensions     : Number of cells along each direction in [x][y][z] coordinate.
//                left_edge      : Grid left  edge in code units
//                right_edge     : Grid right edge in code units
//                particle_count : Nunber of particles in this grid
//                level          : AMR level (0 for the root level)
//                id             : Grid ID (0-indexed ==> must be in the range 0 <= id < total number of grids)
//                parent_id      : Parent grid ID (0-indexed, -1 for grids on the root level)
//                proc_num       : Process number, grid belong to which MPI rank
//                field_data     : Pointer arrays pointing to the data of each field
//
// Method      :  yt_grid  : Constructor
//               ~yt_grid  : Destructor
//                validate : Check if all data members have been set properly by users
//-------------------------------------------------------------------------------------------------------
struct yt_grid
{

// data members
// ===================================================================================
   double left_edge[3];
   double right_edge[3];

   long   particle_count;
   long   id;
   long   parent_id;

   int    dimensions[3];
   int    level;
   int    proc_num;

   void   **field_data;

   //===================================================================================
   // Method      :  yt_grid
   // Description :  Constructor of the structure "yt_grid"
   //
   // Note        :  Initialize all data members
   //
   // Parameter   :  None
   //===================================================================================
   yt_grid()
   {

//    set defaults
      for (int d=0; d<3; d++) {
      left_edge [d]  = DBL_UNDEFINED;
      right_edge[d]  = DBL_UNDEFINED; }

      for (int d=0; d<3; d++) {
      dimensions[d]  = INT_UNDEFINED; }

      particle_count = LNG_UNDEFINED;
      id             = LNG_UNDEFINED;
      parent_id      = LNG_UNDEFINED;
      level          = INT_UNDEFINED;

      proc_num       = INT_UNDEFINED;

      field_data     = NULL;

   } // METHOD : yt_grid


   //===================================================================================
   // Method      :  ~yt_grid
   // Description :  Destructor of the structure "yt_grid"
   //
   // Note        :  1. Not used currently
   //                2. We do not free the pointer arrays "field_list" and "field_data" here
   //                   ==> They are freed by yt_free_gridsPtr
   //
   // Parameter   :  None
   //===================================================================================
   ~yt_grid()
   {

   } // METHOD : ~yt_grid


   //===================================================================================
   // Method      :  validate
   // Description :  Check if all data members have been set properly by users
   //
   // Note        :  1. This function does not perform checks that depend on the input
   //                   YT parameters (e.g., whether left_edge lies within the simulation domain)
   //                   ==> These checks are performed in yt_commit_grids()
   //
   // Parameter   :  None
   //
   // Return      :  YT_SUCCESS or YT_FAIL
   //===================================================================================
   int validate() const
   {

      for (int d=0; d<3; d++) {
      if ( left_edge [d]  == DBL_UNDEFINED    )   YT_ABORT( "\"%s[%d]\" has not been set for grid id [%ld]!\n", "left_edge",  d,  id );
      if ( right_edge[d]  == DBL_UNDEFINED    )   YT_ABORT( "\"%s[%d]\" has not been set for grid id [%ld]!\n", "right_edge", d,  id ); }

      for (int d=0; d<3; d++) {
      if ( dimensions[d]  == INT_UNDEFINED    )   YT_ABORT( "\"%s[%d]\" has not been set for grid id [%ld]!\n", "dimensions", d,  id ); }
      if ( particle_count == LNG_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "particle_count", id );
      if ( id             == LNG_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "id",             id );
      if ( parent_id      == LNG_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "parent_id",      id );
      if ( level          == INT_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "level",          id );
      if ( proc_num       == INT_UNDEFINED    )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "proc_num",       id );
      if ( field_data     == NULL             )   YT_ABORT(     "\"%s\" has not been set for grid id [%ld]!\n", "field_data",     id );

//    additional checks
      for (int d=0; d<3; d++) {
      if ( dimensions[d] <= 0 )   YT_ABORT( "\"%s[%d]\" == %d <= 0 for grid [%ld]!\n", "dimensions", d, dimensions[d], id ); }
      if ( particle_count < 0 )   YT_ABORT( "\"%s\" == %d < 0 for grid [%ld]!\n", "particle_count", particle_count, id );
      if ( id < 0 )               YT_ABORT( "\"%s\" == %d < 0!\n", "id", id );
      if ( level < 0 )            YT_ABORT( "\"%s\" == %d < 0 for grid [%ld]!\n", "level", level, id );

      return YT_SUCCESS;

   } // METHOD : validate

}; // struct yt_grid



#endif // #ifndef __YT_TYPE_GRID_H__
